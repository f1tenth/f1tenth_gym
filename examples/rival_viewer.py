"""Standalone pyglet 1.5 compatibility-OpenGL race viewer (no Gym renderer).

Importing this module, including its geometry/camera/map helpers, never imports
pyglet or opens a display. Constructing RaceViewer *does* need an OpenGL display,
even with visible=False (use Xvfb for offscreen CI). Call update on the window's
owning/main thread; it pumps events and presents one frame, without app.run().

Cars are centred metric rectangles, +X forward, +Y left, yaw about +Z. Their
extrusions are illustrative RC-car geometry, not a vehicle/steering model. Map
selection/threshold/orientation match rival_geometry.TrackMap: base + extension,
vertically flipped pixels >128, YAML resolution and full x/y/yaw origin; YAML
image/negate/occupancy thresholds do not override the simulator's bitmap.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

CAMERA_MODES = ('follow', 'chase', 'isometric', 'topdown')
CYAN = (0.12, 0.86, 0.96)
RED = (1., 0.23, 0.31)
GOLD = (1., 0.76, 0.28)


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return value


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def car_footprint(x, y, yaw, length=.58, width=.31):
    """Counterclockwise XY corners of the exact centred vehicle envelope."""
    x, y, yaw = _finite([x, y, yaw], (3,), 'pose')
    length, width = _positive(length, 'length'), _positive(width, 'width')
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length / 2, width / 2]
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


@dataclass
class MapSurface:
    """Bottom-row-first source pixels and their four world-space outer corners."""
    pixels: np.ndarray
    resolution: float
    origin: np.ndarray

    @property
    def corners(self):
        height, width = self.pixels.shape
        local = np.array([[0, 0], [width, 0], [width, height], [0, height]]) * self.resolution
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        return local @ np.array([[c, s], [-s, c]]) + self.origin[:2]

    def rgb(self):
        """Dark asphalt/free space and charcoal obstacles, from actual pixels.

        The faint edge accent is derived solely from occupied/free neighbours;
        no track width, artificial kerbs, or driveable surface is inferred.
        """
        free = self.pixels > 128
        tone = self.pixels.astype(float) / 255.
        rgb = np.where(free[..., None], [48., 58., 70.], [13., 20., 29.])
        rgb += tone[..., None] * np.where(free[..., None], 7., 3.)
        edge = np.zeros_like(free)
        edge[1:] |= free[1:] != free[:-1]
        edge[:-1] |= free[:-1] != free[1:]
        edge[:, 1:] |= free[:, 1:] != free[:, :-1]
        edge[:, :-1] |= free[:, :-1] != free[:, 1:]
        rgb[edge & free] = [66, 87, 98]
        return np.ascontiguousarray(rgb, dtype=np.uint8)


def load_map_surface(map_base, map_ext):
    """Read the same selected image and metric transform as TrackMap."""
    base = Path(map_base)
    metadata_path = base if base.suffix in ('.yaml', '.yml') else Path(str(base) + '.yaml')
    if base.suffix in ('.yaml', '.yml'):
        base = base.with_suffix('')
    with metadata_path.open() as stream:
        metadata = yaml.safe_load(stream)
    resolution = _positive(metadata['resolution'], 'map resolution')
    origin = _finite(metadata['origin'], (3,), 'map origin')
    extension = map_ext if map_ext.startswith('.') else '.' + map_ext
    with Image.open(str(base) + extension) as image:
        pixels = np.array(image.convert('L'))[::-1].copy()
    return MapSurface(pixels, resolution, origin)


def look_at(eye, focus, up=(0., 0., 1.)) -> np.ndarray:
    """Row-major world-to-eye matrix; rejects degenerate camera bases."""
    eye, focus, up = (_finite(v, (3,), name) for v, name in
                      ((eye, 'eye'), (focus, 'focus'), (up, 'up')))
    forward = focus - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        raise ValueError('eye and focus must differ')
    forward /= norm
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-12:
        raise ValueError('up must not be parallel to the view direction')
    right /= norm
    matrix = np.eye(4)
    matrix[:3, :3] = [right, np.cross(right, forward), -forward]
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def perspective(aspect, fov=52., near=.05, far=2000.) -> np.ndarray:
    """Row-major OpenGL perspective matrix (fov in degrees)."""
    aspect, near, far = (_positive(v, n) for v, n in
                         ((aspect, 'aspect'), (near, 'near'), (far, 'far')))
    if not np.isfinite(fov) or not 0 < fov < 180 or far <= near:
        raise ValueError('Require 0 < fov < 180 and far > near')
    f = 1. / math.tan(math.radians(fov) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])


def camera_pose(poses, mode='isometric', zoom=1.):
    """Return eye/focus/up from actual Nx3 poses, with wrap-safe yaw handling.

    Follow/isometric/topdown frame both cars; chase follows defender only.
    Zoom is distance scale, bounded here and by scroll input to [.3, 6].
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('poses must be a nonempty finite Nx3 array')
    if mode not in CAMERA_MODES:
        raise ValueError(f'Unknown camera mode: {mode}')
    zoom = float(np.clip(_positive(zoom, 'zoom'), .3, 6.))
    forward = np.array([math.cos(poses[0, 2]), math.sin(poses[0, 2]), 0.])
    left = np.array([-forward[1], forward[0], 0.])
    centre = np.r_[np.mean(poses[:, :2], axis=0), .15]
    spread = float(np.max(np.linalg.norm(poses[:, :2] - centre[:2], axis=1)))
    scale = max(1., spread / 3.) * zoom
    up = np.array([0., 0., 1.])
    if mode == 'chase':
        centre = np.r_[poses[0, :2], .18] + forward * 1.2
        eye = np.r_[poses[0, :2], .18] - forward * (3.4 * zoom) + [0, 0, 1.9 * zoom]
    elif mode == 'follow':
        eye = centre - 6. * scale * forward - 3. * scale * left + [0, 0, 5.5 * scale]
    elif mode == 'isometric':
        eye = centre + np.array([-6., -7., 9.]) * scale
    else:
        eye = centre + [0., 0., 13. * scale]
        up = np.array([0., 1., 0.])
    return eye, centre, up


def project_world(point, view, projection, width, height):
    """Return logical screen xy/depth, or None outside the view/behind camera."""
    clip = projection @ view @ np.r_[_finite(point, (3,), 'point'), 1.]
    if clip[3] <= 0:
        return None
    ndc = clip[:3] / clip[3]
    if np.any(np.abs(ndc) > 1):
        return None
    return np.array([(ndc[0] + 1) * width / 2, (ndc[1] + 1) * height / 2, ndc[2]])


def target_on_line(line, position, offset, preview=1.):
    """Target at projected s + preview, offset along the line's LEFT normal."""
    position = _finite(position, (2,), 'position')
    offset, preview = _finite([offset, preview], (2,), 'offset and preview')
    return np.asarray(line.sample(line.project(position)[0] + preview, offset), dtype=float)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def observation_state(obs):
    """Copy the two actual poses/speed magnitudes; never integrate commands."""
    values = []
    for name in ('poses_x', 'poses_y', 'poses_theta', 'linear_vels_x', 'linear_vels_y'):
        array = np.asarray(obs[name], dtype=float)
        if array.ndim != 1 or len(array) < 2 or not np.isfinite(array[:2]).all():
            raise ValueError(f'{name} must contain at least two finite values')
        values.append(array[:2].copy())
    return np.column_stack(values[:3]), np.hypot(values[3], values[4])


def _path_xy(path):
    if path is None:
        return np.empty((0, 2))
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 2 or not np.isfinite(path[:, :2]).all():
        raise ValueError('rollout_path must be a finite Nx2 (or NxM, M >= 2) array')
    return path[:, :2].copy()


class RaceViewer:
    """Synchronous standalone viewer; index 0 defender/cyan, 1 attacker/red.

    update(obs, *, elapsed, decision, attacker_target, progress, lap_goal,
           scenario, defense, target_point=None, rollout_path=None) -> bool

    decision may be an object/dict (target_offset, mode, reason, feasible).
    attacker_target is a lateral offset, NOT an XY point. Targets default to
    each car's projected s + 1 m. target_point overrides the defender marker
    with an exact world XY point. rollout_path optionally accepts Nx2/Nx7 XY-
    first world samples; decision.rollout_path is also accepted. progress is
    two unwrapped signed distances in metres, lap_goal is a lap count.

    C cycles cameras, wheel zooms, H hides HUD AND world overlays, Esc closes.
    False means closed, including all updates after close(). visible=False
    hides the window, not its GL context. capture(path) writes a PNG frame.
    """

    def __init__(self, line, track, map_base, map_ext, car_length=.58,
                 car_width=.31, visible=True):
        self.line, self.track = line, track
        self.car_length = _positive(car_length, 'car_length')
        self.car_width = _positive(car_width, 'car_width')
        self.surface = load_map_surface(map_base, map_ext)
        self._line_points = np.asarray(line.points, dtype=float).copy()
        if (self._line_points.ndim != 2 or self._line_points.shape[1] != 2
                or len(self._line_points) < 3 or not np.isfinite(self._line_points).all()):
            raise ValueError('line.points must be a finite Nx2 closed raceline')
        self._line_length = _positive(line.length, 'line.length')
        # All rendered positions are rebased for stable single-precision GL.
        self._origin = np.mean(self.surface.corners, axis=0)
        self.camera_mode, self.zoom, self.overlays = 'isometric', 1., True
        self._closed, self._frame = False, None
        self._dispatching = False
        self._window, self._texture = None, None
        self._car_lists, self._labels = [], {}
        self._view = self._projection = np.eye(4)

        # Import ONLY here: importing helpers must be safe on display-less hosts.
        import pyglet
        from pyglet import gl
        self._pyglet, self._gl = pyglet, gl
        try:
            try:
                config = gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
                self._window = pyglet.window.Window(1280, 800, caption='RIVAL / RACE CONTROL',
                                                    resizable=True, visible=visible, config=config)
            except pyglet.window.NoSuchConfigException:
                config = gl.Config(double_buffer=True, depth_size=24)
                self._window = pyglet.window.Window(1280, 800, caption='RIVAL / RACE CONTROL',
                                                    resizable=True, visible=visible, config=config)
            self._window.switch_to()
            self._window.set_minimum_size(720, 480)
            self._window.set_vsync(False)
            self._window.push_handlers(on_draw=self._draw, on_close=self._on_close,
                                       on_key_press=self._on_key, on_mouse_scroll=self._on_scroll)
            rgb = self.surface.rgb()
            height, width = rgb.shape[:2]
            image = pyglet.image.ImageData(width, height, 'RGB', rgb.tobytes(), pitch=width * 3)
            self._texture = image.get_texture()
            gl.glBindTexture(self._texture.target, self._texture.id)
            # Nearest filtering preserves the occupancy boundary, not invented tarmac.
            for parameter in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(self._texture.target, parameter, gl.GL_NEAREST)
            gl.glTexParameteri(self._texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(self._texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            for color in (CYAN, RED):
                identifier = gl.glGenLists(1)
                if not identifier:
                    raise RuntimeError('OpenGL could not allocate car geometry')
                self._car_lists.append(identifier)
                gl.glNewList(identifier, gl.GL_COMPILE)
                self._car_mesh(color)
                gl.glEndList()
        except BaseException:
            self.close()
            raise

    @property
    def window(self):
        """Owned pyglet window, e.g. for integration event dispatch/testing."""
        return self._window

    def _on_close(self):
        self.close()
        return self._pyglet.event.EVENT_HANDLED

    def _on_key(self, symbol, modifiers):
        key = self._pyglet.window.key
        if symbol == key.ESCAPE:
            self.close()
        elif symbol == key.C:
            self.camera_mode = CAMERA_MODES[(CAMERA_MODES.index(self.camera_mode) + 1) % len(CAMERA_MODES)]
        elif symbol == key.H:
            self.overlays = not self.overlays
        return self._pyglet.event.EVENT_HANDLED

    def _on_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom = float(np.clip(self.zoom * math.exp(-.12 * float(np.clip(scroll_y, -20, 20))), .3, 6.))

    def update(self, obs, *, elapsed, decision, attacker_target, progress, lap_goal,
               scenario, defense, target_point=None, rollout_path=None) -> bool:
        """Pump events and draw one actual-observation frame; return not closed."""
        if self._closed:
            return False
        assert self._window is not None
        self._window.switch_to()
        self._dispatching = True
        try:
            self._window.dispatch_events()
        finally:
            self._dispatching = False
            if self._closed:
                self._release()
        if self._closed:
            return False
        poses, speeds = observation_state(obs)
        elapsed = float(elapsed)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('elapsed must be finite and nonnegative')
        progress = _finite(progress, (2,), 'progress').copy()
        lap_goal = _positive(lap_goal, 'lap_goal')
        targets = np.array([
            target_on_line(self.line, poses[0, :2], _field(decision, 'target_offset', 0.))
            if target_point is None else _finite(target_point, (2,), 'target_point'),
            target_on_line(self.line, poses[1, :2], attacker_target)])
        path = _path_xy(_field(decision, 'rollout_path') if rollout_path is None else rollout_path)
        self._frame = {'poses': poses, 'speeds': speeds, 'elapsed': elapsed, 'decision': decision,
                       'progress': progress, 'lap_goal': lap_goal, 'scenario': str(scenario),
                       'defense': bool(defense), 'targets': targets, 'path': path}
        self._draw()
        self._window.flip()
        return True

    def capture(self, path) -> Path:
        """Redraw/read the BACK buffer before swap; reliable for hidden windows.

        Call after update. This uses pyglet's PNG writer and does not create
        parent directories. Returns the destination Path.
        """
        if self._closed or self._frame is None:
            raise RuntimeError('capture requires an open viewer with an updated frame')
        assert self._window is not None
        self._window.switch_to()
        self._draw()
        self._gl.glReadBuffer(self._gl.GL_BACK)
        self._gl.glFinish()
        with Path(path).open('wb') as stream:
            self._pyglet.image.get_buffer_manager().get_color_buffer().save(str(path), file=stream)
        return Path(path)

    def close(self) -> None:
        """Release only this viewer's resources/context. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        # Xlib's dispatch_events reads native handles AFTER draining queued
        # events. Destroying the window from a queued Esc/close handler would
        # invalidate those handles while pyglet is still using them.
        if not self._dispatching:
            self._release()

    def _release(self):
        if self._window is not None:
            try:
                self._window.switch_to()
                for label in self._labels.values():
                    label.delete()
                for identifier in self._car_lists:
                    self._gl.glDeleteLists(identifier, 1)
                if self._texture is not None:
                    # pyglet 1.5 Texture has no delete(); non-power-of-two maps
                    # can be TextureRegions, whose owner holds the actual ID.
                    owner = self._texture
                    while hasattr(owner, 'owner'):
                        owner = owner.owner
                    self._gl.glDeleteTextures(1, (self._gl.GLuint * 1)(owner.id))
                    owner.id = 0  # Prevent deferred __del__ from deleting a reused ID.
                    self._texture = None
                self._car_lists.clear()
                self._labels.clear()
            finally:
                self._window.close()

    def _color(self, rgb, alpha=1.):
        self._gl.glColor4f(*rgb, alpha)

    def _box(self, bounds, color):
        """Six outward-normal faces, rather than a flat footprint sprite."""
        gl = self._gl
        x0, x1, y0, y1, z0, z1 = bounds
        vertices = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                    (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        self._color(color)
        gl.glBegin(gl.GL_QUADS)
        for normal, indices in [((0, 0, -1), (3, 2, 1, 0)), ((0, 0, 1), (4, 5, 6, 7)),
                                ((0, -1, 0), (0, 1, 5, 4)), ((1, 0, 0), (1, 2, 6, 5)),
                                ((0, 1, 0), (2, 3, 7, 6)), ((-1, 0, 0), (3, 0, 4, 7))]:
            gl.glNormal3f(*normal)
            for index in indices:
                gl.glVertex3f(*vertices[index])
        gl.glEnd()

    def _wheel(self, x, y, radius, thickness):
        """Cylinder axle parallel to local Y; tyre and alloy hub both extruded."""
        gl = self._gl
        z = radius + .012
        angles = np.linspace(0, 2 * np.pi, 17)
        self._color((.035, .045, .06))
        gl.glBegin(gl.GL_QUAD_STRIP)
        for angle in angles:
            c, s = math.cos(angle), math.sin(angle)
            gl.glNormal3f(c, 0, s)
            for side in (-1, 1):
                gl.glVertex3f(x + radius * c, y + side * thickness / 2, z + radius * s)
        gl.glEnd()
        for side in (-1, 1):
            for size, color, inset in [(radius, (.07, .09, .12), 0.),
                                       (radius * .55, (.48, .56, .63), -.0005)]:
                self._color(color)
                gl.glBegin(gl.GL_TRIANGLE_FAN)
                gl.glNormal3f(0, side, 0)
                yy = y + side * (thickness / 2 + inset)
                gl.glVertex3f(x, yy, z)
                for angle in angles[::side]:
                    gl.glVertex3f(x + size * math.cos(angle), yy, z + size * math.sin(angle))
                gl.glEnd()

    def _car_mesh(self, color):
        length, width = self.car_length, self.car_width
        height = min(length * .24, width * .48)
        # Every component fits INSIDE the supplied centred XY envelope. A thin
        # lower chassis reaches its exact length/width, as used by safety tests.
        self._box((-length / 2, length / 2, -width / 2, width / 2, .025, .05), (.07, .1, .14))
        self._box((-.48 * length, .48 * length, -.34 * width, .34 * width, .05, height), color)
        self._box((-.23 * length, .12 * length, -.27 * width, .27 * width,
                   height, height * 1.65), tuple(v * .55 for v in color))
        self._box((.121 * length, .125 * length, -.23 * width, .23 * width,
                   height * 1.08, height * 1.55), (.09, .17, .24))
        self._box((-.2 * length, .1 * length, -.045 * width, .045 * width,
                   height * 1.65, height * 1.67), (.85, .92, .97))
        # White front and red rear light bars establish heading without making
        # assumptions about steering angle, suspension or F1 aerodynamic parts.
        for y in (-.23 * width, .23 * width):
            self._box((.48 * length, .485 * length, y - .04 * width, y + .04 * width,
                       .065, height * .82), (.86, .96, 1.))
            self._box((-.485 * length, -.48 * length, y - .04 * width, y + .04 * width,
                       .065, height * .82), (1., .08, .12))
        radius, thickness = min(length * .115, width * .23), width * .16
        for x in (-length * .29, length * .29):
            for side in (-1, 1):
                self._wheel(x, side * (width / 2 - thickness / 2), radius, thickness)

    def _polyline(self, points, z, color, alpha=1., closed=False):
        gl = self._gl
        self._color(color, alpha)
        gl.glBegin(gl.GL_LINE_LOOP if closed else gl.GL_LINE_STRIP)
        for x, y in np.asarray(points) - self._origin:
            gl.glVertex3f(x, y, z)
        gl.glEnd()

    def _ring(self, xy, radius, z, color, alpha):
        angle = np.linspace(0, 2 * np.pi, 49)[:-1]
        points = np.column_stack((np.cos(angle), np.sin(angle))) * radius + xy
        self._polyline(points, z, color, alpha, closed=True)

    def _draw(self):
        if self._closed:
            return
        gl, window = self._gl, self._window
        if window is None:
            return
        width, height = window.get_size()
        if width <= 0 or height <= 0:
            return
        viewport = window.get_viewport_size() if hasattr(window, 'get_viewport_size') else (width, height)
        gl.glViewport(0, 0, *viewport)
        gl.glClearColor(.026, .041, .064, 1.)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self._frame is None:
            return
        poses = self._frame['poses'].copy()
        poses[:, :2] -= self._origin
        eye, focus, up = camera_pose(poses, self.camera_mode, self.zoom)
        self._view = look_at(eye, focus, up)
        far = max(200., float(np.linalg.norm(eye)) + float(np.max(
            np.linalg.norm(self.surface.corners - self._origin, axis=1))) * 2 + 100.)
        self._projection = perspective(width / height, far=far)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixf((gl.GLfloat * 16)(*self._projection.T.ravel()))
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadMatrixf((gl.GLfloat * 16)(*self._view.T.ravel()))
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_LIGHTING)

        # Opaque pass: the ONLY ground is the actual metadata-aligned bitmap.
        texture = self._texture
        assert texture is not None
        gl.glEnable(texture.target)
        gl.glBindTexture(texture.target, texture.id)
        self._color((1., 1., 1.))
        gl.glBegin(gl.GL_QUADS)
        for corner, uv in zip(self.surface.corners - self._origin,
                              np.asarray(texture.tex_coords).reshape(4, 3)):
            gl.glTexCoord3f(*uv)
            gl.glVertex3f(*corner, 0.)
        gl.glEnd()
        gl.glDisable(texture.target)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (gl.GLfloat * 4)(.48, .5, .56, 1.))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(-3., -4., 8., 0.))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(.82, .88, 1., 1.))
        for pose, identifier in zip(poses, self._car_lists):
            gl.glPushMatrix()
            gl.glTranslatef(pose[0], pose[1], 0.)
            gl.glRotatef(math.degrees(pose[2]), 0., 0., 1.)
            gl.glCallList(identifier)
            gl.glPopMatrix()
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_COLOR_MATERIAL)

        if self.overlays:
            # Translucent pass: retain depth TEST so cars occlude floor marks;
            # no depth WRITES, and each coplanar layer has a distinct height.
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glLineWidth(1.)
            self._polyline(self._line_points, .008, CYAN, .35, closed=True)
            if len(self._frame['path']) > 1:
                self._polyline(self._frame['path'], .014, GOLD, .9)
            for i, color in enumerate((CYAN, RED)):
                xy, target = self._frame['poses'][i, :2], self._frame['targets'][i]
                self._ring(xy, self.car_length * .72, .01, color, .45)
                self._polyline([xy, target], .018, color, .5)
                self._ring(target, .17, .022, color, .95)
                self._ring(target, .055, .023, color, .8)
                self._polyline([target - [.24, 0], target + [.24, 0]], .025, color, .9)
                self._polyline([target - [0, .24], target + [0, .24]], .025, color, .9)
            gl.glDepthMask(gl.GL_TRUE)
            self._hud(width, height)
        # Explicitly restore the depth-write invariant before next clear/capture.
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

    def _panel(self, x, y, width, height, color, alpha=1.):
        gl = self._gl
        self._color(color, alpha)
        gl.glBegin(gl.GL_QUADS)
        for xx, yy in ((x, y), (x + width, y), (x + width, y + height), (x, y + height)):
            gl.glVertex2f(xx, yy)
        gl.glEnd()

    def _text(self, name, text, x, y, size=11, color=(210, 225, 238, 255), bold=False):
        label = self._labels.get(name)
        if label is None:
            label = self._pyglet.text.Label(font_name='DejaVu Sans', font_size=size,
                                            bold=bold, anchor_y='baseline', color=color)
            self._labels[name] = label
        label.text, label.x, label.y, label.color = str(text), x, y, color
        label.draw()

    def _hud(self, width, height):
        gl, frame = self._gl, self._frame
        if frame is None:
            return
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self._panel(20, height - 92, width - 40, 72, (.025, .043, .07), .95)
        self._panel(20, height - 92, 4, 72, CYAN)
        self._text('title', 'RIVAL', 40, height - 54, 24, bold=True)
        self._text('subtitle', 'RACE CONTROL / LIVE TELEMETRY', 155, height - 50, 10)
        self._text('scenario', frame['scenario'][:65].upper(), 42, height - 76, 10,
                   (133, 158, 181, 255))
        self._text('time', f"{frame['elapsed']:08.2f} s", width - 215, height - 53, 19, bold=True)
        self._text('camera', self.camera_mode.upper(), width - 215, height - 75, 10)

        card_width = min(270, (width - 64) / 2)
        for i, (name, color) in enumerate((('DEFENDER', CYAN), ('ATTACKER', RED))):
            x, y = 20 + i * (card_width + 12), 85
            tint = tuple(int(v * 255) for v in color) + (255,)
            self._panel(x, y, card_width, 118, (.025, .043, .07), .94)
            self._panel(x, y + 115, card_width, 3, color)
            self._text(f'name{i}', f'0{i + 1} / {name}', x + 14, y + 92, 11, tint, True)
            self._text(f'speed{i}', f"{frame['speeds'][i]:.2f}", x + 14, y + 51, 28, bold=True)
            self._text(f'units{i}', 'm/s', x + 126, y + 53, 10)
            laps = frame['progress'][i] / self._line_length
            self._text(f'lap{i}', f"LAPS {laps:.2f} / {frame['lap_goal']:g}   {frame['progress'][i]:.1f} m",
                       x + 14, y + 26, 10)
            self._panel(x + 14, y + 12, card_width - 28, 3, (.12, .17, .22))
            fraction = float(np.clip(laps / frame['lap_goal'], 0., 1.))
            self._panel(x + 14, y + 12, (card_width - 28) * fraction, 3, color)

        decision = frame['decision']
        status = str(_field(decision, 'mode', 'unavailable')).upper()
        feasible = bool(_field(decision, 'feasible', False))
        status_color = (90, 230, 185, 255) if feasible else (255, 180, 85, 255)
        self._panel(20, 20, width - 40, 55, (.025, .043, .07), .96)
        self._text('status', f"{status[:24]} / DEFENSE {'ON' if frame['defense'] else 'OFF'}",
                   34, 53, 10, status_color, True)
        reason = str(_field(decision, 'reason', 'No decision telemetry'))
        self._text('reason', reason[:max(28, int((width - 85) / 7.5))], 34, 32, 10)
        if width >= 1000:
            x, y = width - 290, 85
            self._panel(x, y, 270, 150, (.025, .043, .07), .94)
            self._panel(x, y + 147, 270, 3, GOLD)
            self._text('strategy_title', 'STRATEGY / MODEL CHECKS', x + 14, y + 123, 10,
                       (255, 206, 117, 255), True)
            phase = str(_field(decision, 'racecraft_phase', 'unavailable')).upper()
            self._text('strategy_phase', phase[:22], x + 14, y + 95, 16, bold=True)
            target = float(_field(decision, 'target_offset', 0.))
            self._text('strategy_target', f'TARGET OFFSET  {target:+.2f} m', x + 14, y + 72, 10)
            move = 'USED' if _field(decision, 'racecraft_move_used', False) else 'AVAILABLE / GATED'
            self._text('strategy_move', f'DEFENSIVE MOVE  {move}', x + 14, y + 49, 9)
            self._text('strategy_check', 'SAMPLED CHECKS PASS' if feasible else 'UNRESOLVED / CHECK LIMIT',
                       x + 14, y + 27, 9, status_color)
            self._text('strategy_warning', 'SIMULATION ONLY / NOT SAFETY PROOF',
                       x + 14, y + 10, 8, (145, 167, 187, 255))
        self._text('keys', 'C camera   /   SCROLL zoom   /   H overlays   /   ESC close',
                   24, 218, 9, (145, 167, 187, 255))
        for i, (name, color) in enumerate((('DEFENDER', CYAN), ('ATTACKER', RED))):
            xy = frame['poses'][i, :2] - self._origin
            screen = project_world([*xy, .42], self._view, self._projection, width, height)
            if screen is not None and 245 < screen[1] < height - 112:
                xx = float(np.clip(screen[0] - 45, 24, width - 135))
                yy = float(screen[1] + 12)
                self._panel(xx - 6, yy - 5, 112, 23, (.025, .043, .07), .9)
                self._text(f'floating{i}', f'0{i + 1} {name}', xx, yy, 9,
                           tuple(int(v * 255) for v in color) + (255,), True)
