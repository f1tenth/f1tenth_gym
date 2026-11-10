"""Closed-polyline geometry and conservative raster-map safety queries.

Distances are metres; angles are radians. Positive lateral offset and curvature
mean left of travel. Map paths follow config_example_map.yaml: resolve its
map_path relative to examples (or pass an absolute base), with optional .yaml.
Like the laser model, the image is flipped top-to-bottom (not matrix-transposed),
then pixels >128 are free. YAML image/negate/occupancy thresholds are deliberately
not used: map_base + map_ext selects the bitmap, matching the simulator.

Pixels represent square areas, with YAML origin at the lower-left outer corner.
Clearance is a conservative EDT-derived lower bound, not subpixel-exact distance:
we subtract two pixel circumradii from a bilinearly interpolated EDT (one
for obstacle area, one for interpolation distance).
Outside the bitmap is occupied. These choices can reject genuinely safe poses,
but never rely on pixel-centre distance as clearance to an obstacle surface.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml


def _xy(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError('Expected a finite two-dimensional position')
    return value


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


class ClosedRaceline:
    """Periodic piecewise-linear raceline; a repeated closing point is removed.

    Other consecutive duplicates are rejected. Projection ties choose the first
    segment. At a vertex heading uses the outgoing segment. Curvature is the
    linearly interpolated signed three-vertex circumcircle estimate, since an
    exact polygon has zero curvature on edges and undefined curvature at corners.
    """

    def __init__(self, points):
        points = np.array(points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError('points must be a finite Nx2 array')
        if len(points) > 1 and np.array_equal(points[0], points[-1]):
            points = points[:-1].copy()
        if len(points) < 3:
            raise ValueError('At least three vertices are required')
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths == 0):
            raise ValueError('Consecutive duplicate vertices are not allowed')
        self._points = points
        self._edges = edges
        self._lengths = lengths
        self._starts = np.r_[0., np.cumsum(lengths)]
        self._tangents = edges / lengths[:, None]
        previous = np.roll(edges, 1, axis=0)
        chords = np.linalg.norm(previous + edges, axis=1)
        denominator = np.roll(lengths, 1) * lengths * chords
        cross = previous[:, 0] * edges[:, 1] - previous[:, 1] * edges[:, 0]
        self._curvatures = np.divide(2 * cross, denominator,
                                     out=np.zeros_like(cross), where=denominator > 0)
        self._points.flags.writeable = False

    @property
    def points(self):
        """Read-only vertices, without a repeated closing endpoint."""
        return self._points.view()

    @property
    def length(self):
        return float(self._starts[-1])

    def _segment(self, s):
        if not np.isfinite(s):
            raise ValueError('s must be finite')
        s = float(s) % self.length
        i = min(int(np.searchsorted(self._starts, s, side='right') - 1),
                len(self._points) - 1)
        return i, (s - self._starts[i]) / self._lengths[i]

    def project(self, position):
        """Return nearest arc length and signed normal displacement (s, d)."""
        relative = _xy(position) - self._points
        fractions = np.clip(np.sum(relative * self._edges, axis=1)
                            / self._lengths ** 2, 0., 1.)
        residual = relative - fractions[:, None] * self._edges
        i = int(np.argmin(np.sum(residual ** 2, axis=1)))
        tangent = self._tangents[i]
        d = tangent[0] * residual[i, 1] - tangent[1] * residual[i, 0]
        s = (self._starts[i] + fractions[i] * self._lengths[i]) % self.length
        return float(s), float(d)

    def sample(self, s, d=0):
        """Return xy at wrapped arc length, shifted along the left normal."""
        if not np.isfinite(d):
            raise ValueError('d must be finite')
        i, fraction = self._segment(s)
        tangent = self._tangents[i]
        return (self._points[i] + fraction * self._edges[i]
                + float(d) * np.array([-tangent[1], tangent[0]]))

    def heading(self, s):
        i, _ = self._segment(s)
        return float(np.arctan2(self._edges[i, 1], self._edges[i, 0]))

    def curvature(self, s):
        i, fraction = self._segment(s)
        return float((1 - fraction) * self._curvatures[i]
                     + fraction * self._curvatures[(i + 1) % len(self._points)])

    def signed_distance(self, target_s, origin_s):
        """Signed shortest arc displacement in [-length/2, length/2)."""
        if not np.isfinite(target_s) or not np.isfinite(origin_s):
            raise ValueError('Arc lengths must be finite')
        return float((target_s - origin_s + self.length / 2) % self.length
                     - self.length / 2)


class TrackMap:
    """Laser-compatible bitmap map with conservative metric clearance."""

    def __init__(self, map_base, map_ext='.png'):
        base = Path(map_base)
        if base.suffix in ('.yaml', '.yml'):
            metadata_path = base
            base = base.with_suffix('')
        else:
            metadata_path = Path(str(base) + '.yaml')
        with metadata_path.open() as stream:
            metadata = yaml.safe_load(stream)
        self.resolution = float(metadata['resolution'])
        self.origin = np.asarray(metadata['origin'], dtype=float)
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError('Map resolution must be finite and positive')
        if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
            raise ValueError('Map origin must contain finite x, y, yaw')
        extension = map_ext if map_ext.startswith('.') else '.' + map_ext
        with Image.open(str(base) + extension) as image:
            pixels = np.asarray(image.convert('L'))[::-1]
        self._free = pixels > 128
        self._height, self._width = self._free.shape
        self._dt = distance_transform_edt(np.pad(self._free, 1)) * self.resolution
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        self._world_to_map = np.array([[c, s], [-s, c]])

    def clear(self, position):
        """Conservative distance to occupied pixel areas/outside; zero if blocked."""
        return float(self.clear_many(_xy(position)[None, :])[0])

    def clear_many(self, points):
        """Return conservative clearances as an (N,) array for finite Nx2 points.

        Uses the same EDT bound as clear(); outside points return zero. An empty
        (0, 2) array is accepted. Invalid shapes/nonfinite points raise ValueError.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError('points must be a finite Nx2 array')
        local = (points - self.origin[:2]) @ self._world_to_map.T
        extent = np.array([self._width, self._height]) * self.resolution
        inside = np.all((local >= 0) & (local < extent), axis=1)
        result = np.zeros(len(points), dtype=float)
        local = local[inside]
        grid = local / self.resolution + .5  # Padded pixel-centre coordinates.
        col, row = np.floor(grid).astype(int).T
        tx, ty = (grid - np.column_stack((col, row))).T
        interpolated = ((1 - ty) * ((1 - tx) * self._dt[row, col]
                                  + tx * self._dt[row, col + 1])
                        + ty * ((1 - tx) * self._dt[row + 1, col]
                                + tx * self._dt[row + 1, col + 1]))
        raster_bound = interpolated - np.sqrt(2) * self.resolution
        boundary = np.min(np.minimum(local, extent - local), axis=1)
        result[inside] = np.maximum(0., np.minimum(raster_bound, boundary))
        return result

    def safe(self, position, clearance):
        clearance = _nonnegative(clearance, 'clearance')
        distance = self.clear(position)
        return bool(distance > 0 and distance >= clearance)

    def corridor(self, line, s, clearance, max_offset=3., step=.05):
        """Return connected safe offsets (lo, hi) containing zero.

        Raise ValueError if the reference is unsafe. Search stops at max_offset
        or the first uncertifiable interval, never jumping to a separate free
        region. Each interval is certified using midpoint clearance plus its
        half-length times sqrt(2) (the interpolated bound is sqrt(2)-Lipschitz).
        Both sides' candidate intervals are checked in one batch; only the
        first failure on each side is bisected to tolerance <=1e-6 metres.
        Bounds may be conservative
        because clear() itself is a raster-derived lower bound.
        """
        clearance = _nonnegative(clearance, 'clearance')
        max_offset = _nonnegative(max_offset, 'max_offset')
        step = _nonnegative(step, 'step')
        if step == 0:
            raise ValueError('step must be positive')
        centre = line.sample(s)
        normal = line.sample(s, 1.) - centre
        step = min(step, self.resolution)
        count = int(np.ceil(max_offset / step))
        edges = np.minimum(np.arange(count + 1) * step, max_offset)
        spans = np.diff(edges)
        midpoints = edges[:-1] + spans / 2
        directions = np.array([-1., 1.])
        offsets = (directions[:, None] * midpoints).ravel()
        positions = centre + offsets[:, None] * normal
        distances = self.clear_many(np.vstack((centre, positions)))
        if distances[0] <= 0 or distances[0] < clearance:
            raise ValueError('Reference point has insufficient clearance')
        if count == 0:
            return 0., 0.
        required = clearance + np.sqrt(2) * spans / 2
        certified = distances[1:].reshape(2, count) >= required
        # Only the prefix before the first failure is reachable from zero.
        failed = ~certified
        has_failure = np.any(failed, axis=1)
        first = np.argmax(failed, axis=1)
        bounds = np.full(2, max_offset)
        sides = np.flatnonzero(has_failure)
        base = edges[first[sides]]
        low = np.zeros(len(sides))
        high = spans[first[sides]].copy()
        while np.any(high - low > 1e-6):
            candidate = (low + high) / 2
            offsets = directions[sides] * (base + candidate / 2)
            distances = self.clear_many(centre + offsets[:, None] * normal)
            certified = distances >= clearance + np.sqrt(2) * candidate / 2
            low = np.where(certified, candidate, low)
            high = np.where(certified, high, candidate)
        bounds[sides] = base + low
        return float(-bounds[0]), float(bounds[1])

    def footprints_safe(self, poses, length, width, margin):
        """Certify finite Nx3 (x, y, yaw) poses with footprint_safe's grid.

        Return an (N,) boolean array using one clear_many query across all
        rectangles. Empty (0, 3) input returns an empty array without a query.
        Dimensions and margin must be finite and nonnegative, even when empty.
        """
        poses = np.asarray(poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3 or not np.all(np.isfinite(poses)):
            raise ValueError('poses must be a finite Nx3 array')
        length = _nonnegative(length, 'length')
        width = _nonnegative(width, 'width')
        margin = _nonnegative(margin, 'margin')
        if len(poses) == 0:
            return np.empty(0, dtype=bool)
        spacing = self.resolution / 2
        nx = max(1, int(np.ceil(length / spacing)))
        ny = max(1, int(np.ceil(width / spacing)))
        dx, dy = length / nx, width / ny
        required = margin + np.hypot(dx, dy) / 2
        x = -length / 2 + (np.arange(nx) + .5) * dx
        y = -width / 2 + (np.arange(ny) + .5) * dy
        gx, gy = np.meshgrid(x, y, indexing='ij')
        local = np.column_stack((gx.ravel(), gy.ravel()))
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        rotations = np.stack((np.stack((c, -s), axis=-1),
                              np.stack((s, c), axis=-1)), axis=1)
        samples = local @ rotations.swapaxes(1, 2) + poses[:, None, :2]
        distances = self.clear_many(samples.reshape(-1, 2)).reshape(len(poses), -1)
        return np.all((distances > 0) & (distances >= required), axis=1)

    def footprint_safe(self, position, yaw, length, width, margin):
        """Conservatively certify the whole centred rectangle plus margin.

        Tile the local rectangle with cells of side lengths <= resolution/2
        and query their centres after rotating by yaw. Each centre must have
        clearance >= margin plus the cell's half-diagonal: every rectangle
        point is within that covering radius of a sample, so the 1-Lipschitz
        property of true obstacle distance certifies the entire interior.
        Length/width are full dimensions; zero dimensions are supported.
        """
        if not np.isfinite(yaw):
            raise ValueError('yaw must be finite')
        length = _nonnegative(length, 'length')
        width = _nonnegative(width, 'width')
        margin = _nonnegative(margin, 'margin')
        position = _xy(position)
        spacing = self.resolution / 2
        nx = max(1, int(np.ceil(length / spacing)))
        ny = max(1, int(np.ceil(width / spacing)))
        dx, dy = length / nx, width / ny
        required = margin + np.hypot(dx, dy) / 2
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        x = -length / 2 + (np.arange(nx) + .5) * dx
        y = -width / 2 + (np.arange(ny) + .5) * dy
        gx, gy = np.meshgrid(x, y, indexing='ij')
        samples = np.column_stack((gx.ravel(), gy.ravel())) @ rotation.T + position
        distances = self.clear_many(samples)
        return bool(np.all((distances > 0) & (distances >= required)))
