"""Focused tests; no display required unless RIVAL_VIEWER_GL_TEST=1.

    .venv-py315/bin/python -m unittest discover -s examples -p test_rival_viewer.py
    RIVAL_VIEWER_GL_TEST=1 xvfb-run -a .venv-py315/bin/python -m unittest discover -s examples -p test_rival_viewer.py

The opt-in GL smoke creates hidden pyglet 1.5 windows; no Gym environment,
physics simulation, controller, or full project test suite is involved.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image
from rival_geometry import ClosedRaceline, TrackMap
from rival_viewer import (
    CAMERA_MODES,
    CYAN,
    RaceViewer,
    _path_xy,
    camera_pose,
    car_footprint,
    load_map_surface,
    look_at,
    observation_state,
    perspective,
    project_world,
    target_on_line,
)

EXAMPLES = Path(__file__).resolve().parent


def observation():
    return {'poses_x': [1., 2.], 'poses_y': [1., 1.], 'poses_theta': [0., .2],
            'linear_vels_x': [3., 4.], 'linear_vels_y': [4., 0.]}


def telemetry():
    return {'elapsed': 2.5, 'decision': SimpleNamespace(target_offset=.2, mode='follow',
            reason='Sampled path telemetry; not a safety certificate', feasible=True),
            'attacker_target': -.15, 'progress': [3., 2.], 'lap_goal': 2,
            'scenario': 'viewer smoke', 'defense': True}


class ImportTests(unittest.TestCase):
    def test_import_and_helpers_do_not_import_pyglet_or_gym(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('pyglet', 'gym', 'f110_gym'):
        raise AssertionError('Unexpected renderer import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import rival_viewer as v
v.car_footprint(0, 0, 0)
v.look_at([2, 2, 2], [0, 0, 0])
v.camera_pose([[0, 0, 0]])
'''
        env = dict(os.environ)
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
        result = subprocess.run([sys.executable, '-c', code], cwd=EXAMPLES,
                                env=env, text=True, capture_output=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.line = ClosedRaceline([[0, 0], [4, 0], [4, 4], [0, 4]])

    def test_footprint_centred_dimensions_heading_and_area(self):
        for yaw in (0., np.pi / 2, -.73):
            corners = car_footprint(12, -8, yaw)
            np.testing.assert_allclose(corners.mean(axis=0), [12, -8])
            self.assertAlmostEqual(float(np.linalg.norm(corners[1] - corners[0])), .58)
            self.assertAlmostEqual(float(np.linalg.norm(corners[2] - corners[1])), .31)
            area = np.sum(corners[:, 0] * np.roll(corners[:, 1], -1)
                          - corners[:, 1] * np.roll(corners[:, 0], -1)) / 2
            self.assertAlmostEqual(float(area), .58 * .31)
        corners = car_footprint(0, 0, np.pi / 2)
        np.testing.assert_allclose(np.ptp(corners, axis=0), [.31, .58])

    def test_full_car_mesh_including_wheels_fits_exact_xy_envelope(self):
        class VertexRecorder:
            GL_QUADS, GL_QUAD_STRIP, GL_TRIANGLE_FAN = 1, 2, 3

            def __init__(self):
                self.vertices = []

            def glVertex3f(self, *xyz):
                self.vertices.append(xyz)

            def glColor4f(self, *args):
                pass

            def glBegin(self, *args):
                pass

            def glEnd(self):
                pass

            def glNormal3f(self, *args):
                pass

        viewer = RaceViewer.__new__(RaceViewer)
        viewer.car_length, viewer.car_width = .58, .31
        recorder = VertexRecorder()
        with patch.object(viewer, '_gl', recorder, create=True):
            viewer._car_mesh(CYAN)
        vertices = np.asarray(recorder.vertices)
        self.assertGreater(len(vertices), 400)
        np.testing.assert_allclose(vertices[:, :2].min(axis=0), [-.29, -.155])
        np.testing.assert_allclose(vertices[:, :2].max(axis=0), [.29, .155])
        self.assertGreater(vertices[:, 2].max(), .15)
        self.assertGreaterEqual(vertices[:, 2].min(), 0.)

    def test_targets_are_previewed_line_offsets_not_global_coordinates(self):
        np.testing.assert_allclose(target_on_line(self.line, [1, 0], .3), [2, .3])
        # The preview wraps at the end, and uses the outgoing segment normal.
        np.testing.assert_allclose(target_on_line(self.line, [0, .5], -.2), [.5, -.2])
        np.testing.assert_allclose(target_on_line(self.line, [4, 1], .2), [3.8, 2])

    def test_observation_actual_velocity_magnitude_and_copies(self):
        obs = observation()
        poses, speeds = observation_state(obs)
        np.testing.assert_allclose(poses, [[1, 1, 0], [2, 1, .2]])
        np.testing.assert_allclose(speeds, [5, 4])
        obs['poses_x'][0] = 900
        self.assertEqual(poses[0, 0], 1)
        obs['linear_vels_y'][0] = float('nan')
        with self.assertRaises(ValueError):
            observation_state(obs)
        with self.assertRaises(ValueError):
            observation_state({**observation(), 'poses_x': [1.]})

    def test_optional_rollout_xy_first_and_copy(self):
        path = np.arange(21.).reshape(3, 7)
        xy = _path_xy(path)
        np.testing.assert_array_equal(xy, path[:, :2])
        path[0, 0] = -99
        self.assertEqual(xy[0, 0], 0)
        self.assertEqual(_path_xy(None).shape, (0, 2))
        self.assertEqual(_path_xy(np.empty((0, 7))).shape, (0, 2))
        for bad in ([1, 2], [[1]], [[np.nan, 2]]):
            with self.assertRaises(ValueError):
                _path_xy(bad)

    def test_invalid_geometry(self):
        for kwargs in ({'length': 0}, {'width': -1}, {'yaw': float('nan')}):
            args = {'x': 0., 'y': 0., 'yaw': 0., **kwargs}
            with self.assertRaises(ValueError):
                car_footprint(**args)


class CameraTests(unittest.TestCase):
    def test_look_at_basis_and_projection(self):
        eye, focus = np.array([2., -4., 3.]), np.array([1., 1., 0.])
        view = look_at(eye, focus)
        np.testing.assert_allclose(view @ np.r_[eye, 1.], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(view[:3, :3] @ view[:3, :3].T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(view[:3, :3])), 1.)
        projection = perspective(16 / 10)
        screen = project_world(focus, view, projection, 1280, 800)
        assert screen is not None
        np.testing.assert_allclose(screen[:2], [640, 400])
        self.assertIsNone(project_world(eye + (eye - focus), view, projection, 1280, 800))
        self.assertIsNone(project_world([1e6, 0, 0], view, projection, 1280, 800))

    def test_every_camera_zoom_and_yaw_wrap_is_finite(self):
        for mode in CAMERA_MODES:
            for zoom in (.01, .3, 1., 6., 100.):
                poses = np.array([[1., 2., np.pi - 1e-8], [2., 3., 0.]])
                eye, focus, up = camera_pose(poses, mode, zoom)
                self.assertTrue(np.isfinite(look_at(eye, focus, up)).all())
                poses[0, 2] -= 2 * np.pi
                eye2, focus2, up2 = camera_pose(poses, mode, zoom)
                np.testing.assert_allclose(eye, eye2, atol=1e-12)
                np.testing.assert_allclose(focus, focus2, atol=1e-12)
                np.testing.assert_array_equal(up, up2)

    def test_pair_cameras_expand_for_separation(self):
        for mode in ('follow', 'isometric', 'topdown'):
            near = camera_pose([[0, 0, 0], [1, 0, 0]], mode)
            far = camera_pose([[0, 0, 0], [30, 0, 0]], mode)
            self.assertGreater(np.linalg.norm(far[0] - far[1]), np.linalg.norm(near[0] - near[1]))
            view = look_at(*far)
            projection = perspective(1.6)
            for point in ([0, 0, .1], [30, 0, .1]):
                self.assertIsNotNone(project_world(point, view, projection, 1280, 800))

    def test_degenerate_inputs_rejected(self):
        for args in (([0, 0, 0], [0, 0, 0]), ([0, 0, 1], [0, 0, 0], [0, 0, 1])):
            with self.assertRaises(ValueError):
                look_at(*args)
        for kwargs in ({'aspect': 0}, {'aspect': 1, 'near': 2, 'far': 1},
                       {'aspect': 1, 'fov': 180}):
            with self.assertRaises(ValueError):
                perspective(**kwargs)
        for poses in ([], [[0, 0]], [[np.nan, 0, 0]]):
            with self.assertRaises(ValueError):
                camera_pose(poses)
        with self.assertRaises(ValueError):
            camera_pose([[0, 0, 0]], 'not a camera')


class MapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'map'
        # Asymmetric source catches transposition, wrong vertical flip, and a
        # threshold of >=128 instead of TrackMap's strict >128.
        self.source = np.array([[0, 128, 129], [255, 40, 200]], dtype=np.uint8)
        Image.fromarray(self.source).save(str(self.base) + '.png')
        self.metadata = {'resolution': .5, 'origin': [10., -3., np.pi / 2],
                         'image': 'deliberately-not-selected.png', 'negate': 1,
                         'occupied_thresh': .99, 'free_thresh': .01}
        self.metadata_path = Path(str(self.base) + '.yaml')
        self.metadata_path.write_text(yaml.safe_dump(self.metadata))

    def test_bitmap_flip_metric_origin_yaw_and_selected_image(self):
        surface = load_map_surface(self.base, 'png')
        np.testing.assert_array_equal(surface.pixels, self.source[::-1])
        np.testing.assert_allclose(surface.corners, [[10, -3], [10, -1.5], [9, -1.5], [9, -3]])
        np.testing.assert_array_equal(load_map_surface(self.metadata_path, '.png').pixels, surface.pixels)
        track = TrackMap(self.base, '.png')
        np.testing.assert_array_equal(surface.pixels > 128, track._free)
        np.testing.assert_array_equal(surface.origin, track.origin)
        self.assertEqual(surface.resolution, track.resolution)

    def test_texture_preserves_occupied_and_free_topology(self):
        surface = load_map_surface(self.base, '.png')
        rgb = surface.rgb()
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb.mean(axis=2) > 40, surface.pixels > 128)
        self.assertTrue(rgb.flags.c_contiguous)

    def test_invalid_metadata(self):
        for key, value in (('resolution', 0), ('resolution', float('nan')),
                           ('origin', [0, 0]), ('origin', [0, 0, float('inf')])):
            metadata = {**self.metadata, key: value}
            self.metadata_path.write_text(yaml.safe_dump(metadata))
            with self.assertRaises(ValueError):
                load_map_surface(self.base, '.png')


class LifecycleTests(unittest.TestCase):
    def test_closed_viewer_never_needs_gl_or_valid_observation(self):
        viewer = RaceViewer.__new__(RaceViewer)
        viewer._closed = True
        self.assertFalse(viewer.update(None, **telemetry()))
        viewer.close()
        with self.assertRaises(RuntimeError):
            viewer.capture('unused.png')


@unittest.skipUnless(os.environ.get('RIVAL_VIEWER_GL_TEST') == '1',
                     'Opt in with RIVAL_VIEWER_GL_TEST=1 on a display/Xvfb')
class OpenGLSmokeTests(unittest.TestCase):
    def test_hidden_window_all_cameras_capture_events_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'map'
            pixels = np.zeros((96, 96), dtype=np.uint8)
            pixels[8:88, 8:88] = 255
            Image.fromarray(pixels).save(str(base) + '.png')
            Path(str(base) + '.yaml').write_text(yaml.safe_dump(
                {'resolution': .05, 'origin': [-1., -1., .1]}))
            line = ClosedRaceline([[0, 0], [3, 0], [3, 3], [0, 3]])
            viewer = RaceViewer(line, TrackMap(base), base, '.png', visible=False)
            window = viewer.window
            assert window is not None
            try:
                from pyglet import gl
                from pyglet.window import key
                for mode in CAMERA_MODES:
                    viewer.camera_mode = mode
                    self.assertTrue(viewer.update(observation(), **telemetry(),
                        target_point=[2.1, 1.3], rollout_path=[[1, 1], [1.5, 1.1], [2, 1.3]]))
                    assert viewer._frame is not None
                    np.testing.assert_allclose(viewer._frame['targets'][0], [2.1, 1.3])
                    image_path = viewer.capture(Path(directory) / (mode + '.png'))
                    with Image.open(image_path) as image:
                        self.assertEqual(image.size, window.get_viewport_size())
                        rgb = np.asarray(image.convert('RGB'))
                        # More than a clear colour: asphalt, lit cars and HUD.
                        self.assertGreater(np.std(rgb.astype(float)), 15)
                        self.assertGreater(np.count_nonzero((rgb[:, :, 1] > 150)
                                                           & (rgb[:, :, 2] > 150)), 500)
                    self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
                window.dispatch_event('on_key_press', key.C, 0)
                self.assertTrue(viewer.update(observation(), **telemetry()))
                self.assertEqual(viewer.camera_mode, 'follow')
                window.dispatch_event('on_mouse_scroll', 0, 0, 0, 100)
                window.dispatch_event('on_key_press', key.H, 0)
                self.assertTrue(viewer.update(observation(), **telemetry()))
                self.assertGreaterEqual(viewer.zoom, .3)
                self.assertLess(viewer.zoom, 1.)
                self.assertFalse(viewer.overlays)
                viewer.capture(Path(directory) / 'clean.png')
                window.dispatch_event('on_key_press', key.ESCAPE, 0)
                self.assertFalse(viewer.update(None, **telemetry()))
                viewer.close()
            finally:
                viewer.close()
            # OS/window-manager close takes the same idempotent path.
            viewer = RaceViewer(line, None, base, '.png', visible=False)
            try:
                window = viewer.window
                assert window is not None
                window.dispatch_event('on_close')
                self.assertFalse(viewer.update(None, **telemetry()))
            finally:
                viewer.close()


if __name__ == '__main__':
    unittest.main()
