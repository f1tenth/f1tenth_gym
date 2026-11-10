"""Run with python -m unittest discover -s examples -p test_rival_geometry.py."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from rival_geometry import ClosedRaceline, TrackMap


class RacelineTests(unittest.TestCase):
    def setUp(self):
        self.line = ClosedRaceline([[0, 0], [4, 0], [4, 4], [0, 4]])

    def test_wrap_sample_and_distance(self):
        self.assertEqual(self.line.length, 16.)
        np.testing.assert_allclose(self.line.sample(-1), [0, 1])
        np.testing.assert_allclose(self.line.sample(17, .5), [1, .5])
        self.assertAlmostEqual(self.line.signed_distance(.2, 15.8), .4)
        self.assertAlmostEqual(self.line.signed_distance(15.8, .2), -.4)
        self.assertEqual(self.line.signed_distance(8, 0), -8.)
        self.assertEqual(self.line.signed_distance(40, 0), -8.)
        self.assertAlmostEqual(self.line.heading(16), 0.)
        self.assertAlmostEqual(self.line.heading(-1), -np.pi / 2)

    def test_projection_sign_and_closing_segment(self):
        for position, expected in [([2, .5], (2, .5)),
                                   ([2, -.5], (2, -.5)),
                                   ([.25, 1], (15, .25)),
                                   ([-.25, 1], (15, -.25)),
                                   ([0, 0], (0, 0))]:
            with self.subTest(position=position):
                np.testing.assert_allclose(self.line.project(position), expected)
        for s in (1., 5., 9., 13., 15.5):
            np.testing.assert_allclose(self.line.project(self.line.sample(s, .1)),
                                       [s, .1], atol=1e-12)

    def test_closing_endpoint_and_invalid_vertices(self):
        closed = ClosedRaceline([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]])
        np.testing.assert_array_equal(closed.points, self.line.points)
        self.assertEqual(closed.length, 16)
        for points in ([[0, 0], [1, 0]],
                       [[0, 0], [1, 0], [1, 0], [0, 1]],
                       [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0]],
                       [[0, 0], [1, 0], [np.nan, 1]], [1, 2, 3]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                ClosedRaceline(points)
        with self.assertRaises(ValueError):
            self.line.points[0, 0] = 99

    def test_signed_curvature(self):
        angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        points = 3 * np.column_stack((np.cos(angles), np.sin(angles)))
        ccw, cw = ClosedRaceline(points), ClosedRaceline(points[::-1])
        for s in (0, .1, 2, ccw.length - .01):
            self.assertAlmostEqual(ccw.curvature(s), 1 / 3, places=12)
            self.assertAlmostEqual(cw.curvature(s), -1 / 3, places=12)
            self.assertAlmostEqual(ccw.curvature(s + ccw.length), 1 / 3)


class TrackMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'synthetic'

    def make_map(self, free, resolution=.1, origin=(0., 0., 0.)):
        # Input rows run bottom to top, opposite to image-file row order.
        Image.fromarray((np.asarray(free, dtype=np.uint8) * 255)[::-1]).save(
            str(self.base) + '.png')
        with Path(str(self.base) + '.yaml').open('w') as stream:
            yaml.safe_dump({'resolution': resolution, 'origin': [float(value) for value in origin],
                            'image': 'ignored_like_laser_model.png'}, stream)
        return TrackMap(self.base)

    def test_flip_origin_rotation_and_outside(self):
        free = np.ones((20, 30), dtype=bool)
        free[2, 7] = False
        origin = np.array([5., -3., .63])
        track = self.make_map(free, origin=origin)
        c, s = np.cos(origin[2]), np.sin(origin[2])
        rotation = np.array([[c, -s], [s, c]])

        def world(local):
            return origin[:2] + rotation @ local

        self.assertEqual(track.clear(world([.75, .25])), 0.)
        self.assertTrue(track.safe(world([.75, 1.65]), .2))
        for local in ([-.01, 1], [3.01, 1], [1, -.01], [1, 2.01]):
            self.assertEqual(track.clear(world(local)), 0.)
            self.assertFalse(track.safe(world(local), 0.))
        self.assertLessEqual(track.clear(world([.02, 1.])), .02 + 1e-12)
        self.assertEqual(TrackMap(str(self.base) + '.yaml').resolution, .1)

    def test_edt_clearance_is_conservative(self):
        free = np.ones((50, 50), dtype=bool)
        free[:, 10] = False
        track = self.make_map(free)
        distance = track.clear([2.05, 2.05])
        self.assertGreater(distance, .8)
        self.assertLessEqual(distance, .95)  # Wall ends at x=1.1.
        self.assertTrue(track.safe([2.05, 2.05], .8))
        self.assertFalse(track.safe([2.05, 2.05], 1.))

    def test_corridor_margin_and_disconnected_free_region(self):
        free = np.ones((80, 100), dtype=bool)
        free[:20] = False
        free[60] = False  # More free space beyond this wall must not be reached.
        track = self.make_map(free)
        line = ClosedRaceline([[3, 4], [7, 4], [7, 5], [3, 5]])
        lo, hi = track.corridor(line, 1., .3, max_offset=3.)
        self.assertGreater(lo, -1.7)
        self.assertLess(lo, -1.4)
        self.assertGreater(hi, 1.4)
        self.assertLess(hi, 1.7)
        for d in np.linspace(lo, hi, 301):
            self.assertTrue(track.safe(line.sample(1., d), .3))
        tighter = track.corridor(line, 1., .6)
        self.assertGreater(tighter[0], lo)
        self.assertLess(tighter[1], hi)
        np.testing.assert_allclose(track.corridor(line, 1., .3, max_offset=.5),
                                   [-.5, .5])
        with self.assertRaises(ValueError):
            track.corridor(line, 1., 3.)
        with self.assertRaises(ValueError):
            track.corridor(line, 1., .3, step=0)

    def test_corridor_batches_both_sides_and_refines_only_first_failure(self):
        free = np.ones((80, 100), dtype=bool)
        free[:20] = False
        free[60] = False
        track = self.make_map(free)
        line = ClosedRaceline([[3, 4], [7, 4], [7, 5], [3, 5]])
        with patch.object(track, 'clear_many', wraps=track.clear_many) as query:
            with patch.object(track, 'clear', side_effect=AssertionError('Scalar query')):
                bounds = track.corridor(line, 1., .3, max_offset=.523, step=.05)
        np.testing.assert_allclose(bounds, [-.523, .523])
        query.assert_called_once()
        self.assertEqual(query.call_args.args[0].shape, (23, 2))

        with patch.object(track, 'clear_many', wraps=track.clear_many) as query:
            with patch.object(track, 'clear', side_effect=AssertionError('Scalar query')):
                lo, hi = track.corridor(line, 1., .3, max_offset=3., step=.05)
        self.assertLessEqual(query.call_count, 17)  # One scan + ceil(log2(.05/1e-6)).
        self.assertEqual(query.call_args_list[0].args[0].shape, (121, 2))
        for call in query.call_args_list[1:]:
            self.assertEqual(call.args[0].shape, (2, 2))
        self.assertLess(hi, 1.7)  # Never skip the wall to reach free space beyond it.
        for bound in (lo, hi):
            direction = np.sign(bound)
            distance = abs(bound)
            edges = np.minimum(np.arange(int(np.ceil(distance / .05)) + 1) * .05,
                               distance)
            for start, end in zip(edges[:-1], edges[1:]):
                midpoint = direction * (start + end) / 2
                self.assertGreaterEqual(track.clear(line.sample(1., midpoint)),
                                        .3 + np.sqrt(2) * (end - start) / 2)
        self.assertEqual(track.corridor(line, 1., .3, max_offset=0), (0., 0.))
        with self.assertRaises(ValueError):
            track.corridor(line, 1., 3., max_offset=0)

    def test_map_boundary_corridor(self):
        track = self.make_map(np.ones((60, 60), dtype=bool))
        line = ClosedRaceline([[2, 3], [4, 3], [4, 4], [2, 4]])
        lo, hi = track.corridor(line, .5, .2, max_offset=5.)
        self.assertGreater(lo, -2.8)
        self.assertLess(lo, -2.6)
        self.assertGreater(hi, 2.6)
        self.assertLess(hi, 2.8)
        self.assertFalse(track.safe([2.5, .1], .2))

    def test_footprint_interior_and_margin(self):
        free = np.ones((60, 60), dtype=bool)
        free[30, 30] = False
        track = self.make_map(free)
        # All four corners are free, but an obstacle lies inside the rectangle.
        for x in (2.55, 3.55):
            for y in (2.55, 3.55):
                self.assertTrue(track.safe([x, y], 0))
        self.assertFalse(track.footprint_safe([3.05, 3.05], 0, 1, 1, 0))
        self.assertTrue(track.footprint_safe([1.5, 1.5], .7, .6, .3, .1))
        self.assertFalse(track.footprint_safe([.45, 1.5], 0, .6, .3, .2))
        with self.assertRaises(ValueError):
            track.footprint_safe([1, 1], 0, -1, 1, 0)

    def test_footprint_orientation_in_narrow_lane(self):
        free = np.ones((100, 150), dtype=bool)
        free[:35] = False
        free[65:] = False  # Horizontal lane: 0.7 <= y < 1.3 metres.
        track = self.make_map(free, resolution=.02)
        position = [1.5, 1.]
        length, width, margin = .58, .31, .02
        self.assertFalse(track.safe(position, np.hypot(length, width) / 2 + margin))
        self.assertTrue(track.footprint_safe(position, 0, length, width, margin))
        self.assertTrue(track.footprint_safe(position, np.pi, length, width, margin))
        self.assertFalse(track.footprint_safe(position, np.pi / 2,
                                              length, width, margin))
        self.assertFalse(track.footprint_safe(position, 0, length, width, .15))

    def test_footprint_grid_spacing_covering_radius_and_rotation(self):
        track = self.make_map(np.ones((60, 60), dtype=bool))
        length, width, margin = .23, .11, .02
        nx, ny = 5, 3  # ceil(dimension / (resolution / 2)).
        dx, dy = length / nx, width / ny
        required = margin + np.hypot(dx, dy) / 2
        position, yaw = np.array([2., 3.]), .73
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        with patch.object(track, 'clear_many',
                          return_value=np.full(nx * ny, required)) as query:
            with patch.object(track, 'clear', side_effect=AssertionError('Scalar query')):
                self.assertTrue(track.footprint_safe(position, yaw, length, width, margin))
        query.assert_called_once()
        samples = query.call_args.args[0]
        local = (samples - position) @ rotation
        expected = [[-length / 2 + (i + .5) * dx,
                     -width / 2 + (j + .5) * dy]
                    for i in range(nx) for j in range(ny)]
        np.testing.assert_allclose(local, expected, atol=1e-14)
        self.assertLessEqual(dx, track.resolution / 2)
        self.assertLessEqual(dy, track.resolution / 2)
        with patch.object(track, 'clear_many',
                          return_value=np.full(nx * ny, required - 1e-9)):
            self.assertFalse(track.footprint_safe(position, yaw, length, width, margin))
        self.assertTrue(track.footprint_safe(position, yaw, 0, 0, margin))
        self.assertTrue(track.footprint_safe(position, yaw, length, 0, margin))
        with self.assertRaises(ValueError):
            track.footprint_safe([np.nan, 0], yaw, length, width, margin)

    def test_footprints_safe_matches_scalar_randomized(self):
        free = np.ones((80, 100), dtype=bool)
        free[20:35, 30:45] = False
        free[50, :70] = False
        origin = np.array([5., -3., .63])
        track = self.make_map(free, origin=origin)
        rng = np.random.default_rng(123)
        c, s = np.cos(origin[2]), np.sin(origin[2])
        rotation = np.array([[c, -s], [s, c]])
        local = rng.uniform([-1., -1.], [11., 9.], (300, 2))
        poses = np.column_stack((local @ rotation.T + origin[:2],
                                 rng.uniform(-8 * np.pi, 8 * np.pi, len(local))))
        for length, width, margin in ((.58, .31, .02), (.23, .11, .1),
                                      (0, 0, 0), (.4, 0, .03), (0, .3, 0)):
            with self.subTest(length=length, width=width, margin=margin):
                expected = np.array([track.footprint_safe(p[:2], p[2], length,
                                                         width, margin) for p in poses])
                actual = track.footprints_safe(poses, length, width, margin)
                self.assertEqual(actual.shape, (len(poses),))
                self.assertEqual(actual.dtype, np.dtype(bool))
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.any(actual))
                self.assertTrue(np.any(~actual))

    def test_footprints_safe_single_query_grid_and_threshold(self):
        track = self.make_map(np.ones((60, 60), dtype=bool))
        poses = np.array([[2., 3., .73], [1., 2., -1.2], [3., 2., np.pi]])
        length, width, margin = .23, .11, .02
        nx, ny = 5, 3
        required = margin + np.hypot(length / nx, width / ny) / 2
        scalar_samples = []
        with patch.object(track, 'clear_many',
                          return_value=np.full(nx * ny, required)) as query:
            for pose in poses:
                track.footprint_safe(pose[:2], pose[2], length, width, margin)
                scalar_samples.append(query.call_args.args[0])
        distances = np.full((len(poses), nx * ny), required)
        distances[1, 7] -= 1e-9
        with patch.object(track, 'clear_many', return_value=distances.ravel()) as query:
            with patch.object(track, 'footprint_safe', side_effect=AssertionError('Scalar footprint')):
                with patch.object(track, 'clear', side_effect=AssertionError('Scalar query')):
                    actual = track.footprints_safe(poses, length, width, margin)
        np.testing.assert_array_equal(actual, [True, False, True])
        query.assert_called_once()
        np.testing.assert_array_equal(query.call_args.args[0],
                                      np.concatenate(scalar_samples))
        with patch.object(track, 'clear_many', return_value=np.array([0., 1., 0.])) as query:
            np.testing.assert_array_equal(track.footprints_safe(poses, 0, 0, 0),
                                          [False, True, False])
        query.assert_called_once()
        self.assertEqual(query.call_args.args[0].shape, (3, 2))

    def test_footprints_safe_validation_and_empty(self):
        track = self.make_map(np.ones((20, 20), dtype=bool))
        with patch.object(track, 'clear_many', side_effect=AssertionError('Empty query')):
            result = track.footprints_safe(np.empty((0, 3)), .6, .3, .1)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype(bool))
        for invalid in ([], [1, 2, 3], [[1, 2]], [[1, 2, 3, 4]],
                        np.empty((0, 2)), np.zeros((1, 1, 3)),
                        [[np.nan, 0, 0]], [[0, np.inf, 0]], [[0, 0, -np.inf]]):
            with self.subTest(poses=invalid), self.assertRaises(ValueError):
                track.footprints_safe(invalid, .6, .3, .1)
        for poses in (np.empty((0, 3)), [[1, 1, 0]]):
            for index in range(3):
                for invalid in (-1, np.nan, np.inf):
                    dimensions = [.6, .3, .1]
                    dimensions[index] = invalid
                    with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                        track.footprints_safe(poses, *dimensions)

    def test_clear_many_matches_scalar_including_outside(self):
        free = np.ones((20, 30), dtype=bool)
        free[4:8, 10:13] = False
        origin = np.array([5., -3., .63])
        track = self.make_map(free, origin=origin)
        c, s = np.cos(origin[2]), np.sin(origin[2])
        rotation = np.array([[c, -s], [s, c]])
        rng = np.random.default_rng(42)
        local = np.vstack((rng.uniform([-.5, -.5], [3.5, 2.5], (300, 2)),
                           [[1.15, .55], [.75, 1.65], [-.01, 1], [3.01, 1],
                            [1, -.01], [1, 2.01], [0, 0], [3, 2]]))
        points = local @ rotation.T + origin[:2]
        batch = track.clear_many(points)
        scalar = np.array([track.clear(point) for point in points])
        self.assertEqual(batch.shape, (len(points),))
        np.testing.assert_allclose(batch, scalar, atol=1e-14, rtol=1e-14)
        self.assertEqual(batch[300], 0.)  # Inside occupied rectangle.
        self.assertGreater(batch[301], 0.)
        np.testing.assert_array_equal(batch[302:306], np.zeros(4))
        np.testing.assert_array_equal(track.clear_many(points[302:306]), np.zeros(4))
        self.assertEqual(track.clear_many(np.empty((0, 2))).shape, (0,))
        for invalid in ([1, 2], [[1, 2, 3]], [[np.nan, 0]], [[0, np.inf]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                track.clear_many(invalid)

    def test_repository_example_config(self):
        examples = Path(__file__).resolve().parent
        with (examples / 'config_example_map.yaml').open() as stream:
            config = yaml.safe_load(stream)
        track = TrackMap(examples / config['map_path'], config['map_ext'])
        waypoints = np.loadtxt(examples / config['wpt_path'],
                               delimiter=config['wpt_delim'],
                               skiprows=config['wpt_rowskip'])
        line = ClosedRaceline(waypoints[:, [config['wpt_xind'], config['wpt_yind']]])
        self.assertGreater(line.length, 10)
        self.assertTrue(track.safe([config['sx'], config['sy']], 0.))
        self.assertGreater(track.clear(line.sample(0)), 0.)


if __name__ == '__main__':
    unittest.main()
