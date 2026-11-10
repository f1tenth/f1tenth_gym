"""Reference regressions; run directly to also benchmark the actual map envelope."""
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from rival_reference import smoothed_reference, clearance_reference, _reference_poses


class ReferenceTests(unittest.TestCase):
    def test_pinch_transition_anticipates(self):
        s = np.arange(0., 80., .5)
        lo, hi = np.full_like(s, -2.), np.full_like(s, 2.)
        hi[(s >= 49) & (s <= 50)] = .1
        lo[(s >= 55) & (s <= 59)] = 1.1
        ref, diag = smoothed_reference(s, lo, hi, 80.)
        self.assertTrue(diag['solver_success'])
        self.assertTrue(np.all(ref >= lo - 1e-10))
        self.assertTrue(np.all(ref <= hi + 1e-10))
        self.assertGreater(ref[s == 53].item(), .35)
        self.assertLessEqual(ref[s == 50].item(), .1 + 1e-10)
        self.assertLess(diag['max_slope'], .5)
        self.assertLess(diag['max_second_derivative'], .4)

    def test_wrap_is_rotation_equivariant_and_diagnostics_include_closure(self):
        s = np.arange(64.)
        lo, hi = np.full(64, -2.), np.full(64, 2.)
        lo[:4] = 1.
        ref, diag = smoothed_reference(s, lo, hi, 64.)
        shifted, _ = smoothed_reference(s, np.roll(lo, 17), np.roll(hi, 17), 64.)
        np.testing.assert_allclose(shifted, np.roll(ref, 17), atol=2e-5)
        self.assertGreater(ref[-1], .5)
        self.assertAlmostEqual(diag['max_slope'], np.max(np.abs(np.roll(ref, -1) - ref)))
        self.assertAlmostEqual(diag['max_second_derivative'],
                               np.max(np.abs(np.roll(ref, -1) - 2 * ref + np.roll(ref, 1))))
        forced = np.zeros(64)
        forced[-1] = 2.
        _, diag = smoothed_reference(s, forced, forced, 64.)
        self.assertEqual(diag['max_slope'], 2.)
        self.assertEqual(diag['max_second_derivative'], 4.)

    def test_constant_shifted_and_zero_smoothing(self):
        s = np.arange(32.)
        for lo, hi, expected in [(-1., 1., 0.), (1.1, 2., 1.1), (-2., -.7, -.7), (.4, .4, .4)]:
            for length in (0., .5, 3., 10.):
                ref, diag = smoothed_reference(s, np.full(32, lo), np.full(32, hi), 32., length)
                np.testing.assert_allclose(ref, expected, atol=1e-10)
                self.assertEqual(diag['max_slope'], 0.)
        lo = np.sin(s)
        ref, _ = smoothed_reference(s, lo, lo + 1., 32., 0.)
        np.testing.assert_allclose(ref, np.clip(0., lo, lo + 1.), atol=1e-10)

    def test_invalid_inputs(self):
        s = np.arange(4.)
        cases = [(s, [0]*4, [-1]*4, 4., 3.),
                 ([], [], [], 4., 3.), (s, [0]*3, [1]*4, 4., 3.),
                 ([0, 1, 1, 3], [0]*4, [1]*4, 4., 3.),
                 ([0, 1, 2, 3.1], [0]*4, [1]*4, 4., 3.),
                 ([.1, 1, 2, 3], [0]*4, [1]*4, 4., 3.)]
        cases += [(s, [0]*4, [1]*4, p, l) for p, l in
                  [(0, 3), (np.inf, 3), (4, -1), (4, np.nan), (4, 1e200)]]
        cases += [(s, [0, 0, np.nan, 0], [1]*4, 4., 3.),
                  (s, [0]*4, [1, 1, np.inf, 1], 4., 3.)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                smoothed_reference(*args)

    def test_deterministic_finite_and_solver_failure(self):
        s = np.arange(0., 40., .5)
        lo = .6 * np.sin(s / 4) - .2
        hi = lo + .8
        first, diag = smoothed_reference(s, lo, hi, 40.)
        second, other = smoothed_reference(s, lo, hi, 40.)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(diag, other)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.isfinite(diag['objective']))
        with patch('rival_reference.minimize', return_value=SimpleNamespace(
                success=False, message='iteration limit')):
            with self.assertRaisesRegex(ValueError, 'iteration limit'):
                smoothed_reference(s, lo, hi, 40.)


class ClearanceTests(unittest.TestCase):
    def test_rotated_narrow_road_and_empty_body_corridor(self):
        from PIL import Image
        from rival_geometry import ClosedRaceline, TrackMap

        resolution, angle = .04, .63
        coords = (np.arange(240) + .5) * resolution
        x, y = np.meshgrid(coords, coords)
        radius = np.hypot(x - 4.8, y - 4.8)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        points = np.column_stack((4.8 + 3.35 * np.cos(theta), 4.8 + 3.35 * np.sin(theta)))
        line = ClosedRaceline(points @ rotation.T)
        nodes = np.linspace(0, line.length, 96, endpoint=False)
        lo, hi = np.full(96, -.1), np.full(96, .7)
        originals = lo.copy(), hi.copy()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'road'
            Image.fromarray(np.uint8((radius >= 2.55) & (radius <= 3.45)) * 255).transpose(
                Image.Transpose.FLIP_TOP_BOTTOM).save(str(base) + '.png')
            base.with_suffix('.yaml').write_text(
                f'resolution: {resolution}\norigin: [0, 0, {angle}]\n')
            track = TrackMap(base)
            old, _ = smoothed_reference(nodes, lo, hi, line.length)
            self.assertFalse(np.all(track.footprints_safe(_reference_poses(line, nodes, old), .58, .31, .08)))
            ref, diag = clearance_reference(line, track, nodes, lo, hi, .58, .31, .08)
            self.assertTrue(np.all(track.footprints_safe(_reference_poses(line, nodes, ref), .58, .31, .08)))
            self.assertTrue(np.all((ref >= lo) & (ref <= hi)))
            self.assertTrue(diag['footprint_success'])
            self.assertEqual(diag['refinements'], 2)
            self.assertGreater(diag['min_bandwidth'], 0.)
            np.testing.assert_array_equal(lo, originals[0])
            np.testing.assert_array_equal(hi, originals[1])
            with self.assertRaisesRegex(ValueError, 'Empty body-clear legal band'):
                clearance_reference(line, track, nodes, lo, hi, .58, 2., .08)
            for kwargs in ({'iterations': 0}, {'step': 0}, {'step': .03}, {'margin': -.1}):
                args = dict(length=.58, width=.31, margin=.08)
                args.update(kwargs)
                with self.assertRaises(ValueError):
                    clearance_reference(line, track, nodes, lo, hi, **args)


def actual_geometry_report():
    """Optional real-data prototype; deliberately no module-scope robust import."""
    from rival_geometry import ClosedRaceline, TrackMap
    from rival_strategy import Settings
    from rival_robust import RobustConfig, TrackEnvelope

    base = Path(__file__).resolve().parent
    points = np.loadtxt(base / 'example_waypoints.csv', delimiter=';', skiprows=3)
    line = ClosedRaceline(points[:, 1:3])
    track, vehicle = TrackMap(base / 'example_map'), Settings()
    envelope = TrackEnvelope(line, track, vehicle, RobustConfig())
    start = time.perf_counter()
    reference, diagnostics = smoothed_reference(envelope.s, envelope.lo, envelope.hi, line.length)
    elapsed = time.perf_counter() - start
    print(f'Actual geometry: {len(reference)} nodes, {elapsed * 1000:.2f} ms; {diagnostics}')
    old = reference
    start = time.perf_counter()
    reference, diagnostics = clearance_reference(line, track, envelope.s, envelope.lo, envelope.hi,
                                                vehicle.car_length, vehicle.car_width, .08)
    valid = track.footprints_safe(_reference_poses(line, envelope.s, reference),
                                  vehicle.car_length, vehicle.car_width, .08)
    assert np.all(valid), 'Actual reference fails requested .08 footprint margin'
    print(f'Body-clear reference: {1000 * (time.perf_counter() - start):.2f} ms; {diagnostics}')
    print(f'Actual footprint checks at .08: {np.count_nonzero(valid)}/{len(valid)} pass')
    print('s, lower, upper, old reference, body-clear reference (metres)')
    for i in np.flatnonzero((envelope.s >= 58) & (envelope.s <= 62)):
        print(f'{envelope.s[i]:.3f}, {envelope.lo[i]:.6f}, {envelope.hi[i]:.6f}, {old[i]:.6f}, {reference[i]:.6f}')


if __name__ == '__main__':
    result = unittest.main(exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    actual_geometry_report()
