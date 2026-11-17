"""Exact helper parity and optional warmed 144-candidate CPU benchmark.

.venv-py315/bin/python examples/test_rival_defender_checks.py
.venv-py315/bin/python examples/test_rival_defender_checks.py --benchmark --profile

No Gym episodes or parent patches are installed. Benchmark uses actual map,
line, envelope, SpeedProfile, and six saved fast_headway_trial truth snapshots.
It measures the three helpers together, not a full solve or pipeline speedup.
"""
import argparse
import copy
import cProfile
import csv
import io
import math
import pstats
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from rival_baseline import load
from rival_defender_checks import generate_hypotheses, trajectory_progress, trajectory_separation
from rival_geometry import ClosedRaceline
from rival_perception import OpponentEstimate
from rival_robust import RobustConfig, RobustDefender, TrackEnvelope, opponent_hypotheses, sat_clearance
from rival_rollout import simulate_trajectory
from rival_speed import SpeedProfile
from rival_strategy import CarState, Settings


HERE = Path(__file__).resolve().parent


def reference_separation(path, predictions, inflation, c, r):
    minimum = math.inf
    if len(predictions):
        gaps = sat_clearance(path[None, :, :3], predictions, c.car_length, c.car_width)
        relative = predictions[:, :, :2] - path[None, :, :2]
        translation = np.linalg.norm(np.diff(relative, axis=1), axis=2) / 2
        opponent_turn = (np.diff(predictions[:, :, 2], axis=1) + np.pi) % (2 * np.pi) - np.pi
        ego_turn = (np.diff(path[:, 2]) + np.pi) % (2 * np.pi) - np.pi
        rotation = math.hypot(c.car_length, c.car_width) / 2 * (np.abs(opponent_turn) + np.abs(ego_turn)[None, :]) / 2
        sweep = translation + rotation
        padding = np.maximum(np.pad(sweep, ((0, 0), (0, 1))), np.pad(sweep, ((0, 0), (1, 0))))
        minimum = float(np.min(gaps - inflation[None, :] - padding - c.separation_margin - r.lateral_tracking_margin))
    return minimum


def reference_progress(path, line):
    return sum(line.signed_distance(b, a) for a, b in zip(path[:-1, 4], path[1:, 4]))


def estimate(line, s=0., d=.1, speed=2., *, age=.1, position_sigma=.03, velocity_sigma=.1,
             longitudinal_acceleration=.2, lateral_speed=.1, fresh=True, valid=True):
    x, y = line.sample(s, d)
    q = CarState(s, d, speed, x, y, line.heading(s))
    return OpponentEstimate(q, age, position_sigma, velocity_sigma,
                            longitudinal_acceleration, lateral_speed, fresh, valid)


class DefenderChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual_c, cls.actual_line, cls.actual_track, _, _ = load(HERE / 'rival_advanced_fast.yaml')

    def setUp(self):
        self.c, self.r = Settings(), RobustConfig()
        self.line = ClosedRaceline([[0., 0.], [2., 0.], [3.1, .2], [4., 3.], [-1., 2.]])

    def assert_predictions(self, value, times, line=None):
        line = self.line if line is None else line
        expected = opponent_hypotheses(value, line, times, self.r)
        actual = generate_hypotheses(value, line, times, self.r)
        assert actual is not None
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-13, atol=2e-12)
        np.testing.assert_array_equal(actual[1], expected[1])
        return actual

    def assert_separation(self, path, predictions, inflation):
        expected = reference_separation(path, predictions, inflation, self.c, self.r)
        actual = trajectory_separation(path, predictions, inflation, self.c, self.r)
        assert actual is not None
        self.assertTrue(math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-12), (actual, expected))
        self.assertEqual(actual > 0., expected > 0.)
        return actual

    def test_hypotheses_real_geometry_vertices_and_periodic_seams(self):
        for line in (self.line, self.actual_line):
            for s in (0., -1e-10, line.length - 1e-10, line.length + 1e-10,
                      line._starts[2] - 1e-12, line._starts[2], line._starts[2] + 1e-12):
                for speed in (0., 2., 11.):
                    with self.subTest(s=s, speed=speed, points=len(line.points)):
                        self.assert_predictions(estimate(line, s=s, speed=speed), np.linspace(0., 1.5, 76), line)

    def test_all_cases_order_saturation_and_nonuniform_unsorted_times(self):
        times = np.array([0., 1e-9, .017, .13, .131, 1.499, 3., .2, .2, -.1])
        for speed in (-2., 0., .01, 3., 14.):
            for acceleration, lateral in ((-12., -3.), (0., 0.), (12., 3.)):
                value = estimate(self.line, speed=speed, longitudinal_acceleration=acceleration,
                                 lateral_speed=lateral, velocity_sigma=2.)
                self.assert_predictions(value, times)
        predictions, _ = self.assert_predictions(estimate(self.line), np.array([0., .02]))
        for longitudinal in range(3):
            for lateral in range(3):
                yaw = predictions[3 * longitudinal + lateral, 0, 2]
                rate = (-self.r.max_opponent_lateral_speed, .1, self.r.max_opponent_lateral_speed)[lateral]
                initial = (1.8, 2., 2.2)[longitudinal]
                self.assertAlmostEqual(yaw, self.line.heading(0.) + math.atan2(rate, initial))

    def test_staleness_does_not_drop_valid_hypotheses_or_change_inflation(self):
        value = estimate(self.line, fresh=False, age=4., position_sigma=.7, velocity_sigma=.6)
        times = np.array([0., .03, .6, 1.5])
        predictions, inflation = self.assert_predictions(value, times)
        self.assertEqual(predictions.shape, (9, 4, 3))
        np.testing.assert_array_equal(inflation, self.r.position_margin + self.r.uncertainty_multiplier * .7
                                      + self.r.lateral_tracking_margin * times)
        fresh = copy.copy(value)
        fresh.fresh, fresh.age = True, 0.
        current = self.assert_predictions(fresh, times)
        np.testing.assert_array_equal(predictions, current[0])
        np.testing.assert_array_equal(inflation, current[1])

    def test_measured_overspeed_is_not_clamped_to_configured_bound(self):
        value = estimate(self.line, speed=10., longitudinal_acceleration=0., lateral_speed=0.)
        times = np.array([0., .03, .2])
        predictions, _ = self.assert_predictions(value, times)
        np.testing.assert_allclose(predictions[4, -1, :2], self.line.sample(2., .1), atol=2e-12)  # pyright: ignore[reportArgumentType]
        self.assertGreater(np.linalg.norm(predictions[4, -1, :2] - predictions[4, 0, :2]), .5)

    def test_no_estimate_invalid_estimate_single_and_empty_times(self):
        for value in (None, estimate(self.line, valid=False), estimate(self.line)):
            for times in (np.empty(0), np.array([0.]), np.array([.017, .1])):
                self.assert_predictions(value, times)
        missing = generate_hypotheses(None, self.line, np.array([0.]), None)
        assert missing is not None
        self.assertEqual(missing[0].shape, (0, 1, 3))

    def test_mutable_config_and_estimate_are_repacked_without_global_state(self):
        value = estimate(self.line)
        times = np.array([0., .2, .8])
        old = self.assert_predictions(value, times)
        self.r.max_opponent_acceleration = 0.
        self.r.max_opponent_lateral_speed = 0.
        self.r.max_braking = 0.
        self.r.max_opponent_speed = 9.
        self.r.position_margin = .4
        self.r.uncertainty_multiplier = .5
        self.r.lateral_tracking_margin = .3
        value.state.speed = 5.
        new = self.assert_predictions(value, times)
        self.assertFalse(np.array_equal(old[0], new[0]))
        self.assertFalse(np.array_equal(old[1], new[1]))

    def test_custom_geometry_and_estimates_use_scalar_fallback(self):
        class Custom(ClosedRaceline):
            def heading(self, s):
                raise AssertionError('Must not evaluate custom geometry')
        custom = Custom(self.line.points)
        times = np.array([0., .1])
        self.assertIsNone(generate_hypotheses(None, custom, times, self.r))
        self.assertIsNone(generate_hypotheses(object(), self.line, times, self.r))
        self.assertIsNone(generate_hypotheses(None, object(), times, self.r))
        self.assertIsNone(generate_hypotheses(None, self.line, np.zeros((2, 2)), self.r))
        self.assertIsNone(generate_hypotheses(None, self.line, np.array([np.nan]), self.r))
        self.assertIsNone(generate_hypotheses(estimate(self.line), self.line, times.astype(np.float32), self.r))

    def test_separation_random_orientations_shapes_and_all_hypotheses(self):
        rng = np.random.default_rng(20261117)
        for count in (1, 2, 3, 19, 76):
            for hypotheses in (0, 1, 4, 9, 13):
                path = rng.normal(size=(count, 7))
                predictions = rng.normal(size=(hypotheses, count, 3))
                path[:, 2] *= 9.
                predictions[:, :, 2] *= 9.
                inflation = rng.uniform(0., .5, count)
                self.assert_separation(path, predictions, inflation)
        # No accidental early exit at the first hypothesis or first collision.
        path = np.zeros((3, 7))
        predictions = np.zeros((9, 3, 3))
        predictions[:, :, 0] = 10.
        predictions[8, 1, 0] = 0.
        self.assertLess(self.assert_separation(path, predictions, np.zeros(3)), 0.)

    def test_sweep_padding_translational_tunnelling_and_both_endpoints(self):
        path = np.zeros((3, 7))
        predictions = np.zeros((1, 3, 3))
        predictions[0, :, 0] = [-3., 3., 3.]
        result = self.assert_separation(path, predictions, np.zeros(3))
        self.assertLess(result, 0.)
        self.assertGreater(float(np.min(sat_clearance(path[None, :, :3], predictions,
                                                      self.c.car_length, self.c.car_width))), 0.)
        self.assert_separation(path[::-1], predictions[:, ::-1], np.zeros(3))

    def test_sweep_yaw_short_angle_seam_and_mutable_vehicle_dimensions(self):
        path = np.zeros((3, 7))
        path[:, 2] = [math.pi - .01, -math.pi + .01, math.pi - .03]
        predictions = np.zeros((2, 3, 3))
        predictions[:, :, 0] = 2.
        predictions[:, :, 2] = [-math.pi + .02, math.pi - .02, -math.pi + .03]
        for length, width, margin in ((.58, .31, .12), (0., 0., 0.), (1.3, .7, .3)):
            self.c.car_length, self.c.car_width, self.c.separation_margin = length, width, margin
            self.r.lateral_tracking_margin = margin / 2
            self.assert_separation(path, predictions, np.array([.1, .2, .3]))

    def test_separation_threshold_precision(self):
        path = np.zeros((2, 7))
        predictions = np.zeros((9, 2, 3))
        inflation = np.array([.1, .1])
        for delta in (-1e-10, 0., 1e-10):
            predictions[:, :, 1] = (self.c.car_width + self.c.separation_margin
                                     + self.r.lateral_tracking_margin + .1 + delta)
            self.assert_separation(path, predictions, inflation)

    def test_readonly_strided_arrays_and_inputs_unchanged(self):
        path = np.arange(70., dtype=float).reshape(10, 7)[::2]
        predictions = np.arange(9 * 10 * 3., dtype=float).reshape(9, 10, 3)[:, ::2]
        inflation = np.arange(10., dtype=float)[::2]
        originals = [a.copy() for a in (path, predictions, inflation)]
        for array in (path, predictions, inflation):
            array.flags.writeable = False
        self.assert_separation(path, predictions, inflation)
        self.assertEqual(trajectory_progress(path, self.line), reference_progress(path, self.line))
        for actual, expected in zip((path, predictions, inflation), originals):
            np.testing.assert_array_equal(actual, expected)

    def test_progress_seams_half_lap_ties_reverse_and_large_wrapped_stations(self):
        length = self.line.length
        sets = [[], [0.], [length - .01, .01, .03], [.01, length - .01, length - .03],
                [0., length / 2, 0., -length / 2],
                [20 * length + .01, -20 * length - .01, 50 * length + .02]]
        rng = np.random.default_rng(42)
        sets += [rng.uniform(-3 * length, 3 * length, 2000)]
        for stations in sets:
            path = np.zeros((len(stations), 7))
            path[:, 4] = stations
            self.assertEqual(trajectory_progress(path, self.line), reference_progress(path, self.line))

    def test_progress_nonuniform_line_vertex_sequences_and_geometry_mutation(self):
        for line in (self.line, self.actual_line):
            stations = np.r_[line._starts[:-1], line._starts[:-1][::-1], line.length + line._starts[:-1]]
            path = np.zeros((len(stations), 7))
            path[:, 4] = stations
            self.assertEqual(trajectory_progress(path, line), reference_progress(path, line))
        path = np.zeros((4, 7))
        path[:, 4] = [0., .1, .9, 2.]
        line = ClosedRaceline(self.line.points * 2.)
        self.assertEqual(trajectory_progress(path, line), reference_progress(path, line))

    def test_unsupported_numeric_inputs_return_none(self):
        path = np.zeros((2, 7))
        predictions = np.zeros((9, 2, 3))
        inflation = np.zeros(2)
        self.assertIsNone(trajectory_separation(path.astype(np.float32), predictions, inflation, self.c, self.r))
        self.assertIsNone(trajectory_separation(path, predictions[:, :1], inflation, self.c, self.r))
        self.assertIsNone(trajectory_separation(path, predictions, np.zeros(3), self.c, self.r))
        self.assertIsNone(trajectory_progress(path.astype(np.float32), self.line))
        self.assertIsNone(trajectory_progress(path, object()))
        path[0, 4] = np.nan
        self.assertIsNone(trajectory_progress(path, self.line))
        predictions[0, 0, 0] = np.nan
        self.assertIsNone(trajectory_separation(path, predictions, inflation, self.c, self.r))

    def test_reference_helpers_match_actual_parent_rollout(self):
        # Check the independent Python equation copies against the actual parent
        # method, with only its trajectory producer stubbed (map checks are real).
        path = np.zeros((4, 7))
        path[:, 4] = [18., 18.01, 18.04, 18.1]
        for index, s in enumerate(path[:, 4]):
            path[index, :2] = self.actual_line.sample(s, .6)  # pyright: ignore[reportArgumentType]
            path[index, 2] = self.actual_line.heading(s)
        predictions, inflation = opponent_hypotheses(estimate(self.actual_line, s=20.), self.actual_line,
                                                    np.array([0., .01, .04, .1]), self.r)
        parent = RobustDefender.__new__(RobustDefender)
        parent.vehicle, parent.config = self.c, self.r
        parent.line, parent.track = self.actual_line, self.actual_track
        parent.envelope = TrackEnvelope.__new__(TrackEnvelope)
        parent.target = parent._lateral_rate = parent._acceleration = parent._steering = parent._speed_command = 0.
        parent.speed_profile = None
        with patch('rival_rollout.simulate_trajectory', return_value=(path, 0., 0.)):
            checked = parent._rollout(estimate(self.actual_line, s=18.).state, 0., 1.,
                                      np.array([0., .01, .04, .1]), predictions, inflation)
        self.assertEqual(checked.min_separation, reference_separation(path, predictions, inflation, self.c, self.r))
        self.assertEqual(checked.progress, reference_progress(path, self.actual_line))
        actual = trajectory_separation(path, predictions, inflation, self.c, self.r)
        assert actual is not None
        self.assertTrue(math.isclose(actual, checked.min_separation, rel_tol=2e-13, abs_tol=2e-12))


def benchmark_fixture():
    c, line, track, _, _ = load(HERE / 'rival_advanced_fast.yaml')
    with (HERE / 'rival_advanced_fast.yaml').open() as stream:
        r = RobustConfig.from_dict(yaml.safe_load(stream)['robust'])
    envelope = TrackEnvelope(line, track, c, r)
    profile = SpeedProfile(line, envelope, c, r, c.defender_speed)
    # Keep the Python oracle independent of future fast-only parent dispatch.
    # The explicit SpeedProfile below still caps every generated trajectory.
    r.track_speed_planning = False
    saved = HERE / 'rival_results/fast_headway_trial/00_adaptive_robust_defense.csv'
    with saved.open() as stream:
        rows = list(csv.DictReader(stream))
    times = np.linspace(0., r.horizon, int(np.ceil(r.horizon / min(c.timestep, r.prediction_dt))) + 1)
    cases = []
    for index in (100, 200, 500, 1000, 2000, 3000):
        row, previous = rows[index], rows[index - 1]
        ego = CarState(*(float(row['ego_truth_eval_' + key]) for key in ('s', 'd', 'speed', 'x', 'y', 'yaw')))
        other = CarState(*(float(row['opponent_truth_eval_' + key]) for key in ('s', 'd', 'speed', 'x', 'y', 'yaw')))
        # CSV does not store tracker packets. Use recorded truth with explicit,
        # representative uncertainty; this is NOT an exact perception replay.
        value = OpponentEstimate(other, .1, .03, .1, float(row.get('opponent_truth_acceleration') or 0.),
                                 float(row.get('opponent_truth_lateral_speed') or 0.), True, True)
        controls = tuple(float(previous.get(key) or 0.) for key in
                         ('target', 'ego_target_rate', 'ego_command_acceleration', 'steering', 'speed'))
        paths = []
        for bias in (-.3, -.1, .1, .3):
            for fraction in r.rollout_speed_fractions:
                path, _, _ = simulate_trajectory(line, envelope, ego, controls, bias,
                                                 c.defender_speed * fraction, times, c, r, speed_profile=profile)
                paths.append(path)
        cases.append((value, paths))
    return c, r, line, times, cases


def benchmark(repeat=7, profile=False):
    c, r, line, times, cases = benchmark_fixture()

    def run(compiled):
        result = []
        generate = generate_hypotheses if compiled else opponent_hypotheses
        separate = trajectory_separation if compiled else reference_separation
        progress = trajectory_progress if compiled else reference_progress
        for value, paths in cases:
            generated = generate(value, line, times, r)
            assert generated is not None
            predictions, inflation = generated
            result.append((predictions, inflation))
            for path in paths:
                result.append((separate(path, predictions, inflation, c, r), progress(path, line)))
        return result

    old, new = run(False), run(True)  # Warm kernels and check all outputs first.
    for a, b in zip(old, new):
        for expected, actual in zip(a, b):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)
    count = sum(len(paths) for _, paths in cases)
    durations = {False: [], True: []}
    for _ in range(repeat):
        for compiled in (False, True):
            start = time.perf_counter()
            run(compiled)
            durations[compiled].append(time.perf_counter() - start)
    old_ms, new_ms = (float(np.median(durations[key])) * 1000 for key in (False, True))
    print(f'Parity: {count} candidates, {len(cases)} nine-hypothesis generations, {len(times)} unchanged nodes; tolerance 2e-12')
    print(f'Combined helpers: Python {old_ms:.3f} ms; compiled {new_ms:.3f} ms; {old_ms / new_ms:.2f}x')
    print(f'Amortized per candidate: {old_ms / count:.4f} -> {new_ms / count:.4f} ms')
    print('Includes wrappers/packing/generation/separation/progress. Excludes fixture I/O, trajectory generation, warmup, map/racecraft checks.')
    print('Six actual saved truth states + representative estimate uncertainties; not exact sensor replay or full-solve acceleration.')
    if profile:
        for compiled in (False, True):
            profiler = cProfile.Profile()
            profiler.runcall(run, compiled)
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('cumtime').print_stats(12)
            print(('Compiled' if compiled else 'Python') + ' helper profile:\n' + text.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--repeat', type=int, default=7)
    args, remaining = parser.parse_known_args()
    if args.benchmark:
        benchmark(args.repeat, args.profile)
    else:
        unittest.main(argv=[__file__, *remaining])
