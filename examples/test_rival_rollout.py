"""Independent parent-loop parity tests and an opt-in warmed benchmark.

Run: .venv-py315/bin/python examples/test_rival_rollout.py --benchmark
"""
import math
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from rival_geometry import ClosedRaceline
from rival_robust import RobustConfig, RobustDefender, TrackEnvelope
from rival_rollout import _profile_limit, _simulate, simulate_trajectory
from rival_speed import SpeedProfile
from rival_strategy import CarState, Settings


class BroadTrack:
    def corridor(self, *args, **kwargs):
        return -2., 2.

    def clear_many(self, points):
        return np.full(len(points), 100.)

    def footprint_safe(self, *args):
        return True


def fixture():
    angles = np.linspace(0., 2 * np.pi, 128, endpoint=False)
    line = ClosedRaceline(np.column_stack((12 * np.cos(angles), 8 * np.sin(angles))))
    c = Settings(legal_min=-3., legal_max=3.)
    r = RobustConfig(region='track')
    track = BroadTrack()
    envelope = TrackEnvelope(line, track, c, r)
    return line, envelope, c, r, track


def reference(line, envelope, ego, controls, bias, desired, times, c, r, track, speed_profile=None):
    parent = RobustDefender(line, track, c, r, controls[0], envelope=envelope)
    (parent.target, parent._lateral_rate, parent._acceleration,
     parent._steering, parent._speed_command) = controls
    setattr(parent, 'speed_profile', speed_profile)
    commands = []
    command = parent._command

    def record(state, target, lateral_rate, acceleration, steering, command_speed,
               bias, desired_speed, dt):
        # Keep the oracle usable before/after caller-side profile integration.
        # Applying the same cap again in the parent is idempotent.
        profile = getattr(parent, 'speed_profile', None)
        if profile is not None:
            desired_speed = min(desired_speed, profile.limit(state.s, state.speed))
        result = command(state, target, lateral_rate, acceleration, steering, command_speed,
                         bias, desired_speed, dt)
        commands.append(result)
        return result

    parent._command = record
    path, raw_violation, raw_peak = parent._trajectory_python(ego, bias, desired, times)
    peak = max((row[3] ** 2 * abs(math.tan(cmd[3])) / c.wheelbase
                for row, cmd in zip(path[1:], commands)), default=0.)
    # Reconstruct raw legal violation independently of the parent helper.
    lo, hi = envelope.interval(ego.s)
    initial_error = max(0., lo - ego.d, ego.d - hi)
    violation = 0.
    for row, t in zip(path[1:], times[1:]):
        lo, hi = envelope.interval(row[4])
        plo, phi = envelope.interval(row[4] + c.lookahead)
        tolerance = r.lateral_tracking_margin + max(0., initial_error - .1 * t)
        violation = max(violation, lo - row[5] - tolerance, row[5] - hi - tolerance,
                        plo - row[6] - tolerance, row[6] - phi - tolerance)
        if lo > hi or plo > phi:
            violation = max(violation, 1.)
    np.testing.assert_allclose(raw_violation, violation, atol=1e-12)
    np.testing.assert_allclose(raw_peak, peak, atol=1e-12)
    return path, raw_violation, raw_peak


class RolloutTests(unittest.TestCase):
    def test_parent_parity(self):
        line, envelope, c, r, track = fixture()
        envelope.reference_offsets[:] = .6 * np.sin(2 * np.pi * envelope.s / line.length)
        envelope._reference_slopes[:] = np.diff(np.r_[envelope.reference_offsets, envelope.reference_offsets[0]]) / np.diff(np.r_[envelope.s, line.length])
        times = np.arange(76) * .02
        rng = np.random.default_rng(12)
        cases = [(line.length - .2, 0., 3., 0., (.1, -.2, 0., .1, 3.), 0.),
                 (1., .5, .01, 0., (.4, .2, -1., 0., .01), 0.),
                 (2., -.4, 0., 0., (.3, -.3, 0., 0., 0.), 4.)]
        for angle in (1.4, math.pi, -.9):
            cases.append((3., .2, 1., angle, (.5, -.3, -.4, .2, 2.), 0.))
        for _ in range(12):
            cases.append((rng.uniform(0, line.length), rng.uniform(-2.5, 2.5),
                          rng.uniform(-1, 5), rng.uniform(-2, 2),
                          tuple(rng.uniform(-.3, .3, 4)) + (rng.uniform(0, 5),), rng.uniform(0, 6)))
        for invalid in (False, True):
            envelope.lo[:] = -.8 + .2 * np.sin(envelope.s)
            envelope.hi[:] = .9 + .2 * np.cos(envelope.s)
            if invalid:
                envelope.lo[::3], envelope.hi[::3] = 1., -1.
            for s, d, speed, angle, controls, desired in cases:
                x, y = line.sample(s, d)
                ego = CarState(s, d, speed, x, y, line.heading(s) + angle)
                with self.subTest(invalid=invalid, s=s, angle=angle):
                    actual = simulate_trajectory(line, envelope, ego, controls, .35, desired, times, c, r)
                    expected = reference(line, envelope, ego, controls, .35, desired, times, c, r, track)
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11)

    def test_no_profile_packed_and_legacy_kernel_are_bit_identical(self):
        line, envelope, c, r, _track = fixture()
        s = line.length - .2
        x, y = line.sample(s, .2)
        ego = CarState(s, .2, 3., x, y, line.heading(s) + .3)
        args = (line, envelope, ego, (.3, -.2, -.4, .1, 3.), .4, 5.,
                np.arange(76) * .02, c, r)
        with patch("rival_rollout._simulate", wraps=_simulate) as kernel:
            default = simulate_trajectory(*args)
            packed = kernel.call_args.args
        speed_nodes, speed_squared, response = packed[-1]
        self.assertEqual(len(speed_nodes), 0)
        self.assertEqual(len(speed_squared), 0)
        self.assertEqual(response, (0., 0., 0., 0., 0.))
        explicit_none = simulate_trajectory(*args, speed_profile=None)
        legacy_kernel = _simulate(*packed[:-1])
        for actual in (explicit_none, legacy_kernel):
            for a, b in zip(actual, default):
                self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())

    def test_compiled_profile_limit_matches_actual_arrays_near_wrap_and_vertices(self):
        line, envelope, c, r, _track = fixture()
        envelope.reference_offsets[:] = .4 * np.sin(2 * np.pi * envelope.s / line.length)
        envelope._reference_slopes[:] = np.diff(np.r_[envelope.reference_offsets,
            envelope.reference_offsets[0]]) / np.diff(np.r_[envelope.s, line.length])
        profile = SpeedProfile(line, envelope, c, r, 8.)
        response = (profile.length, profile._lookahead, profile._response_time,
                    profile._response_acceleration_distance, profile._global_min)
        rng = np.random.default_rng(41)
        positions = np.r_[rng.uniform(-2 * line.length, 2 * line.length, 40),
                          line._starts, line._starts - 1e-10, line._starts + 1e-10,
                          profile.s[::17]]
        for s in positions:
            for measured in (-2., 0., .3, 5., 25., 1e308):
                actual = _profile_limit(s, measured, profile._wrapped_s, profile._squared, response)
                expected = profile.limit(s, measured)
                self.assertTrue(np.isfinite(actual))
                self.assertAlmostEqual(actual, expected, delta=2e-12)
        for s, measured in [(np.nan, 0.), (np.inf, 0.), (0., np.nan), (0., np.inf)]:
            with self.assertRaises(ValueError):
                _profile_limit(s, measured, profile._wrapped_s, profile._squared, response)

    def test_compiled_profile_does_not_skip_internal_corner_or_seam(self):
        vertices = np.array([[0., 0.], [40., 0.], [40., 40.], [0., 40.]])
        points = np.concatenate([start + np.arange(40)[:, None] / 40 * (end - start)
                                 for start, end in zip(vertices, np.roll(vertices, -1, axis=0))])
        line = ClosedRaceline(points)
        c, r = Settings(), RobustConfig(region="track")
        profile = SpeedProfile(line, None, c, r, 20.)
        response = (profile.length, profile._lookahead, profile._response_time,
                    profile._response_acceleration_distance, profile._global_min)
        span = 25.
        measured = float((span - profile._lookahead - profile._response_acceleration_distance)
                         / profile._response_time)
        for start in (30., line.length - 10.):
            actual = _profile_limit(start, measured, profile._wrapped_s, profile._squared, response)
            self.assertAlmostEqual(actual, profile.limit(start, measured), places=12)
            self.assertLess(actual ** 2, min(profile._at(start), profile._at(start + span)))
        self.assertAlmostEqual(_profile_limit(30., 1e6, profile._wrapped_s,
                               profile._squared, response), np.min(profile.speeds), places=12)

    def test_parent_parity_with_speed_profile(self):
        line, envelope, c, r, track = fixture()
        envelope.reference_offsets[:] = .35 * np.sin(2 * np.pi * envelope.s / line.length)
        envelope._reference_slopes[:] = np.diff(np.r_[envelope.reference_offsets,
            envelope.reference_offsets[0]]) / np.diff(np.r_[envelope.s, line.length])
        cases = [(line.length - .2, .1, 3., 0., (.1, -.2, 0., .1, 3.), .25),
                 (line._starts[5], -.1, 2., 1.2, (.1, .1, .5, -.1, 2.5), -.2),
                 (3., 0., -1., math.pi, (0., 0., 0., 0., 1.), 0.),
                 (0., 0., 0., 0., (0., 0., 0., 0., 0.), 0.)]
        timesets = (np.arange(76) * .02, np.array([0., .02, .04, .051, .071, .103]))
        for speed_limit in (0., 1.1, 8.):
            profile = SpeedProfile(line, envelope, c, r, speed_limit)
            for times in timesets:
                for s, d, speed, angle, controls, bias in cases:
                    x, y = line.sample(s, d)
                    ego = CarState(s, d, speed, x, y, line.heading(s) + angle)
                    args = (line, envelope, ego, controls, bias, 8., times, c, r)
                    with self.subTest(limit=speed_limit, s=s, angle=angle, nodes=len(times)):
                        # The compiled loop must use numeric arrays, not callbacks.
                        with patch.object(profile, "limit", side_effect=AssertionError("Python callback")):
                            actual = simulate_trajectory(*args, speed_profile=profile)
                        expected = reference(*args, track, speed_profile=profile)
                        self.assertEqual(actual[0].shape, (len(times), 7))
                        for a, b in zip(actual, expected):
                            self.assertTrue(np.all(np.isfinite(a)))
                            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11)
        # A profile must affect command planning, without clipping measured speed.
        s = 3.
        x, y = line.sample(s)
        ego = CarState(s, 0., 2., x, y, line.heading(s))
        args = (line, envelope, ego, (0., 0., 0., 0., 2.), 0., 8., timesets[0], c, r)
        baseline = simulate_trajectory(*args)
        capped = simulate_trajectory(*args, speed_profile=SpeedProfile(line, envelope, c, r, 1.1))
        self.assertEqual(capped[0][0, 3], baseline[0][0, 3])
        self.assertLess(capped[0][-1, 3], baseline[0][-1, 3] - .5)

    def test_reference_slope_speed_cap(self):
        line, envelope, c, r, track = fixture()
        s = 3.
        x, y = line.sample(s)
        ego = CarState(s, 0., 2., x, y, line.heading(s))
        args = (line, envelope, ego, (0., 0., 0., 0., 2.), 0., 5.,
                np.arange(51) * .02, c, r)
        baseline = simulate_trajectory(*args)
        # Deliberately steep stored slopes isolate the cap from goal tracking.
        # Mutating precomputed arrays must be observed on the very next call.
        envelope._reference_slopes[:] = 4.
        actual = simulate_trajectory(*args)
        for a, b in zip(actual, reference(*args, track)):
            np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)
        self.assertLess(actual[0][-1, 3], baseline[0][-1, 3] - .5)

    def test_pinch_target_uses_future_band(self):
        line = ClosedRaceline([[0, 0], [40, 0], [40, 40], [0, 40]])
        c = Settings(legal_min=-3., legal_max=3., lookahead=2.)
        r = RobustConfig(region='track', envelope_spacing=.1)
        track = BroadTrack()
        envelope = TrackEnvelope(line, track, c, r)
        envelope.lo[:] = -.2
        envelope.hi[:] = np.where(envelope.s < 11., .1, 1.)
        envelope.reference_offsets[:] = .6
        envelope._reference_slopes[:] = 0.
        ego = CarState(10., 0., 0., 10., 0., 0.)
        args = (line, envelope, ego, (.6, 0., 0., 0., 0.), 0., 0.,
                np.array([0., .02]), c, r)
        actual = simulate_trajectory(*args)
        for a, b in zip(actual, reference(*args, track)):
            np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)
        self.assertGreater(actual[0][-1, 6], envelope.interval(actual[0][-1, 4])[1])
        self.assertEqual(actual[1], 0.)

    def test_projection_ties_and_nonuniform_times(self):
        line, envelope, c, r, track = fixture()
        line = ClosedRaceline([[0, 0], [40, 0], [40, 40], [0, 40]])
        envelope = TrackEnvelope(line, track, c, r)
        for s in (0., 40., line.length - .01):
            x, y = line.sample(s)
            ego = CarState(s, 0., 0., x, y, line.heading(s))
            args = (line, envelope, ego, (.4, -.2, 0., 0., 0.), -.3, 2.,
                    np.array([0., .02, .04, .051, .071]), c, r)
            for a, b in zip(simulate_trajectory(*args), reference(*args, track)):
                np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)


def benchmark():
    line, envelope, c, r, track = fixture()
    s = line.length - .3
    x, y = line.sample(s, .2)
    ego = CarState(s, .2, 3., x, y, line.heading(s) + .3)
    args = (line, envelope, ego, (.3, -.2, -.4, .1, 3.), .4, 2., np.arange(76) * .02, c, r)
    start = time.perf_counter()
    actual = simulate_trajectory(*args)
    first = time.perf_counter() - start
    start = time.perf_counter()
    _simulate.recompile()
    compile_time = time.perf_counter() - start
    expected = reference(*args, track)
    error = max(float(np.max(np.abs(np.asarray(a) - b))) for a, b in zip(actual, expected))
    n = 50
    start = time.perf_counter()
    for _ in range(n):
        simulate_trajectory(*args)
    compiled = (time.perf_counter() - start) / n
    start = time.perf_counter()
    for _ in range(n):
        reference(*args, track)
    python = (time.perf_counter() - start) / n
    print(f'First call (compile/cache load): {first:.6f}s; forced compilation: {compile_time:.6f}s; '
          f'warmed kernel: {compiled * 1e3:.4f}ms; '
          f'Python reference: {python * 1e3:.4f}ms; speedup: {python / compiled:.1f}x; '
          f'max numeric error: {error:.3g}')


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
