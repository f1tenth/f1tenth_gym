"""Independent parent-loop parity tests and an opt-in warmed benchmark.

Run: .venv-py315/bin/python examples/test_rival_rollout.py --benchmark
"""
import math
import sys
import time
import unittest
from unittest.mock import Mock

import numpy as np

from rival_geometry import ClosedRaceline
from rival_robust import RobustConfig, RobustDefender, TrackEnvelope
from rival_rollout import _simulate, simulate_trajectory
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


def reference(line, envelope, ego, controls, bias, desired, times, c, r, track):
    # A real mock is intentional: future parent dispatch must retain this path.
    proxy = Mock(spec=['interval'])
    proxy.interval.side_effect = envelope.interval
    parent = RobustDefender(line, track, c, r, controls[0], envelope=proxy)
    (parent.target, parent._lateral_rate, parent._acceleration,
     parent._steering, parent._speed_command) = controls
    commands = []
    command = parent._command

    def record(*args):
        result = command(*args)
        commands.append(result)
        return result

    parent._command = record
    result = parent._rollout(ego, bias, desired, times,
                             np.empty((0, len(times), 3)), np.zeros(len(times)))
    peak = max((row[3] ** 2 * abs(math.tan(cmd[3])) / c.wheelbase
                for row, cmd in zip(result.trajectory[1:], commands)), default=0.)
    # Independently reconstruct the raw legal violation, before parent's final
    # max with the lateral acceleration excess.
    lo, hi = envelope.interval(ego.s)
    initial_error = max(0., lo - ego.d, ego.d - hi)
    violation = 0.
    for row, t in zip(result.trajectory[1:], times[1:]):
        lo, hi = envelope.interval(row[4])
        tolerance = r.lateral_tracking_margin + max(0., initial_error - .1 * t)
        violation = max(violation, lo - row[5] - tolerance, row[5] - hi - tolerance,
                        lo - row[6] - tolerance, row[6] - hi - tolerance)
        if lo > hi:
            violation = max(violation, 1.)
    np.testing.assert_allclose(result.violation, max(violation, peak - r.max_lateral_accel, 0.), atol=1e-12)
    return result.trajectory, violation, peak


class RolloutTests(unittest.TestCase):
    def test_parent_parity(self):
        line, envelope, c, r, track = fixture()
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
