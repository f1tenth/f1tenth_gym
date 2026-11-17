"""CPU-only parity tests and saved fast_headway_trial preview benchmark.

.venv-py315/bin/python examples/test_rival_attack_rollout.py
.venv-py315/bin/python examples/test_rival_attack_rollout.py --benchmark --profile

No Gym import/episode. Benchmark times include parameter packing, numeric rollout,
map certification, SAT, and Preview construction, but exclude warmup/fixture I/O.
CSV has projected speed, not body vx: replay uses preview's standard fallback.
"""
import argparse
import copy
import cProfile
import csv
import io
import json
import math
import pstats
import time
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from rival_adversary import Adversary
from rival_attack_rollout import compiled_preview
from rival_baseline import load
from rival_geometry import ClosedRaceline, TrackMap
from rival_robust import RobustConfig, TrackEnvelope
from rival_speed import SpeedProfile
from rival_strategy import CarState


HERE = Path(__file__).resolve().parent
TRIAL = HERE / 'rival_results/fast_headway_trial/00_adaptive_robust_defense'


def fixture():
    cfg, line, track, _, _ = load(HERE / 'rival_advanced_fast.yaml')
    with (HERE / 'rival_advanced_fast.yaml').open() as stream:
        motion = RobustConfig.from_dict(yaml.safe_load(stream)['robust'])
    # The fast YAML enables dispatch; parity and timing need an independent
    # scalar oracle, including tests that observe scalar footprint queries.
    motion.compiled_attack_preview = False
    envelope = TrackEnvelope(line, track, cfg, motion)
    profile = SpeedProfile(line, envelope, cfg, motion, cfg.attacker_speed)
    return cfg, motion, line, track, envelope, profile


def state(line, s, d, velocity=1., angle=0.):
    x, y = line.sample(s, d)
    return CarState(s % line.length, d, velocity * math.cos(angle), x, y,
                    line.heading(s) + angle)


def assert_preview(case, actual, expected):
    case.assertIsNotNone(actual)
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, (float, np.floating)):
            case.assertTrue(math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-11),
                            f'{field.name}: {a!r} != {b!r}; {expected.reason}')
        else:
            case.assertEqual(a, b, field.name)


def snapshots(data, indices=(0, 50, 200, 500, 1000, 2000, 3000, 4000)):
    """Reconstruct pre-plan state from truth columns and PREVIOUS commands.

    Row i contains post-plan commands but pre-step truth, so using row i's own
    command state would advance jerk/target twice. The previous row supplies it.
    This is representative candidate replay, not a replay of all policy memory.
    """
    cfg, motion, line, _track, envelope, profile = data
    with TRIAL.with_suffix('.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for index in indices:
        if index >= len(rows):
            continue
        row, previous = rows[index], rows[index - 1] if index else None

        def truth(prefix, row=row):
            return CarState(*(float(row[f'{prefix}_truth_eval_{name}'])
                              for name in ('s', 'd', 'speed', 'x', 'y', 'yaw')))

        attacker, defender = truth('opponent'), truth('ego')
        target = (float(previous['attacker_target']) if previous else
                  envelope.nominal(attacker.s + cfg.lookahead))
        policy = Adversary(target, tracking_margin=motion.lateral_tracking_margin,
                           speed_profile=profile, motion_config=motion)
        policy.speed = float(previous['attacker_speed_command']) if previous else 0.
        policy._acceleration = float(previous['attacker_command_acceleration']) if previous else 0.
        steering = float(previous['attacker_steering']) if previous else 0.
        paths = policy._defender_paths(defender, line, envelope, cfg)
        # Same integration grid and all four dynamically evolving hypotheses
        # reused across this plan's full/hold/brake/recovery candidate demands.
        goals = (float(row['attack_desired']), envelope.nominal(attacker.s + cfg.lookahead), target)
        for goal in goals:
            for speed in (cfg.attacker_speed, .7 * cfg.attacker_speed, 0.):
                result.append((float(row['time']), policy, attacker, defender, goal, speed,
                               {'previous_steering': steering, 'defender_paths': paths,
                                'body_recovery': True}))
    return result


class AttackRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()

    def setUp(self):
        self.cfg, self.motion, self.line, self.track, envelope, self.profile = self.data
        self.cfg, self.motion = copy.copy(self.cfg), copy.copy(self.motion)
        self.envelope = copy.copy(envelope)
        for name in ('lo', 'hi', 'reference_offsets', '_reference_slopes'):
            setattr(self.envelope, name, getattr(envelope, name).copy())

    def policy(self, target=None, margin=.08):
        if target is None:
            target = self.envelope.nominal(18. + self.cfg.lookahead)
        policy = Adversary(target, tracking_margin=margin,
                           speed_profile=self.profile, motion_config=self.motion)
        policy.speed = 1.
        return policy

    def compare(self, policy, attacker, defender=None, goal=None, speed=3.4, **kwargs):
        if defender is None:
            defender = state(self.line, attacker.s + 10., self.envelope.reference(attacker.s + 10.))
        if goal is None:
            goal = self.envelope.nominal(attacker.s + self.cfg.lookahead)
        if 'defender_paths' not in kwargs:
            kwargs['defender_paths'] = policy._defender_paths(defender, self.line, self.envelope, self.cfg)
        before = policy.__dict__.copy()
        args = attacker, defender, self.line, self.envelope, self.track, self.cfg, goal, speed
        expected = policy.preview(*args, **kwargs)
        actual = compiled_preview(policy, *args, **kwargs)
        assert_preview(self, actual, expected)
        self.assertEqual(policy.__dict__, before)
        # Exact Python first-command calculation, including rejected candidates.
        self.assertEqual(actual.target, expected.target)
        self.assertEqual(actual.speed, expected.speed)
        self.assertEqual(actual.acceleration, expected.acceleration)
        return actual

    def test_saved_truth_snapshots_and_candidate_demands(self):
        if not TRIAL.with_suffix('.csv').exists():
            self.skipTest('optional saved fast_headway_trial CSV unavailable')
        cfg, _, line, track, envelope, _ = self.data
        for t, policy, attacker, defender, goal, speed, kwargs in snapshots(self.data):
            with self.subTest(time=t, goal=goal, speed=speed):
                args = attacker, defender, line, envelope, track, cfg, goal, speed
                assert_preview(self, compiled_preview(policy, *args, **kwargs), policy.preview(*args, **kwargs))

    def test_random_actual_geometry_commands_yaw_and_wrapped_stations(self):
        rng = np.random.default_rng(20261117)
        for index in range(90):
            s = rng.uniform(0., self.line.length)
            lo, hi = self.envelope.interval(s)
            d = rng.uniform(lo - .12, hi + .12)
            angle = rng.choice([0., -.2, .3, math.pi, math.pi / 2, -2 * math.pi])
            attacker = state(self.line, s, d, rng.uniform(-.5, 4.), angle)
            policy = self.policy(self.envelope.nominal(s + self.cfg.lookahead), margin=rng.choice([0., .08, .2]))
            policy.speed = rng.uniform(0., 4.)
            policy._acceleration = rng.uniform(-2., 2.)
            if index % 3 == 0:
                attacker.s += self.line.length * rng.choice([-2., 2.])
            with self.subTest(index=index):
                self.compare(policy, attacker, goal=rng.uniform(-1.3, 1.3), speed=rng.uniform(0., 5.),
                             previous_steering=rng.uniform(-.4, .4), body_recovery=bool(index % 2),
                             measured_forward_speed=None if index % 2 else rng.uniform(-.5, 4.),
                             nominal=bool(index % 3 == 0))

    def test_body_budget_and_invalid_body_precede_footprint_and_collision(self):
        attacker = state(self.line, 18., .65)
        self.envelope.lo[:] = -.1
        self.envelope.hi[:] = .1
        self.envelope.reference_offsets[:] = 0.
        policy = self.policy(0., margin=0.)
        attacker.x, attacker.y = self.track.origin[:2] - 1.
        result = self.compare(policy, attacker, defender=attacker)
        self.assertEqual(result.reason, 'map_or_legal_preview:strict_body_band')
        self.assertEqual(result.safe_time, 0.)
        self.assertAlmostEqual(result.final_body_error, .55)
        self.envelope.lo[:] = .8
        self.envelope.hi[:] = .9
        result = self.compare(self.policy(.85, margin=.08), attacker)
        self.assertEqual(result.reason, 'map_or_legal_preview:body_tracking_budget_exceeded')

    def test_wall_precedes_collision_and_lateral_and_preserves_initial_summary(self):
        attacker = state(self.line, 18., .65, 4.)
        attacker.x, attacker.y = self.track.origin[:2] - 1.
        result = self.compare(self.policy(), attacker, defender=attacker, previous_steering=.4)
        self.assertEqual(result.reason, 'map_or_legal_preview:footprint')
        self.assertEqual(result.safe_time, 0.)
        self.assertEqual(result.progress, 0.)
        self.assertEqual(result.final_speed, attacker.speed)

    def test_later_raster_wall_returns_first_failed_node_not_terminal_summary(self):
        from scipy.ndimage import distance_transform_edt
        self.envelope.lo[:] = -1.2
        self.envelope.hi[:] = 1.2
        self.envelope.reference_offsets[:] = .65
        self.envelope._reference_slopes[:] = 0.
        attacker = state(self.line, 18., .65, 1.)
        policy = self.policy(.65)
        clear = self.compare(policy, attacker, goal=.65, speed=1.)
        self.assertTrue(clear.feasible)
        self.track = copy.copy(self.track)
        self.track._free = self.track._free.copy()
        obstacle = self.line.sample(18.8, .65)  # pyright: ignore[reportArgumentType]
        local = self.track._world_to_map @ (obstacle - self.track.origin[:2])
        col, row = np.floor(local / self.track.resolution).astype(int)
        self.track._free[row, col] = False
        self.track._dt = distance_transform_edt(np.pad(self.track._free, 1)) * self.track.resolution
        result = self.compare(policy, attacker, goal=.65, speed=1.)
        self.assertEqual(result.reason, 'map_or_legal_preview:footprint')
        self.assertGreater(result.safe_time, 0.)
        self.assertLess(result.safe_time, policy.preview_horizon)
        self.assertGreater(result.progress, 0.)
        self.assertLess(result.progress, clear.progress)

    def test_evolving_collision_precedes_lateral_and_sat(self):
        attacker = state(self.line, 18., .65, 4.)
        result = self.compare(self.policy(), attacker, defender=attacker, previous_steering=.4)
        self.assertEqual(result.reason, 'sampled_collision_preview:evolving_defender')
        self.assertEqual(result.safe_time, 0.)

    def test_new_and_current_steering_lateral_acceleration(self):
        attacker = state(self.line, 18., .65, 4.)
        result = self.compare(self.policy(), attacker, previous_steering=.4)
        self.assertEqual(result.reason, 'lateral_acceleration_preview')
        self.assertEqual(result.safe_time, 0.)
        # A low limit allows current straight steering but rejects the new slew
        # before integrating: tau is +dt while progress/final_speed remain old.
        self.motion.max_lateral_accel = .001
        result = self.compare(self.policy(), attacker, goal=-1., measured_forward_speed=4.)
        self.assertEqual(result.reason, 'lateral_acceleration_preview')
        self.assertEqual(result.safe_time, self.cfg.timestep)
        self.assertEqual(result.progress, 0.)
        self.assertEqual(result.final_speed, attacker.speed)

    def test_target_is_slewed_not_projected_and_goal_clipping_reported(self):
        policy = self.policy(1.1)
        self.envelope.lo[:] = -.1
        self.envelope.hi[:] = .8
        attacker = state(self.line, 18., .65, 0.)
        result = self.compare(policy, attacker, goal=2.)
        self.assertEqual(result.reason, 'map_or_legal_preview:lookahead_target')
        self.assertTrue(result.goal_clipped)
        self.assertGreater(result.target, .8)

    def test_recovery_monotonicity_and_insufficient_descent(self):
        self.envelope.lo[:] = -.8
        self.envelope.hi[:] = .5
        self.envelope.reference_offsets[:] = .3
        self.envelope._reference_slopes[:] = 0.
        for angle in (-.25, 0., .25):
            with self.subTest(angle=angle):
                result = self.compare(self.policy(.3, margin=.08), state(self.line, 18., .65, 1., angle),
                                      goal=.3, body_recovery=True)
                self.assertTrue(result.body_band_recovery)
                if angle == .25:
                    self.assertEqual(result.reason, 'body_band_recovery_error_increased')
                    self.assertEqual(result.safe_time, self.cfg.timestep)
                else:
                    self.assertEqual(result.reason, 'body_band_recovered')
                    self.assertEqual(result.final_body_error, 0.)
        policy = self.policy(.3)
        policy.preview_horizon = .2
        result = self.compare(policy, state(self.line, 18., .65, 1., -.25), goal=.3, body_recovery=True)
        self.assertEqual(result.reason, 'body_band_descent_partial')
        self.assertTrue(result.partial_body_recovery)
        self.assertLess(result.final_body_error, result.initial_body_error)
        policy = self.policy(.3)
        policy.speed = 0.
        result = self.compare(policy, state(self.line, 18., .65, 0.), speed=0., body_recovery=True)
        self.assertEqual(result.reason, 'body_band_recovery_insufficient_descent')

    def test_deferred_sat_retains_terminal_summary_and_hypothesis_order(self):
        self.envelope.lo[:] = -1.2
        self.envelope.hi[:] = 1.2
        self.envelope.reference_offsets[:] = .65
        self.envelope._reference_slopes[:] = 0.
        attacker = state(self.line, 18., .65, .2)
        defender = state(self.line, 28., .65, 0.)
        policy = self.policy(.65)
        policy.speed = .2
        paths = policy._defender_paths(defender, self.line, self.envelope, self.cfg)
        # Keep the longitudinal hypothesis far away but its actual rectangle at
        # the attacker. This isolates deferred oriented SAT from the Frenet gate.
        paths[:, :, :3] = [attacker.x, attacker.y, attacker.yaw + math.pi / 2]
        result = self.compare(policy, attacker, defender, speed=.2, defender_paths=paths)
        self.assertEqual(result.reason, 'sampled_rectangle_collision_preview:hold_constant')
        self.assertEqual(result.safe_time, 0.)
        self.assertGreater(result.progress, .1)
        self.assertNotEqual(result.final_speed, attacker.speed)

    def test_parent_dispatch_matches_explicit_backend_with_scalar_oracle(self):
        self.assertFalse(self.motion.compiled_attack_preview)
        attacker = state(self.line, 18., .65, 1.)
        defender = state(self.line, 28., .1, 1.)
        policy = self.policy()
        paths = policy._defender_paths(defender, self.line, self.envelope, self.cfg)
        args = attacker, defender, self.line, self.envelope, self.track, self.cfg, policy.target, 1.
        expected = policy.preview(*args, defender_paths=paths)
        self.motion.compiled_attack_preview = True
        actual = policy.preview(*args, defender_paths=paths)
        assert_preview(self, actual, expected)
        assert_preview(self, actual, compiled_preview(policy, *args, defender_paths=paths))

    def test_numeric_path_matches_every_reference_map_query(self):
        attacker = state(self.line, 18., .65, 1.)
        policy = self.policy()
        poses = []
        scalar = self.track.footprint_safe
        batch = self.track.footprints_safe

        def record(position, yaw, *args):
            poses.append([*position, yaw])
            return scalar(position, yaw, *args)

        defender = state(self.line, 28., .1, 1.)
        paths = policy._defender_paths(defender, self.line, self.envelope, self.cfg)
        args = attacker, defender, self.line, self.envelope, self.track, self.cfg, policy.target, 1.
        with patch.object(self.track, 'footprint_safe', side_effect=record):
            expected = policy.preview(*args, defender_paths=paths)
        with patch.object(self.track, 'footprints_safe', wraps=batch) as query:
            actual = compiled_preview(policy, *args, defender_paths=paths)
        assert_preview(self, actual, expected)
        np.testing.assert_allclose(query.call_args.args[0], poses, rtol=2e-11, atol=2e-11)
        self.assertEqual(len(poses), math.ceil(policy.preview_horizon / self.cfg.timestep) + 1)

    def test_rotated_map_origin_and_yaw_seam(self):
        # Transform the real map and line together; preserve the raster and EDT,
        # exercise its world-to-map yaw transform rather than an always-safe mock.
        angle = 2.8
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        shift = np.array([8., -3.])
        self.line = ClosedRaceline(self.line.points @ rotation.T + shift)
        self.track = copy.copy(self.track)
        self.track.origin = np.r_[self.track.origin[:2] @ rotation.T + shift, self.track.origin[2] + angle]
        c, s = np.cos(self.track.origin[2]), np.sin(self.track.origin[2])
        self.track._world_to_map = np.array([[c, s], [-s, c]])
        self.envelope.line = self.line
        for station in (18., self.line.length - .02):
            for yaw_offset in (0., -2 * math.pi):
                attacker = state(self.line, station, self.envelope.reference(station), .5, yaw_offset)
                self.compare(self.policy(self.envelope.nominal(station + self.cfg.lookahead)), attacker)

    def test_fractional_horizon_and_nondefault_action_grid(self):
        self.cfg.timestep = .03
        policy = self.policy()
        policy.preview_horizon = .071
        self.compare(policy, state(self.line, 18., .65, 1.))
        policy.preview_horizon = 0.
        self.compare(policy, state(self.line, 18., .65, 1.))

    def test_numeric_tolerance_edges_do_not_expand_body_budget(self):
        self.envelope.lo[:] = -.8
        self.envelope.hi[:] = .5
        self.envelope.reference_offsets[:] = .3
        policy = self.policy(.3)
        policy.preview_horizon = 0.
        for delta in (-1e-12, 1e-12):
            d = .5 + policy.tracking_margin + policy.legal_numeric_tolerance + delta
            result = self.compare(policy, state(self.line, 18., d, 0.), speed=0.)
            self.assertEqual(result.feasible, delta < 0.)
            self.assertFalse(result.strict_body_legal)
            self.assertEqual(result.within_tracking_budget, delta < 0.)

    def test_jerk_release_through_stop_and_negative_body_speed(self):
        for command in (0., 1e-5, .01, .2):
            for forward in (-.2, 0., 1.):
                policy = self.policy()
                policy.speed, policy._acceleration = command, -self.motion.max_braking
                self.compare(policy, state(self.line, 18., .65, forward), speed=0.,
                             measured_forward_speed=forward)

    def test_footprint_geometry_dimensions_and_margin(self):
        for length, width, margin in ((0., 0., 0.), (.58, .31, .04), (.8, .45, .12)):
            self.cfg.car_length, self.cfg.car_width, self.cfg.boundary_margin = length, width, margin
            self.compare(self.policy(), state(self.line, 18., .65, .5))

    def test_future_invalid_nominal_raises_but_not_before_an_earlier_map_failure(self):
        self.envelope.lo[:] = -.8
        self.envelope.hi[:] = .8
        self.envelope.reference_offsets[:] = 0.
        self.envelope._reference_slopes[:] = 0.
        attacker = state(self.line, 18., .65, 1.)
        defender = state(self.line, 28., .1, 1.)
        policy = self.policy(0.)
        paths = policy._defender_paths(defender, self.line, self.envelope, self.cfg)
        invalid = np.searchsorted(self.envelope.s, attacker.s + self.cfg.lookahead) + 1
        self.envelope.lo[invalid], self.envelope.hi[invalid] = 1., -1.
        args = attacker, defender, self.line, self.envelope, self.track, self.cfg, 0., 1.
        with self.assertRaises(ValueError) as expected:
            policy.preview(*args, defender_paths=paths)
        with self.assertRaises(ValueError) as actual:
            compiled_preview(policy, *args, defender_paths=paths)
        self.assertEqual(str(actual.exception), str(expected.exception))
        # Change the actual raster clearance data, not the certification methods.
        # The numeric loop still encounters the later invalid band, but reporting
        # must stop at the initial wall and must not propagate that later error.
        self.track = copy.copy(self.track)
        self.track._dt = np.zeros_like(self.track._dt)
        result = self.compare(policy, attacker, defender, goal=0., speed=1., defender_paths=paths)
        self.assertEqual(result.reason, 'map_or_legal_preview:footprint')
        self.assertEqual(result.safe_time, 0.)

    def test_optional_paths_computed_once_and_no_cross_call_cache(self):
        policy = self.policy()
        attacker = state(self.line, 18., .65, 1.)
        defender = state(self.line, 28., .1, 1.)
        args = attacker, defender, self.line, self.envelope, self.track, self.cfg, policy.target, 3.4
        with patch.object(policy, '_defender_paths', wraps=policy._defender_paths) as build:
            first = compiled_preview(policy, *args)
            self.assertEqual(build.call_count, 1)
        self.compare(policy, attacker, defender, speed=3.4)
        self.cfg.car_width *= 1.2
        self.envelope.hi[:] -= .05
        self.motion.max_jerk = 2.
        second = self.compare(policy, attacker, defender, speed=3.4)
        self.assertNotEqual(first.acceleration, second.acceleration)

    def test_custom_and_legacy_types_fall_back_without_evaluation(self):
        attacker = state(self.line, 18., .65)
        policy = self.policy()
        args = [attacker, attacker, self.line, self.envelope, self.track, self.cfg, .65, 1.]
        legacy = Adversary(.65)
        self.assertIsNone(compiled_preview(legacy, *args))
        self.assertIsNone(compiled_preview(policy, *args, maneuver=object()))
        self.assertIsNone(compiled_preview(policy, *args, horizon=2.))
        self.assertIsNone(compiled_preview(policy, *args, defender_paths=np.zeros((3, 41, 6))))
        for index in (2, 3, 4):
            changed = args.copy()
            changed[index] = object()
            self.assertIsNone(compiled_preview(policy, *changed))
        class CustomMap(TrackMap):
            pass
        custom = CustomMap.__new__(CustomMap)
        changed = args.copy()
        changed[4] = custom
        self.assertIsNone(compiled_preview(policy, *changed))
        policy.speed_profile = object()
        self.assertIsNone(compiled_preview(policy, *args))


def benchmark(repeat=5, profile=False):
    data = fixture()
    cfg, _, line, track, envelope, _ = data
    cases = snapshots(data)
    check = unittest.TestCase()

    def run(compiled):
        results = []
        for _, policy, attacker, defender, goal, speed, kwargs in cases:
            args = attacker, defender, line, envelope, track, cfg, goal, speed
            results.append(compiled_preview(policy, *args, **kwargs) if compiled else policy.preview(*args, **kwargs))
        return results

    expected, actual = run(False), run(True)  # warm Numba and geometry outside timing
    for a, b in zip(actual, expected):
        assert_preview(check, a, b)
    print(f'Parity: {len(cases)} saved-state candidate previews; all Preview fields within 2e-11')
    print('Reasons:', {reason: sum(p.reason == reason for p in expected) for reason in sorted({p.reason for p in expected})})
    timings = {False: [], True: []}
    for _ in range(repeat):
        for compiled in (False, True):
            start = time.perf_counter()
            run(compiled)
            timings[compiled].append(time.perf_counter() - start)
    scalar, compiled = (float(np.median(timings[key])) for key in (False, True))
    print(f'{len(cases)} candidates: Python {scalar * 1000:.3f} ms; compiled {compiled * 1000:.3f} ms; {scalar / compiled:.2f}x')
    print(f'Per candidate: Python {scalar / len(cases) * 1000:.3f} ms; compiled {compiled / len(cases) * 1000:.3f} ms')
    # Per-plan defender hypotheses are common cost, not hidden in the speedup.
    start = time.perf_counter()
    unique = cases[::9]
    for _, policy, _, defender, _, _, _ in unique:
        policy._defender_paths(defender, line, envelope, cfg)
    paths_ms = (time.perf_counter() - start) * 1000
    print(f'Four-hypothesis prediction: {paths_ms:.3f} ms / {len(unique)} snapshot plans (once/plan, shared across candidates)')
    print(f'Including once/plan prediction: {(scalar * 1000 + paths_ms) / (compiled * 1000 + paths_ms):.2f}x')
    if TRIAL.with_suffix('.json').exists():
        with TRIAL.with_suffix('.json').open() as stream:
            saved = json.load(stream)
        print('Saved trial:', {k: saved[k] for k in ('elapsed', 'wall_seconds', 'action_cycles')})
        print('Saved pipeline:', saved['pipeline_latency'])
    if profile:
        for compiled in (False, True):
            profiler = cProfile.Profile()
            profiler.runcall(run, compiled)
            output = io.StringIO()
            pstats.Stats(profiler, stream=output).strip_dirs().sort_stats('cumtime').print_stats(14)
            print(('Compiled' if compiled else 'Python') + ' warmed candidate profile:\n' + output.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--repeat', type=int, default=5)
    args, remaining = parser.parse_known_args()
    if args.benchmark:
        benchmark(args.repeat, args.profile)
    else:
        unittest.main(argv=[__file__, *remaining])
