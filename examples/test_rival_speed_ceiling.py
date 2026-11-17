"""CPU-only longitudinal ceiling and scalar/compiled attack parity regressions.

Run: .venv-py315/bin/python -m unittest discover -s examples -p test_rival_speed_ceiling.py
"""
import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from rival_adversary import Adversary, advance_speed
from rival_attack_rollout import _advance, compiled_preview
from rival_control import advance_speed_command
from test_rival_attack_rollout import assert_preview, fixture, state


class SpeedCeilingTests(unittest.TestCase):
    dt = .01
    limits = (2., 3., 6.)

    def step(self, speed, acceleration, goal, dt=None):
        dt = self.dt if dt is None else dt
        result = advance_speed_command(speed, acceleration, goal, dt, *self.limits)
        new_speed, new_acceleration = result
        self.assertTrue(all(math.isfinite(v) for v in result))
        self.assertGreaterEqual(new_speed, 0.)
        self.assertGreaterEqual(new_acceleration, -self.limits[1] - 1e-12)
        self.assertLessEqual(new_acceleration, self.limits[0] + 1e-12)
        self.assertLessEqual(abs(new_acceleration - acceleration), self.limits[2] * dt + 1e-12)
        # Check the emitted speed derivatives too, not just a reported state.
        self.assertAlmostEqual((new_speed - speed) / dt, new_acceleration, delta=1e-12)
        self.assertLessEqual(abs((new_speed - speed) / dt - acceleration) / dt,
                             self.limits[2] + 1e-9)
        return result

    def test_constant_ceiling_from_rest_including_small_and_coarse_steps(self):
        for dt in (.01, .05, .1):
            for goal in (3.4, 3., .1):
                with self.subTest(dt=dt, goal=goal):
                    speed = acceleration = 0.
                    released_below_goal = False
                    for _ in range(math.ceil(8. / dt)):
                        old_acceleration = acceleration
                        speed, acceleration = self.step(speed, acceleration, goal, dt)
                        self.assertLessEqual(speed, goal + 1e-12)
                        released_below_goal |= acceleration < old_acceleration and speed < goal - .001
                    self.assertTrue(released_below_goal)
                    self.assertAlmostEqual(speed, goal, delta=1e-10)
                    self.assertAlmostEqual(acceleration, 0., delta=1e-10)

    def test_infinite_goal_then_reachable_finite_and_mutable_goals(self):
        speed = acceleration = 0.
        for _ in range(60):
            speed, acceleration = self.step(speed, acceleration, math.inf)
        self.assertEqual(acceleration, self.limits[0])
        # All goal changes here leave enough positive-acceleration headroom.
        for goal in (3.4, 4., 5.):
            for _ in range(500):
                speed, acceleration = self.step(speed, acceleration, goal)
                self.assertLessEqual(speed, goal + 1e-12)
            self.assertAlmostEqual(speed, goal, delta=1e-10)
        # A mutable, falling goal may cease to be reachable, but never jumps.
        for goal in np.linspace(5., 1., 600):
            speed, acceleration = self.step(speed, acceleration, goal)
        for _ in range(600):
            speed, acceleration = self.step(speed, acceleration, 1.)
        self.assertAlmostEqual(speed, 1., delta=1e-10)

    def test_sudden_lower_infeasible_goal_releases_at_jerk_limit(self):
        speed, acceleration = 3.3, 2.
        next_speed, next_acceleration = self.step(speed, acceleration, .1)
        self.assertAlmostEqual(next_acceleration, 2. - 6. * self.dt)
        self.assertGreater(next_speed, speed)
        speed, acceleration = next_speed, next_acceleration
        saw_braking = False
        for _ in range(1000):
            speed, acceleration = self.step(speed, acceleration, .1)
            saw_braking |= acceleration < -1.
        self.assertTrue(saw_braking)
        self.assertAlmostEqual(speed, .1, delta=1e-10)
        self.assertAlmostEqual(acceleration, 0., delta=1e-10)

    def test_initial_overspeed_is_not_clipped(self):
        speed, acceleration = self.step(4., 1., 3.4)
        self.assertAlmostEqual(acceleration, .94)
        self.assertAlmostEqual(speed, 4.0094)
        speed, acceleration = self.step(4., 0., 3.4)
        self.assertAlmostEqual(acceleration, -.06)
        self.assertAlmostEqual(speed, 3.9994)

    def test_stop_restart_preserves_braking_budget_and_derivatives(self):
        speed = acceleration = 0.
        for goal in (3.4, 0., .1, 0., 3., 0., 3.4):
            for _ in range(1000):
                speed, acceleration = self.step(speed, acceleration, goal)
            self.assertAlmostEqual(speed, goal, delta=1e-10)
            self.assertAlmostEqual(acceleration, 0., delta=1e-10)
        # No positive headroom logic changes the old stopping release budget.
        speed, acceleration = 2., 0.
        legacy = SimpleNamespace(max_acceleration=2., max_braking=3., max_jerk=6.)
        for _ in range(500):
            expected = advance_speed(speed, acceleration, 0., self.dt, legacy)
            speed, acceleration = self.step(speed, acceleration, 0.)
            self.assertEqual((speed, acceleration), expected)

    def test_shared_python_jit_wrapper_and_numeric_step_parity(self):
        cfg = SimpleNamespace(max_acceleration=2., max_braking=3., max_jerk=6., corner_tracking=True)
        motion = (2., 3., 6., 8.)
        speed = acceleration = 0.
        for goal in (math.inf, 3.4, .1, 0., 3.):
            for _ in range(300):
                args = speed, acceleration, goal, self.dt
                actual = advance_speed_command(*args, *self.limits)
                self.assertEqual(actual, advance_speed_command.py_func(*args, *self.limits))
                self.assertEqual(actual, advance_speed(*args, cfg))
                self.assertEqual(actual, _advance(*args, motion, True))
                speed, acceleration = actual

    def test_legacy_default_math_unchanged(self):
        cfg = SimpleNamespace(max_acceleration=2., max_braking=3., max_jerk=6.)
        motion = (2., 3., 6., 8.)
        speed = acceleration = 0.
        maximum = 0.
        for goal in (3.4, 0., .1, 3.):
            for _ in range(500):
                desired = float(np.clip((goal - speed) / self.dt, -3., 2.))
                a = acceleration + float(np.clip(desired - acceleration, -.06, .06))
                a = max(a, -max(0., math.sqrt(.06 ** 2 + 12. * speed) - .06))
                expected = max(0., speed + a * self.dt), a
                self.assertEqual(advance_speed(speed, acceleration, goal, self.dt, cfg), expected)
                np.testing.assert_allclose(_advance(speed, acceleration, goal, self.dt, motion),
                                           expected, rtol=0., atol=1e-14)
                cfg.corner_tracking = False
                self.assertEqual(advance_speed(speed, acceleration, goal, self.dt, cfg), expected)
                speed, acceleration = expected
                if goal == 3.4:
                    maximum = max(maximum, speed)
        self.assertGreater(maximum, 3.6)  # Prove the opt-in fixes, not rewrites, legacy math.


class CornerAttackParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()

    def test_all_preview_fields_and_first_acceleration_without_mutation(self):
        cfg, motion, line, track, envelope, profile = self.data
        motion = copy.copy(motion)
        motion.corner_tracking = True
        motion.compiled_attack_preview = False
        rng = np.random.default_rng(103)
        cases = [(103., 3.3, 2., 3.4), (18., 0., 0., .1),
                 (18., 4., 1., 3.), (18., .1, -.3, 0.), (18., 0., 0., 3.4)]
        cases += [(rng.uniform(0., line.length), rng.uniform(.5, 4.),
                   rng.uniform(-1., 2.), rng.choice([0., .1, 3., 3.4])) for _ in range(20)]
        for station, command_speed, acceleration, goal_speed in cases:
            with self.subTest(s=station, command_speed=command_speed, goal=goal_speed):
                goal = envelope.nominal(station + cfg.lookahead)
                attacker = state(line, station, envelope.reference(station), command_speed)
                defender = state(line, station + 10., envelope.reference(station + 10.), 1.)
                policy = Adversary(goal, tracking_margin=.08,
                                   speed_profile=profile, motion_config=motion)
                policy.speed, policy._acceleration = command_speed, acceleration
                paths = policy._defender_paths(defender, line, envelope, cfg)
                before = policy.__dict__.copy()
                args = attacker, defender, line, envelope, track, cfg, goal, goal_speed
                expected = policy.preview(*args, defender_paths=paths)
                actual = compiled_preview(policy, *args, defender_paths=paths)
                assert_preview(self, actual, expected)
                self.assertEqual(actual.speed, expected.speed)
                self.assertEqual(actual.acceleration, expected.acceleration)
                demand = min(goal_speed, policy.reference_speed_cap(station, envelope, cfg),
                             profile.limit(station, attacker.speed))
                first = advance_speed_command(command_speed, acceleration, demand, cfg.timestep, 2., 3., 6.)
                self.assertEqual((actual.speed, actual.acceleration), first)
                self.assertEqual(policy.__dict__, before)
                motion.compiled_attack_preview = True
                assert_preview(self, policy.preview(*args, defender_paths=paths), expected)
                self.assertEqual(policy.__dict__, before)
                motion.compiled_attack_preview = False


if __name__ == '__main__':
    unittest.main()
