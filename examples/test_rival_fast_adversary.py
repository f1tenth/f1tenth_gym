"""Opt-in attacker regressions; no Gym episodes or recorded output dependencies."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rival_adversary import Adversary, Preview, advance_speed
from rival_geometry import ClosedRaceline
from rival_speed import SpeedProfile
from rival_strategy import CarState, Settings, pursuit
from test_rival_adversary import Envelope, Line, Track, state


class Profile:
    def __init__(self, limit=20.):
        self.value = limit
        self.calls = []

    def limit(self, s, measured_speed=0.):
        self.calls.append((s, measured_speed))
        return self.value(s) if callable(self.value) else self.value


def motion(**kwargs):
    return SimpleNamespace(**dict(dict(max_acceleration=2., max_braking=3.,
                                       max_jerk=6., max_lateral_accel=3.), **kwargs))


class FastAdversaryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Settings(), attacker_speed=20., defender_speed=20.)
        self.motion = motion()

    def policy(self, target=0., profile=None, **kwargs):
        return Adversary(target, speed_profile=profile or Profile(),
                         motion_config=self.motion, **kwargs)

    def plan(self, policy, t=3., scenario='approach', car=None, defender=None, envelope=None,
             track=None):
        car = car or state(5., speed=policy.speed)
        return policy.plan(t, scenario, car, defender or state(40.), Line(),
                           envelope or Envelope(), track or Track(), self.cfg,
                           measured_forward_speed=car.speed)

    def test_opt_in_requires_both_arguments_and_valid_motion_limits(self):
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            Adversary(0., speed_profile=Profile())
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            Adversary(0., motion_config=self.motion)
        for name in vars(self.motion):
            for value in (0., -1., math.inf, math.nan):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Adversary(0., speed_profile=Profile(), motion_config=motion(**{name: value}))

    def test_defaults_keep_legacy_first_command(self):
        legacy = self.plan(Adversary(0.))
        fast = self.plan(self.policy())
        self.assertAlmostEqual(legacy.speed, .04)
        self.assertAlmostEqual(fast.speed, .0024)
        self.assertAlmostEqual(fast.acceleration, .12)
        self.assertTrue(math.isinf(legacy.profile_speed_cap))

    def test_pure_command_step_jerk_through_stop_and_restart(self):
        dt = self.cfg.timestep
        speed, acceleration = 2., 0.
        for index, desired in enumerate([0.] * 300 + [2.] * 200 + [0.] * 300):
            previous_speed, previous_acceleration = speed, acceleration
            speed, acceleration = advance_speed(speed, acceleration, desired, dt, self.motion)
            emitted_acceleration = (speed - previous_speed) / dt
            self.assertAlmostEqual(emitted_acceleration, acceleration, places=10)
            self.assertGreaterEqual(speed, 0.)
            self.assertLessEqual(acceleration, self.motion.max_acceleration + 1e-10)
            self.assertGreaterEqual(acceleration, -self.motion.max_braking - 1e-10)
            self.assertLessEqual(abs(acceleration - previous_acceleration),
                                 self.motion.max_jerk * dt + 1e-10)
            if index in (299, 799):
                self.assertLess(speed, 1e-10)
                self.assertLess(abs(acceleration), 1e-9)
            if index == 499:
                self.assertGreater(speed, 1.9)

    def test_plan_emitted_jerk_through_stop_and_restart(self):
        profile = Profile(1.)
        policy = self.policy(profile=profile)
        policy.preview_horizon = .1
        previous_speed = previous_acceleration = 0.
        for index, limit in enumerate([1.] * 40 + [0.] * 120 + [1.] * 40):
            profile.value = limit
            command = self.plan(policy, t=3. + index * self.cfg.timestep)
            acceleration = (command.speed - previous_speed) / self.cfg.timestep
            self.assertAlmostEqual(acceleration, command.acceleration, places=10)
            self.assertLessEqual(abs(acceleration - previous_acceleration),
                                 self.motion.max_jerk * self.cfg.timestep + 1e-9)
            self.assertGreaterEqual(command.speed, 0.)
            self.assertEqual(command.profile_speed_cap, limit)
            previous_speed, previous_acceleration = command.speed, acceleration
            if index == 159:
                self.assertLess(command.speed, 1e-10)
        self.assertGreater(command.speed, .5)

    def test_preview_is_pure_and_plan_commits_first_not_terminal_acceleration(self):
        policy = self.policy(profile=Profile(1.2))
        policy.speed, policy._acceleration = .6, .4
        car = state(5., speed=.6)
        expected = advance_speed(.6, .4, 1.2, self.cfg.timestep, self.motion)
        for goal in (0., .3, -.3):
            prediction = policy.preview(car, state(40.), Line(), Envelope(), Track(), self.cfg,
                                        goal, 20., measured_forward_speed=.6)
            self.assertTrue(prediction.feasible)
            self.assertAlmostEqual(prediction.speed, expected[0])
            self.assertAlmostEqual(prediction.acceleration, expected[1])
            self.assertEqual((policy.target, policy.speed, policy._acceleration), (0., .6, .4))
        command = self.plan(policy, car=car)
        self.assertAlmostEqual(command.speed, expected[0])
        self.assertAlmostEqual(command.acceleration, expected[1])
        self.assertEqual(policy._acceleration, command.acceleration)
        self.assertEqual(pursuit(car, Line(), command.target, command.speed, 0., self.cfg)[1],
                         prediction.speed)

    def test_guarded_candidates_clone_command_state_and_commit_selected_first(self):
        policy = self.policy()
        policy.speed, policy._acceleration = .6, .4
        predictions = []
        original = policy.preview

        def checked(*args, **kwargs):
            self.assertEqual((policy.speed, policy._acceleration), (.6, .4))
            prediction = original(*args, **kwargs)
            self.assertEqual((policy.speed, policy._acceleration), (.6, .4))
            predictions.append(prediction)
            return prediction

        with patch.object(policy, 'preview', side_effect=checked):
            command = self.plan(policy, scenario='left-pass', defender=state(10., -.5, 0.),
                                envelope=Envelope(-.1, .1))
        self.assertTrue(command.selected_path_feasible)
        self.assertEqual(len(predictions), 3)
        selected = max(predictions[1:], key=lambda p: p.progress)
        self.assertEqual((command.speed, command.acceleration), (selected.speed, selected.acceleration))
        self.assertAlmostEqual(command.acceleration, .52)

    def test_profile_caps_initial_scenario_demand_and_reports_cap(self):
        policy = self.policy(profile=Profile(.3))
        with patch('rival_adversary.advance_speed', wraps=advance_speed) as step:
            command = self.plan(policy)
        self.assertEqual(command.profile_speed_cap, .3)
        self.assertTrue(all(call.args[2] == .3 for call in step.call_args_list))
        self.assertAlmostEqual(command.speed, .0024)

    def test_caps_are_queried_at_predicted_stations_with_predicted_speed(self):
        profile = Profile(lambda s: 1. if s < 5.04 else .2)
        policy = self.policy(profile=profile)
        policy.speed = 1.
        with patch('rival_adversary.advance_speed', wraps=advance_speed) as step:
            prediction = policy.preview(state(5., speed=1.), state(40.), Line(), Envelope(),
                Track(), self.cfg, 0., 20., measured_forward_speed=1.)
        self.assertTrue(prediction.feasible)
        demands = [call.args[2] for call in step.call_args_list]
        self.assertIn(1., demands)
        self.assertIn(.2, demands)
        self.assertTrue(any(s > 5.04 for s, _ in profile.calls))
        self.assertTrue(any(v < .99 for _, v in profile.calls))
        self.assertEqual(policy._acceleration, 0.)

    def test_initial_cap_is_not_frozen_for_entire_preview(self):
        profile = Profile(lambda s: .2 if s < 5.01 else 2.)
        policy = self.policy(profile=profile)
        policy.speed = 1.
        with patch('rival_adversary.advance_speed', wraps=advance_speed) as step:
            policy.preview(state(5., speed=1.), state(40.), Line(), Envelope(), Track(),
                           self.cfg, 0., 20., measured_forward_speed=1.)
        self.assertEqual(step.call_args_list[0].args[2], .2)
        self.assertTrue(any(call.args[2] == 2. for call in step.call_args_list))

    def test_real_periodic_profile_queries_cross_lap_closure(self):
        angles = np.linspace(0., 2 * np.pi, 80, endpoint=False)
        line = ClosedRaceline(8 * np.column_stack((np.cos(angles), np.sin(angles))))
        profile = SpeedProfile(line, None, self.cfg, self.motion, speed_limit=20.)
        policy = self.policy(profile=profile)
        policy.speed = 1.
        s = line.length - .05
        car = CarState(s, 0., 1., *line.sample(s), line.heading(s))
        other_s = line.length / 2
        other = CarState(other_s, 0., 1., *line.sample(other_s), line.heading(other_s))
        with patch.object(profile, 'limit', wraps=profile.limit) as limit:
            prediction = policy.preview(car, other, line, Envelope(), Track(), self.cfg,
                                        0., 20., measured_forward_speed=1.)
        self.assertTrue(prediction.feasible, prediction.reason)
        stations = [call.args[0] for call in limit.call_args_list]
        self.assertTrue(any(s > line.length - .1 for s in stations))
        self.assertTrue(any(s < 1. for s in stations))
        self.assertGreater(prediction.progress, .5)

    def test_lateral_acceleration_uses_body_velocity_and_actual_steering(self):
        car = CarState(5., 0., .1, 5., 0., math.acos(.1 / 3.))
        kwargs = dict(previous_steering=.2, measured_forward_speed=3.)
        result = self.policy().preview(car, state(40.), Line(), Envelope(), Track(),
                                      self.cfg, 0., 20., **kwargs)
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, 'lateral_acceleration_preview')
        self.assertEqual(result.safe_time, 0.)

    def test_lateral_acceleration_checks_new_slew_limited_steering(self):
        policy = self.policy(1.)
        policy.speed = 3.
        result = policy.preview(state(5., speed=3.), state(40.), Line(), Envelope(), Track(),
                                self.cfg, 1., 3., measured_forward_speed=3.)
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, 'lateral_acceleration_preview')
        self.assertGreater(result.safe_time, 0.)

    def test_guarded_nominal_can_keep_useful_speed_without_attack_credit(self):
        policy = self.policy()
        policy.speed = 1.
        result = self.plan(policy, scenario='left-pass', defender=state(10., -.5, 0.),
                           envelope=Envelope(-.1, .1))
        self.assertTrue(result.guard_intervened)
        self.assertTrue(result.selected_path_feasible)
        self.assertTrue(result.fallback.startswith('nominal:'))
        self.assertGreater(result.speed, 1.)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_twenty_mps_does_not_turn_repair_crawl_into_four(self):
        policy = self.policy()
        policy.speed = 1.
        demands = []

        def preview(*args, **kwargs):
            goal, speed = args[-2:]
            demands.append(speed)
            feasible = speed <= .5
            return Preview(feasible, 'checked' if feasible else 'blocked', .8, 0., .1,
                           .8 * speed if feasible else 0., final_speed=speed if feasible else 0.)

        with patch.object(policy, 'preview', side_effect=preview):
            result = self.plan(policy, scenario='left-pass', defender=state(10., -.5, 0.),
                               envelope=Envelope(-.1, .1))
        self.assertTrue(result.selected_path_feasible)
        self.assertTrue(result.fallback.endswith(':0.5'))
        self.assertNotIn(4., demands)
        self.assertNotIn(2., demands)
        self.assertLessEqual(len(demands), 15)

    def test_meaningful_moving_candidate_beats_epsilon_or_coasting_stop(self):
        for residual_progress in (1e-7, .3):
            policy = self.policy()

            def preview(*args, **kwargs):
                goal, _ = args[-2:]
                if abs(goal - .25) < 1e-9:
                    return Preview(False, 'blocked_request', 0., 0., 0.)
                if kwargs.get('nominal'):
                    return Preview(True, 'coasting_stop', .8, 0., 0., residual_progress,
                                   final_speed=0.)
                return Preview(True, 'forward_hold', .8, .001, .1, .2, final_speed=.3)

            with self.subTest(residual_progress=residual_progress), patch.object(
                    policy, 'preview', side_effect=preview) as checked:
                result = self.plan(policy, scenario='left-pass', car=state(5., .05, 0.),
                                   defender=state(10., -.5, 0.), envelope=Envelope(-.1, .1))
            self.assertTrue(result.fallback.startswith('hold:'))
            self.assertGreater(result.speed, 0.)
            self.assertEqual(checked.call_count, 3)

    def test_repair_continuation_is_rechecked_and_expires(self):
        policy = self.policy()
        allow_attack = False

        def preview(*args, **kwargs):
            goal, _ = args[-2:]
            feasible = abs(goal) < 1e-9 or allow_attack
            return Preview(feasible, 'checked' if feasible else 'blocked', .8, 0., .2,
                           .2 if feasible else 0., final_speed=.3)

        with patch.object(policy, 'preview', side_effect=preview) as checked:
            first = self.plan(policy, t=3., scenario='left-pass', defender=state(10., -.5, 0.))
            allow_attack = True
            second = self.plan(policy, t=3.02, scenario='left-pass', defender=state(10., -.5, 0.))
            self.assertIn('continuing_checked_recovery', second.reason)
            self.assertEqual(second.fallback, first.fallback)
            self.assertTrue(checked.call_args.kwargs['nominal'])
            self.assertEqual(policy.executed_attack_steps, 0)
            third = self.plan(policy, t=3.22, scenario='left-pass', defender=state(10., -.5, 0.))
        self.assertEqual(third.fallback, 'none')
        self.assertFalse(third.guard_intervened)
        self.assertEqual(policy.executed_attack_steps, 1)

    def test_continuation_never_bypasses_a_new_unsafe_preview(self):
        policy = self.policy()
        policy.speed = 1.
        self.plan(policy, scenario='left-pass', defender=state(10., -.5, 0.),
                  envelope=Envelope(-.1, .1))
        self.assertIsNotNone(policy._fast_recovery)

        class Wall(Track):
            def footprint_safe(self, *args):
                return False

        result = self.plan(policy, t=3.02, track=Wall())
        self.assertTrue(result.unresolved)
        self.assertFalse(result.selected_path_feasible)
        self.assertIsNone(policy._fast_recovery)

    def test_hard_initial_wall_margin_remains_unresolved(self):
        margins = []

        class WallMargin(Track):
            def footprint_safe(self, position, yaw, length, width, margin):
                margins.append(margin)
                return margin == 0.

        policy = self.policy()
        result = self.plan(policy, track=WallMargin(), car=state(5., speed=0.))
        self.assertTrue(result.unresolved)
        self.assertFalse(result.selected_path_feasible)
        self.assertTrue(result.fallback.startswith('unresolved_brake:'))
        self.assertEqual(result.speed, 0.)
        self.assertTrue(all(margin == self.cfg.boundary_margin for margin in margins))
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_body_repair_requires_descent_and_never_counts_as_attack(self):
        policy = self.policy(.98)
        policy.speed = .5
        result = self.plan(policy, car=state(5., 1.02, .5), envelope=Envelope(-1., 1.))
        self.assertTrue(result.body_band_recovery)
        self.assertTrue(result.selected_path_feasible)
        self.assertLessEqual(result.predicted_max_body_error, result.initial_body_error + 1e-6)
        self.assertGreaterEqual(result.initial_body_error - result.predicted_final_body_error,
                                min(.005, .2 * result.initial_body_error))
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_collision_and_actual_target_band_are_not_relaxed(self):
        policy = self.policy(tracking_margin=.08)
        collision = self.plan(policy, car=state(5., speed=0.), defender=state(5., speed=0.))
        self.assertTrue(collision.unresolved)
        self.assertIn('sampled_collision_preview', collision.reason)
        policy.target = 1.04
        prediction = policy.preview(state(5., 1.04, 0.), state(40.), Line(), Envelope(-1., 1.),
            Track(), self.cfg, 1., 0., measured_forward_speed=0.)
        self.assertFalse(prediction.feasible)
        self.assertIn('lookahead_target', prediction.reason)
        self.assertAlmostEqual(prediction.target, 1.035)


class DefenderPredictionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Settings(), attacker_speed=3.4, defender_speed=3., offset_rate=.45)
        self.motion = motion(max_lateral_rate=.45)

    def policy(self, target=0.):
        return Adversary(target, speed_profile=Profile(3.4), motion_config=self.motion)

    @staticmethod
    def corner(sign, narrow=False):
        class Corner(Envelope):
            def nominal(self, s):
                return sign * (.4 + .6 * float(np.clip(s - 10., 0., 1.)))
        return Corner(-1.2, 1.2) if narrow else Corner()

    def test_reference_corner_squeeze_rejected_for_both_passing_sides(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                envelope = self.corner(sign)
                car, defender = state(9., sign, 2.), state(10., .4 * sign, 1.)
                legacy = Adversary(float(sign))
                legacy.speed = 2.
                fast = self.policy(float(sign))
                fast.speed = 2.
                args = (car, defender, Line(), envelope, Track(), self.cfg, 1.1 * sign, 2.)
                old = legacy.preview(*args, measured_forward_speed=2.)
                new = fast.preview(*args, measured_forward_speed=2.)
                self.assertTrue(old.feasible, old.reason)
                self.assertFalse(new.feasible)
                self.assertIn('collision_preview', new.reason)
                self.assertGreater(new.safe_time, 0.)
                self.assertLess(new.safe_time, fast.preview_horizon)

    def test_narrowing_room_triggers_following_before_alongside_and_reuses_paths(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                policy = self.policy(.4 * sign)
                policy.speed = 2.
                car, defender = state(8.2, .4 * sign, 2.), state(10., .4 * sign, 1.5)
                with patch.object(policy, '_defender_paths', wraps=policy._defender_paths) as predicted:
                    command = policy.plan(3., 'left-pass' if sign > 0 else 'right-pass', car,
                        defender, Line(), self.corner(sign, narrow=True), Track(), self.cfg,
                        measured_forward_speed=2.)
                self.assertEqual(predicted.call_count, 1)
                self.assertGreater(10. - car.s, self.cfg.car_length + self.cfg.separation_margin)
                self.assertIn('narrowing_defender_passing_corridor', command.reason)
                self.assertIn('rear_end_guard', command.reason)
                self.assertFalse(command.requested_corridor_valid)
                self.assertTrue(command.selected_path_feasible, command.reason)
                self.assertFalse(command.unresolved)
                self.assertLess(command.speed, 2.)
                self.assertEqual(policy.executed_attack_steps, 0)

    def test_hypotheses_start_at_actual_pose_and_bound_offset_and_heading(self):
        policy = self.policy()
        for sign in (-1, 1):
            defender = CarState(10., .4 * sign, 1., 10.03, .42 * sign, .1 * sign)
            paths = policy._defender_paths(defender, Line(), self.corner(sign), self.cfg)
            np.testing.assert_array_equal(paths[:, 0], np.tile(
                [defender.x, defender.y, defender.yaw, defender.s, defender.d, 0.], (4, 1)))
            np.testing.assert_allclose(paths[[0, 2], :, 4], defender.d)
            self.assertTrue(np.all(sign * (paths[[1, 3], -1, 4] - defender.d) > .1))
            self.assertLessEqual(np.max(np.abs(np.diff(paths[:, :, 4], axis=1))),
                                 self.cfg.offset_rate * self.cfg.timestep + 1e-12)
            self.assertLessEqual(np.max(np.abs(np.diff(paths[:, :, 2], axis=1))),
                                 self.motion.max_lateral_accel * self.cfg.timestep + 1e-12)
            self.assertGreater(paths[2, -1, 5], paths[0, -1, 5])
            np.testing.assert_allclose(paths[[0, 1], -1, 5], policy.preview_horizon)
            # The pose residual is anchored, not snapped to the raceline at t=dt.
            np.testing.assert_allclose(paths[:, 1, 0] - defender.x, paths[:, 1, 5])

    def test_prediction_retains_fixed_current_reference_bias(self):
        policy = self.policy()
        defender = state(10., .2, 1.)  # reference .4, measured bias -.2
        paths = policy._defender_paths(defender, Line(), self.corner(1), self.cfg)
        self.assertAlmostEqual(paths[1, 1, 4], .2 + self.cfg.offset_rate * self.cfg.timestep)
        self.assertLessEqual(paths[1, -1, 4], .2 + self.cfg.offset_rate * policy.preview_horizon + 1e-12)
        self.assertAlmostEqual(paths[0, -1, 4], .2)

    def test_oriented_rectangles_reject_clear_frenet_boxes_at_actual_pose(self):
        car = state(5., speed=0.)
        defender = CarState(5.55, .44, 0., 5.55, .44, math.pi / 4)
        args = (car, defender, Line(), Envelope(), Track(), self.cfg, 0., 0.)
        old = Adversary(0.).preview(*args, measured_forward_speed=0.)
        new = self.policy().preview(*args, measured_forward_speed=0.)
        self.assertTrue(old.feasible)
        self.assertFalse(new.feasible)
        self.assertIn('sampled_rectangle_collision_preview', new.reason)
        self.assertEqual(new.safe_time, 0.)

    def test_batched_sweep_padding_catches_translation_tunnelling(self):
        ego = np.array([[0., 0., 0.], [2., 0., 0.]])
        defender = np.array([[[1., 0., 0.], [1., 0., 0.]]])
        gaps = self.policy()._rectangle_gaps(ego, defender, self.cfg)
        self.assertEqual(gaps.shape, (1, 2))
        self.assertTrue(np.all(gaps <= 0.))
        # Each endpoint alone is separated even with the unchanged margin.
        self.assertGreater(1. - self.cfg.car_length - self.cfg.separation_margin, 0.)

    def test_sweep_rotation_uses_short_angle_across_heading_seam(self):
        from rival_robust import sat_clearance
        ego = np.array([[0., 0., math.radians(350)], [0., 0., math.radians(370)]])
        defender = np.array([[[2., 0., math.radians(10)], [2., 0., math.radians(-10)]]])
        raw = sat_clearance(ego[None], defender, self.cfg.car_length, self.cfg.car_width)
        rotation = math.hypot(self.cfg.car_length, self.cfg.car_width) / 2 * math.radians(20)
        expected = raw - self.cfg.separation_margin - rotation
        np.testing.assert_allclose(self.policy()._rectangle_gaps(ego, defender, self.cfg), expected)

    def test_reference_hypotheses_cross_track_seam_without_xy_or_yaw_teleport(self):
        angles = np.linspace(0., 2 * np.pi, 80, endpoint=False)
        line = ClosedRaceline(8 * np.column_stack((np.cos(angles), np.sin(angles))))
        s = line.length - .03
        defender = CarState(s, 0., 1., *line.sample(s), line.heading(s) + 2 * math.pi)
        paths = self.policy()._defender_paths(defender, line, Envelope(), self.cfg)
        self.assertTrue(np.any(paths[:, 1:, 3] < 1.))
        self.assertLess(np.max(np.linalg.norm(np.diff(paths[:, :, :2], axis=1), axis=2)), .1)
        self.assertLess(np.max(np.abs(np.diff(paths[:, :, 2], axis=1))), .1)
        np.testing.assert_allclose(paths[:, 0, 2], defender.yaw)
        self.assertTrue(np.all(np.diff(paths[:, :, 5], axis=1) > 0.))

    def test_labels_are_explicit_assumptions_and_legacy_does_not_predict(self):
        legacy = Adversary(0.)
        with patch.object(legacy, '_defender_paths', side_effect=AssertionError('fast only')):
            command = legacy.plan(3., 'approach', state(5., speed=0.), state(40.), Line(),
                                  Envelope(), Track(), self.cfg, measured_forward_speed=0.)
        self.assertEqual(command.defender_prediction_model, 'constant_frenet')
        self.assertNotIn('defender_model=', command.reason)
        fast = self.policy()
        command = fast.plan(3., 'approach', state(5., speed=0.), state(40.), Line(),
                            Envelope(), Track(), self.cfg, measured_forward_speed=0.)
        self.assertIn('hold_and_reference_bias', command.defender_prediction_model)
        self.assertIn('not_future_truth_or_exhaustive_defense=True', command.reason)
        prediction = fast.preview(state(5., speed=0.), state(40.), Line(), Envelope(), Track(),
                                  self.cfg, 0., 1., measured_forward_speed=0.)
        self.assertEqual(prediction.defender_prediction_model, command.defender_prediction_model)


class ScalarEnvelope:
    """Delegate to real rules without exposing their vectorizable implementation."""
    def __init__(self, envelope):
        self.envelope = envelope

    def interval(self, s):
        return self.envelope.interval(s)

    def nominal(self, s):
        return self.envelope.nominal(s)


class EnvelopeBatchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Settings(), attacker_speed=3.4, defender_speed=3., offset_rate=.45)
        self.policy = Adversary(0., speed_profile=Profile(3.4), motion_config=motion(max_lateral_rate=.45))

    @staticmethod
    def envelope():
        from rival_robust import TrackEnvelope
        # Real interpolation methods with inexpensive, deterministic table data.
        envelope = object.__new__(TrackEnvelope)
        envelope.line = Line()
        envelope.s = np.arange(0., envelope.line.length, .25)
        envelope.lo = np.full_like(envelope.s, -1.2)
        envelope.hi = np.full_like(envelope.s, 1.2)
        envelope.reference_offsets = .15 * np.sin(2 * np.pi * envelope.s / envelope.line.length)
        return envelope

    def paths(self, envelope, s=10., speed=2.):
        car = state(s, envelope.nominal(s), speed)
        return self.policy._defender_paths(car, envelope.line, envelope, self.cfg)

    def test_batched_interpolation_is_exact_at_nodes_and_periodic_seams(self):
        envelope = self.envelope()
        queries = np.array([[-200.1, -.1, 0.], [99.999, 100., 300.25]])
        lo, hi, reference = self.policy._envelope_samples(envelope, queries)
        expected_lo = np.array([envelope.interval(s)[0] for s in queries.ravel()]).reshape(queries.shape)
        expected_hi = np.array([envelope.interval(s)[1] for s in queries.ravel()]).reshape(queries.shape)
        expected_reference = np.array([envelope.nominal(s) for s in queries.ravel()]).reshape(queries.shape)
        np.testing.assert_array_equal(lo, expected_lo)
        np.testing.assert_array_equal(hi, expected_hi)
        np.testing.assert_array_equal(reference, expected_reference)
        self.assertIsNone(self.policy._envelope_samples(Envelope(), queries))

    def test_custom_subclass_keeps_scalar_nominal_override(self):
        from rival_robust import TrackEnvelope

        class Custom(TrackEnvelope):
            def nominal(self, s):
                return 1.1

        custom = object.__new__(Custom)
        custom.__dict__.update(self.envelope().__dict__)
        self.assertIsNone(self.policy._envelope_samples(custom, [0., 1.]))
        paths = self.paths(custom)
        self.assertFalse(self.policy._passing_room(paths, 1, 0., custom, self.cfg))

    def test_defender_paths_match_scalar_reference_exactly(self):
        envelope = self.envelope()
        scalar = ScalarEnvelope(envelope)
        for s in (10., 99.8):
            for speed in (-.5, 0., 3.4):
                with self.subTest(s=s, speed=speed):
                    car = CarState(s, .2, speed, s + .01, .2, .1)
                    expected = self.policy._defender_paths(car, envelope.line, scalar, self.cfg)
                    actual = self.policy._defender_paths(car, envelope.line, envelope, self.cfg)
                    np.testing.assert_array_equal(actual, expected)

    def test_passing_room_parity_both_sides_biases_and_seam(self):
        envelope = self.envelope()
        scalar = ScalarEnvelope(envelope)
        for s in (10., 97.):
            for speed in (0., 2., 6.):
                paths = self.paths(envelope, s, speed)
                for side in (-1, 1):
                    for bias in (-1.4, -.2, .2, 1.4):
                        with self.subTest(s=s, speed=speed, side=side, bias=bias):
                            expected = self.policy._passing_room(paths, side, bias, scalar, self.cfg)
                            self.assertEqual(self.policy._passing_room(
                                paths, side, bias, envelope, self.cfg), expected)

    def test_extended_reserve_rejects_narrowing_beyond_short_horizon_and_seam(self):
        for start, peak in ((10., 16.5), (97., 3.)):
            for side in (-1, 1):
                with self.subTest(start=start, side=side):
                    envelope = self.envelope()
                    distance = abs((envelope.s - peak + 50.) % 100. - 50.)
                    envelope.reference_offsets = side * 1.1 * np.maximum(0., 1. - distance / .6)
                    paths = self.paths(envelope, start)
                    # All short-rollout stations plus the original two lookaheads fit.
                    for s in np.unique(paths[:, :, 3]):
                        for ds in (0., self.cfg.lookahead, 2 * self.cfg.lookahead):
                            lo, hi = envelope.interval(s + ds)
                            goal = envelope.nominal(s + ds) + side * (
                                self.cfg.car_width + self.cfg.separation_margin)
                            self.assertLessEqual(lo, goal)
                            self.assertLessEqual(goal, hi)
                    self.assertFalse(self.policy._passing_room(paths, side, 0., envelope, self.cfg))
                    self.assertFalse(self.policy._passing_room(
                        paths, side, 0., ScalarEnvelope(envelope), self.cfg))

    def test_reserve_sampling_stations_and_quarter_metre_spacing_are_unchanged(self):
        envelope = self.envelope()
        paths = self.paths(envelope, s=97., speed=6.)
        with patch.object(self.policy, '_envelope_samples', wraps=self.policy._envelope_samples) as sampled:
            self.policy._passing_room(paths, 1, 0., envelope, self.cfg)
        speed = max(0., float(np.max(paths[:, :, 5])) / self.policy.preview_horizon)
        reserve = speed * (self.cfg.pass_offset / self.cfg.offset_rate
                           + self.policy.motion_config.max_acceleration / self.policy.motion_config.max_jerk)
        reserve += speed ** 2 / (2 * self.policy.motion_config.max_braking)
        future = paths[0, 0, 3] + np.linspace(0., reserve, max(2, int(math.ceil(reserve / .25)) + 1))
        self.assertLessEqual(np.max(np.diff(future)), .25 + 1e-12)
        expected = np.array([s + ds for s in np.unique(np.r_[paths[:, :, 3].ravel(), future])
                             for ds in (0., self.cfg.lookahead, 2 * self.cfg.lookahead)])
        np.testing.assert_array_equal(sampled.call_args.args[1], expected)

    def test_batching_avoids_scalar_envelope_queries(self):
        envelope = self.envelope()
        car = state(10., envelope.nominal(10.), 3.4)
        with patch.object(envelope, 'interval', wraps=envelope.interval) as interval, patch.object(
                envelope, 'nominal', wraps=envelope.nominal) as nominal:
            paths = self.policy._defender_paths(car, envelope.line, envelope, self.cfg)
            self.assertTrue(self.policy._passing_room(paths, 1, 0., envelope, self.cfg))
            self.assertEqual(interval.call_count, 0)
            self.assertEqual(nominal.call_count, 0)
        expected = self.policy._defender_paths(car, envelope.line, ScalarEnvelope(envelope), self.cfg)
        np.testing.assert_array_equal(paths, expected)

    def test_mutated_reference_is_rechecked_without_cross_step_cache(self):
        envelope = self.envelope()
        paths = self.paths(envelope)
        self.assertTrue(self.policy._passing_room(paths, 1, 0., envelope, self.cfg))
        envelope.reference_offsets[:] = 1.1
        self.assertFalse(self.policy._passing_room(paths, 1, 0., envelope, self.cfg))
        envelope.reference_offsets[:] = 0.
        self.assertTrue(self.policy._passing_room(paths, 1, 0., envelope, self.cfg))

    def test_numeric_tolerance_and_first_failure_order_match_scalar(self):
        envelope = self.envelope()
        envelope.reference_offsets[:] = 0.
        paths = self.paths(envelope)
        for delta in (-1e-10, 0., 1e-10):
            envelope.hi[:] = (self.cfg.car_width + self.cfg.separation_margin
                              - self.policy.legal_numeric_tolerance + delta)
            self.assertEqual(self.policy._passing_room(paths, 1, 0., envelope, self.cfg),
                             self.policy._passing_room(paths, 1, 0., ScalarEnvelope(envelope), self.cfg))
        center = self.cfg.car_width + self.cfg.separation_margin
        envelope.lo[:], envelope.hi[:] = center - .5, center + .5
        index = np.searchsorted(envelope.s, 12.)
        envelope.lo[index], envelope.hi[index] = center + .5, center - .5
        for source in (envelope, ScalarEnvelope(envelope)):
            with self.assertRaisesRegex(ValueError, 'No estimated legal corridor'):
                self.policy._passing_room(paths, 1, 0., source, self.cfg)
            with self.assertRaisesRegex(ValueError, 'No estimated legal corridor'):
                self.policy._defender_paths(state(10., speed=2.), envelope.line, source, self.cfg)
        # Earlier lack of passing room short-circuits a later invalid band.
        envelope.hi[:] = .1
        envelope.hi[index] = -2.
        for source in (envelope, ScalarEnvelope(envelope)):
            self.assertFalse(self.policy._passing_room(paths, 1, 0., source, self.cfg))


class RecoveryDemandTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Settings(), attacker_speed=3.4, defender_speed=3., offset_rate=.45)
        self.policy = Adversary(0., speed_profile=Profile(3.4), motion_config=motion())
        self.policy.speed = .4
        self.policy._fast_recovery = ('hold', .2, False, .25)
        self.policy._recovery_until = 3.2

    def recover(self, preview, speed=1.5, body_recovery=False):
        with patch.object(self.policy, 'preview', side_effect=preview) as checked:
            result = self.policy._fast_fallback(3.02, state(5., .2, .4), state(10.), Line(),
                Envelope(), Track(), self.cfg, 0., speed, body_recovery,
                dict(previous_steering=0., measured_forward_speed=.4))
        return result, checked

    @staticmethod
    def prediction(feasible, speed):
        return Preview(feasible, 'checked' if feasible else 'unsafe', .8, 0., .4,
                       .8 * speed if feasible else 0., final_speed=speed)

    def test_old_hold_demand_does_not_hide_newly_safe_full_speed_nominal(self):
        def preview(*args, **kwargs):
            return self.prediction(True, args[-1])
        (selected, label, unresolved), checked = self.recover(preview)
        self.assertEqual(label, 'nominal:1.5')
        self.assertFalse(unresolved)
        self.assertTrue(selected.feasible)
        self.assertEqual(checked.call_count, 1)
        self.assertTrue(checked.call_args.kwargs['nominal'])
        self.assertEqual(checked.call_args.args[-1], 1.5)

    def test_continued_hold_can_accelerate_when_nominal_is_still_unsafe(self):
        def preview(*args, **kwargs):
            return self.prediction(not kwargs['nominal'], args[-1])
        (_, label, unresolved), checked = self.recover(preview)
        self.assertEqual(label, 'hold:1.5')
        self.assertFalse(unresolved)
        self.assertEqual(checked.call_count, 2)
        self.assertFalse(checked.call_args.kwargs['nominal'])

    def test_unsafe_faster_choices_keep_checked_old_repair(self):
        def preview(*args, **kwargs):
            speed = args[-1]
            return self.prediction(not kwargs['nominal'] and speed <= .25, speed)
        (_, label, unresolved), checked = self.recover(preview)
        self.assertEqual(label, 'hold:0.25')
        self.assertFalse(unresolved)
        self.assertEqual(checked.call_count, 3)

    def test_body_repair_keeps_continuation_priority_and_new_guard_is_never_exceeded(self):
        def preview(*args, **kwargs):
            return self.prediction(True, args[-1])
        (_, label, _), checked = self.recover(preview, body_recovery=True)
        self.assertEqual(label, 'hold:0.25')
        self.assertTrue(checked.call_args.kwargs['body_recovery'])
        self.policy._fast_recovery = ('hold', .2, False, .25)
        (_, label, _), checked = self.recover(preview, speed=.1)
        self.assertEqual(label, 'nominal:0.1')
        self.assertTrue(all(call.args[-1] <= .1 for call in checked.call_args_list))


class FullPassAndFollowingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Settings(), attacker_speed=3.4, defender_speed=3., offset_rate=.45)
        self.policy = Adversary(0., speed_profile=Profile(3.4), motion_config=motion(max_lateral_rate=.45))

    def test_not_closing_defers_pass_before_entering_lane_toward_distant_bend(self):
        class NarrowAhead(Envelope):
            def interval(self, s):
                return (-2., .2) if s > 25. else (-2., 2.)

        envelope = NarrowAhead()
        car, defender = state(6., speed=0.), state(10., speed=0.)
        paths = self.policy._defender_paths(defender, Line(), envelope, self.cfg)
        self.assertTrue(self.policy._passing_room(paths, 1, 0., envelope, self.cfg))
        self.assertFalse(self.policy._passing_room(paths, 1, 0., envelope, self.cfg,
            gap=-4., attacker_speed=0., defender_speed=0., track_length=100.))
        command = self.policy.plan(1., 'left-pass', car, defender, Line(), envelope, Track(),
                                   self.cfg, measured_forward_speed=0.)
        self.assertTrue(command.requested)
        self.assertFalse(command.requested_corridor_valid)
        self.assertIn('narrowing_defender_passing_corridor', command.reason)
        self.assertTrue(command.fallback.startswith('nominal:'))
        self.assertEqual(command.target, 0.)
        self.assertGreater(command.speed, 0.)
        self.assertEqual(self.policy.executed_attack_steps, 0)

    def test_completion_reserve_detects_both_sides_across_track_seam(self):
        for side in (-1, 1):
            envelope = EnvelopeBatchTests.envelope()
            distance = abs((envelope.s - 5. + 50.) % 100. - 50.)
            envelope.reference_offsets = side * 1.1 * np.maximum(0., 1. - distance / .6)
            paths = self.policy._defender_paths(state(90., speed=0.), Line(), envelope, self.cfg)
            for source in (envelope, ScalarEnvelope(envelope)):
                self.assertTrue(self.policy._passing_room(paths, side, 0., source, self.cfg))
                self.assertFalse(self.policy._passing_room(paths, side, 0., source, self.cfg,
                    gap=-4., attacker_speed=0., defender_speed=0., track_length=100.))

    def test_completion_reserve_handles_tiny_negative_closing_and_huge_gap_without_overflow(self):
        envelope = EnvelopeBatchTests.envelope()
        paths = self.policy._defender_paths(state(10., speed=0.), Line(), envelope, self.cfg)
        for attacker_speed, defender_speed, gap in ((0., 0., -4.), (1., 3., -4.),
                                                  (3. + 1e-12, 3., -1e308)):
            with patch.object(self.policy, '_envelope_samples', wraps=self.policy._envelope_samples) as sampled:
                self.assertTrue(self.policy._passing_room(paths, 1, 0., envelope,
                    replace(self.cfg, attacker_speed=3., defender_speed=3.), gap=gap,
                    attacker_speed=attacker_speed, defender_speed=defender_speed, track_length=100.))
            queries = sampled.call_args.args[1]
            self.assertTrue(np.all(np.isfinite(queries)))
            self.assertLessEqual(len(queries), 3 * (401 + paths.shape[0] * paths.shape[1]))

    def test_invalid_room_prefers_nominal_over_tiny_hold_progress_win_without_cache(self):
        def preview(*args, **kwargs):
            progress = .2 if kwargs['nominal'] else .200001
            return Preview(True, 'checked', .8, 0., 1., progress, final_speed=1.)

        with patch.object(self.policy, 'preview', side_effect=preview) as checked:
            selected, label, unresolved = self.policy._fast_fallback(3., state(5., .8, 1.),
                state(10.), Line(), Envelope(), Track(), self.cfg, 0., 1., False,
                dict(previous_steering=0., measured_forward_speed=1.), prefer_nominal=True)
        self.assertEqual(label, 'nominal:1')
        self.assertTrue(selected.feasible)
        self.assertFalse(unresolved)
        self.assertEqual(checked.call_count, 1)

    def test_invalid_room_never_forces_return_through_alongside_defender(self):
        self.policy.target = .45
        self.policy.speed = 1.
        car, defender = state(10., .45, 1.), state(10., 0., 1.)
        with patch.object(self.policy, 'preview', wraps=self.policy.preview) as checked:
            selected, label, unresolved = self.policy._fast_fallback(3., car, defender, Line(),
                Envelope(), Track(), self.cfg, 0., 1., False,
                dict(previous_steering=0., measured_forward_speed=1.), prefer_nominal=True)
        self.assertTrue(checked.call_args_list[0].kwargs['nominal'])
        self.assertFalse(label.startswith('nominal:'))
        self.assertTrue(selected.feasible)
        self.assertFalse(unresolved)

    def test_fast_following_cap_anticipates_stationary_or_braking_lead(self):
        self.policy.speed = 3.
        stopped = self.policy._following_speed_cap(4., 3., 0., self.cfg)
        moving = self.policy._following_speed_cap(4., 3., 3., self.cfg)
        self.assertLess(stopped, moving)
        old_cap = math.sqrt(2 * self.cfg.comfortable_braking * (
            4. - self.cfg.car_length - self.cfg.separation_margin))
        self.assertLess(stopped, old_cap)
        self.assertLess(stopped, 3.)
        # At 4 m headway feedback dominates; use 6 m to isolate the braking budget.
        braking_limited = self.policy._following_speed_cap(6., 3., 0., self.cfg)
        self.policy._acceleration = 2.
        self.assertLess(self.policy._following_speed_cap(6., 3., 0., self.cfg), braking_limited)
        self.policy._acceleration = 0.
        self.policy.motion_config = motion(max_jerk=2.)
        self.assertLess(self.policy._following_speed_cap(6., 3., 0., self.cfg), braking_limited)

    def test_following_demand_respects_headway_and_stays_finite_nonnegative(self):
        self.policy.speed = 3.
        normal = self.policy._following_speed_cap(3., 3., 2., self.cfg)
        larger_headway = self.policy._following_speed_cap(3., 3., 2.,
                                                        replace(self.cfg, time_headway=1.2))
        self.assertLess(larger_headway, normal)
        for distance in (0., 1.4, 2., 40., 1e100):
            for lead_speed in (0., 3., 1e100):
                cap = self.policy._following_speed_cap(distance, 3., lead_speed,
                                                      replace(self.cfg, time_headway=0.))
                self.assertTrue(math.isfinite(cap))
                self.assertGreaterEqual(cap, 0.)
        self.assertGreater(self.policy._following_speed_cap(40., 3., 0., self.cfg), 3.4)

    def test_following_changes_demand_not_already_moving_command_instantly(self):
        self.policy.speed = 3.
        command = self.policy.plan(3., 'approach', state(5., speed=3.), state(9., speed=0.),
            Line(), Envelope(), Track(), self.cfg, measured_forward_speed=3.)
        self.assertIn('rear_end_guard', command.reason)
        self.assertLess(command.speed, 3.)
        self.assertGreater(command.speed, 3. - self.policy.motion_config.max_braking * self.cfg.timestep)
        self.assertAlmostEqual((command.speed - 3.) / self.cfg.timestep, command.acceleration)
        self.assertLessEqual(abs(command.acceleration), self.policy.motion_config.max_jerk * self.cfg.timestep)

    def test_distant_catchup_and_no_attack_do_not_freeze_or_scan_passing_room(self):
        for scenario in ('adaptive', 'no-attack'):
            policy = Adversary(0., speed_profile=Profile(3.4), motion_config=motion())
            with patch.object(policy, '_passing_room', side_effect=AssertionError('not a pass')):
                command = policy.plan(3., scenario, state(5., speed=0.), state(40., speed=0.),
                    Line(), Envelope(), Track(), self.cfg, measured_forward_speed=0.)
            self.assertGreater(command.speed, 0.)
            self.assertFalse(command.requested)
            self.assertFalse(command.guard_intervened)
            self.assertEqual(command.fallback, 'none')


if __name__ == '__main__':
    unittest.main()
