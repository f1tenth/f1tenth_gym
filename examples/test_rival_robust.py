"""Focused robust-controller regressions; no simulator or real map required."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from rival_geometry import ClosedRaceline
from rival_perception import OpponentEstimate, OpponentTracker, Measurement, SensorConfig
from rival_strategy import CarState, Settings
from rival_robust import (RobustConfig, TrackEnvelope, RobustDefender,
                          stopping_distance, advance_target, sat_clearance,
                          opponent_hypotheses)


class BroadTrack:
    def corridor(self, line, s, margin, **kwargs):
        return -2., 2.

    def clear_many(self, points):
        return np.full(len(points), 10.)

    def footprint_safe(self, *args):
        return True


class RobustTests(unittest.TestCase):
    def setUp(self):
        self.line = ClosedRaceline([[0, 0], [40, 0], [40, 40], [0, 40]])
        self.vehicle = Settings(legal_min=-3., legal_max=3.)
        self.cfg = RobustConfig(horizon=.4, prediction_dt=.1, region='track',
                                envelope_spacing=10.)
        self.track = BroadTrack()
        self.envelope = Mock()
        self.envelope.interval.return_value = (-2., 2.)
        self.envelope.nominal.return_value = 0.

    def state(self, s=10., d=0., speed=2.):
        x, y = self.line.sample(s, d)
        return CarState(s, d, speed, float(x), float(y), self.line.heading(s))

    def estimate(self, s=7., d=.8, speed=3., **kwargs):
        values = dict(state=self.state(s, d, speed), age=0., position_sigma=.01,
                      velocity_sigma=.02, longitudinal_acceleration=.4,
                      lateral_speed=.2, fresh=True, valid=True)
        values.update(kwargs)
        return OpponentEstimate(**values)

    def controller(self, target=0.):
        return RobustDefender(self.line, self.track, self.vehicle, self.cfg,
                              target, envelope=self.envelope)

    def test_stopping_distance_including_latency_and_reverse(self):
        self.assertAlmostEqual(stopping_distance(6., 3., .25), 7.5)
        self.assertEqual(stopping_distance(0., 3., .25), 0.)
        self.assertEqual(stopping_distance(-2., 3., .25), 0.)
        self.assertAlmostEqual(stopping_distance(6., 3., .5)
                               - stopping_distance(6., 3.), 3.)
        for args in ((1, 0, 0), (1, -1, 0), (1, 1, -1),
                     (np.nan, 1, 0), (1, np.inf, 0), (1, 1, np.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                stopping_distance(*args)

    def test_second_order_target_rate_acceleration_and_reversal(self):
        target = velocity = 0.
        dt = .02
        for goal in [1.] * 100 + [-1.] * 200:
            old_target, old_velocity = target, velocity
            target, velocity = advance_target(target, velocity, goal, dt, self.cfg)
            self.assertLessEqual(abs(velocity), self.cfg.max_lateral_rate + 1e-12)
            self.assertLessEqual(abs(velocity - old_velocity),
                                 self.cfg.max_lateral_acceleration * dt + 1e-12)
            self.assertAlmostEqual(target - old_target, velocity * dt)
        self.assertLess(target, 0.)
        self.assertEqual(advance_target(0., 0., 0., dt, self.cfg), (0., 0.))

    def test_sat_broadcast_rotation_overlap_and_touch(self):
        ego = np.zeros((2, 1, 3))
        opponents = np.array([[3., 0., 0.], [0., 0., np.pi / 2],
                              [2., 0., 0.]])[None, :, :]
        gaps = sat_clearance(ego, opponents, 2., 1.)
        self.assertEqual(gaps.shape, (2, 3))
        np.testing.assert_allclose(gaps, [[1., -1.5, 0.]] * 2, atol=1e-12)
        angle = .7
        rotated = np.array([3 * np.cos(angle), 3 * np.sin(angle), angle])
        self.assertAlmostEqual(sat_clearance([0, 0, angle], rotated, 2., 1.), 1.)
        self.assertGreater(sat_clearance([0, 0, 0], [0, 2., np.pi / 2], 2., 1.), 0.)

    def test_hypotheses_acceleration_lateral_uncertainty_and_wrap(self):
        times = np.array([0., .2, .5])
        estimate = self.estimate(s=10.)
        paths, inflation = opponent_hypotheses(estimate, self.line, times, self.cfg)
        self.assertEqual(paths.shape, (9, 3, 3))
        np.testing.assert_allclose(paths[:, 0, :2], np.tile([10., .8], (9, 1)))
        self.assertGreater(np.ptp(paths[:, -1, 0]), .1)
        self.assertGreater(np.ptp(paths[:, -1, 1]), .5)
        self.assertTrue(np.all(np.diff(inflation) > 0))
        older = replace(estimate, age=1., position_sigma=.4, velocity_sigma=.5)
        self.assertTrue(np.all(opponent_hypotheses(older, self.line, times, self.cfg)[1] > inflation))
        tracker = OpponentTracker(self.line, SensorConfig(noise_std=0))
        tracker.update(0., Measurement(0., *self.line.sample(159.8), 2.))
        wrapped = tracker.update(.2, None)
        self.assertAlmostEqual(wrapped.state.s, .2)
        paths, _ = opponent_hypotheses(wrapped, self.line, times, self.cfg)
        self.assertTrue(np.all(np.isfinite(paths)))
        self.assertTrue(np.all(paths[:, -1, 0] > paths[:, 0, 0]))
        for absent in (None, replace(estimate, valid=False)):
            empty, margin = opponent_hypotheses(absent, self.line, times, self.cfg)
            self.assertEqual(empty.shape, (0, 3, 3))
            np.testing.assert_array_equal(margin, 0.)

    def test_finite_config_and_control_validation(self):
        excluded = {'region', 'boundaries_path', 'rollout_speed_fractions'}
        for key in RobustConfig.__dataclass_fields__.keys() - excluded:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    replace(self.cfg, **{key: value})
        for fractions in ((), (np.nan,), (-.1,), (1.1,)):
            with self.assertRaises(ValueError):
                replace(self.cfg, rollout_speed_fractions=fractions)
        with self.assertRaises(ValueError):
            RobustConfig.from_dict({'typo': 1})
        for field in vars(self.state()):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.controller().step(0., replace(self.state(), **{field: np.nan}), None)
        controller = self.controller()
        controller.step(0., self.state(), None)
        for t in (0., -.1, np.nan, np.inf):
            with self.subTest(t=t), self.assertRaises(ValueError):
                controller.step(t, self.state(), None)

    def test_csv_boundary_body_margin_periodicity_and_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bounds.csv'
            path.write_text('s,lo,hi\n0,-2,3\n50,-2,3\n100,-2,3\n')
            envelope = TrackEnvelope(self.line, self.track, self.vehicle, self.cfg, path)
            margin = self.vehicle.car_width / 2 + self.vehicle.boundary_margin
            np.testing.assert_allclose(envelope.interval(5.), [-2 + margin, 3 - margin])
            self.assertEqual(envelope.interval(5.), envelope.interval(165.))
            self.assertEqual(envelope.nominal(5.), 0.)
            for rows in ('0,-2,3\n50,-2,3', '0,-2,3\n0,-2,3\n100,-2,3',
                         '1,-2,3\n50,-2,3\n100,-2,3',
                         '0,-2,3\n50,3,-2\n100,-2,3',
                         '0,-2,3\n50,nan,3\n100,-2,3',
                         '0,-2,3\n50,-2,3\n160,-2,3'):
                path.write_text('s,lo,hi\n' + rows + '\n')
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    TrackEnvelope(self.line, self.track, self.vehicle, self.cfg, path)
            path.write_text('s,lo,hi\n0,-.01,.01\n50,-.01,.01\n100,-.01,.01\n')
            narrow = TrackEnvelope(self.line, self.track, self.vehicle, self.cfg, path)
            self.assertGreater(*narrow.interval(10.))
            with self.assertRaises(ValueError):
                narrow.nominal(10.)

    def test_inside_region_for_both_turn_directions(self):
        for curvature in (.2, -.2):
            line = Mock(length=160.)
            line.curvature.return_value = curvature
            inside = TrackEnvelope(line, self.track, self.vehicle,
                                   replace(self.cfg, region='inside'))
            whole = TrackEnvelope(line, self.track, self.vehicle, self.cfg)
            lo, hi = inside.interval(0.)
            self.assertLess(lo, hi)
            self.assertGreaterEqual(lo, whole.interval(0.)[0])
            self.assertLessEqual(hi, whole.interval(0.)[1])
            if curvature > 0:
                self.assertGreater(lo, 0.)
            else:
                self.assertLess(hi, 0.)

    def test_alongside_never_selects_motion_toward_opponent(self):
        for side in (-1., 1.):
            controller = self.controller()
            decision = controller.step(0., self.state(), self.estimate(s=10., d=side))
            self.assertLessEqual(controller._bias * side, 0.)
            self.assertLessEqual(decision.target_offset * side, 0.)
            self.assertNotEqual(decision.mode, 'defend')

    def test_absent_stale_invalid_disable_defense_and_feasibility(self):
        for estimate in (None, self.estimate(fresh=False, age=1.),
                         self.estimate(valid=False)):
            with self.subTest(estimate=estimate):
                controller = self.controller()
                result = controller.step(0., self.state(), estimate)
                self.assertNotEqual(result.mode, 'defend')
                self.assertFalse(result.feasible)
                self.assertAlmostEqual(controller._bias, 0.)
                self.assertLessEqual(controller._desired_speed, self.vehicle.defender_speed * .65)

    def test_no_defense_disables_lateral_tactic(self):
        controller = self.controller()
        result = controller.step(0., self.state(), self.estimate(s=7., speed=2.5), defense=False)
        self.assertNotEqual(result.mode, 'defend')
        self.assertAlmostEqual(controller._bias, 0.)
        self.assertAlmostEqual(result.target_offset, 0.)

    def test_command_steering_speed_acceleration_jerk_slew(self):
        controller = self.controller()
        target = rate = acceleration = steering = speed = 0.
        dt = .02
        for desired in [4.] * 150 + [0.] * 150:
            previous = acceleration, steering, speed
            target, rate, acceleration, steering, speed = controller._command(
                self.state(), target, rate, acceleration, steering, speed,
                .6, desired, dt)
            self.assertGreaterEqual(speed, 0.)
            self.assertLessEqual(abs(steering), self.vehicle.max_steer + 1e-12)
            self.assertLessEqual(abs(steering - previous[1]), self.vehicle.max_steer_rate * dt + 1e-12)
            self.assertLessEqual(abs(acceleration - previous[0]), self.cfg.max_jerk * dt + 1e-12)
            self.assertGreaterEqual(acceleration, -self.cfg.max_braking - 1e-12)
            self.assertLessEqual(acceleration, self.cfg.max_acceleration + 1e-12)
            self.assertGreaterEqual(speed - previous[2], -self.cfg.max_braking * dt - 1e-12)
            self.assertLessEqual(speed - previous[2], self.cfg.max_acceleration * dt + 1e-12)

    def test_emitted_speed_jerk_through_complete_stop(self):
        controller = self.controller()
        dt = .02
        for initial_speed in (.02, .3, 2.):
            with self.subTest(initial_speed=initial_speed):
                command = (0., 0., 0., 0., initial_speed)
                speeds = [initial_speed]
                accelerations = [0.]
                for _ in range(250):
                    command = controller._command(self.state(), *command, 0., 0., dt)
                    speeds.append(command[-1])
                    accelerations.append((speeds[-1] - speeds[-2]) / dt)
                    self.assertAlmostEqual(accelerations[-1], command[2], places=10)
                speeds = np.asarray(speeds)
                accelerations = np.asarray(accelerations)
                self.assertTrue(np.all(speeds >= 0.))
                self.assertTrue(np.all(np.diff(speeds) <= 1e-12))
                self.assertLessEqual(np.max(np.abs(np.diff(accelerations))) / dt,
                                     self.cfg.max_jerk + 1e-9)
                self.assertGreaterEqual(np.min(accelerations), -self.cfg.max_braking - 1e-12)
                self.assertLess(speeds[-1], 1e-10)
                self.assertLess(abs(accelerations[-1]), 1e-9)

    def test_rollout_matches_actuation_grid_command_and_misaligned_kinematics(self):
        controller = self.controller(target=.15)
        controller._lateral_rate = .1
        controller._acceleration = .3
        controller._steering = .08
        controller._speed_command = 1.4
        ego = replace(self.state(speed=1.6 * np.cos(.4)), yaw=.4)
        forecast_dt = min(self.vehicle.timestep, self.cfg.prediction_dt)
        times = np.linspace(0., self.cfg.horizon,
                            int(np.ceil(self.cfg.horizon / forecast_dt)) + 1)
        np.testing.assert_allclose(np.diff(times), self.vehicle.timestep, atol=1e-15)
        rollout_spy = Mock(wraps=controller._rollout)
        controller._rollout = rollout_spy
        controller._solve(0., ego, None, False)
        for call in rollout_spy.call_args_list:
            np.testing.assert_array_equal(call.args[3], times)
            self.assertEqual(call.args[4].shape, (0, len(times), 3))
        bias, desired_speed = .3, 1.8
        actual = controller._rollout(ego, bias, desired_speed, times,
                                     np.empty((0, len(times), 3)), np.zeros(len(times)))
        state = ego
        forward = 1.6
        command = (controller.target, controller._lateral_rate, controller._acceleration,
                   controller._steering, controller._speed_command)
        expected = [[ego.x, ego.y, ego.yaw, forward, ego.s, ego.d, command[0]]]
        for dt in np.diff(times):
            command = controller._command(state, *command, bias, desired_speed, dt)
            speed = max(0., forward + np.clip(command[-1] - forward,
                        -self.cfg.max_braking * dt, self.cfg.max_acceleration * dt))
            average = (forward + speed) / 2
            turn = average / self.vehicle.wheelbase * np.tan(command[3]) * dt
            x = state.x + average * np.cos(state.yaw + turn / 2) * dt
            y = state.y + average * np.sin(state.yaw + turn / 2) * dt
            yaw = state.yaw + turn
            s, d = self.line.project([x, y])
            state = CarState(s, d, speed * np.cos(yaw - self.line.heading(s)), x, y, yaw)
            forward = speed
            expected.append([x, y, yaw, speed, s, d, command[0]])
        np.testing.assert_allclose(actual.trajectory, expected, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(actual.trajectory[0, 3], 1.6)
        self.assertGreater(actual.trajectory[-1, 1], ego.y)

    def test_alongside_residual_lateral_momentum_cannot_claim_feasible_yield(self):
        controller = self.controller(target=.5)
        controller._lateral_rate = self.cfg.max_lateral_rate
        controller._steering = .25
        controller._speed_command = 2.
        result = controller.step(0., self.state(), self.estimate(s=10., d=1.8, speed=2.))
        approach = np.max(controller._best.trajectory[:, 5])
        self.assertGreater(approach, .02, 'fixture must retain lateral momentum toward opponent')
        self.assertFalse(controller._best.separation_safe)
        self.assertFalse(result.feasible)
        self.assertEqual(result.mode, 'emergency')

    def test_defense_toggle_replans_before_period_expires(self):
        controller = self.controller()
        controller._solve = Mock(wraps=controller._solve)
        ego, estimate = self.state(), self.estimate(s=7., speed=2.5)
        controller.step(0., ego, estimate, defense=True)
        deadline = controller._next_plan
        controller.step(.02, ego, estimate, defense=True)
        self.assertEqual(controller._solve.call_count, 1)
        self.assertLess(.04, deadline)
        controller.step(.04, ego, estimate, defense=False)
        self.assertEqual(controller._solve.call_count, 2)
        self.assertFalse(controller._solve.call_args.args[3])
        controller.step(.06, ego, estimate, defense=True)
        self.assertEqual(controller._solve.call_count, 3)
        self.assertTrue(controller._solve.call_args.args[3])

    def test_nonfinite_valid_opponent_packet_is_degraded(self):
        estimate = self.estimate()
        packets = []
        for value in (np.nan, np.inf, -np.inf):
            for field in vars(estimate.state):
                packets.append((field, value, replace(estimate, state=replace(
                    estimate.state, **{field: value}))))
            for field in ('age', 'position_sigma', 'velocity_sigma',
                          'longitudinal_acceleration', 'lateral_speed'):
                packets.append((field, value, replace(estimate, **{field: value})))
        for field, value, packet in packets:
            with self.subTest(field=field, value=value):
                controller = self.controller()
                # Exercise rejection after a fresh plan, not only initialization.
                controller.step(0., self.state(), estimate)
                result = controller.step(.02, self.state(), packet)
                self.assertEqual(result.mode, 'degraded')
                self.assertFalse(result.feasible)
                self.assertEqual(result.hypothesis_count, 0)
                self.assertTrue(np.all(np.isfinite([
                    result.steering, result.speed, result.target_offset])))

    def test_known_acceleration_bound_violation_disables_defense_without_replan_storm(self):
        controller = self.controller()
        controller._solve = Mock(wraps=controller._solve)
        packet = self.estimate(longitudinal_acceleration=self.cfg.max_opponent_acceleration + .5)
        first = controller.step(0., self.state(), packet)
        self.assertEqual(first.mode, 'degraded')
        self.assertFalse(first.feasible)
        self.assertIn('prediction bounds', first.reason)
        controller.step(.02, self.state(), packet)
        self.assertEqual(controller._solve.call_count, 1)
        self.assertFalse(controller._solve.call_args.args[3])

    def test_minimax_fallback_is_explicitly_unsafe(self):
        controller = self.controller()
        result = controller.step(0., self.state(), self.estimate(s=10., d=0., speed=2.))
        self.assertEqual(result.mode, 'emergency')
        self.assertFalse(result.feasible)
        self.assertIn('fallback', result.reason)
        self.assertIn('NOT certified safe', result.reason)
        self.assertLessEqual(result.min_separation, 0.)

    def test_sampled_map_obstacle_rejects_rollout(self):
        controller = self.controller()
        times = np.linspace(0., .4, 5)
        empty = np.empty((0, len(times), 3))
        safe = controller._rollout(self.state(), 0., 2., times, empty, np.zeros(5))
        self.assertTrue(safe.map_safe)
        clear = np.full(5, 10.)
        clear[2] = 0.
        self.track.clear_many = Mock(return_value=clear)
        self.track.footprint_safe = Mock(return_value=False)
        blocked = controller._rollout(self.state(), 0., 2., times, empty, np.zeros(5))
        self.assertFalse(blocked.map_safe)
        self.assertFalse(blocked.feasible)
        self.track.footprint_safe.assert_called_once()
        np.testing.assert_allclose(self.track.footprint_safe.call_args.args[0],
                                   blocked.trajectory[2, :2])
        self.assertLess(blocked.min_clearance, 0.)


if __name__ == '__main__':
    unittest.main()
