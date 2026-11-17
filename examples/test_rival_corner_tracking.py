"""Focused command/preview regressions; no simulator episodes or physics changes."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from rival_control import advance_speed_command
from rival_geometry import ClosedRaceline
from rival_robust import RobustConfig, RobustDefender, TrackEnvelope, advance_target
from rival_rollout import simulate_trajectory
from rival_speed import SpeedProfile
from rival_strategy import CarState, Settings, rate_limit


class BroadTrack:
    def corridor(self, *args, **kwargs):
        return -10., 10.

    def clear_many(self, points):
        return np.full(len(points), 100.)

    def footprint_safe(self, *args):
        return True


def fixture(**kwargs):
    line = ClosedRaceline([[0., 0.], [20., 0.], [20., 20.], [0., 20.]])
    vehicle = Settings(legal_min=-10., legal_max=10.)
    config = RobustConfig(region='track', corner_tracking=True, **kwargs)
    track = BroadTrack()
    envelope = TrackEnvelope(line, track, vehicle, config)
    parent = RobustDefender(line, track, vehicle, config, 0., envelope=envelope)
    return parent


def set_reference(parent, values):
    envelope = parent.envelope
    envelope.reference_offsets[:] = values
    envelope._reference_slopes[:] = np.diff(np.r_[values, values[0]]) / np.diff(
        np.r_[envelope.s, parent.line.length])


def state_at(parent, s, speed=2., d=0., angle=0.):
    x, y = parent.line.sample(s, d)
    return CarState(s, d, speed, x, y, parent.line.heading(s) + angle)


def legacy_command(parent, state, controls, bias, desired_speed, dt):
    """Frozen flag-off arithmetic, independent of the new command branches."""
    target, lateral_rate, acceleration, steering, command_speed = controls
    r, c = parent.config, parent.vehicle
    target, lateral_rate = advance_target(target, lateral_rate, parent._goal(state, bias), dt, r)
    curvature = max(abs(parent.line.curvature(state.s + ds))
                    for ds in (0., c.lookahead, 2 * c.lookahead))
    desired_speed = min(desired_speed, math.sqrt(r.max_lateral_accel / max(curvature, 1e-3)))
    if parent.speed_profile is not None:
        desired_speed = min(desired_speed, parent.speed_profile.limit(state.s, state.speed))
    slope = max(abs(parent.envelope.reference_slope(state.s + ds))
                for ds in (0., c.lookahead, 2 * c.lookahead))
    desired_speed = min(desired_speed, .8 * r.max_lateral_rate / max(slope, 1e-6))
    desired_accel = np.clip((desired_speed - command_speed) / max(dt, 1e-6),
                            -r.max_braking, r.max_acceleration)
    acceleration = rate_limit(acceleration, desired_accel, r.max_jerk, dt)
    release = r.max_jerk * dt
    braking_limit = max(0., math.sqrt(release ** 2 + 2 * r.max_jerk * command_speed) - release)
    acceleration = max(acceleration, -braking_limit)
    next_speed = max(0., command_speed + acceleration * dt)
    point = parent.line.sample(state.s + c.lookahead, target)
    delta = point - [state.x, state.y]
    lateral = -math.sin(state.yaw) * delta[0] + math.cos(state.yaw) * delta[1]
    desired_steer = math.atan2(2 * c.wheelbase * lateral, max(float(delta @ delta), 1e-6))
    desired_steer = float(np.clip(desired_steer, -c.max_steer, c.max_steer))
    steering = rate_limit(steering, desired_steer, c.max_steer_rate, dt)
    return target, lateral_rate, acceleration, steering, next_speed


class CornerTrackingTests(unittest.TestCase):
    def test_defaults_and_strict_validation(self):
        default = RobustConfig()
        self.assertIs(default.corner_tracking, False)
        self.assertEqual(default.target_tracking_gain, 4.)
        self.assertTrue(RobustConfig.from_dict({'corner_tracking': True}).corner_tracking)
        for value in (0, 1, 'true', None, np.bool_(True), np.nan):
            with self.subTest(flag=value), self.assertRaises(ValueError):
                RobustConfig.from_dict({'corner_tracking': value})
        for value in (0., -1., np.nan, np.inf, -np.inf, True, np.bool_(True), '4', None):
            with self.subTest(gain=value), self.assertRaises(ValueError):
                RobustConfig.from_dict({'target_tracking_gain': value})
        self.assertEqual(RobustConfig(target_tracking_gain=.25).target_tracking_gain, .25)

    def test_feedforward_tracks_moving_reference_in_both_directions(self):
        for sign in (-1., 1.):
            parent = fixture()
            s = parent.envelope.s
            set_reference(parent, sign * .1 * np.minimum(s, parent.line.length - s))
            dt, speed, bias = .02, 2., .17
            state = state_at(parent, 3., speed)
            goal = parent._goal(state, bias)
            # At zero error a stationary-reference filter would brake this rate.
            controls = (goal, sign * .2, 0., 0., speed)
            for k in range(60):
                state = state_at(parent, 3. + k * speed * dt, speed)
                controls = parent._command(state, *controls, bias, speed, dt)
                expected = parent.envelope.reference(state.s + parent.vehicle.lookahead + speed * dt) + bias
                with self.subTest(sign=sign, k=k):
                    self.assertAlmostEqual(controls[0], expected, places=12)
                    self.assertAlmostEqual(controls[1], sign * .2, places=12)
            # From rest the feedforward itself must also enter through the slew bound.
            target, rate, *_ = parent._command(state, parent._goal(state, 0.), 0., 0., 0., speed,
                                               0., speed, dt)
            self.assertAlmostEqual(rate, sign * parent.config.max_lateral_acceleration * dt)
            self.assertAlmostEqual(target, parent._goal(state, 0.) + rate * dt)

    def test_exact_clipped_goal_feedback_and_projected_speed(self):
        parent = fixture()
        s = parent.envelope.s
        set_reference(parent, .1 * np.minimum(s, parent.line.length - s))
        state, dt = state_at(parent, 3., speed=1.3, angle=.8), .02
        preview = state.s + parent.vehicle.lookahead
        reference_rate = (parent.envelope.reference(preview + state.speed * dt)
                          - parent.envelope.reference(preview)) / dt
        goal = parent._goal(state, 0.)
        target = goal - .01
        expected_rate = reference_rate + 4. * .01
        controls = parent._command(state, target, expected_rate, 0., 0., 7., 0., 7., dt)
        self.assertAlmostEqual(controls[1], expected_rate, places=12)
        self.assertAlmostEqual(controls[0], target + expected_rate * dt, places=12)
        # A negative projected speed gives zero FF, not backward or absolute-speed FF.
        reverse = replace(state, speed=-1.3)
        controls = parent._command(reverse, goal, 0., 0., 0., 7., 0., 7., dt)
        self.assertEqual(controls[1], 0.)
        # Feedback uses _goal's legal clipping, not the unbounded nominal+bias.
        parent.envelope.lo[:] = -.05
        parent.envelope.hi[:] = .05
        controls = parent._command(state, .04, expected_rate, 0., 0., 7., 20., 7., dt)
        desired = np.clip(reference_rate + 4. * (.05 - .04),
                          -parent.config.max_lateral_rate, parent.config.max_lateral_rate)
        self.assertAlmostEqual(controls[1], rate_limit(expected_rate, desired,
                               parent.config.max_lateral_acceleration, dt), places=12)

    def test_feedforward_is_forward_difference_across_reference_knot(self):
        parent = fixture()
        values = np.zeros_like(parent.envelope.s)
        values[1] = .2
        set_reference(parent, values)
        dt, speed = .02, 2.
        preview = parent.envelope.s[1] - .01
        state = state_at(parent, preview - parent.vehicle.lookahead, speed=speed)
        goal = parent._goal(state, 0.)
        rate = (parent.envelope.reference(preview + speed * dt)
                - parent.envelope.reference(preview)) / dt
        self.assertLess(abs(rate), parent.config.max_lateral_rate)
        controls = parent._command(state, goal, rate, 0., 0., 2., 0., 2., dt)
        self.assertAlmostEqual(controls[1], rate, places=12)
        self.assertAlmostEqual(controls[0], goal + rate * dt, places=12)
        self.assertGreater(abs(rate - speed * parent.envelope.reference_slope(preview)), .1)

    def test_tactical_step_reversal_bounds_total_rate_and_acceleration(self):
        for sign in (-1., 1.):
            parent = fixture()
            s = parent.envelope.s
            # FF alone exceeds the allowed rate: even then TOTAL motion stays bounded.
            set_reference(parent, sign * np.minimum(s, parent.line.length - s))
            state = state_at(parent, 3.)
            controls = (parent._goal(state, 0.), 0., 0., 0., 0.)
            rates = []
            for k in range(180):
                dt = (.02, .013, .027)[k % 3]
                bias = sign * (6. if k < 90 else -6.)
                previous = controls
                controls = parent._command(state, *controls, bias, 0., dt)
                rates.append(controls[1])
                self.assertLessEqual(abs(controls[1]), parent.config.max_lateral_rate + 1e-12)
                self.assertLessEqual(abs(controls[1] - previous[1]),
                                     parent.config.max_lateral_acceleration * dt + 1e-12)
                self.assertAlmostEqual(controls[0] - previous[0], controls[1] * dt, places=12)
            self.assertAlmostEqual(rates[89], sign * parent.config.max_lateral_rate)
            self.assertAlmostEqual(rates[-1], -sign * parent.config.max_lateral_rate)
            self.assertGreater(sign * rates[90], 0.)  # Reversal cannot erase momentum.

    def test_custom_envelope_without_reference_has_zero_feedforward(self):
        class CustomEnvelope:
            def interval(self, s):
                return -.8, .8

            def nominal(self, s):
                return 0.

        parent = fixture()
        parent.envelope = CustomEnvelope()
        state = state_at(parent, 3., speed=20.)
        target, rate, *_ = parent._command(state, .2, -.1, 0., 0., 4., 0., 4., .02)
        expected_rate = rate_limit(-.1, -.8, parent.config.max_lateral_acceleration, .02)
        self.assertAlmostEqual(rate, expected_rate)
        self.assertAlmostEqual(target, .2 + rate * .02)

    def test_empty_band_preserves_fallback_rejection_and_never_hops(self):
        parent = fixture()
        parent.target = .4
        parent.envelope.lo[:] = 1.
        parent.envelope.hi[:] = -1.
        set_reference(parent, np.full_like(parent.envelope.s, 9.))
        state, dt = state_at(parent, 3.), .02
        controls = (.4, .2, 0., 0., 2.)
        self.assertEqual(parent._goal(state, 8.), parent.target)
        with patch.object(parent.envelope, 'reference', side_effect=AssertionError('invalid reference')):
            command = parent._command(state, *controls, 8., 2., dt)
        self.assertAlmostEqual(command[1], .2 - parent.config.max_lateral_acceleration * dt)
        self.assertAlmostEqual(command[0], .4 + command[1] * dt)
        self.assertNotEqual(command[0], parent.target)  # No teleport even on rejection.
        (parent.target, parent._lateral_rate, parent._acceleration,
         parent._steering, parent._speed_command) = controls
        times = np.array([0., .02, .04, .051, .071])
        compiled = simulate_trajectory(parent.line, parent.envelope, state, controls, 8., 2.,
                                       times, parent.vehicle, parent.config)
        python = parent._trajectory_python(state, 8., 2., times)
        for a, b in zip(compiled, python):
            np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)
        self.assertGreaterEqual(compiled[1], 1.)
        rollout = parent._rollout(state, 8., 2., times, np.empty((0, len(times), 3)), np.zeros(len(times)))
        self.assertFalse(rollout.legal_safe)
        self.assertFalse(rollout.feasible)

    def test_seam_feedforward_is_finite_and_periodic(self):
        parent = fixture()
        set_reference(parent, .6 * np.sin(2 * np.pi * parent.envelope.s / parent.line.length))
        s = parent.line.length - parent.vehicle.lookahead - .005
        dt = .02
        state = state_at(parent, s)
        preview = s + parent.vehicle.lookahead
        expected_rate = (parent.envelope.reference(preview + state.speed * dt)
                         - parent.envelope.reference(preview)) / dt
        goal = parent._goal(state, 0.)
        commands = [parent._command(state_at(parent, s + lap * parent.line.length),
                    goal, expected_rate, 0., 0., 2., 0., 2., dt) for lap in (-1, 0, 1)]
        for command in commands:
            self.assertTrue(np.all(np.isfinite(command)))
            self.assertAlmostEqual(command[1], expected_rate, places=12)
            np.testing.assert_allclose(command, commands[1], atol=2e-12, rtol=2e-12)

    def test_shared_speed_helper_gets_absolute_command_state(self):
        parent = fixture()
        state = state_at(parent, 3., speed=.7)
        with patch('rival_robust.advance_speed_command', wraps=advance_speed_command) as helper:
            command = parent._command(state, 0., 0., .6, 0., 1.1, 0., 1.2, .02)
        helper.assert_called_once_with(1.1, .6, 1.2, .02, parent.config.max_acceleration,
                                       parent.config.max_braking, parent.config.max_jerk)
        speed, acceleration = advance_speed_command(1.1, .6, 1.2, .02,
            parent.config.max_acceleration, parent.config.max_braking, parent.config.max_jerk)
        self.assertEqual(command[4], speed)
        self.assertEqual(command[2], acceleration)

    def test_jit_python_actuation_parity_and_same_time_grid(self):
        parent = fixture()
        set_reference(parent, .6 * np.sin(2 * np.pi * parent.envelope.s / parent.line.length))
        cases = [(3., 2., 0., (.3, -.2, .6, .1, 1.95), .25, 2.),
                 (parent.line.length - .03, 2., 0., (.1, .2, -.6, 0., 2.), -.3, 0.),
                 (3., -1., math.pi, (.1, -.1, 0., 0., 1.), .2, 2.),
                 (3., .01, 0., (0., 0., -.1, 0., .01), 0., 0.),
                 (3., 0., 0., (0., 0., 0., 0., 0.), 0., 2.)]
        timesets = (np.arange(26) * .02, np.array([0., .02, .04, .051, .071, .103]))
        for profile_on in (False, True):
            parent.speed_profile = (SpeedProfile(parent.line, parent.envelope, parent.vehicle,
                                    parent.config, 2.) if profile_on else None)
            for times in timesets:
                for s, speed, angle, controls, bias, desired in cases:
                    state = state_at(parent, s, speed=speed, angle=angle)
                    (parent.target, parent._lateral_rate, parent._acceleration,
                     parent._steering, parent._speed_command) = controls
                    with self.subTest(profile=profile_on, s=s, speed=speed, nodes=len(times)):
                        compiled = simulate_trajectory(parent.line, parent.envelope, state, controls,
                            bias, desired, times, parent.vehicle, parent.config,
                            speed_profile=parent.speed_profile)
                        python = parent._trajectory_python(state, bias, desired, times)
                        self.assertEqual(compiled[0].shape, (len(times), 7))
                        for a, b in zip(compiled, python):
                            self.assertTrue(np.all(np.isfinite(a)))
                            np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)
                    if angle == 0. and speed == controls[4] and speed > .1:
                        # One integration step independently exposes target rate,
                        # speed/acceleration and steering from the compiled pose.
                        command = parent._command(state, *controls, bias, desired, times[1])
                        row = compiled[0][1]
                        dt = times[1]
                        self.assertAlmostEqual((row[6] - controls[0]) / dt, command[1], places=11)
                        self.assertAlmostEqual(row[3], command[4], places=11)
                        self.assertAlmostEqual((row[3] - controls[4]) / dt, command[2], places=11)
                        average = (speed + row[3]) / 2
                        steering = math.atan((row[2] - state.yaw) * parent.vehicle.wheelbase / (average * dt))
                        self.assertAlmostEqual(steering, command[3], places=11)

    def test_flag_off_keeps_legacy_command_arithmetic_and_ignores_gain(self):
        parent = fixture()
        parent.config = replace(parent.config, corner_tracking=False)
        set_reference(parent, .6 * np.sin(2 * np.pi * parent.envelope.s / parent.line.length))
        state = state_at(parent, 3., speed=2., angle=.4)
        controls = (.3, -.2, .6, .1, 1.95)
        times = np.array([0., .02, .04, .051, .071, .103])
        with patch('rival_robust.advance_speed_command', side_effect=AssertionError('flag off')):
            for dt in (.02, .013):
                actual = parent._command(state, *controls, .25, 2., dt)
                expected = legacy_command(parent, state, controls, .25, 2., dt)
                self.assertEqual(np.asarray(actual).tobytes(), np.asarray(expected).tobytes())
        args = (parent.line, parent.envelope, state, controls, .25, 2., times, parent.vehicle)
        default = simulate_trajectory(*args, replace(parent.config, target_tracking_gain=4.))
        for gain in (.01, 1e6):
            explicit_off = simulate_trajectory(*args, replace(parent.config, target_tracking_gain=gain))
            for a, b in zip(default, explicit_off):
                self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())


if __name__ == '__main__':
    unittest.main()
