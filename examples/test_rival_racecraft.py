"""Racecraft permissions and model-output regressions; no simulator required.

Synthetic rollouts isolate policy rejection/utility, not vehicle feasibility.
The initial-defense and degraded tests also exercise the real robust rollouts.
"""
from dataclasses import asdict, replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rival_geometry import ClosedRaceline
from rival_perception import OpponentEstimate
from rival_racecraft import RacecraftLayer, trajectory_headings
from rival_robust import Decision, RobustConfig, RobustDefender, _Rollout, opponent_hypotheses
from rival_strategy import CarState, Settings
from test_rival_robust import BroadTrack


class RacecraftTests(unittest.TestCase):
    def setUp(self):
        self.line = ClosedRaceline([[0, 0], [40, 0], [40, 40], [0, 40]])
        self.vehicle = Settings(legal_min=-3., legal_max=3.)
        self.config = RobustConfig(horizon=.4, region='track', envelope_spacing=10.)
        self.envelope = Mock()
        self.envelope.interval.return_value = (-2., 2.)
        self.envelope.nominal.return_value = 0.
        self.track = BroadTrack()

    def state(self, s=10., d=0., speed=2.):
        x, y = self.line.sample(s, d)
        return CarState(s, d, speed, float(x), float(y), self.line.heading(s))

    def estimate(self, s=6., d=1., speed=3., **kwargs):
        values = dict(state=self.state(s, d, speed), age=0., position_sigma=.01,
                      velocity_sigma=.02, longitudinal_acceleration=0.,
                      lateral_speed=0., fresh=True, valid=True)
        values.update(kwargs)
        return OpponentEstimate(**values)

    def controller(self):
        return RobustDefender(self.line, self.track, self.vehicle, self.config,
                              0., envelope=self.envelope)

    def policy(self):
        return RacecraftLayer(self.line, self.envelope, self.vehicle, self.config)

    def path(self, end_d=0., bias=0., safe=True, progress=.4):
        trajectory = np.array([[10., 0., 0., 2., 10., 0., 0.],
                               [10.4, end_d, 0., 2., 10.4, end_d, end_d]])
        return _Rollout(bias, 2., trajectory, safe, True, True, 1., 1., 0., progress)

    def commit(self, policy, side=1.):
        policy.update(0., self.state(), self.estimate(d=side), True, 0., 0.)
        lane = policy.tactical_bias
        self.assertIsNotNone(lane)
        policy.selected(self.path(bias=lane), self.state(), 0.)
        self.assertTrue(policy.move_used)
        return lane

    def test_initial_early_defense_uses_real_model_on_both_sides(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                controller = self.controller()
                decision = controller.step(0., self.state(), self.estimate(d=side))
                self.assertTrue(decision.feasible)
                self.assertEqual(decision.hypothesis_count, 9)
                self.assertEqual(decision.mode, 'defend')
                self.assertEqual(decision.racecraft_phase, 'committed')
                self.assertTrue(decision.racecraft_move_used)
                self.assertAlmostEqual(decision.racecraft_committed_bias,
                                       side * self.config.racecraft_lane_offset)
                self.assertTrue(controller._best.map_safe)
                self.assertTrue(controller._best.legal_safe)
                self.assertTrue(controller._best.separation_safe)

    def test_one_fixed_lane_never_tracks_or_switches_with_noisy_opponent(self):
        policy = self.policy()
        lane = self.commit(policy)
        # Time alone (including the old commitment_time) never grants a move.
        for t, d, rate in ((.1, .6, -.8), (.5, -1., -.5),
                           (2., 1.8, .8), (8., -1.5, 0.)):
            policy.update(t, self.state(), self.estimate(d=d, lateral_speed=rate), True, .1, 0.)
            self.assertEqual(policy.tactical_bias, lane)
            self.assertEqual(policy.committed_bias, lane)
            self.assertGreater(policy.utility(lane), policy.utility(-lane))
            policy.selected(self.path(bias=lane), self.state(), 0.)
        self.assertEqual(policy.encounter, 1)

    def test_infeasible_proposal_is_not_a_commit_or_permission_to_try_other_side(self):
        policy = self.policy()
        policy.update(0., self.state(), self.estimate(), True, 0., 0.)
        lane = policy.tactical_bias
        policy.selected(self.path(bias=lane, safe=False), self.state(), 0.)
        self.assertFalse(policy.move_used)
        self.assertIsNone(policy.committed_bias)
        policy.update(1., self.state(), self.estimate(d=-1.), True, 0., 0.)
        self.assertEqual(policy.tactical_bias, lane)

    def test_late_attack_latches_even_when_ttc_estimate_recovers(self):
        controller = self.controller()
        result = controller.step(0., self.state(), self.estimate(s=8.5, speed=4.))
        self.assertNotEqual(result.mode, 'defend')
        self.assertEqual(result.racecraft_phase, 'late')
        self.assertFalse(result.racecraft_move_used)
        policy = controller.racecraft
        for t, s in ((.1, 6.), (.2, 8.5), (2., 6.)):
            policy.update(t, self.state(), self.estimate(s=s), True, 0., 0.)
            self.assertEqual(policy.lock, 'late')
            self.assertIsNone(policy.tactical_bias)
        # A late block cannot hide behind a good terminal separation.
        policy.update(2.1, self.state(), self.estimate(s=8.5, speed=4.), True, 0., 0.)
        self.assertFalse(policy.accepts(self.path(end_d=.1, bias=.3), self.state(),
            self.estimate(), 0., 0., np.empty((0, 2, 3)), np.zeros(2)))

    def test_late_permission_change_replans_before_planner_deadline(self):
        controller = self.controller()
        controller._solve = Mock(wraps=controller._solve)
        controller.step(0., self.state(), self.estimate())
        deadline = controller._next_plan
        self.assertEqual(controller._solve.call_count, 1)
        self.assertLess(.02, deadline)
        result = controller.step(.02, self.state(), self.estimate(s=8.5, speed=4.))
        self.assertEqual(controller._solve.call_count, 2)
        self.assertEqual(result.racecraft_phase, 'late')
        self.assertNotEqual(result.mode, 'defend')
        self.assertTrue(result.racecraft_move_used)

    def test_acceleration_uncertainty_and_age_close_the_early_window(self):
        for packet in (self.estimate(longitudinal_acceleration=2.),
                       self.estimate(velocity_sigma=1.),
                       self.estimate(age=2., fresh=True)):
            with self.subTest(packet=packet):
                policy = self.policy()
                policy.update(0., self.state(), packet, True, 0., 0.)
                self.assertEqual(policy.phase, 'late')
                self.assertIsNone(policy.tactical_bias)

    def test_early_window_uses_actual_opposite_lane_displacement(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                policy = self.policy()
                ego = self.state(d=-.45 * side)
                policy.update(0., ego, self.estimate(s=7., d=1.5 * side), True, ego.d, 0.)
                self.assertAlmostEqual(policy.proposed_bias, .45 * side)
                self.assertEqual(policy.phase, 'late')
                self.assertIsNone(policy.tactical_bias)
                self.assertAlmostEqual(policy._move_time(ego.d, .45 * side, 0.), 2.5625)

    def test_early_window_uses_clipped_goal_not_unclipped_lane_offset(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                self.envelope.interval.return_value = (-2., .05) if side > 0 else (-.05, 2.)
                policy = self.policy()
                ego = self.state(d=-.45 * side)
                policy.update(0., ego, self.estimate(s=7., d=side), True, ego.d, 0.)
                self.assertEqual(policy.phase, 'early')
                self.assertAlmostEqual(policy.tactical_bias, .05 * side)

    def test_signed_lateral_momentum_can_close_early_window(self):
        for side in (-1., 1.):
            for rate, phase in ((.2, 'early'), (-.2, 'late')):
                with self.subTest(side=side, rate=rate):
                    policy = self.policy()
                    policy.update(0., self.state(), self.estimate(s=7., d=side), True, 0., 0.,
                                  lateral_rate=rate * side)
                    self.assertEqual(policy.phase, phase)

    def test_controller_passes_actual_lateral_rate_at_both_update_sites(self):
        controller = self.controller()
        controller._lateral_rate = -.2
        controller.racecraft.update = Mock(wraps=controller.racecraft.update)
        result = controller.step(0., self.state(), self.estimate(s=7.))
        self.assertEqual(result.racecraft_phase, 'late')
        self.assertEqual(controller.racecraft.update.call_count, 2)
        for call in controller.racecraft.update.call_args_list:
            self.assertAlmostEqual(call.kwargs['lateral_rate'], -.2)

    def test_overlap_rejects_motion_and_residual_momentum_toward_both_sides(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                policy = self.policy()
                packet = self.estimate(s=10., d=side)
                policy.update(0., self.state(), packet, True, 0., 0.)
                self.assertEqual(policy.phase, 'overlap')
                predictions = np.array([[[10., side, 0.], [10.4, side, 0.]]])
                for path in (self.path(end_d=.1 * side, bias=.3 * side),
                             self.path(end_d=.03 * side, bias=-.3 * side)):
                    # The second goal is AWAY, but the body still squeezes.
                    self.assertFalse(policy.accepts(path, self.state(), packet, 0., 0., predictions, np.zeros(2)))
                for path in (self.path(), self.path(end_d=-.1 * side, bias=-.3 * side)):
                    self.assertTrue(policy.accepts(path, self.state(), packet, 0., 0., predictions, np.zeros(2)))

    def test_overlap_side_noise_never_unlocks_or_reverses_protection(self):
        policy = self.policy()
        policy.update(0., self.state(), self.estimate(s=10., d=1.), True, 0., 0.)
        for t, d in ((.1, -.01), (.2, .01), (1., -1.)):
            policy.update(t, self.state(), self.estimate(s=10., d=d), True, 0., 0.)
            self.assertEqual(policy.opponent_side, 1.)
            self.assertTrue(policy.overlap)
            self.assertIsNone(policy.tactical_bias)

    def test_overlap_anchor_stops_small_cumulative_squeezes_on_both_sides(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                policy = self.policy()
                packet = self.estimate(s=10., d=1.5 * side, speed=2.)
                predictions = np.array([[[10., 1.5 * side, 0.], [10.4, 1.5 * side, 0.]]])
                for step in range(3):
                    d, next_d = step * .009 * side, (step + 1) * .009 * side
                    ego = self.state(d=d)
                    policy.update(step * .1, ego, packet, True, d, 0.)
                    candidate = self.path(end_d=next_d, bias=next_d)
                    candidate.trajectory[0, [1, 5]] = d
                    accepted = policy.accepts(candidate, ego, packet, 0., d, predictions, np.zeros(2))
                    self.assertEqual(accepted, step < 2)
                    self.assertAlmostEqual(policy._overlap_anchor, 0.)
                self.assertEqual(policy.phase, 'overlap')

    def test_overlap_anchor_follows_retreat_but_not_a_return_or_dropout(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                policy = self.policy()
                packet = self.estimate(s=10., d=1.5 * side, speed=2.)
                policy.update(0., self.state(), packet, True, 0., 0.)
                ego = self.state(d=-.1 * side)
                policy.update(.1, ego, packet, True, ego.d, 0.)
                self.assertAlmostEqual(policy._overlap_anchor, ego.d)
                predictions = np.array([[[10., 1.5 * side, 0.], [10.4, 1.5 * side, 0.]]])
                for end_d, accepted in ((-.2 * side, True), (-.07 * side, False)):
                    candidate = self.path(end_d=end_d, bias=end_d)
                    candidate.trajectory[0, [1, 5]] = ego.d
                    self.assertEqual(policy.accepts(candidate, ego, packet, 0., ego.d,
                        predictions, np.zeros(2)), accepted)
                policy.update(.2, ego, None, True, ego.d, 0.)
                self.assertAlmostEqual(policy._overlap_anchor, ego.d)
                self.assertFalse(policy.accepts(candidate, ego, None, 0., ego.d, predictions, np.zeros(2)))

    def test_overlap_anchor_is_body_reference_relative_through_shift_and_wrap(self):
        class MovingReference:
            s = np.array([0., 40., 80., 120.])
            reference_offsets = np.array([0., .4, 0., -.4])

            def reference(self, s):
                return float(np.interp(s % 160., self.s, self.reference_offsets, period=160.))

            nominal = reference

            def interval(self, s):
                return -2., 2.

        self.envelope = MovingReference()
        for side in (-1., 1.):
            for start, end in ((10., 20.), (159.8, 160.2)):
                with self.subTest(side=side, start=start):
                    policy = self.policy()
                    reference = self.envelope.reference
                    ego = self.state(s=start, d=reference(start) + .1 * side)
                    packet = self.estimate(s=start, d=reference(start) + 1.2 * side, speed=2.)
                    nominal = reference(start + self.vehicle.lookahead)
                    target = nominal + .1 * side
                    policy.update(0., ego, packet, True, target, nominal)
                    candidate = self.path(bias=.1 * side)
                    predictions = np.empty((1, 2, 3))
                    for index, s in enumerate((start, end)):
                        d = reference(s) + .1 * side
                        candidate.trajectory[index, :3] = [*self.line.sample(s, d), self.line.heading(s)]
                        candidate.trajectory[index, 4:6] = [s, d]
                        predictions[0, index] = [*self.line.sample(s, reference(s) + 1.2 * side), self.line.heading(s)]
                    self.assertAlmostEqual(policy._overlap_anchor, .1 * side)
                    self.assertTrue(policy.accepts(candidate, ego, packet, nominal, target, predictions, np.zeros(2)))
                    candidate.trajectory[-1, 5] += .03 * side
                    self.assertFalse(policy.accepts(candidate, ego, packet, nominal, target, predictions, np.zeros(2)))

    def test_old_overlap_anchor_does_not_constrain_separated_corner_recovery(self):
        policy = self.policy()
        policy.update(0., self.state(), self.estimate(s=10., d=1.5, speed=2.), True, 0., 0.)
        self.assertAlmostEqual(policy._overlap_anchor, 0.)
        ego, packet = self.state(d=.25), self.estimate(s=3., d=1.5, speed=0.)
        policy.update(.1, ego, packet, True, .25, 0.)
        self.assertFalse(policy.protect_line)
        self.assertTrue(policy.overlap)
        candidate = self.path(end_d=.35, bias=.3)
        candidate.trajectory[0, [1, 5]] = .25
        predictions, inflation = opponent_hypotheses(packet, self.line, np.array([0., .4]), self.config)
        self.assertTrue(policy.accepts(candidate, ego, packet, 0., .25, predictions, inflation))
        passed = self.estimate(s=12., speed=2.)
        policy.update(.2, ego, passed, True, .25, 0.)
        policy.update(1.21, ego, passed, True, .25, 0.)
        self.assertFalse(policy.active)
        self.assertIsNone(policy._overlap_anchor)

    def test_room_to_boundary_on_both_sides_even_without_body_collision(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                policy = self.policy()
                packet = self.estimate(s=10., d=side)
                policy.update(0., self.state(), packet, True, 0., 0.)
                predictions = np.array([[[10., side, 0.], [10.4, side, 0.]]])
                hold = self.path()
                self.assertTrue(policy.accepts(hold, self.state(), packet, 0., 0., predictions, np.full(2, .1)))
                # The geometric body path is unchanged; an occupied side cannot
                # be squeezed to a boundary when the region has room elsewhere.
                self.envelope.interval.return_value = (-2., .4) if side > 0 else (-.4, 2.)
                self.assertFalse(policy.accepts(hold, self.state(), packet, 0., 0., predictions, np.full(2, .1)))
                self.assertTrue(hold.map_safe and hold.legal_safe and hold.separation_safe)
                self.envelope.interval.return_value = (-2., 2.)

    def test_future_overlap_and_uncertainty_room_are_checked_not_only_endpoint(self):
        policy = self.policy()
        packet = self.estimate()
        policy.update(0., self.state(), packet, True, 0., 0.)
        for side in (-1., 1.):
            predictions = np.array([[[8., side * .8, 0.], [10.4, side * .8, 0.]]])
            hold = self.path()
            self.assertTrue(policy.accepts(hold, self.state(), packet, 0., 0., predictions, np.full(2, .1)))
            self.assertFalse(policy.accepts(hold, self.state(), packet, 0., 0., predictions, np.full(2, .4)))

    def test_corner_approach_uses_braking_preview_both_turn_directions(self):
        for turn in (-.2, .2):
            with self.subTest(turn=turn):
                self.line.curvature = Mock(side_effect=lambda s: turn if s >= 15. else 0.)
                policy = self.policy()
                policy.update(0., self.state(speed=4.), self.estimate(s=4., speed=5.), True, 0., 0., threat=True)
                self.assertEqual(policy.phase, 'corner')
                self.assertIsNone(policy.tactical_bias)
                # Lower speed cannot see this bend inside the braking preview.
                slow = self.policy()
                slow.update(0., self.state(speed=.5), self.estimate(s=4., speed=1.), True, 0., 0., threat=True)
                self.assertEqual(slow.phase, 'early')
                self.line.curvature = Mock(return_value=0.)
                policy.update(1., self.state(speed=4.), self.estimate(s=4., speed=5.), True, 0., 0., threat=True)
                self.assertEqual(policy.lock, 'corner')

    def test_existing_lane_is_frozen_on_corner_entry_and_not_restarted(self):
        policy = self.policy()
        lane = self.commit(policy)
        self.line.curvature = Mock(return_value=.2)
        policy.update(.1, self.state(), self.estimate(), True, .1, 0.)
        self.assertEqual(policy.lock, 'corner')
        self.assertAlmostEqual(policy.hold_bias, .1)
        self.assertIsNone(policy.tactical_bias)
        self.assertEqual(policy.committed_bias, lane)
        self.line.curvature = Mock(return_value=0.)
        policy.update(1., self.state(), self.estimate(), True, .1, 0.)
        self.assertIsNone(policy.tactical_bias)
        self.assertTrue(policy.move_used)

    def test_stale_lost_invalid_tracks_disable_movement_not_memory(self):
        controller = self.controller()
        first = controller.step(0., self.state(), self.estimate())
        lane = first.racecraft_committed_bias
        self.assertIsNotNone(lane)
        packets = (self.estimate(fresh=False, age=1.), None,
                   self.estimate(valid=False), self.estimate(position_sigma=np.nan))
        for index, packet in enumerate(packets, 1):
            decision = controller.step(index * .1, self.state(), packet)
            self.assertFalse(decision.feasible)
            self.assertNotEqual(decision.mode, 'defend')
            self.assertEqual(decision.racecraft_phase, 'stale')
            self.assertEqual(decision.racecraft_committed_bias, lane)
            self.assertTrue(decision.racecraft_move_used)
            self.assertIsNone(controller.racecraft.tactical_bias)
        # Reacquisition near overlap cannot restart the unfinished lane change.
        decision = controller.step(.5, self.state(), self.estimate(s=10., d=1.5))
        self.assertEqual(decision.racecraft_phase, 'overlap')
        self.assertNotEqual(decision.mode, 'defend')
        self.assertEqual(decision.racecraft_committed_bias, lane)

    def test_recovery_during_dropout_or_disable_cannot_rearm_original_lane(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                policy = self.policy()
                lane = self.commit(policy)
                packet = None if missing else self.estimate()
                policy.update(.1, self.state(), packet, missing, .1, 0.)
                self.assertIsNone(policy.tactical_bias)
                policy.selected(self.path(bias=-.3), self.state(), 0.)
                self.assertEqual(policy.lock, 'recovery')
                self.assertEqual(policy.phase, 'stale' if missing else 'disabled')
                policy.update(.2, self.state(), self.estimate(), True, -.1, 0.)
                self.assertEqual(policy.committed_bias, lane)
                self.assertTrue(policy.move_used)
                self.assertIsNone(policy.tactical_bias)

    def test_zero_goal_error_recovery_cannot_rearm_defensive_lane(self):
        for side in (-1., 1.):
            for held_target in (0., .1 * side):
                with self.subTest(side=side, target=held_target):
                    policy = self.policy()
                    lane = self.commit(policy, side)
                    packet = self.estimate(d=side)
                    policy.update(.1, self.state(), packet, True, held_target, 0.)
                    recovery = self.path(end_d=-.1 * side, bias=0.)
                    predictions, inflation = opponent_hypotheses(packet, self.line,
                        np.array([0., .4]), self.config)
                    self.assertTrue(policy.accepts(recovery, self.state(), packet, 0., held_target,
                        predictions, inflation))
                    policy.selected(recovery, self.state(), 0.)
                    self.assertEqual(policy.lock, 'recovery')
                    self.assertEqual(policy.committed_bias, lane)
                    self.assertTrue(policy.move_used)
                    policy.update(.2, self.state(), packet, True, 0., 0.)
                    self.assertIsNone(policy.tactical_bias)

    def test_holding_command_without_a_reversal_does_not_spend_another_move(self):
        policy = self.policy()
        lane = self.commit(policy)
        policy.update(.1, self.state(), self.estimate(), True, .1, 0.)
        policy.selected(self.path(bias=.1), self.state(), 0.)
        self.assertEqual(policy.lock, '')
        self.assertEqual(policy.tactical_bias, lane)
        policy.selected(self.path(bias=lane, safe=False), self.state(), 0.)
        self.assertEqual(policy.lock, '')
        self.assertEqual(policy.committed_bias, lane)

    def test_unreliable_bound_violation_cannot_reset_encounter(self):
        controller = self.controller()
        lane = controller.step(0., self.state(), self.estimate()).racecraft_committed_bias
        for t in (.1, 2.):
            result = controller.step(t, self.state(), self.estimate(
                s=13., longitudinal_acceleration=self.config.max_opponent_acceleration + .1))
            self.assertFalse(result.feasible)
            self.assertEqual(result.racecraft_committed_bias, lane)
            self.assertTrue(result.racecraft_move_used)

    def test_reset_behind_requires_continuous_clear_dwell_with_wrap_and_noise(self):
        policy = self.policy()
        lane = self.commit(policy)
        ego = self.state(s=1.)
        clear = self.estimate(s=self.line.length - 7., speed=1.)
        policy.update(.1, ego, clear, True, 0., 0.)
        # Near the threshold: uncertainty means it isn't clearly separated.
        policy.update(.9, ego, self.estimate(s=self.line.length - 5., speed=1.), True, 0., 0.)
        policy.update(1., ego, clear, True, 0., 0.)
        policy.update(1.9, ego, clear, True, 0., 0.)
        self.assertEqual(policy.committed_bias, lane)
        policy.update(2.01, ego, clear, True, 0., 0.)
        self.assertFalse(policy.active)
        self.assertFalse(policy.move_used)
        self.assertIsNone(policy.committed_bias)
        policy.update(2.1, ego, self.estimate(s=self.line.length - 3., d=-1.), True, 0., 0.)
        self.assertEqual(policy.encounter, 2)
        self.assertLess(policy.tactical_bias, 0.)

    def test_safe_pass_dwell_and_dropout_do_not_reset_early(self):
        policy = self.policy()
        lane = self.commit(policy)
        policy.update(.1, self.state(), self.estimate(s=10., d=1.), True, 0., 0.)
        policy.update(.5, self.state(), self.estimate(s=10.9, d=1.), True, 0., 0.)
        self.assertEqual(policy.committed_bias, lane)
        passed = self.estimate(s=12., d=1.)
        policy.update(1., self.state(), passed, True, 0., 0.)
        policy.update(1.5, self.state(), None, True, 0., 0.)
        policy.update(3., self.state(), passed, True, 0., 0.)
        self.assertEqual(policy.committed_bias, lane)
        policy.update(4.01, self.state(), passed, True, 0., 0.)
        self.assertFalse(policy.active)
        self.assertIsNone(policy.committed_bias)

    def test_defense_disabled_keeps_baseline_and_never_clears_spent_move(self):
        controller = self.controller()
        result = controller.step(0., self.state(), self.estimate(), defense=False)
        self.assertNotEqual(result.mode, 'defend')
        self.assertAlmostEqual(controller._bias, 0.)
        self.assertFalse(result.racecraft_move_used)
        policy = self.policy()
        lane = self.commit(policy)
        policy.update(.1, self.state(), self.estimate(), False, .1, 0.)
        self.assertEqual(policy.phase, 'disabled')
        self.assertIsNone(policy.tactical_bias)
        policy.update(.2, self.state(), self.estimate(s=8.5, speed=4.), False, .1, 0.)
        policy.update(.3, self.state(), self.estimate(), True, .1, 0.)
        self.assertIsNone(policy.tactical_bias)
        self.assertEqual(policy.committed_bias, lane)
        self.assertTrue(policy.move_used)

    def test_committed_move_allows_checked_forward_recovery_away_from_opponent(self):
        controller = self.controller()
        controller.step(0., self.state(), self.estimate(d=-1.))
        lane = controller.racecraft.committed_bias
        self.assertLess(lane, 0.)

        def model(ego, bias, speed, times, predictions, inflation):
            path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, bias], (len(times), 1))
            repair = abs(bias - .3) < 1e-5 and abs(speed - .4) < 1e-5
            return _Rollout(bias, speed, path, repair or speed == 0., True, True,
                            1., 1., 0., .2 if repair else 0.)

        controller._rollout = Mock(side_effect=model)
        result = controller.step(.1, self.state(speed=0.), self.estimate(d=-1., speed=1.))
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(controller._best.bias, .3)
        self.assertGreater(result.speed, 0.)
        self.assertEqual(result.racecraft_phase, 'recovery')
        self.assertEqual(result.racecraft_committed_bias, lane)
        self.assertTrue(result.racecraft_move_used)
        self.assertIsNone(controller.racecraft.tactical_bias)

    def test_old_encounter_lock_does_not_strand_longitudinally_separated_corner_recovery(self):
        controller = self.controller()
        controller.step(0., self.state(), self.estimate(s=8.5, speed=4.))
        self.assertEqual(controller.racecraft.lock, 'late')

        def model(ego, bias, speed, times, predictions, inflation):
            path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, bias], (len(times), 1))
            repair = abs(bias - .3) < 1e-5 and abs(speed - .4) < 1e-5
            path[:, 5] = np.linspace(ego.d, ego.d + .1, len(times))
            return _Rollout(bias, speed, path, repair or speed == 0., True, True,
                            1., 1., 0., .2 if repair else 0.)

        controller._rollout = Mock(side_effect=model)
        # It is still the same locked encounter (no reset dwell), but the
        # opponent is now 7 m behind. Numerical d direction alone is not a block.
        result = controller.step(.1, self.state(speed=0.), self.estimate(s=3., speed=0.))
        self.assertEqual(controller.racecraft.lock, 'late')
        self.assertFalse(controller.racecraft.protect_line)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(controller._best.bias, .3)
        self.assertGreater(result.speed, 0.)
        self.assertNotEqual(result.mode, 'defend')
        self.assertFalse(result.racecraft_move_used)

    def test_no_feasible_room_remains_unresolved_emergency(self):
        controller = self.controller()
        self.envelope.interval.return_value = (-2., .4)
        result = controller.step(0., self.state(), self.estimate(s=10., d=1.8, speed=2.))
        self.assertEqual(result.mode, 'emergency')
        self.assertFalse(result.feasible)
        self.assertFalse(result.racecraft_safe)
        self.assertFalse(controller._best.feasible)
        self.assertIn('NOT certified safe', result.reason)
        self.assertEqual(self.envelope.interval.return_value, (-2., .4))

    def test_assumption_failure_does_not_hide_unresolved_emergency(self):
        controller = self.controller()
        self.track.clear_many = lambda points: np.zeros(len(points))
        self.track.footprint_safe = lambda *args: False
        result = controller.step(0., self.state(), self.estimate(position_sigma=np.nan))
        self.assertEqual(result.mode, 'emergency')
        self.assertFalse(result.feasible)
        self.assertIn('NOT certified safe', result.reason)
        self.assertIn('invalid opponent packet', result.reason)

    def test_zero_lane_settings_do_not_claim_a_defensive_move(self):
        self.config = replace(self.config, racecraft_lane_offset=0., racecraft_min_move=0.)
        controller = self.controller()
        result = controller.step(0., self.state(), self.estimate())
        self.assertNotEqual(result.mode, 'defend')
        self.assertFalse(result.racecraft_move_used)
        self.assertIsNone(result.racecraft_committed_bias)
        self.assertAlmostEqual(controller._bias, 0.)

    def test_zero_reset_settings_still_require_full_car_separation(self):
        self.config = replace(self.config, racecraft_reset_gap=0., racecraft_reset_dwell=0.)
        policy = self.policy()
        lane = self.commit(policy)
        for t in (.1, .2, 1.):
            policy.update(t, self.state(), self.estimate(s=9.5, speed=1.), True, 0., 0.)
            self.assertEqual(policy.committed_bias, lane)
            self.assertTrue(policy.move_used)
            self.assertEqual(policy.encounter, 1)

    def test_appended_decision_defaults_and_config_roundtrip(self):
        old = Decision(0., 0., 0., 'cruise', '', True, 1., 1., 0., 9)
        self.assertEqual(old.racecraft_phase, 'idle')
        self.assertIsNone(old.racecraft_committed_bias)
        self.assertFalse(old.racecraft_move_used)
        self.assertEqual(asdict(self.config), asdict(RobustConfig.from_dict(asdict(self.config))))
        for key in asdict(self.config):
            if key.startswith('racecraft_'):
                for value in (-1., np.nan, np.inf):
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        replace(self.config, **{key: value})


class HeadingTests(unittest.TestCase):
    def test_vectorized_matches_scalar_at_vertices_wrap_and_random_nodes(self):
        rng = np.random.default_rng(71)
        line = ClosedRaceline([[2., -1.], [6., 2.], [3., 9.], [-4., 3.]])
        nodes = np.concatenate((rng.uniform(-5 * line.length, 5 * line.length, 2000),
            line._starts, line._starts - 2 * line.length,
            np.nextafter(line._starts, -np.inf), np.nextafter(line._starts, np.inf)))
        expected = np.array([line.heading(s) for s in nodes])
        with patch.object(line, '_segment', side_effect=AssertionError('scalar segment lookup')):
            np.testing.assert_array_equal(trajectory_headings(line, nodes), expected)
        self.assertEqual(trajectory_headings(line, np.array([])).shape, (0,))
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trajectory_headings(line, np.array([0., value]))

    def test_custom_line_and_overridden_method_use_scalar_fallback(self):
        class CustomLine:
            def __init__(self):
                self.heading = Mock(side_effect=lambda s: .25 + .01 * s)

        nodes = np.array([-2., 0., 3., 160.])
        custom = CustomLine()
        np.testing.assert_array_equal(trajectory_headings(custom, nodes), .25 + .01 * nodes)
        self.assertEqual([call.args[0] for call in custom.heading.call_args_list], list(nodes))
        line = ClosedRaceline([[0., 0.], [40., 0.], [40., 40.], [0., 40.]])
        with patch.object(line, 'heading', side_effect=lambda s: .25 + .01 * s) as heading:
            np.testing.assert_array_equal(trajectory_headings(line, nodes), .25 + .01 * nodes)
            self.assertEqual(heading.call_count, len(nodes))

    def test_acceptance_matches_scalar_heading_path_for_room_checks(self):
        fixture = RacecraftTests()
        fixture.setUp()
        policy = fixture.policy()
        scalar = lambda line, nodes: np.array([line.heading(s) for s in nodes])
        rng = np.random.default_rng(83)
        for gap in (-5., -.5, 0., .5, 5.):
            for side in (-1., 1.):
                for lateral in (.3, .8, 1.5):
                    path = fixture.path()
                    path.trajectory[:, 4] = [39.9, 40.1]
                    predictions = np.empty((9, 2, 3))
                    for index, s in enumerate(path.trajectory[:, 4]):
                        path.trajectory[index, :3] = [*fixture.line.sample(s), fixture.line.heading(s)]
                        for hypothesis in range(9):
                            predictions[hypothesis, index] = [*fixture.line.sample(s + gap,
                                lateral * side + rng.uniform(-.1, .1)), fixture.line.heading(s + gap)]
                    args = (path, fixture.state(), None, 0., 0., predictions, np.full(2, .1))
                    vectorized = policy.accepts(*args)
                    with patch('rival_racecraft.trajectory_headings', scalar):
                        self.assertEqual(policy.accepts(*args), vectorized)


if __name__ == '__main__':
    unittest.main()
