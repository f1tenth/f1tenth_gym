"""Deterministic policy tests independent of Gym and the robust implementation."""
import unittest
import numpy as np
from rival_adversary import Adversary
from rival_strategy import CarState, Settings


class Line:
    length = 100.
    def signed_distance(self, a, b):
        return (a - b + 50) % 100 - 50
    def project(self, position):
        return float(position[0]) % self.length, float(position[1])
    def sample(self, s, d):
        return np.array([s, d])
    def heading(self, s):
        return 0.


class Envelope:
    def __init__(self, lo=-2., hi=2.):
        self.lo, self.hi = lo, hi
    def interval(self, s):
        return self.lo, self.hi
    def nominal(self, s):
        return float(np.clip(0., self.lo, self.hi))
    def reference(self, s):
        return self.nominal(s)
    def reference_slope(self, s):
        return 0.


class Track:
    def footprint_safe(self, *args):
        return True


def state(s, d=0., speed=2.):
    return CarState(s, d, speed, s, d, 0.)


class AdversaryTests(unittest.TestCase):
    def plan(self, policy, scenario='adaptive', t=2., ego=None, opponent=None, envelope=None):
        return policy.plan(t, scenario, opponent or state(5.), ego or state(10., .5),
                           Line(), envelope or Envelope(), Track(), Settings())

    def test_adaptive_free_side_deterministic(self):
        for _ in range(2):
            policy = Adversary(0.)
            result = self.plan(policy)
            self.assertEqual(policy.side, -1)
            self.assertLess(result.target, 0.)
            self.assertTrue(result.requested)

    def test_feint_switches_exactly_once(self):
        policy = Adversary(0.)
        self.plan(policy, 'feint', t=1.)
        side = policy.side
        self.plan(policy, 'feint', t=3.)
        self.assertEqual(policy.side, -side)
        self.plan(policy, 'feint', t=6.)
        self.assertEqual(policy.side, -side)
        self.assertTrue(policy.switched)

    def test_rearend_guard_reduces_command(self):
        policy = Adversary(0.)
        policy.speed = 2.
        result = self.plan(policy, 'approach', ego=state(10., 0., 0.), opponent=state(9., 0., 2.))
        self.assertLess(result.speed, 2.)
        self.assertTrue(result.guard_intervened)
        self.assertIn('rear_end', result.reason)

    def test_clipped_attack_not_feasible_credit(self):
        policy = Adversary(0.)
        result = self.plan(policy, 'left-pass', ego=state(10.), envelope=Envelope(-.1, .1))
        self.assertTrue(result.requested)
        self.assertFalse(result.requested_corridor_valid)
        self.assertTrue(result.guard_intervened)
        self.assertIn('clipped', result.reason)
        self.assertEqual(policy.requested_steps, 1)
        self.assertEqual(policy.corridor_valid_steps, 0)

    def test_cut_in_guard(self):
        policy = Adversary(0.)
        result = self.plan(policy, 'cut-in', ego=state(10.), opponent=state(10., .1))
        self.assertTrue(result.guard_intervened)
        self.assertIn('alongside', result.reason)

    def test_map_preview_blocked(self):
        class Blocked(Track):
            def footprint_safe(self, *args):
                return False
        policy = Adversary(0.)
        result = policy.plan(2., 'adaptive', state(5.), state(10.), Line(), Envelope(), Blocked(), Settings())
        self.assertFalse(result.sampled_path_feasible)
        self.assertTrue(result.guard_intervened)
        self.assertIn('map_or_legal', result.reason)


    def test_adaptive_rechecks_before_attack_with_commitment(self):
        policy = Adversary(0.)
        self.plan(policy, t=0., ego=state(10.))
        self.assertEqual(policy.side, 1)
        # Defender closes initially selected left corridor, opening right.
        self.plan(policy, t=.5, ego=state(10., 1.8))
        self.assertEqual(policy.side, 1)  # Initial .8s commitment.
        result = self.plan(policy, t=1., ego=state(10., 1.8))
        self.assertEqual(policy.side, -1)
        self.assertTrue(result.requested_corridor_valid)
        self.plan(policy, t=1.1, ego=state(10., -1.8))
        self.assertEqual(policy.side, -1)  # Bounded reevaluation frequency.
        self.plan(policy, t=1.5, ego=state(10., -1.8))
        self.assertEqual(policy.side, -1)  # New .8s commitment.
        self.plan(policy, t=2., ego=state(10., -.02))
        self.assertEqual(policy.side, -1)  # Small utility gain below hysteresis.

    def test_no_alongside_side_switch(self):
        policy = Adversary(-1.)
        self.plan(policy, t=0., ego=state(10., .8), opponent=state(5., -1.))
        self.assertEqual(policy.side, -1)
        self.plan(policy, t=1., ego=state(10., -.8), opponent=state(10., -1.))
        self.assertEqual(policy.side, -1)

    def test_opposite_clear_endpoints_still_cross_occupied_interval(self):
        from rival_adversary import swept_overlap
        self.assertTrue(swept_overlap(-1., .75, 0., .43))
        policy = Adversary(-1.)
        result = self.plan(policy, 'left-pass', ego=state(10.), opponent=state(10., -1.))
        self.assertGreater(abs(-1.), .43)
        self.assertGreater(abs(result.desired), .43)
        self.assertTrue(result.guard_intervened)
        self.assertIn('alongside', result.reason)
        self.assertEqual(result.target, -1.)

    def test_continuing_target_detects_future_narrow_section(self):
        class Narrowing(Envelope):
            def interval(self, s):
                return (-2., .02) if s >= 6. else (-2., 2.)
        policy = Adversary(0.)
        policy.speed = 2.
        result = self.plan(policy, 'left-pass', ego=state(10.),
                           opponent=state(5.), envelope=Narrowing())
        self.assertFalse(result.requested_corridor_valid)
        self.assertFalse(result.sampled_path_feasible)
        self.assertTrue(result.guard_intervened)
        self.assertIn('map_or_legal_preview', result.reason)
        self.assertEqual(result.target, 0.)
        self.assertEqual(policy.corridor_valid_steps, 0)
        self.assertEqual(policy.sampled_path_feasible_steps, 0)


    def test_xy_preview_heading_toward_wall_not_ideal_offset(self):
        class Wall(Track):
            def footprint_safe(self, xy, yaw, length, width, margin):
                return xy[1] + width / 2 + margin < .5
        policy = Adversary(0.)
        policy.speed = 2.
        heading = .7
        car = CarState(5., 0., 2. * np.cos(heading), 5., 0., heading)
        prediction = policy.preview(car, state(20.), Line(), Envelope(), Wall(),
            Settings(), 0., 2., previous_steering=0., measured_forward_speed=2.)
        self.assertFalse(prediction.feasible)
        self.assertIn('map_or_legal', prediction.reason)
        straight = policy.preview(state(5.), state(20.), Line(), Envelope(), Wall(),
            Settings(), 0., 2., measured_forward_speed=2.)
        self.assertTrue(straight.feasible)

    def test_preview_accepts_only_roundoff_at_legal_boundary(self):
        policy = Adversary(0.)
        boundary = Envelope(0., 1.)
        barely_below = state(5., -1e-15, 0.)
        result = policy.preview(barely_below, state(20.), Line(), boundary,
                                Track(), Settings(), 0., 0.)
        self.assertTrue(result.feasible)
        outside = policy.preview(state(5., -.001, 0.), state(20.), Line(), boundary,
                                 Track(), Settings(), 0., 0.)
        self.assertFalse(outside.feasible)

    def test_all_unsafe_is_unresolved_bounded_brake_not_attack(self):
        class Blocked(Track):
            def footprint_safe(self, *args):
                return False
        policy = Adversary(0.)
        policy.speed = 2.
        result = policy.plan(2., 'adaptive', state(5.), state(20.), Line(), Envelope(),
            Blocked(), Settings(), measured_forward_speed=2.)
        self.assertTrue(result.unresolved)
        self.assertFalse(result.selected_path_feasible)
        self.assertEqual(policy.executed_attack_steps, 0)
        self.assertAlmostEqual(result.speed, 2. - Settings().comfortable_braking * Settings().timestep)


    def test_lookahead_goal_not_compared_to_current_narrow_body_band(self):
        class Shifted(Envelope):
            def interval(self, s):
                return (-.02, .02) if s < .5 else (.19, .21)
            def nominal(self, s):
                return 0. if s < .5 else .2
        cfg = Settings(defender_speed=0., attacker_speed=0.)
        policy = Adversary(.2)
        result = policy.plan(0., 'no-attack', state(0., speed=0.), state(20.),
                             Line(), Shifted(), Track(), cfg, measured_forward_speed=0.)
        self.assertTrue(result.requested_corridor_valid)
        self.assertTrue(result.sampled_path_feasible)
        self.assertTrue(result.selected_path_feasible)
        self.assertFalse(result.guard_intervened)
        self.assertAlmostEqual(result.target, .2)
        self.assertGreater(result.target, Shifted().interval(0.)[1])

    def test_future_goal_follows_nominal_plus_fixed_bias(self):
        from unittest.mock import patch
        from rival_strategy import pursuit as real_pursuit
        class Sloping(Envelope):
            def nominal(self, s):
                return .01 * s
            def reference_slope(self, s):
                return .01
        cfg, envelope = Settings(), Sloping()
        initial_goal = envelope.nominal(5. + cfg.lookahead) + .1
        policy = Adversary(initial_goal)
        policy.speed = 1.
        targets = []
        def record(car, line, target, speed, steering, settings):
            targets.append((car.s, target))
            return real_pursuit(car, line, target, speed, steering, settings)
        with patch('rival_adversary.pursuit', side_effect=record):
            result = policy.preview(state(5., speed=1.), state(20.), Line(), envelope,
                Track(), cfg, initial_goal, 1., measured_forward_speed=1.)
        self.assertTrue(result.feasible)
        self.assertGreater(targets[-1][1], targets[0][1])
        for s, target in targets:
            self.assertAlmostEqual(target, envelope.nominal(s + cfg.lookahead) + .1)

    def test_recovery_correction_beats_first_safe_standstill_without_attack_credit(self):
        from unittest.mock import patch
        from rival_adversary import Preview
        policy = Adversary(0.)
        calls = []
        def preview(*args, **kwargs):
            goal, speed = args[-2:]
            calls.append((goal, speed))
            if abs(goal + .3) < 1e-9 and speed > 0:
                return Preview(True, 'checked_correction', .8, -.005, .04, .1)
            if abs(goal) < 1e-9:
                return Preview(True, 'checked_stationary', .8, 0., 0., 0.)
            return Preview(False, 'blocked', 0., 0., 0.)
        with patch.object(policy, 'preview', side_effect=preview):
            result = self.plan(policy, 'left-pass', ego=state(10.), opponent=state(5., speed=0.))
        self.assertTrue(result.guard_intervened)
        self.assertTrue(result.selected_path_feasible)
        self.assertGreater(result.speed, 0.)
        self.assertLess(result.target, 0.)
        self.assertIn('correction:-0.3', result.fallback)
        self.assertEqual(policy.executed_attack_steps, 0)
        self.assertTrue(any(abs(goal) < 1e-9 for goal, _ in calls))

    def test_crawl_correction_checked_after_safe_nonprogress_candidates(self):
        from unittest.mock import patch
        from rival_adversary import Preview
        policy = Adversary(0.)
        def preview(*args, **kwargs):
            goal, speed = args[-2:]
            if abs(goal - .3) < 1e-9 and abs(speed - .1 * Settings().attacker_speed) < 1e-9:
                return Preview(True, 'crawl_correction', .8, .005, .04, .02)
            if abs(goal) < 1e-9:
                return Preview(True, 'stationary', .8, 0., 0., 0.)
            return Preview(False, 'blocked', 0., 0., 0.)
        with patch.object(policy, 'preview', side_effect=preview):
            result = self.plan(policy, 'left-pass', ego=state(10.), opponent=state(5., speed=0.))
        self.assertIn('correction:+0.3:0.1', result.fallback)
        self.assertGreater(result.speed, 0.)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_raw_illegal_recovery_bias_is_checked_without_attack_credit(self):
        from unittest.mock import patch
        from rival_adversary import Preview
        policy, goals = Adversary(0.), []
        def preview(*args, **kwargs):
            goal, speed = args[-2:]
            goals.append(goal)
            return Preview(abs(goal) <= .1, 'checked', .8, 0., 0., 0.)
        with patch.object(policy, 'preview', side_effect=preview):
            result = self.plan(policy, 'left-pass', ego=state(10.), opponent=state(5., speed=0.),
                               envelope=Envelope(-.1, .1))
        self.assertGreater(goals[0], .1)  # Original rejected request remains visible.
        self.assertTrue(any(abs(goal) > .1 for goal in goals[1:]))
        self.assertTrue(result.selected_path_feasible)
        self.assertFalse(result.requested_corridor_valid)
        self.assertTrue(result.guard_intervened)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_no_forward_feasible_keeps_safe_stop_not_backward_candidate(self):
        from unittest.mock import patch
        from rival_adversary import Preview
        policy = Adversary(0.)
        def preview(*args, **kwargs):
            goal, speed = args[-2:]
            if abs(goal) < 1e-9:
                return Preview(True, 'stationary', .8, 0., 0., 0.)
            if abs(goal + .3) < 1e-9:
                return Preview(True, 'backward', .8, -.005, 0., -.1)
            return Preview(False, 'blocked', 0., 0., 0.)
        with patch.object(policy, 'preview', side_effect=preview):
            result = self.plan(policy, 'left-pass', ego=state(10.), opponent=state(5., speed=0.))
        self.assertEqual(result.speed, 0.)
        self.assertIn(':0.0', result.fallback)
        self.assertTrue(result.selected_path_feasible)
        self.assertFalse(result.unresolved)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_preview_progress_is_accumulated_signed_projected_distance(self):
        policy = Adversary(0.)
        policy.speed = 1.
        result = policy.preview(state(99.8, speed=1.), state(80.), Line(), Envelope(),
                                Track(), Settings(), 0., 1., measured_forward_speed=1.)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.progress, policy.preview_horizon)

    def test_recorded_s49_stall_pose_has_checked_forward_recovery(self):
        from pathlib import Path
        from rival_baseline import load
        from rival_robust import RobustConfig, TrackEnvelope
        # Last attacker pose/command in lookahead_adaptive60s CSV. Keep this
        # fixture literal so the regression does not depend on output files.
        cfg, line, track, _, _ = load(Path(__file__).with_name('rival_advanced.yaml'))
        envelope = TrackEnvelope(line, track, cfg, RobustConfig())
        car = CarState(49.02029354273702, .21346893354449062, 1e-323,
                       -37.93932406095216, 25.005973561266565, 2.6823364816180373)
        s, speed = 100., 2.
        d = envelope.nominal(s)
        x, y = line.sample(s, d)
        defender = CarState(s, d, speed, x, y, line.heading(s))
        policy = Adversary(.0864645872546161)
        result = policy.plan(60., 'adaptive', car, defender, line, envelope, track, cfg,
            previous_steering=.12778304719090974, measured_forward_speed=0.)
        self.assertTrue(result.guard_intervened)
        self.assertTrue(result.selected_path_feasible)
        self.assertGreater(result.speed, 0.)
        self.assertFalse(result.unresolved)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_future_raw_bias_clipped_but_actual_target_is_not_projected(self):
        from unittest.mock import patch
        from rival_strategy import pursuit as real_pursuit
        class Narrowing(Envelope):
            def interval(self, s):
                return (-.1, 1.) if s > 1. else (-1., 1.)
        policy = Adversary(0.)
        policy.speed = .5
        predicted_d = []
        def record(car, *args):
            predicted_d.append(car.d)
            return real_pursuit(car, *args)
        with patch('rival_adversary.pursuit', side_effect=record):
            result = policy.preview(state(0., d=.2, speed=.5), state(20.), Line(), Narrowing(),
                Track(), Settings(), -.3, .5, measured_forward_speed=.5)
        self.assertTrue(result.feasible)
        self.assertLess(predicted_d[-1], predicted_d[0])
        self.assertTrue(result.goal_clipped)
        self.assertGreater(result.progress, 0.)
        # The initial goal is clipped, but a target too far outside cannot be
        # snapped into the legal band in one action; it must remain rejected.
        policy = Adversary(-.8)
        result = policy.preview(state(0., speed=0.), state(20.), Line(), Envelope(-.1, .1),
            Track(), Settings(), -.3, 0., measured_forward_speed=0.)
        self.assertFalse(result.feasible)
        self.assertTrue(result.goal_clipped)
        self.assertAlmostEqual(result.target, -.795)

    def test_future_clipped_request_never_counts_as_executed_attack(self):
        from unittest.mock import patch
        from rival_adversary import Preview
        policy = Adversary(0.)
        with patch.object(policy, 'preview', return_value=Preview(True, 'checked_clipped', .8,
                                                               0., .04, .1, True)):
            result = self.plan(policy, 'left-pass', ego=state(10.), opponent=state(5., speed=0.))
        self.assertFalse(result.requested_corridor_valid)
        self.assertTrue(result.sampled_path_feasible)
        self.assertTrue(result.guard_intervened)
        self.assertIn('clipped_future_passing_corridor', result.reason)
        self.assertEqual(policy.executed_attack_steps, 0)
        self.assertEqual(policy.corridor_valid_steps, 0)

    def test_recorded_s54_narrowing_goal_uses_clipped_moving_recovery(self):
        from pathlib import Path
        from rival_baseline import load
        from rival_robust import RobustConfig, TrackEnvelope
        cfg, line, track, _, _ = load(Path(__file__).with_name('rival_advanced.yaml'))
        envelope = TrackEnvelope(line, track, cfg, RobustConfig())
        # Final attacker observation/commands in longrun300_stream_diagnostic.
        car = CarState(54.2833083679789, 1.0489185988358765, 0.,
                       -42.729808049820704, 23.780951250617992, 3.637089369633178)
        s = 100.
        d = envelope.nominal(s)
        x, y = line.sample(s, d)
        defender = CarState(s, d, 0., x, y, line.heading(s))
        policy = Adversary(1.1077467991736503)
        kwargs = dict(previous_steering=-.0328077500988247, measured_forward_speed=0.)
        nominal = envelope.nominal(car.s + cfg.lookahead)
        moving_nominal = policy.preview(car, defender, line, envelope, track, cfg,
            nominal, .2 * policy.reference_speed_cap(car.s, envelope, cfg), **kwargs)
        self.assertFalse(moving_nominal.feasible)
        stationary = policy.preview(car, defender, line, envelope, track, cfg,
                                    nominal, 0., **kwargs)
        self.assertTrue(stationary.feasible)
        self.assertEqual(stationary.progress, 0.)
        correction = policy.preview(car, defender, line, envelope, track, cfg,
            nominal + .3, .1 * policy.reference_speed_cap(car.s, envelope, cfg), **kwargs)
        self.assertTrue(correction.feasible)
        self.assertTrue(correction.goal_clipped)
        self.assertGreater(correction.progress, 0.)
        result = policy.plan(90., 'adaptive', car, defender, line, envelope, track, cfg, **kwargs)
        self.assertTrue(result.selected_path_feasible)
        self.assertGreater(result.speed, 0.)
        # The clearer reference can make a hold-based moving repair safe too;
        # require checked progress rather than one specific candidate name.
        self.assertNotEqual(result.fallback, 'none')
        self.assertFalse(result.unresolved)
        self.assertFalse(result.requested)  # Distant opponent is nominal catch-up.
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_minor_body_error_requires_explicit_repair_and_strict_reentry(self):
        policy = Adversary(0.)
        envelope = Envelope(-1., 1.)
        car = state(5., d=1.000078, speed=0.)
        kwargs = dict(measured_forward_speed=0.)
        strict = policy.preview(car, state(20.), Line(), envelope, Track(), Settings(), 0., .5, **kwargs)
        self.assertFalse(strict.feasible)
        repair = policy.preview(car, state(20.), Line(), envelope, Track(), Settings(), 0., .5,
                                body_recovery=True, **kwargs)
        self.assertTrue(repair.feasible)
        self.assertTrue(repair.body_band_recovery)
        self.assertGreater(repair.progress, 0.)
        self.assertAlmostEqual(repair.initial_body_error, .000078)
        self.assertLessEqual(repair.max_body_error, policy.legal_recovery_tolerance)
        self.assertLessEqual(repair.final_body_error, policy.legal_numeric_tolerance)
        stationary = policy.preview(car, state(20.), Line(), envelope, Track(), Settings(), 0., 0.,
                                    body_recovery=True, **kwargs)
        self.assertFalse(stationary.feasible)
        self.assertEqual(stationary.reason, 'body_band_recovery_insufficient_descent')

    def test_body_recovery_keeps_map_target_hard_without_fixed_error_lockout(self):
        class Blocked(Track):
            def footprint_safe(self, *args):
                return False
        policy = Adversary(0.)
        kwargs = dict(body_recovery=True, measured_forward_speed=0.)
        car = state(5., d=1.000078, speed=0.)
        blocked = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Blocked(), Settings(),
                                 0., .5, **kwargs)
        self.assertFalse(blocked.feasible)
        self.assertIn('footprint', blocked.reason)
        large = policy.preview(state(5., d=1.021, speed=0.), state(20.), Line(), Envelope(-1., 1.),
                               Track(), Settings(), 0., .5, **kwargs)
        self.assertTrue(large.feasible)
        self.assertLessEqual(large.max_body_error, large.initial_body_error + policy.recovery_error_allowance)
        self.assertLess(large.final_body_error, large.initial_body_error)
        policy.target = 1.2
        unreachable = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                     0., .5, **kwargs)
        self.assertFalse(unreachable.feasible)
        self.assertIn('lookahead_target', unreachable.reason)
        result = self.plan(Adversary(0.), 'adaptive', ego=state(20.),
                           opponent=state(5., d=2.021, speed=0.))
        self.assertFalse(result.unresolved)
        self.assertIn('body_band_recovery', result.reason)
        self.assertIn('rule_violation_present=True', result.reason)

    def test_recovery_never_increases_initial_rule_error_during_rollout(self):
        class Tightens(Envelope):
            def interval(self, s):
                return (-1., 1.) if s <= 5.1 else (-1., .97)
        policy = Adversary(0.)
        policy.speed = 1.
        result = policy.preview(state(5., d=1.000078, speed=1.), state(20.), Line(), Tightens(),
                                Track(), Settings(), 0., 1., body_recovery=True,
                                measured_forward_speed=1.)
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, 'body_band_recovery_error_increased')
        self.assertGreater(result.max_body_error, .02)
        # Merely enabling repair cannot create a tolerance for a body that
        # started strictly legal; only an existing minor error qualifies.
        result = policy.preview(state(5., d=.9999, speed=1.), state(20.), Line(), Tightens(),
                                Track(), Settings(), 0., 1., body_recovery=True,
                                measured_forward_speed=1.)
        self.assertFalse(result.feasible)
        self.assertFalse(result.body_band_recovery)
        self.assertIn('strict_body_band', result.reason)

    def test_body_repair_is_guarded_and_never_passing_credit(self):
        policy = Adversary(0.)
        result = self.plan(policy, 'adaptive', ego=state(20.),
                           opponent=state(5., d=2.000078, speed=0.))
        self.assertTrue(result.body_band_recovery)
        self.assertTrue(result.guard_intervened)
        self.assertIn('body_band_recovery', result.reason)
        self.assertIn('body_error_m=', result.reason)
        self.assertTrue(result.selected_path_feasible)
        self.assertFalse(result.sampled_path_feasible)
        self.assertGreater(result.speed, 0.)
        self.assertLessEqual(result.predicted_final_body_error, policy.legal_numeric_tolerance)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_recorded_s41_minor_error_can_reenter_strict_band(self):
        from pathlib import Path
        from rival_baseline import load
        from rival_robust import RobustConfig, TrackEnvelope
        cfg, line, track, _, _ = load(Path(__file__).with_name('rival_advanced.yaml'))
        envelope = TrackEnvelope(line, track, cfg, RobustConfig())
        car = CarState(41.66662852772548, 1.2000783792392904, 0.,
                       -32.090986404510886, 21.023245644574498, 2.4964796149507413)
        s = 100.
        d = envelope.nominal(s)
        x, y = line.sample(s, d)
        defender = CarState(s, d, 0., x, y, line.heading(s))
        policy = Adversary(1.2)
        result = policy.plan(120., 'adaptive', car, defender, line, envelope, track, cfg,
            previous_steering=.00983774148884641, measured_forward_speed=0.)
        self.assertTrue(result.body_band_recovery)
        self.assertTrue(result.selected_path_feasible)
        self.assertFalse(result.unresolved)
        self.assertGreater(result.speed, 0.)
        self.assertLessEqual(result.predicted_max_body_error, .02)
        self.assertLessEqual(result.predicted_final_body_error, policy.legal_numeric_tolerance)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_rule_error_above_two_cm_can_partially_descend_without_legal_status(self):
        policy = Adversary(.98)
        policy.speed = .5
        car = state(5., d=1.02092, speed=.5)
        kwargs = dict(body_recovery=True, measured_forward_speed=.5)
        repair = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                .98, .5, **kwargs)
        self.assertTrue(repair.feasible)
        self.assertTrue(repair.partial_body_recovery)
        self.assertEqual(repair.reason, 'body_band_descent_partial')
        self.assertGreater(repair.final_body_error, policy.legal_numeric_tolerance)
        self.assertLessEqual(repair.max_body_error, repair.initial_body_error + 1e-6)
        required = min(.005, .2 * repair.initial_body_error)
        self.assertGreaterEqual(repair.initial_body_error - repair.final_body_error, required)
        too_little = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                    1., .5, **kwargs)
        self.assertFalse(too_little.feasible)
        self.assertEqual(too_little.reason, 'body_band_recovery_insufficient_descent')
        class Blocked(Track):
            def footprint_safe(self, *args):
                return False
        obstacle = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Blocked(), Settings(),
                                  .98, .5, **kwargs)
        self.assertFalse(obstacle.feasible)
        self.assertIn('footprint', obstacle.reason)
        increasing = CarState(5., 1.02092, .5, 5., 1.02092, .2)
        rejected = policy.preview(increasing, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                  .98, .5, **kwargs)
        self.assertFalse(rejected.feasible)
        self.assertEqual(rejected.reason, 'body_band_recovery_error_increased')

    def test_executed_rule_error_growth_is_observable_not_attack_credit(self):
        policy = Adversary(0.)
        first = self.plan(policy, 'adaptive', ego=state(20.), opponent=state(5., d=2.02092, speed=0.))
        self.assertTrue(first.body_band_recovery)
        self.assertTrue(first.selected_path_feasible)
        second = self.plan(policy, 'adaptive', t=2.02, ego=state(20.),
                           opponent=state(5., d=2.022, speed=0.))
        self.assertTrue(second.actual_recovery_error_increased)
        self.assertGreater(second.actual_body_error_delta, 1e-6)
        self.assertIn('actual_recovery_error_increased=True', second.reason)
        self.assertIn('rule_violation_present=True', second.reason)
        self.assertEqual(policy.executed_violation_steps, 2)
        self.assertEqual(policy.recovery_error_growth_steps, 1)
        self.assertEqual(policy.executed_attack_steps, 0)

    def test_distant_opponent_is_nominal_catchup_not_passing_edge(self):
        policy = Adversary(0.)
        result = policy.plan(3., 'adaptive', state(5., speed=2.), state(50., d=1.5),
                             Line(), Envelope(), Track(), Settings(), measured_forward_speed=2.)
        self.assertFalse(result.passing_window)
        self.assertFalse(result.requested)
        self.assertEqual(result.desired, 0.)
        self.assertGreater(result.speed, 0.)
        self.assertEqual(policy.requested_steps, 0)
        self.assertEqual(policy.executed_attack_steps, 0)
        surge = Adversary(0.)
        surge.speed = 2.
        command = surge.plan(3., 'surge', state(5., speed=2.), state(50.),
                             Line(), Envelope(), Track(), Settings(), measured_forward_speed=2.)
        self.assertFalse(command.requested)
        self.assertEqual(command.desired, 0.)
        self.assertGreater(command.speed, Settings().defender_speed)

    def test_passing_window_covers_close_behind_alongside_not_fully_ahead(self):
        for attacker_s, speed, expected in ((5., 0., True), (3., 4., True),
                                           (2., 0., False), (10., 2., True), (10.8, 2., False)):
            with self.subTest(attacker_s=attacker_s, speed=speed):
                command = Adversary(0.).plan(3., 'adaptive', state(attacker_s, speed=speed),
                    state(10.), Line(), Envelope(), Track(), Settings(), measured_forward_speed=speed)
                self.assertEqual(command.passing_window, expected)
                self.assertEqual(command.requested, expected)
                if not expected:
                    self.assertEqual(command.desired, 0.)

    def test_tracking_margin_is_explicit_validated_and_defaults_to_strict(self):
        self.assertEqual(Adversary(0.).tracking_margin, 0.)
        self.assertEqual(Adversary(0., tracking_margin=.08).tracking_margin, .08)
        for margin in (-.01, float('nan'), float('inf')):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                Adversary(0., tracking_margin=margin)

    def test_positive_tracking_budget_allows_lag_without_claiming_strict_legality(self):
        car = state(5., d=1.04, speed=.5)
        args = (car, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(), 0., .5)
        strict = Adversary(0.).preview(*args, measured_forward_speed=.5)
        self.assertFalse(strict.feasible)
        budgeted = Adversary(0., tracking_margin=.08).preview(*args, measured_forward_speed=.5)
        self.assertTrue(budgeted.feasible)
        self.assertTrue(budgeted.within_tracking_budget)
        self.assertFalse(budgeted.strict_body_legal)
        self.assertFalse(budgeted.body_band_recovery)
        self.assertAlmostEqual(budgeted.initial_body_error, .04)
        self.assertLessEqual(budgeted.max_body_error, .08 + 1e-8)
        stationary = Adversary(0., tracking_margin=.08).preview(
            state(5., d=1.04, speed=0.), state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
            0., 0., measured_forward_speed=0.)
        self.assertTrue(stationary.feasible)
        self.assertFalse(stationary.strict_body_legal)
        self.assertAlmostEqual(stationary.final_body_error, .04)

    def test_plan_reports_within_budget_error_without_enabling_descent(self):
        policy = Adversary(0., tracking_margin=.08)
        command = self.plan(policy, 'adaptive', ego=state(20.),
                            opponent=state(5., d=2.04, speed=0.))
        self.assertTrue(command.selected_path_feasible)
        self.assertFalse(command.body_band_recovery)
        self.assertTrue(command.predicted_within_tracking_budget)
        self.assertFalse(command.predicted_strict_body_legal)
        self.assertAlmostEqual(command.initial_body_error, .04)
        self.assertIn('rule_violation_present=True', command.reason)
        self.assertIn('required_descent_m=0;', command.reason)
        self.assertIn('tracking_budget_m=0.08', command.reason)

    def test_budget_does_not_relax_footprint_collision_or_target_legality(self):
        class Blocked(Track):
            def footprint_safe(self, *args):
                return False
        policy = Adversary(0., tracking_margin=.08)
        car = state(5., d=1.04, speed=0.)
        obstacle = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Blocked(), Settings(),
                                  0., .5, measured_forward_speed=0.)
        self.assertFalse(obstacle.feasible)
        self.assertIn('footprint', obstacle.reason)
        collision = policy.preview(state(5., speed=0.), state(5.), Line(), Envelope(-1., 1.),
                                   Track(), Settings(), 0., .5, measured_forward_speed=0.)
        self.assertFalse(collision.feasible)
        self.assertEqual(collision.reason, 'sampled_collision_preview')
        policy.target = 1.04  # Within body budget, outside strict pursuit band.
        target = policy.preview(car, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                1., .5, measured_forward_speed=0.)
        self.assertFalse(target.feasible)
        self.assertIn('lookahead_target', target.reason)
        self.assertAlmostEqual(target.target, 1.035)  # No instantaneous projection.

    def test_budget_growth_rejected_and_existing_excess_needs_descent(self):
        policy = Adversary(0., tracking_margin=.08)
        outward = CarState(5., 1.07, 1., 5., 1.07, .5)
        rejected = policy.preview(outward, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                  0., 1., measured_forward_speed=1., body_recovery=True)
        self.assertFalse(rejected.feasible)
        self.assertFalse(rejected.body_band_recovery)
        self.assertIn('body_tracking_budget_exceeded', rejected.reason)
        policy = Adversary(.98, tracking_margin=.08)
        policy.speed = .5
        outside = state(5., d=1.1, speed=.5)
        strict_budget = policy.preview(outside, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                       .98, .5, measured_forward_speed=.5)
        self.assertFalse(strict_budget.feasible)
        descent = policy.preview(outside, state(20.), Line(), Envelope(-1., 1.), Track(), Settings(),
                                 .98, .5, measured_forward_speed=.5, body_recovery=True)
        self.assertTrue(descent.feasible)
        self.assertTrue(descent.body_band_recovery)
        self.assertFalse(descent.within_tracking_budget)
        self.assertFalse(descent.strict_body_legal)
        self.assertLessEqual(descent.max_body_error, descent.initial_body_error + 1e-6)
        self.assertGreaterEqual(descent.initial_body_error - descent.final_body_error, .005)

    def test_reference_slope_caps_speed_before_shift(self):
        class SteepAhead(Envelope):
            def reference_slope(self, s):
                return 2. if s > .5 else 0.
        cfg = Settings()
        cap = Adversary.reference_speed_cap(0., SteepAhead(), cfg)
        self.assertAlmostEqual(cap, .8 * cfg.offset_rate / 2.)
        policy = Adversary(0.)
        result = policy.plan(0., 'no-attack', state(0., speed=0.), state(20.),
                             Line(), SteepAhead(), Track(), cfg, measured_forward_speed=0.)
        self.assertLessEqual(result.speed, cap)


class RunnerMetricsTests(unittest.TestCase):
    def test_local_stall_not_hidden_by_large_early_progress(self):
        from rival_advanced import StallMetric
        stalls = StallMetric()
        for t in range(1, 91):
            progress = min(t, 25.) * 2.
            stalls.update(float(t), [progress, float(t)])
        self.assertEqual(stalls.rolling_progress[0], 0.)
        self.assertEqual(stalls.rolling_progress[1], 5.)
        self.assertTrue(stalls.active[0])
        self.assertFalse(stalls.active[1])
        self.assertAlmostEqual(stalls.start[0], 25.)
        self.assertAlmostEqual(stalls.max_time[0], 65.)
        self.assertFalse(50. < .2 * 90.)  # Whole-run criterion misses this stall.
        self.assertTrue(stalls.timed_out(60.))
        stalls.update(91., [51., 91.])
        self.assertFalse(stalls.active[0])
        self.assertIsNone(stalls.start[0])
        self.assertEqual(stalls.last_start[0], 25.)
        self.assertEqual(stalls.max_time[0], 65.)

    def test_stall_window_interpolation_and_reverse_progress(self):
        from rival_advanced import StallMetric
        stalls = StallMetric()
        for t in (1.3, 2.6, 3.9, 5.2, 6.5):
            stalls.update(t, [t, -t])
        self.assertAlmostEqual(stalls.rolling_progress[0], 5.)
        self.assertAlmostEqual(stalls.rolling_progress[1], -5.)
        self.assertFalse(stalls.active[0])
        self.assertTrue(stalls.active[1])
        self.assertTrue(stalls.timed_out(5.))

    def test_attacker_failures_cannot_be_defense_credit(self):
        from rival_advanced import attack_outcome
        for progress, offtrack in (([4., 0.], [False, False]), ([4., 2.], [False, True])):
            result = attack_outcome(progress, 4., [False, False], offtrack, 100, 0, 1)
            self.assertTrue(result['invalid_attack'])
            self.assertTrue(result['attacker_failure'])
            self.assertFalse(result['defended_attack_credit'])
            self.assertFalse(result['defender_failure'])
            self.assertFalse(result['attack_evaluation_eligible'])
    def test_every_step_logged_and_only_actual_solves_counted(self):
        from rival_advanced import StepMetrics
        metrics = StepMetrics(.02, .1, [0., 0.])
        rows = [metrics.update([[0., .02 * i], [0., 0.]], [0., 0.], timing, 1.)
                for i, timing in enumerate((120., 0., 0., 4., 0.))]
        self.assertEqual(len(rows), 5)
        self.assertEqual(metrics.planning, [120., 4.])
        self.assertEqual(metrics.missed, 1)
        self.assertEqual([r['planning_budget_miss_count'] for r in rows], [1] * 5)
        self.assertAlmostEqual(rows[1]['ego_command_acceleration'], 1.)
        self.assertAlmostEqual(rows[1]['ego_command_jerk'], 50.)

    def test_pipeline_deadline_separate_from_solver_budget(self):
        from rival_advanced import StepMetrics
        metrics = StepMetrics(.02, .1, [0., 0.])
        metrics.update([[0., 0.], [0., 0.]], [0., 0.], 5., 6.)
        row = metrics.pipeline(12., 3., 25.)
        self.assertEqual(row['perception_tracking_ms'], 15.)
        self.assertTrue(row['physics_deadline_miss'])
        self.assertEqual(metrics.missed, 0)
        self.assertEqual(metrics.physics_deadline_misses, 1)

    def test_legal_tolerance_and_sensor_bounds_are_separate(self):
        from rival_advanced import EvaluationBounds
        from rival_perception import SensorConfig
        bounds = EvaluationBounds(.1)
        previous = [state(10., .105), state(5.)]
        row = bounds.update([state(10., .11), state(5., .2, 9.)], previous,
                            Envelope(-.1, .1), SensorConfig())
        self.assertFalse(row['ego_legal_violation'])
        self.assertTrue(row['attacker_legal_violation'])
        self.assertAlmostEqual(bounds.duration[1], .1)
        self.assertAlmostEqual(bounds.max_magnitude[0], .01)
        self.assertEqual(bounds.sensor_counts, dict(speed=1, acceleration=1, lateral_speed=1))

    def test_pass_uses_unwrapped_progress_and_onset_timestamp(self):
        from rival_advanced import PassMetric
        metric = PassMetric(3., .7)
        self.assertFalse(metric.update(1., [0., 3.8]))
        self.assertFalse(metric.update(1.4, [0., 3.9]))
        self.assertTrue(metric.update(1.5, [0., 4.]))
        self.assertEqual(metric.time, 1.)
        # A half-lap shortest-gap sign flip must not erase a confirmed pass.
        self.assertFalse(metric.update(2., [0., 70.]))
        self.assertEqual(metric.time, 1.)
        self.assertEqual(metric.lead, 67.)

    def test_wrap_progress_and_interrupted_lead(self):
        from rival_advanced import PassMetric
        line = Line()
        self.assertEqual(line.signed_distance(1., 99.), 2.)
        metric = PassMetric(3., .7)
        metric.update(1., [0., 4.])
        metric.update(1.2, [0., 3.])
        self.assertIsNone(metric.since)
        self.assertFalse(metric.update(2., [0., 4.]))
        self.assertTrue(metric.update(2.5, [0., 4.]))
        self.assertEqual(metric.time, 2.)

    def test_legacy_missing_estimate_nominal_stop(self):
        from rival_advanced import LegacyAdapter
        adapter = LegacyAdapter(Line(), Track(), Settings(), .5, Envelope())
        result = adapter.step(0., state(10.), None)
        self.assertEqual(result.speed, 0.)
        self.assertEqual(result.target_offset, 0.)
        self.assertIn('no_valid_fresh_estimate', result.reason)
        self.assertFalse(result.feasible)


class SimulationSubstepTests(unittest.TestCase):
    @staticmethod
    def observation(x=0., collision=False, yaw_rate=0., theta=0.):
        return dict(poses_x=[x, x + 5.], poses_y=[0., 0.],
                    poses_theta=[theta, 0.], linear_vels_x=[1., 1.],
                    linear_vels_y=[0., 0.], ang_vels_z=[yaw_rate, 0.],
                    collisions=[collision, False])

    class FakeEnv:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.actions = []
        def step(self, action):
            self.actions.append(action.copy())
            obs, done = next(self.responses)
            return obs, 0., done, {}

    def test_schedule_and_held_commands(self):
        from rival_advanced import held_action_steps, integration_schedule
        dt, count = integration_schedule(.02, .01)
        self.assertEqual((dt, count), (.01, 2))
        self.assertEqual(integration_schedule(.025, .01)[1], 3)
        actions = np.array([[.1, 2.], [-.2, 3.]])
        env = self.FakeEnv([(self.observation(), False)] * count)
        results = list(held_action_steps(env, actions, count, dt, 0., Track(), Settings()))
        self.assertEqual(len(env.actions), 2)
        for held in env.actions:
            np.testing.assert_array_equal(held, actions)
        self.assertAlmostEqual(results[-1].elapsed, .02)

    def test_inner_collision_stops_before_clean_later_observation(self):
        from rival_advanced import held_action_steps
        env = self.FakeEnv([(self.observation(), False),
                            (self.observation(collision=True), False),
                            (self.observation(), False)])
        results = list(held_action_steps(env, np.zeros((2, 2)), 4, .005, .3, Track(), Settings()))
        self.assertEqual(len(env.actions), 2)
        self.assertTrue(results[-1].collision[0])
        self.assertEqual(results[-1].reason, 'collision')
        self.assertAlmostEqual(results[-1].elapsed, .31)
        self.assertEqual(results[-1].count, 2)

    def test_inner_done_and_offtrack_stop_immediately(self):
        from rival_advanced import held_action_steps
        env = self.FakeEnv([(self.observation(), True), (self.observation(), False)])
        results = list(held_action_steps(env, np.zeros((2, 2)), 2, .01, 0., Track(), Settings()))
        self.assertEqual(len(env.actions), 1)
        self.assertEqual(results[-1].reason, 'gym_done')
        self.assertAlmostEqual(results[-1].elapsed, .01)
        class Outside(Track):
            def footprint_safe(self, *args):
                return False
        env = self.FakeEnv([(self.observation(), False)] * 2)
        results = list(held_action_steps(env, np.zeros((2, 2)), 2, .01, 0., Outside(), Settings()))
        self.assertEqual(len(env.actions), 1)
        self.assertTrue(results[0].offtrack.all())
        self.assertEqual(results[0].reason, 'offtrack')

    def test_invalid_observed_yaw_and_rate_terminate_without_repair(self):
        from rival_advanced import held_action_steps, numerical_failures
        for obs in (self.observation(theta=float('nan')), self.observation(yaw_rate=101.),
                    self.observation(yaw_rate=float('inf'))):
            self.assertTrue(numerical_failures(obs)[0])
            env = self.FakeEnv([(obs, False), (self.observation(), False)])
            results = list(held_action_steps(env, np.zeros((2, 2)), 2, .01, 0., Track(), Settings()))
            self.assertEqual(len(env.actions), 1)
            self.assertEqual(results[0].reason, 'numerical_instability')
        self.assertFalse(any(numerical_failures(self.observation(yaw_rate=100.))))

    def test_wrap_progress_sampled_each_inner_step(self):
        from rival_advanced import held_action_steps
        line, previous, progress = Line(), 99.9, 0.
        env = self.FakeEnv([(self.observation(x=.1), False), (self.observation(x=.3), False)])
        for result in held_action_steps(env, np.zeros((2, 2)), 2, .01, 0., Track(), Settings()):
            s, _ = line.project([result.obs['poses_x'][0], 0.])
            progress += line.signed_distance(s, previous)
            previous = s
        self.assertAlmostEqual(progress, .4)

    def test_numerical_failure_excludes_attack_eligibility(self):
        from rival_advanced import attack_outcome
        result = attack_outcome([3., 4.], 2., [False, False], [False, False],
                                10, 10, 0, [True, False])
        self.assertTrue(result['defender_failure'])
        self.assertFalse(result['attack_evaluation_eligible'])
        self.assertFalse(result['defended_attack_credit'])


class StreamingOutputTests(unittest.TestCase):
    def test_restarted_run_cannot_retain_a_stale_success_summary(self):
        import tempfile
        from pathlib import Path
        from rival_advanced import RunStream
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            output.with_suffix('.json').write_text('{"completed": true}')
            output.with_suffix('.checkpoint.json').write_text('{"elapsed": 300}')
            with RunStream(output):
                self.assertFalse(output.with_suffix('.json').exists())
                self.assertFalse(output.with_suffix('.checkpoint.json').exists())
            self.assertTrue(output.with_suffix('.csv').exists())

    def test_partial_csv_flush_and_incomplete_checkpoint_on_interrupt(self):
        import csv
        import json
        import tempfile
        from pathlib import Path
        from rival_advanced import RunStream
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested' / 'interrupted'
            stream = RunStream(output)
            with self.assertRaises(TimeoutError):
                with stream:
                    stream.write_row(dict(time=0., step_end_time=.02, physics_step_count=2))
                    stream.checkpoint(dict(elapsed=.02, progress=[0., 0.]))
                    stream.write_row(dict(time=.02, step_end_time=.04, physics_step_count=2))
                    raise TimeoutError('simulated wall timeout')
            self.assertTrue(stream.stream.closed)
            rows = list(csv.DictReader(output.with_suffix('.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertIn('physics_step_count', rows[0])
            checkpoint = json.loads(output.with_suffix('.checkpoint.json').read_text())
            self.assertTrue(checkpoint['incomplete'])
            self.assertFalse(checkpoint['completed'])
            self.assertFalse(output.with_suffix('.json').exists())

    def test_stream_flushes_by_time_or_fifty_actions_without_rewriting(self):
        import csv
        import tempfile
        from pathlib import Path
        from rival_advanced import RunStream
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with RunStream(output) as stream:
                for i in range(50):
                    stream.write_row(dict(time=i * .02, step_end_time=(i + 1) * .02))
                self.assertEqual(len(list(csv.DictReader(output.with_suffix('.csv').read_text().splitlines()))), 50)
                self.assertEqual(stream.row_count, 50)
                stream.write_row(dict(time=1., step_end_time=2.1))
                self.assertEqual(len(list(csv.DictReader(output.with_suffix('.csv').read_text().splitlines()))), 51)

    def run_fake(self, output, failure=None):
        from unittest.mock import patch
        from types import SimpleNamespace
        from rival_advanced import run
        from rival_robust import RobustConfig
        from rival_perception import SensorConfig
        class FakeEnv:
            params = dict(length=.58, width=.31, lf=.1651, lr=.1651)
            calls = 0
            closed = False
            def reset(self, poses):
                self.x = list(poses[:, 0])
                return self.observation(), 0., False, {}
            def observation(self):
                return dict(poses_x=self.x.copy(), poses_y=[0., 0.], poses_theta=[0., 0.],
                    linear_vels_x=[0., 0.], linear_vels_y=[0., 0.],
                    ang_vels_z=[0., 0.], collisions=[False, False])
            def step(self, actions):
                self.calls += 1
                if failure is not None and self.calls == 4:
                    raise failure('interrupted during second action')
                self.x = [x + .001 for x in self.x]
                return self.observation(), 0., False, {}
            def close(self):
                self.closed = True
        env = FakeEnv()
        controller = SimpleNamespace(step=lambda *a, **k: SimpleNamespace(
            steering=0., speed=.1, target_offset=0., mode='test', reason='test',
            feasible=True, min_separation=3., min_clearance=1., planning_ms=1., hypothesis_count=1))
        with patch('gym.make', return_value=env), \
             patch('rival_robust.TrackEnvelope', return_value=Envelope()), \
             patch('rival_robust.RobustDefender', return_value=controller):
            try:
                return run('no-attack', True, Settings(duration=.08), RobustConfig(), SensorConfig(),
                           Line(), Track(), output.parent / 'map', '.png', output), env
            finally:
                self.assertTrue(env.closed)

    def test_runner_exception_preserves_partial_cycle_csv_and_latest_checkpoint(self):
        import csv
        import json
        import tempfile
        from pathlib import Path
        for exception in (TimeoutError, RuntimeError):
            with self.subTest(exception=exception), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'interrupted'
                with self.assertRaises(exception):
                    self.run_fake(output, exception)
                rows = list(csv.DictReader(output.with_suffix('.csv').read_text().splitlines()))
                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(float(rows[0]['step_end_time']), .02)
                self.assertIn('attacker_rolling_progress_5s', rows[0])
                checkpoint = json.loads(output.with_suffix('.checkpoint.json').read_text())
                self.assertAlmostEqual(checkpoint['elapsed'], .03)
                self.assertEqual(checkpoint['csv_rows'], 1)
                self.assertEqual(checkpoint['interruption']['type'], exception.__name__)
                self.assertTrue(checkpoint['incomplete'])
                self.assertFalse(checkpoint['completed'])
                self.assertFalse(output.with_suffix('.json').exists())

    def test_normal_runner_stream_matches_final_summary(self):
        import csv
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'complete'
            result, env = self.run_fake(output)
            rows = list(csv.DictReader(output.with_suffix('.csv').read_text().splitlines()))
            self.assertEqual(len(rows), result['action_cycles'])
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(int(row['physics_step_count']) for row in rows), result['physics_steps'])
            self.assertAlmostEqual(float(rows[-1]['step_end_time']), result['elapsed'])
            self.assertFalse(result['incomplete'])
            from rival_robust import RobustConfig
            shared_margin = RobustConfig().lateral_tracking_margin
            self.assertEqual(result['adversary_prediction_budget']['tracking_margin'], shared_margin)
            self.assertTrue(all(float(row['attacker_tracking_margin']) == shared_margin for row in rows))
            self.assertIn('attacker_predicted_strict_body_legal', rows[0])
            self.assertIn('attacker_predicted_within_tracking_budget', rows[0])
            checkpoint = json.loads(output.with_suffix('.checkpoint.json').read_text())
            self.assertTrue(checkpoint['incomplete'])
            self.assertFalse(checkpoint['completed'])
            self.assertTrue(output.with_suffix('.json').exists())


if __name__ == '__main__':
    unittest.main()
