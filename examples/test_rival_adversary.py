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
        self.assertTrue(result.requested_corridor_valid)
        self.assertFalse(result.sampled_path_feasible)
        self.assertTrue(result.guard_intervened)
        self.assertIn('map_or_legal_preview', result.reason)
        self.assertEqual(result.target, 0.)
        self.assertEqual(policy.corridor_valid_steps, 1)
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


class RunnerMetricsTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
