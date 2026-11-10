"""Run with python -m unittest discover -s examples -p test_rival_strategy.py."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from rival_geometry import ClosedRaceline
from rival_strategy import (
    CarState, Defender, Settings, TrackRules, desired_gap, legal_interval,
    predicted_separation_safe, rate_limit,
)


class LegalIntervalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings(legal_min=-10., legal_max=10.,
                            center_margin=.1, inner_fraction=.8)

    def assertInterval(self, actual, expected):
        for value, wanted in zip(actual, expected):
            self.assertAlmostEqual(value, wanted)

    def test_left_turn_uses_asymmetric_track_center_not_raceline(self):
        # Track center is +1, not the raceline's zero offset.
        self.assertInterval(legal_interval((-1., 3.), .1, self.cfg), (1.1, 2.6))

    def test_right_turn_uses_asymmetric_track_center_not_raceline(self):
        self.assertInterval(legal_interval((-3., 1.), -.1, self.cfg), (-2.6, -1.1))

    def test_straight_and_threshold_boundaries_allow_full_corridor(self):
        for curvature in (0., self.cfg.curvature_threshold,
                          -self.cfg.curvature_threshold):
            with self.subTest(curvature=curvature):
                self.assertInterval(legal_interval((-1., 3.), curvature, self.cfg),
                                    (-1., 3.))

    def test_just_beyond_threshold_selects_inside_half(self):
        for curvature, expected in ((self.cfg.curvature_threshold + 1e-9, (.1, 1.6)),
                                    (-self.cfg.curvature_threshold - 1e-9, (-1.6, -.1))):
            with self.subTest(curvature=curvature):
                self.assertInterval(legal_interval((-2., 2.), curvature, self.cfg), expected)

    def test_legal_limits_clamp_left_right_and_straight(self):
        cfg = replace(self.cfg, legal_min=-.7, legal_max=.9)
        for curvature, expected in ((.1, (.1, .9)), (-.1, (-.7, -.1)),
                                    (0., (-.7, .9))):
            with self.subTest(curvature=curvature):
                self.assertInterval(legal_interval((-2., 2.), curvature, cfg), expected)

    def test_empty_intersection_is_not_made_artificially_legal(self):
        cfg = replace(self.cfg, legal_max=.5)
        lo, hi = legal_interval((-1., 3.), .1, cfg)
        self.assertGreater(lo, hi)

    def test_center_margin_can_make_narrow_inside_half_empty(self):
        lo, hi = legal_interval((-.05, .05), .1, self.cfg)
        self.assertGreater(lo, hi)

    def test_full_inner_fraction_reaches_boundary(self):
        cfg = replace(self.cfg, inner_fraction=1.)
        self.assertInterval(legal_interval((-2., 2.), .1, cfg), (.1, 2.))
        self.assertInterval(legal_interval((-2., 2.), -.1, cfg), (-2., -.1))


class RateLimitTests(unittest.TestCase):
    def test_bounded_steps_in_both_directions(self):
        for current, desired, expected in ((0., 1., .05), (0., -1., -.05),
                                           (1., -1., .95), (-1., 1., -.95)):
            with self.subTest(current=current, desired=desired):
                self.assertAlmostEqual(rate_limit(current, desired, .25, .2), expected)

    def test_nearby_target_is_reached_without_overshoot(self):
        for current, desired in ((0., .01), (0., -.01), (.3, .3), (.3, .29)):
            with self.subTest(current=current, desired=desired):
                self.assertAlmostEqual(rate_limit(current, desired, .25, .2), desired)

    def test_repeated_steps_converge_and_stay_at_target(self):
        for desired in (-.13, .13):
            current = 0.
            for _ in range(20):
                previous = current
                current = rate_limit(current, desired, .25, .2)
                self.assertLessEqual(abs(current - previous), .05 + 1e-12)
                self.assertLessEqual(min(previous, desired), current)
                self.assertGreaterEqual(max(previous, desired), current)
            self.assertAlmostEqual(current, desired)

    def test_zero_elapsed_time_or_rate_holds_current(self):
        self.assertEqual(rate_limit(.3, 1., .25, 0.), .3)
        self.assertEqual(rate_limit(.3, 1., 0., .2), .3)


class DesiredGapTests(unittest.TestCase):
    def test_gap_increases_with_positive_closing_speed(self):
        cfg = Settings()
        gaps = [desired_gap(3., closing, cfg) for closing in (0., .5, 1., 2.)]
        self.assertTrue(all(a < b for a, b in zip(gaps, gaps[1:])))
        self.assertAlmostEqual(gaps[0], cfg.minimum_gap + 3. * cfg.time_headway)
        self.assertAlmostEqual(gaps[2] - gaps[0], .75)

    def test_minimum_gap_is_floor_for_stopped_or_receding_follower(self):
        cfg = Settings()
        for speed, closing in ((0., 0.), (0., 5.), (3., -10.)):
            with self.subTest(speed=speed, closing=closing):
                self.assertEqual(desired_gap(speed, closing, cfg), cfg.minimum_gap)


class StrategyScenarioTests(unittest.TestCase):
    def setUp(self):
        # A long straight keeps these local Frenet scenarios away from corners.
        self.line = ClosedRaceline([[0., 0.], [100., 0.],
                                    [100., 100.], [0., 100.]])
        self.cfg = Settings()

    def state(self, s, d=0., speed=2.):
        x, y = self.line.sample(s, d)
        return CarState(s, d, speed, float(x), float(y), self.line.heading(s))

    def rules(self, interval=(-1., 1.)):
        rules = Mock(spec=['interval', 'planning_interval', 'path_safe'])
        rules.interval.return_value = interval
        rules.planning_interval.return_value = interval
        rules.path_safe.return_value = True
        return rules

    def test_longitudinally_separated_cars_are_safe(self):
        self.assertTrue(predicted_separation_safe(
            self.state(20.), self.state(17., speed=3.), 0., self.line, self.cfg))

    def test_overlapping_rectangles_are_unsafe(self):
        self.assertFalse(predicted_separation_safe(
            self.state(20.), self.state(19.8), 0., self.line, self.cfg))

    def test_lateral_clearance_allows_longitudinal_overlap(self):
        self.assertTrue(predicted_separation_safe(
            self.state(20.), self.state(20., d=.6), 0., self.line, self.cfg))

    def test_fast_closing_sweep_cannot_jump_over_collision(self):
        cfg = replace(self.cfg, prediction_horizon=1.)
        defender, attacker = self.state(20., speed=0.), self.state(18., speed=40.)
        # The first two sample gaps are -2 and +2: both endpoints are clear.
        limit = cfg.car_length + cfg.separation_margin
        self.assertGreater(2., limit)
        self.assertFalse(predicted_separation_safe(
            defender, attacker, 0., self.line, cfg))

    def test_lateral_candidate_sweep_is_unsafe_while_holding_is_safe(self):
        cfg = replace(self.cfg, offset_rate=1.)
        defender, attacker = self.state(20.), self.state(19., d=.6, speed=3.)
        self.assertTrue(predicted_separation_safe(
            defender, attacker, 0., self.line, cfg))
        self.assertFalse(predicted_separation_safe(
            defender, attacker, .6, self.line, cfg))

    def test_separation_prediction_wraps_at_start_finish(self):
        self.assertFalse(predicted_separation_safe(
            self.state(.1), self.state(self.line.length - .1), 0., self.line, self.cfg))

    def test_defender_rejects_collision_candidate_using_real_prediction(self):
        cfg = replace(self.cfg, offset_rate=1.)
        rules = self.rules()
        controller = Defender(self.line, rules, cfg, initial_target=0.)
        target, reason, safe, details = controller.decide(
            self.state(20.), self.state(19., d=.6, speed=3.))
        self.assertTrue(safe)
        self.assertTrue(details['reacting'])
        self.assertFalse(details['active'])
        self.assertEqual(target, 0.)
        self.assertEqual(reason, 'hold:no_useful_shift')
        self.assertIn(.6, [call.args[1] for call in rules.path_safe.call_args_list])

    def test_defender_rejects_track_unsafe_candidate(self):
        rules = self.rules()
        rules.path_safe.side_effect = (
            lambda state, target, speed, command_target=None: target <= 0.)
        controller = Defender(self.line, rules, self.cfg, initial_target=0.)
        target, reason, safe, details = controller.decide(
            self.state(20.), self.state(17., d=.6, speed=3.))
        self.assertTrue(safe)
        self.assertTrue(details['reacting'])
        self.assertFalse(details['active'])
        self.assertEqual(target, 0.)
        self.assertEqual(reason, 'hold:no_useful_shift')
        self.assertIn(.6, [call.args[1] for call in rules.path_safe.call_args_list])

    def test_useful_safe_defense_activates_with_rate_limited_motion(self):
        controller = Defender(self.line, self.rules(), self.cfg, initial_target=0.)
        target, reason, safe, details = controller.decide(
            self.state(20.), self.state(17., d=.6, speed=3.))
        self.assertTrue(safe)
        self.assertTrue(details['reacting'])
        self.assertTrue(details['active'])
        self.assertEqual(reason, 'defend:useful_safe_shift')
        self.assertAlmostEqual(target, self.cfg.offset_rate * self.cfg.timestep)
        self.assertEqual(controller.target, target)

    def test_insufficient_utility_holds_target(self):
        cfg = replace(self.cfg, usefulness_threshold=1.)
        controller = Defender(self.line, self.rules(), cfg, initial_target=0.)
        target, reason, safe, details = controller.decide(
            self.state(20.), self.state(17., d=.6, speed=3.))
        self.assertTrue(safe)
        self.assertTrue(details['reacting'])
        self.assertFalse(details['active'])
        self.assertEqual(target, 0.)
        self.assertEqual(reason, 'hold:no_useful_shift')

    def test_disabled_defense_returns_toward_nominal_without_reacting(self):
        controller = Defender(self.line, self.rules(), self.cfg, initial_target=.4)
        target, reason, safe, details = controller.decide(
            self.state(20., d=.4), self.state(17., d=.6, speed=3.), enabled=False)
        self.assertTrue(safe)
        self.assertFalse(details['reacting'])
        self.assertFalse(details['active'])
        self.assertEqual(reason, 'nominal:legal_raceline')
        self.assertAlmostEqual(target, .4 - self.cfg.offset_rate * self.cfg.timestep)

    def test_no_defense_without_approaching_threat(self):
        for s, speed in ((23., 3.), (17., 2.), (17., 1.), (10., 3.)):
            with self.subTest(attacker_s=s, attacker_speed=speed):
                controller = Defender(self.line, self.rules(), self.cfg, initial_target=0.)
                target, reason, safe, details = controller.decide(
                    self.state(20.), self.state(s, d=.6, speed=speed))
                self.assertTrue(safe)
                self.assertFalse(details['reacting'])
                self.assertFalse(details['active'])
                self.assertEqual(target, 0.)
                self.assertEqual(reason, 'nominal:legal_raceline')

    def test_disabled_defense_still_checks_safety(self):
        controller = Defender(self.line, self.rules(), self.cfg, initial_target=0.)
        target, reason, safe, details = controller.decide(
            self.state(20.), self.state(20.), enabled=False)
        self.assertFalse(safe)
        self.assertFalse(details['reacting'])
        self.assertEqual(target, 0.)
        self.assertEqual(reason, 'stop:no_safe_reachable_target')

    def test_next_command_is_checked_when_final_candidate_reverses_direction(self):
        defender = self.state(20.)
        rules = TrackRules(self.line, Mock(), self.cfg)
        rules.interval = Mock(return_value=(-1., 1.))
        # A negative final candidate is safe, but the lagging positive command
        # first moves the actual car toward an unsafe positive offset.
        rules.footprint = Mock(side_effect=lambda s, d: d <= .1)
        self.assertTrue(rules.path_safe(defender, 0., self.cfg.defender_speed))
        self.assertFalse(rules.path_safe(
            defender, 0., self.cfg.defender_speed, command_target=.495))
        rules.path_safe = Mock(wraps=rules.path_safe)
        controller = Defender(self.line, rules, self.cfg, initial_target=.5)
        target, reason, safe, _ = controller.decide(
            defender, self.state(17., d=.6, speed=3.), enabled=False)
        self.assertFalse(safe)
        self.assertEqual(target, .5)
        self.assertEqual(reason, 'stop:no_safe_reachable_target')
        reversal = next(call for call in rules.path_safe.call_args_list
                        if call.args[1] == 0.)
        self.assertAlmostEqual(reversal.kwargs['command_target'], .495)
        for call in rules.path_safe.call_args_list:
            self.assertAlmostEqual(call.kwargs['command_target'], rate_limit(
                .5, call.args[1], self.cfg.offset_rate, self.cfg.timestep))

    def test_next_command_separation_rejects_safe_final_reversal(self):
        cfg = replace(self.cfg, offset_rate=1.)
        defender, attacker = self.state(20.), self.state(19., d=.6, speed=3.)
        next_target = rate_limit(.6, 0., cfg.offset_rate, cfg.timestep)
        self.assertTrue(predicted_separation_safe(
            defender, attacker, 0., self.line, cfg))
        self.assertFalse(predicted_separation_safe(
            defender, attacker, next_target, self.line, cfg))
        rules = self.rules()
        controller = Defender(self.line, rules, cfg, initial_target=.6)
        target, reason, safe, _ = controller.decide(defender, attacker, enabled=False)
        self.assertFalse(safe)
        self.assertEqual(target, .6)
        self.assertEqual(reason, 'stop:no_safe_reachable_target')
        reversal = next(call for call in rules.path_safe.call_args_list
                        if call.args[1] == 0.)
        self.assertAlmostEqual(reversal.kwargs['command_target'], next_target)

    def test_defender_uses_preview_intersection_for_nominal_target(self):
        rules = self.rules()
        rules.planning_interval.return_value = (.2, .8)
        defender = self.state(20.)
        controller = Defender(self.line, rules, self.cfg, initial_target=.2)
        target, reason, safe, _ = controller.decide(
            defender, self.state(10.), enabled=False)
        self.assertTrue(safe)
        self.assertEqual(target, .2)
        self.assertEqual(reason, 'nominal:legal_raceline')
        rules.planning_interval.assert_called_once_with(defender, self.cfg.defender_speed)
        self.assertTrue(all(.2 <= call.args[1] <= .8
                            for call in rules.path_safe.call_args_list))

    def test_empty_preview_intersection_stops_without_evaluating_paths(self):
        rules = self.rules()
        rules.planning_interval.return_value = (.5, -.5)
        controller = Defender(self.line, rules, self.cfg, initial_target=0.)
        target, reason, safe, _ = controller.decide(self.state(20.), self.state(17.))
        self.assertFalse(safe)
        self.assertEqual(target, 0.)
        self.assertEqual(reason, 'stop:no_preview_legal_interval')
        rules.path_safe.assert_not_called()

    def test_empty_legal_interval_stops_without_evaluating_paths(self):
        rules = self.rules((1., -1.))
        controller = Defender(self.line, rules, self.cfg, initial_target=.2)
        target, reason, safe, _ = controller.decide(
            self.state(20.), self.state(17., speed=3.))
        self.assertFalse(safe)
        self.assertEqual(target, .2)
        self.assertEqual(reason, 'stop:no_legal_interval')
        rules.path_safe.assert_not_called()


class TrackRulesTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()
        self.line = Mock(spec=['length', 'curvature'])
        self.line.length = 10000.
        self.line.curvature.return_value = 0.
        self.track = Mock(spec=['corridor'])
        self.track.corridor.return_value = (-.7, .8)
        self.rules = TrackRules(self.line, self.track, self.cfg)
        self.state = CarState(20., 0., 2., 20., 0., 0.)

    def test_cache_eviction_returns_computed_interval_not_empty_sentinel(self):
        self.rules._cache = {float(i): (-.9, .9) for i in range(4000)}
        self.assertEqual(self.rules.interval(4000.), (-.7, .8))
        self.assertLessEqual(len(self.rules._cache), 4000)
        self.track.corridor.assert_called_once()
        # Eviction may clear or retain the new entry; neither may corrupt it.
        self.assertEqual(self.rules.interval(4000.), (-.7, .8))

    def test_planning_interval_intersects_every_preview_sample(self):
        self.rules.interval = Mock(side_effect=[
            (-1., 1.), (-.8, .9), (-.4, .8), (-.6, .5), (-.5, .7), (-.7, .6)])
        self.assertEqual(self.rules.planning_interval(self.state, 2.), (-.4, .5))
        self.assertEqual(self.rules.interval.call_count, 6)
        for call, expected_s in zip(self.rules.interval.call_args_list,
                                    (20., 20.4, 20.8, 21.2, 21.6, 22.)):
            self.assertAlmostEqual(call.args[0], expected_s)

    def test_planning_interval_preserves_empty_intersection(self):
        self.rules.interval = Mock(side_effect=[(-1., 0.)] * 5 + [(.2, 1.)])
        lo, hi = self.rules.planning_interval(self.state, 2.)
        self.assertAlmostEqual(lo, .2)
        self.assertAlmostEqual(hi, 0.)
        self.assertGreater(lo, hi)

    def test_planning_interval_does_not_predict_backward_for_negative_speed(self):
        self.rules.interval = Mock(return_value=(-.7, .8))
        self.assertEqual(self.rules.planning_interval(self.state, -2.), (-.7, .8))
        self.assertEqual(self.rules.interval.call_count, 6)
        for call in self.rules.interval.call_args_list:
            self.assertEqual(call.args, (self.state.s,))


if __name__ == '__main__':
    unittest.main()
