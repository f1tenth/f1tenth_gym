"""Bounded maneuver tests; no long Gym episodes or recorded-output dependencies.

Run: .venv-py315/bin/python -m unittest discover -s examples -p test_rival_passing.py -v

Controlled-yield setup for a parent-selected config (not a hidden default):
  settings: start_s: 22, initial_gap: 6, defender_speed: .5,
            attacker_speed: 2, minimum_gap: 2.5, offset_rate: .45
  robust: track_speed_planning: true, region: inside
Retain the remaining rival_advanced_fast.yaml limits. The test below establishes
sampled plan feasibility from rest, NOT an executed pass or Gym safety guarantee.
No new scenario/config is installed by these tests. Explicit full-track comparison:
  .venv-py315/bin/python examples/rival_advanced.py \
    --config examples/rival_advanced_fast.yaml --region track \
    --scenario left-pass --duration 20 --output examples/rival_results/track_pass_probe
That command explicitly relaxes the inside-half rule; it does not promise a pass.
"""
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from rival_adversary import Adversary
from rival_baseline import load
from rival_geometry import ClosedRaceline
from rival_robust import RobustConfig, TrackEnvelope
from rival_speed import SpeedProfile
from rival_strategy import CarState, Settings
from test_rival_adversary import Envelope, Line, Track, state


class ConstantProfile:
    def __init__(self, ceiling=2.):
        self.ceiling = ceiling

    def limit(self, s, measured_speed=0.):
        return self.ceiling


def policy(target=0., speed=2., **kwargs):
    result = Adversary(target, speed_profile=ConstantProfile(),
                       motion_config=RobustConfig(track_speed_planning=True), **kwargs)
    result.speed = speed
    return result


def configured():
    return replace(Settings(), attacker_speed=2., defender_speed=.5,
                   offset_rate=.45, minimum_gap=2.5)


class ManeuverTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configured()

    def test_opt_in_is_explicit_and_legacy_is_unchanged(self):
        self.assertFalse(Adversary(0.).maneuver_planning)
        self.assertFalse(Adversary(0., speed_profile=ConstantProfile(),
                                   motion_config=RobustConfig()).maneuver_planning)
        self.assertTrue(policy().maneuver_planning)
        self.assertFalse(policy(maneuver_planning=False).maneuver_planning)
        with self.assertRaisesRegex(ValueError, 'requires speed_profile'):
            Adversary(0., maneuver_planning=True)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            policy(maneuver_planning='yes')

    def check(self, p=None, side=1, envelope=None, attacker=None, defender=None, track=None):
        p = p or policy()
        p.side = side
        plan = p._check_maneuver(3., attacker or state(5., speed=2.),
            defender or state(11., speed=.5), Line(), envelope or Envelope(),
            track or Track(), self.cfg, 2., 0., 2.)
        return p, plan

    def test_wide_full_entry_pass_exit_both_sides(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                p, plan = self.check(side=side)
                self.assertTrue(plan.feasible, plan.reason)
                self.assertLess(0., plan.entry_time)
                self.assertLess(plan.entry_time, plan.pass_time)
                self.assertLess(plan.pass_time, plan.exit_time)
                self.assertGreater(plan.exit_time, p.preview_horizon)
                assert p._maneuver_preview is not None
                self.assertTrue(p._maneuver_preview.feasible)
                self.assertAlmostEqual(plan.goals[-1], 0.)
                self.assertGreater(side * plan.goals[0], self.cfg.car_width)
                self.assertEqual(p.executed_attack_steps, 0)  # planning is not execution

    def test_narrow_entry_rejects_before_long_rollout(self):
        for side in (-1, 1):
            p = policy()
            with patch.object(p, 'preview', side_effect=AssertionError('unnecessary XY work')):
                _, plan = self.check(p=p, side=side, envelope=Envelope(-.2, .2))
            self.assertFalse(plan.feasible)
            self.assertEqual(plan.reason, 'entry_target_corridor')
            self.assertEqual(plan.failure_time, 0.)
            self.assertLess(plan.minimum_room, 0.)

    def test_pass_phase_squeeze_beyond_short_preview_both_sides(self):
        class Squeeze(Envelope):
            def interval(self, s):
                return (-.2, .2) if 12. < s % 100 < 16. else (-2., 2.)

        for side in (-1, 1):
            _, plan = self.check(side=side, envelope=Squeeze())
            self.assertFalse(plan.feasible)
            self.assertEqual(plan.reason, 'pass_target_corridor')
            self.assertGreater(plan.failure_time, .8)

    def test_narrow_section_after_exit_does_not_require_full_lap_lane(self):
        class LaterSqueeze(Envelope):
            def interval(self, s):
                return (-.2, .2) if 35. < s % 100 < 45. else (-2., 2.)

        for side in (-1, 1):
            _, plan = self.check(side=side, envelope=LaterSqueeze())
            self.assertTrue(plan.feasible, plan.reason)
            self.assertLess(5. + 2 * plan.exit_time, 35.)

    def test_no_closing_is_honest_not_a_division_floor(self):
        p = policy()
        cfg = replace(self.cfg, attacker_speed=.5)
        plan = p._maneuver_room(state(5., speed=.5), state(11., speed=.5),
                                 1, Line(), Envelope(), cfg, .5)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, 'no_predicted_speed_advantage')
        self.assertEqual(len(plan.times), 1)

    def test_completion_budget_is_bounded_and_not_global_impossibility(self):
        p = policy(speed=2.)
        plan = p._maneuver_room(state(5., speed=2.), state(11., speed=1.99),
            1, Line(), Envelope(), replace(self.cfg, defender_speed=1.99), 2.)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, 'completion_beyond_planning_budget')
        self.assertLessEqual(plan.times[-1], 30.1)
        self.assertLessEqual(len(plan.times), 302)

    def test_brake_now_rejects_entry_even_if_cruise_gap_would_clear(self):
        p = policy()
        # A cruise-only calculation would keep the two cars apart while entry
        # completes. Brake-now closes their longitudinal envelopes much sooner.
        plan = p._maneuver_room(state(5., speed=2.), state(6.2, speed=2.),
            1, Line(), Envelope(), replace(self.cfg, attacker_speed=3.), 3.)
        self.assertFalse(plan.feasible)
        self.assertIn(plan.reason, ('brake_check_overlap', 'entry_overlap'))
        self.assertLess(plan.failure_time, 1.1)

    def test_front_return_reserves_new_follower_jerk_braking_and_headway(self):
        p = policy()
        normal = p._exit_clearance(2., 2., self.cfg)
        self.assertGreater(normal, self.cfg.minimum_gap + 2 * self.cfg.time_headway)
        self.assertGreater(p._exit_clearance(1., 2., self.cfg), normal)
        p.motion_config = RobustConfig(track_speed_planning=True, max_jerk=2.)
        self.assertGreater(p._exit_clearance(2., 2., self.cfg), normal)

    def test_precheck_defender_is_lazy_and_skips_xy_for_rejected_future(self):
        class Squeeze(Envelope):
            def interval(self, s):
                return (-.2, .2) if 7. < s else (-2., 2.)

        p = policy()
        with patch.object(p, '_reference_defender_samples', wraps=p._reference_defender_samples) as samples:
            plan = p._maneuver_room(state(5., speed=2.), state(11., speed=.5),
                                    1, Line(), Squeeze(), self.cfg, 2.)
        self.assertFalse(plan.feasible)
        self.assertLess(plan.failure_time, 1.)
        self.assertLess(len(plan.times), 12)
        self.assertFalse(samples.call_args.kwargs['poses'])
        self.assertTrue(np.isnan(plan.defender_paths[:, -1, :3]).all())

    def test_defender_recovers_measured_bias_to_own_reference(self):
        class Bend(Envelope):
            def nominal(self, s):
                return min(.6, max(0., (s - 12.) * .1))

        p = policy()
        times, paths = p._reference_defender_paths(state(11., .3, .5), Line(), Bend(),
                                                   self.cfg, 8., .1)
        np.testing.assert_array_equal(paths[:, 0, 4], .3)
        self.assertAlmostEqual(paths[0, -1, 4], Bend().nominal(paths[0, -1, 3]))
        self.assertLessEqual(np.max(np.abs(np.diff(paths[:, :, 4], axis=1))), .045 + 1e-9)
        # No lateral teleport after a brake-now hypothesis has stopped.
        stopped = np.flatnonzero(np.diff(paths[2, :, 5]) == 0.)
        self.assertTrue(len(stopped))
        np.testing.assert_allclose(np.diff(paths[2, :, 4])[stopped], 0.)
        self.assertEqual(times[-1], 8.)

    def test_full_map_and_jerk_limits_remain_conjunctive(self):
        class Wall:
            def footprint_safe(self, *args):
                return False

        p, plan = self.check(track=Wall())
        self.assertFalse(plan.feasible)
        self.assertIn('full_maneuver_map_or_legal_preview:footprint', plan.reason)
        self.assertEqual(p._acceleration, 0.)  # probes are pure
        self.assertEqual(p.target, 0.)
        self.assertEqual(p.speed, 2.)

    def test_overlap_holds_side_and_never_aborts_through_opponent(self):
        class SqueezeAhead(Envelope):
            def interval(self, s):
                return (-.2, .2) if s > 12. else (-.8, .8)

        for side in (-1, 1):
            p = policy(target=side * .75, speed=1.)
            p.side = p._maneuver_side = side
            p._last_evaluation = p._last_switch = 2.
            p.maneuver_phase = 'pass'
            a, b = state(10., side * .75, 1.), state(10., 0., 1.)
            with patch.object(p, 'preview', wraps=p.preview) as checks:
                result = p.plan(3., 'feint', a, b, Line(), SqueezeAhead(), Track(), self.cfg)
            self.assertEqual(p.side, side)
            self.assertTrue(result.guard_intervened)
            # No nominal fallback or far-side request while overlapping.
            self.assertFalse(result.fallback.startswith('nominal:'))
            for call in checks.call_args_list[1:]:
                self.assertFalse(call.kwargs.get('nominal', False))
                self.assertGreater(side * call.args[6], 0.)
            self.assertGreater(side * result.target, 0.)
            self.assertEqual(p.executed_attack_steps, 0)

    def test_entry_then_pass_is_committed_not_a_feint_or_completed_pass(self):
        p = policy()
        entry = p.plan(3., 'left-pass', state(5., speed=2.), state(11., speed=.5),
                       Line(), Envelope(), Track(), self.cfg)
        self.assertFalse(entry.guard_intervened, entry.reason)
        self.assertEqual(p.maneuver_phase, 'entry')
        self.assertGreater(p.executed_attack_steps, 0)
        p.target = .75
        alongside = p.plan(4., 'feint', state(10.5, .75, 2.), state(11., speed=.5),
                           Line(), Envelope(), Track(), self.cfg)
        self.assertFalse(alongside.guard_intervened, alongside.reason)
        self.assertEqual(p.maneuver_phase, 'pass')
        self.assertEqual(p.side, 1)
        self.assertFalse(p.switched)
        self.assertIn('opportunity_is_not_completed_pass=True', alongside.reason)
        self.assertGreaterEqual(p.attacker_plan_ms, 0.)

    def test_failed_commitment_aborts_before_overlap(self):
        p = policy(target=.5)
        p.side = p._maneuver_side = 1
        p._last_evaluation = p._last_switch = 2.
        p.maneuver_phase = 'entry'
        result = p.plan(3., 'left-pass', state(5., .1, 1.), state(11., speed=.5),
                         Line(), Envelope(-.2, .2), Track(), self.cfg)
        self.assertIn('abort_before_overlap', result.reason)
        self.assertTrue(result.guard_intervened)
        self.assertEqual(p.executed_attack_steps, 0)

    def test_fresh_short_guard_is_not_bypassed_by_cached_full_plan(self):
        p, plan = self.check()
        p._last_evaluation = p._last_switch = 3.
        p.maneuver_phase = 'entry'
        class Wall:
            def footprint_safe(self, *args):
                return False
        result = p.plan(3.02, 'left-pass', state(5., speed=2.), state(11., speed=.5),
                         Line(), Envelope(), Wall(), self.cfg)
        self.assertTrue(plan.feasible)
        self.assertTrue(result.unresolved)
        self.assertFalse(result.selected_path_feasible)
        self.assertEqual(p.executed_attack_steps, 0)

    def test_circular_opportunity_runs_are_joined_across_seam(self):
        class Seam(Envelope):
            def interval(self, s):
                return (-2., 2.) if s % 100 < 5. or s % 100 >= 95. else (-.2, .2)
        result = Adversary.passing_opportunities(Line(), Seam(), self.cfg)
        for side in (-1, 1):
            self.assertAlmostEqual(result['sides'][side]['longest_configured_lane_m'], 10.)

    def test_real_closed_line_maneuver_crosses_wrap_in_both_passing_directions(self):
        line = ClosedRaceline([[0., 0.], [60., 0.], [60., 30.], [-60., 30.], [-60., 0.]])
        # Station zero lies WITHIN a collinear straight, not at a corner.
        def car(s, speed):
            return CarState(s % line.length, 0., speed, *line.sample(s, 0), line.heading(s))
        for side in (-1, 1):
            p = policy()
            p.side = side
            plan = p._check_maneuver(3., car(line.length - 8., 2.),
                car(line.length - 2., .5), line, Envelope(), Track(), self.cfg, 2., 0., 2.)
            self.assertTrue(plan.feasible, plan.reason)
            self.assertLess(plan.defender_paths[0, -1, 3], 10.)
            travel = plan.defender_paths[0, :, 5]
            self.assertTrue(np.all(np.diff(travel) >= 0.))


class ActualTrackPassingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('rival_advanced_fast.yaml')
        cls.cfg, cls.line, cls.track, *_ = load(path)
        cls.rc = RobustConfig.from_dict(yaml.safe_load(path.read_text())['robust'])
        cls.inside = TrackEnvelope(cls.line, cls.track, cls.cfg, cls.rc)
        cls.full_track = TrackEnvelope(cls.line, cls.track, cls.cfg, replace(cls.rc, region='track'))

    def car(self, s, speed, envelope):
        d = envelope.nominal(s)
        return CarState(s, d, speed, *self.line.sample(s, d), self.line.heading(s))

    def test_actual_corridor_dimensions_not_an_assumed_two_car_track(self):
        e = self.inside
        raw = e.raw_hi - e.raw_lo
        self.assertGreater(float(np.min(raw)), 2.4)
        self.assertLess(float(np.max(raw)), 3.3)
        diagnostics = Adversary.passing_opportunities(self.line, e, self.cfg)
        self.assertAlmostEqual(diagnostics['track_length_m'], 156.35611902054006)
        self.assertAlmostEqual(diagnostics['required_separation_m'], .43)
        self.assertLess(diagnostics['minimum_center_band_m'], .1)
        self.assertGreater(diagnostics['maximum_center_band_m'], 2.)
        self.assertLess(diagnostics['sides'][1]['longest_configured_lane_m'], 20.)
        self.assertLess(diagnostics['sides'][-1]['longest_configured_lane_m'], 9.)
        full = Adversary.passing_opportunities(self.line, self.full_track, self.cfg)
        self.assertGreater(full['minimum_center_band_m'], 1.3)
        self.assertEqual(e.config.region, 'inside')  # no silent region mutation

    def test_fast_cruise_grid_defers_with_stage_reason_not_global_claim(self):
        cfg, e = self.cfg, self.inside
        p = Adversary(0., speed_profile=SpeedProfile(self.line, e, cfg, self.rc, cfg.attacker_speed),
                      motion_config=self.rc)
        p.speed = cfg.attacker_speed
        reasons = set()
        # A finite grid is a measured negative result, NOT impossibility proof.
        for s in np.arange(0., self.line.length, 4.):
            a = self.car(s, cfg.attacker_speed, e)
            b = self.car((s + cfg.initial_gap) % self.line.length, cfg.defender_speed, e)
            p.target = e.nominal(s + cfg.lookahead)
            for side in (-1, 1):
                plan = p._maneuver_room(a, b, side, self.line, e, cfg, cfg.attacker_speed)
                self.assertFalse(plan.feasible, (s, side))
                reasons.add(plan.reason)
        self.assertIn('entry_target_corridor', reasons)
        self.assertTrue(any(reason.startswith('pass_') for reason in reasons))
        self.assertEqual(p.executed_attack_steps, 0)

    def test_parent_selected_controlled_yield_has_real_full_maneuver_from_rest(self):
        cfg = replace(self.cfg, start_s=22., initial_gap=6., defender_speed=.5, attacker_speed=2.)
        e = self.inside
        p = Adversary(e.nominal(16. + cfg.lookahead),
                      speed_profile=SpeedProfile(self.line, e, cfg, self.rc, 2.), motion_config=self.rc)
        p.side = 1
        plan = p._check_maneuver(2., self.car(16., 0., e), self.car(22., 0., e),
            self.line, e, self.track, cfg, 2., 0., 0.)
        self.assertTrue(plan.feasible, plan.reason)
        assert p._maneuver_preview is not None
        self.assertTrue(p._maneuver_preview.feasible)
        self.assertLess(plan.exit_time, 20.)
        self.assertEqual(p.executed_attack_steps, 0)
        self.assertEqual(e.config.region, 'inside')


if __name__ == '__main__':
    unittest.main()
