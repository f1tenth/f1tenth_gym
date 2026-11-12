"""
Unit tests for IDMBlockingDefender.

Run from repo root:
  python -m unittest discover -s examples -p 'test_rival_*.py' -v

Or directly:
  python examples/heuristic_baseline/test_rival_heuristic_baseline.py
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

_EXAMPLES = Path(__file__).parent.parent
_HERE = Path(__file__).parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rival_geometry import ClosedRaceline
from rival_strategy import CarState, Settings, rate_limit

from heuristic_defender import IDMBlockingDefender


def _square_line():
    """100 m-side square raceline; long straights keep tests away from corners."""
    return ClosedRaceline([[0., 0.], [100., 0.], [100., 100.], [0., 100.]])


def _mock_rules(interval=(-1., 1.)):
    r = Mock(spec=['interval', 'planning_interval', 'path_safe', 'footprint'])
    r.interval.return_value = interval
    r.planning_interval.return_value = interval
    r.path_safe.return_value = True
    r.footprint.return_value = True
    return r


def _state(line, s, d=0., speed=2.):
    x, y = line.sample(s, d)
    return CarState(s, d, speed, float(x), float(y), line.heading(s))


class TestIDMSpeed(unittest.TestCase):
    def setUp(self):
        self.line = _square_line()
        self.cfg = Settings()
        self.defender = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0.)

    def test_speed_starts_at_desired(self):
        # _v_target initialises to v_desired, so first call returns near v_desired
        result = self.defender._idm_speed()
        self.assertAlmostEqual(result, self.cfg.defender_speed, places=3)

    def test_speed_above_desired_decreases_toward_desired(self):
        # Force _v_target above v_desired; IDM should bring it back down
        self.defender._v_target = self.cfg.defender_speed * 1.1
        result = self.defender._idm_speed()
        self.assertLess(result, self.cfg.defender_speed * 1.1)

    def test_speed_never_negative(self):
        # Even with _v_target forced to zero, result is non-negative
        self.defender._v_target = 0.0
        self.assertGreaterEqual(self.defender._idm_speed(), 0.0)

    def test_speed_at_desired_stays_near_desired(self):
        # At steady state (v_target == v_desired), IDM accel == 0 → no change
        v0 = self.cfg.defender_speed
        self.defender._v_target = v0
        result = self.defender._idm_speed()
        self.assertAlmostEqual(result, v0, places=5)

    def test_custom_v_desired_starts_at_that_speed(self):
        cfg = replace(self.cfg, defender_speed=3.0)
        d = IDMBlockingDefender(self.line, _mock_rules(), cfg, 0., v_desired=3.0)
        self.assertAlmostEqual(d._idm_speed(), 3.0, places=3)


class TestIDMBlockingValue(unittest.TestCase):
    def setUp(self):
        self.line = _square_line()
        self.cfg = Settings()
        self.defender = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0.)

    def test_zero_when_laterally_clear(self):
        # Attacker at d=0, defender candidate at d=1 (> car_width + separation)
        bv = self.defender._blocking_value(1.0, 0.0, 2.8, 2.0, 2.0)
        self.assertEqual(bv, 0.0)

    def test_zero_when_attacker_not_behind(self):
        # bumper_gap <= 0 means attacker is not behind (or alongside)
        bv = self.defender._blocking_value(0.0, 0.0, 2.8, -1.0, 2.0)
        self.assertEqual(bv, 0.0)

    def test_positive_when_laterally_overlapping_and_attacker_behind(self):
        # Defender candidate at same d as attacker, attacker clearly behind
        c = self.cfg
        bv = self.defender._blocking_value(0.0, 0.0, 2.8, 5.0, 2.0)
        self.assertGreater(bv, 0.0)

    def test_increases_with_closer_bumper_gap(self):
        # Smaller gap → larger s*/g ratio → more blocking effect
        bv_far = self.defender._blocking_value(0.0, 0.0, 2.8, 10.0, 2.0)
        bv_close = self.defender._blocking_value(0.0, 0.0, 2.8, 2.0, 2.0)
        self.assertGreater(bv_close, bv_far)

    def test_blocked_value_smaller_when_attacker_slower(self):
        # Faster attacker (v > v_desired) has negative free-road accel; leader makes it
        # more negative → higher blocking value. Slower attacker → smaller blocking value.
        bv_fast = self.defender._blocking_value(0.0, 0.0, 3.5, 5.0, 2.0)  # above v_desired
        bv_slow = self.defender._blocking_value(0.0, 0.0, 0.5, 5.0, 2.0)  # below v_desired
        # Both can be positive; fast attacker is decelerated more by the leader
        self.assertGreaterEqual(bv_fast, bv_slow)

    def test_lateral_threshold_is_car_width_plus_separation(self):
        c = self.cfg
        threshold = c.car_width + c.separation_margin
        # Just inside threshold: should block
        bv_in = self.defender._blocking_value(0.0, threshold - 0.01, 2.8, 5.0, 2.0)
        # Just outside threshold: should not block
        bv_out = self.defender._blocking_value(0.0, threshold + 0.01, 2.8, 5.0, 2.0)
        self.assertGreater(bv_in, 0.0)
        self.assertEqual(bv_out, 0.0)


class TestObservationDelay(unittest.TestCase):
    def setUp(self):
        self.line = _square_line()
        self.cfg = Settings(timestep=0.02)

    def test_zero_delay_returns_latest_state(self):
        d = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0., obs_delay=0.0)
        s0 = _state(self.line, 10., d=0.1)
        s1 = _state(self.line, 11., d=0.2)
        d._push(0.0, s0)
        d._push(0.02, s1)
        result = d._delayed(0.02)
        self.assertAlmostEqual(result.s, s1.s)

    def test_positive_delay_returns_older_state(self):
        d = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0., obs_delay=0.1)
        # Use integer millisecond timestamps (multiples of 0.1) to avoid fp boundary
        # issues with i*0.02 accumulation.
        states = [_state(self.line, 10. + i * 0.5, d=i * 0.01) for i in range(10)]
        for i, s in enumerate(states):
            d._push(i * 0.1, s)
        # At t=0.75, delayed t = 0.75 - 0.10 = 0.65 → entry at index 6 (t=0.60)
        result = d._delayed(0.75)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.s, states[6].s, places=5)

    def test_empty_buffer_returns_none(self):
        d = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0., obs_delay=0.1)
        self.assertIsNone(d._delayed(1.0))

    def test_buffer_does_not_grow_unbounded(self):
        d = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0., obs_delay=0.0)
        for i in range(500):
            d._push(float(i) * 0.02, _state(self.line, float(i)))
        # With delay=0, entries older than 1 s are trimmed.
        # At t=499*0.02=9.98 s, cutoff = 9.98 - 0 - 1 = 8.98 s → ~50 entries remain
        self.assertLess(len(d._obs_buf), 100)


class TestDefenderDecide(unittest.TestCase):
    def setUp(self):
        self.line = _square_line()
        self.cfg = Settings()

    def _make(self, rules=None, obs_delay=0.0, initial_target=0.0):
        return IDMBlockingDefender(
            self.line, rules or _mock_rules(), self.cfg,
            initial_target, obs_delay=obs_delay)

    def test_no_reaction_when_attacker_ahead(self):
        d = self._make()
        # Attacker is ahead: gap > 0, no reaction expected
        target, speed, reason, detail = d.decide(
            0.0, _state(self.line, 20.), _state(self.line, 23., speed=3.))
        self.assertFalse(detail['reacting'])
        self.assertFalse(detail['active'])
        self.assertEqual(reason, 'nominal:legal_raceline')

    def test_no_reaction_when_attacker_behind_but_slow(self):
        d = self._make()
        # Attacker behind, but closing speed < closing_threshold
        target, speed, reason, detail = d.decide(
            0.0, _state(self.line, 20., speed=2.),
            _state(self.line, 17., speed=2.0))
        self.assertFalse(detail['reacting'])
        self.assertEqual(reason, 'nominal:legal_raceline')

    def test_reaction_trigger_with_fast_approaching_attacker(self):
        d = self._make()
        # Attacker behind and closing fast
        target, speed, reason, detail = d.decide(
            0.0, _state(self.line, 20., speed=2.),
            _state(self.line, 17., d=0.5, speed=3.))
        self.assertTrue(detail['reacting'])

    def test_active_blocking_shifts_toward_attacker_lateral_position(self):
        d = self._make()
        target, speed, reason, detail = d.decide(
            0.0, _state(self.line, 20., speed=2.),
            _state(self.line, 17., d=0.6, speed=3.))
        self.assertTrue(detail.get('reacting', False))
        # Target should shift toward attacker; defender starts at 0, attacker at 0.6
        self.assertGreaterEqual(target, 0.0)

    def test_disabled_defense_returns_toward_nominal(self):
        d = self._make(initial_target=0.4)
        target, speed, reason, detail = d.decide(
            0.0, _state(self.line, 20., d=0.4),
            _state(self.line, 17., d=0.6, speed=3.), enabled=False)
        self.assertFalse(detail['reacting'])
        self.assertFalse(detail['active'])
        self.assertEqual(reason, 'nominal:legal_raceline')
        # Target should drift toward nominal (0.), not toward attacker
        self.assertLessEqual(target, 0.4)

    def test_speed_command_is_positive_and_bounded(self):
        d = self._make()
        for _ in range(3):  # run a few steps so _v_target is exercised
            _, speed, _, _ = d.decide(
                0.0, _state(self.line, 20., speed=2.0),
                _state(self.line, 17., speed=2.5))
            self.assertGreaterEqual(speed, 0.0)
            self.assertLessEqual(speed, self.cfg.defender_speed * 1.2)

    def test_no_legal_interval_returns_stop(self):
        d = self._make(rules=_mock_rules(interval=(1., -1.)))
        target, speed, reason, detail = d.decide(
            0.0, _state(self.line, 20.), _state(self.line, 17., speed=3.))
        self.assertEqual(reason, 'stop:no_legal_interval')
        self.assertEqual(speed, 0.0)

    def test_target_is_rate_limited(self):
        d = self._make(initial_target=0.0)
        # Trigger a reactive shift toward 1.0 (far from 0.0)
        target, _, _, _ = d.decide(
            0.0, _state(self.line, 20., speed=2.),
            _state(self.line, 17., d=1.0, speed=3.5))
        max_step = self.cfg.offset_rate * self.cfg.timestep
        self.assertLessEqual(abs(target - 0.0), max_step + 1e-9)

    def test_obs_delay_zero_uses_current_state(self):
        d = self._make(obs_delay=0.0)
        atk = _state(self.line, 17., d=0.6, speed=3.)
        _, _, _, detail = d.decide(0.0, _state(self.line, 20., speed=2.), atk)
        self.assertAlmostEqual(detail['delayed_atk_d'], 0.6)

    def test_obs_delay_positive_lags_attacker_lateral_position(self):
        d = self._make(obs_delay=0.1)
        # Push an old state at d=0.0, then current state at d=0.8
        old = _state(self.line, 17., d=0.0, speed=3.)
        new = _state(self.line, 17., d=0.8, speed=3.)
        d._push(0.0, old)
        d._push(0.1, new)
        _, _, _, detail = d.decide(0.1, _state(self.line, 20., speed=2.), new)
        # At t=0.1 with delay=0.1, the delayed state is from t=0.0 → d=0.0
        self.assertAlmostEqual(detail['delayed_atk_d'], 0.0, places=5)

    def test_decision_does_not_violate_legal_interval(self):
        d = self._make()
        for _ in range(50):
            target, _, reason, _ = d.decide(
                0.0, _state(self.line, 20., speed=2.),
                _state(self.line, 17., d=0.5, speed=3.))
        lo, hi = -1., 1.
        # target after rate-limiting must be within legal interval
        self.assertGreaterEqual(target, lo - 1e-9)
        self.assertLessEqual(target, hi + 1e-9)

    def test_detail_dict_contains_required_keys(self):
        d = self._make()
        _, _, _, detail = d.decide(
            0.0, _state(self.line, 20.), _state(self.line, 17., speed=3.))
        for key in ('gap', 'closing_speed', 'lateral_separation', 'desired_gap',
                    'legal_lo', 'legal_hi', 'reacting', 'active'):
            self.assertIn(key, detail, msg=f'missing key: {key}')


class TestDefenderWithRealLine(unittest.TestCase):
    """Integration-style checks with a real ClosedRaceline and TrackRules mock."""

    def setUp(self):
        self.line = _square_line()
        self.cfg = Settings()

    def test_target_enters_blocking_zone_over_many_steps(self):
        """Target should move into the lateral blocking zone (within car_width +
        separation_margin of the attacker), then stabilise there.  The usefulness
        threshold keeps it from wandering further once inside the zone."""
        d = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0.)
        cfg = self.cfg
        atk_d = 0.6
        for i in range(200):
            target, _, _, _ = d.decide(
                i * cfg.timestep,
                _state(self.line, 20., speed=2.),
                _state(self.line, 17., d=atk_d, speed=3.))
        threshold = cfg.car_width + cfg.separation_margin
        self.assertGreater(target, 0.0)
        self.assertLess(abs(target - atk_d), threshold + 1e-9,
                        msg=f'target {target:.4f} not within blocking threshold {threshold:.3f} of atk_d {atk_d}')

    def test_defender_does_not_react_to_attacker_far_behind(self):
        d = IDMBlockingDefender(self.line, _mock_rules(), self.cfg, 0.)
        # Attacker 20 m behind — bumper_gap >> reaction threshold * desired_gap
        _, _, reason, detail = d.decide(
            0.0,
            _state(self.line, 20., speed=2.),
            _state(self.line, 0., d=0.5, speed=3.))
        self.assertFalse(detail['reacting'])
        self.assertEqual(reason, 'nominal:legal_raceline')


if __name__ == '__main__':
    unittest.main()
