"""
IDM/MOBIL-inspired blocking defender for the R.I.V.A.L. F1TENTH gym.

Novel over rival_strategy.Defender:
  1. IDM-based longitudinal speed command (not fixed defender_speed).
  2. IDM blocking-value lane scoring (how much a candidate lane slows the attacker).
  3. Configurable observation-delay buffer (obs_delay > 0 simulates perception lag).

Reused from the repo (reported):
  desired_gap, rate_limit, predicted_separation_safe  → rival_strategy
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rival_strategy import desired_gap, rate_limit, predicted_separation_safe


class IDMBlockingDefender:
    """IDM longitudinal + MOBIL-inspired blocking lane selector.

    Parameters
    ----------
    line : ClosedRaceline
    rules : TrackRules
    cfg : Settings
    initial_target : float
        Starting raceline offset d (metres; positive = left of travel).
    obs_delay : float
        Seconds of observation delay applied to attacker state. Zero = perfect sensing.
    v_desired : float or None
        Defender desired free-road speed (m/s). Defaults to cfg.defender_speed.
    attacker_v_desired : float or None
        Attacker desired free-road speed used in blocking-value computation.
        Defaults to cfg.attacker_speed.
    """

    _IDM_POWER = 4.0  # IDM free-road exponent (Treiber 2000)

    def __init__(self, line, rules, cfg, initial_target,
                 obs_delay=0.0, v_desired=None, attacker_v_desired=None):
        self.line = line
        self.rules = rules
        self.cfg = cfg
        self.target = float(initial_target)
        self.obs_delay = float(obs_delay)
        self._v_desired = float(v_desired) if v_desired is not None else float(cfg.defender_speed)
        self._atk_v_desired = (float(attacker_v_desired) if attacker_v_desired is not None
                               else float(cfg.attacker_speed))
        self._v_target = self._v_desired  # start at desired speed; IDM maintains it
        self._obs_buf = deque()

    # ── Observation delay buffer ─────────────────────────────────────────────

    def _push(self, t, state):
        """Buffer attacker state; trim entries older than obs_delay + 1 s."""
        self._obs_buf.append((t, state))
        cutoff = t - max(self.obs_delay, 0.0) - 1.0
        while len(self._obs_buf) > 1 and self._obs_buf[0][0] < cutoff:
            self._obs_buf.popleft()

    def _delayed(self, t):
        """Return the buffered attacker state from obs_delay seconds ago."""
        if not self._obs_buf:
            return None
        target_t = t - self.obs_delay
        result = self._obs_buf[0][1]
        for entry_t, state in self._obs_buf:
            if entry_t <= target_t:
                result = state
            else:
                break
        return result

    # ── IDM-based longitudinal speed ─────────────────────────────────────────

    def _idm_speed(self):
        """Free-road IDM speed target: maintain v_desired via IDM dynamics.

        Novel: existing Defender commands a fixed cfg.defender_speed; this integrates
        IDM free-road acceleration against an internal speed state so the defender can
        modulate speed dynamically (e.g., slow for a blocking maneuver) while still
        commanding v_desired at steady state.

        self._v_target is initialised to v_desired in __init__, so the defender commands
        full speed from t=0 (identical to rival_baseline at startup) and IDM dynamics
        maintain it rather than ramp from zero.
        """
        c = self.cfg
        v0 = max(self._v_desired, 1e-6)
        accel = c.reference_acceleration * (1.0 - (self._v_target / v0) ** self._IDM_POWER)
        accel = float(np.clip(accel, -c.comfortable_braking, c.reference_acceleration))
        self._v_target = float(np.clip(self._v_target + accel * c.timestep, 0.0, v0 * 1.1))
        return self._v_target

    # ── IDM blocking value ───────────────────────────────────────────────────

    def _blocking_value(self, d_cand, atk_d, v_atk, bumper_gap, v_def):
        """Reduction in attacker IDM accel if defender occupies d_cand.

        Novel: existing Defender scores by proximity to attacker.d.
        This score quantifies the actual slowing effect on the attacker.

        Returns non-negative float; zero when laterally clear or attacker not behind.

        Reuses desired_gap() for the IDM s* term (reported reuse).
        """
        c = self.cfg
        if bumper_gap <= 0.0:
            return 0.0
        if abs(d_cand - atk_d) >= c.car_width + c.separation_margin:
            return 0.0
        v0 = max(self._atk_v_desired, 1e-6)
        free_accel = c.reference_acceleration * (1.0 - (v_atk / v0) ** self._IDM_POWER)
        # Reuse desired_gap() — identical formula to IDM s* (reported reuse)
        s_star = desired_gap(v_atk, v_atk - v_def, c)
        gap = max(bumper_gap, 0.01)
        blocked_accel = c.reference_acceleration * (
            1.0 - (v_atk / v0) ** self._IDM_POWER - (s_star / gap) ** 2)
        return max(0.0, free_accel - blocked_accel)

    # ── Main decision ────────────────────────────────────────────────────────

    def decide(self, t, defender_state, attacker_true, enabled=True):
        """Compute lateral target offset and speed command.

        Parameters
        ----------
        t : float
            Simulation time (s), used for the observation-delay buffer.
        defender_state : CarState
            Current defender Frenet+world state.
        attacker_true : CarState
            Current attacker state; buffered here and returned as a delayed
            snapshot when obs_delay > 0.
        enabled : bool
            If False, hold nominal raceline without active blocking.

        Returns
        -------
        target : float
            Desired lateral offset d (m).
        speed : float
            Desired longitudinal speed (m/s).
        reason : str
        detail : dict
        """
        self._push(t, attacker_true)
        delayed = self._delayed(t) or attacker_true

        c = self.cfg
        dt = c.timestep

        # signed_distance(a.s, b.s) > 0 means a is ahead of b on the raceline.
        gap = self.line.signed_distance(delayed.s, defender_state.s)   # > 0 = attacker ahead
        closing = delayed.speed - defender_state.speed                  # > 0 = attacker closing
        bumper_gap = -gap - c.car_length                                # > 0 = defender leading

        lo, hi = self.rules.interval(defender_state.s)
        detail = dict(
            gap=gap, closing_speed=closing, bumper_gap=bumper_gap,
            lateral_separation=delayed.d - defender_state.d,
            desired_gap=desired_gap(max(0.0, delayed.speed), closing, c),
            legal_lo=lo, legal_hi=hi, obs_delay=self.obs_delay,
            delayed_atk_d=delayed.d, delayed_atk_s=delayed.s,
            reacting=False, active=False,
        )

        if lo > hi:
            return self.target, 0.0, 'stop:no_legal_interval', detail

        future_lo, future_hi = self.rules.planning_interval(defender_state, c.defender_speed)
        if future_lo > future_hi:
            return self.target, 0.0, 'stop:no_preview_legal_interval', detail

        # Reaction trigger — same threshold logic as Defender.decide() in rival_strategy
        react = (enabled
                 and gap < 0
                 and closing > c.closing_threshold
                 and bumper_gap < c.reaction_threshold
                   * desired_gap(max(0.0, delayed.speed), closing, c))
        detail['reacting'] = bool(react)

        # IDM-based speed (novel: maintains v_desired via IDM dynamics vs fixed cfg.defender_speed)
        speed = self._idm_speed()

        # Build lateral target candidates (same set as Defender.decide)
        nominal = float(np.clip(0.0, future_lo, future_hi))
        candidates = [nominal, float(np.clip(self.target, future_lo, future_hi))]
        if react:
            candidates += list(np.linspace(future_lo, future_hi, 7))
            candidates.append(float(np.clip(delayed.d, future_lo, future_hi)))

        # Score and filter — safety checks reuse TrackRules + predicted_separation_safe
        safe = []
        for cand in sorted(set(candidates)):
            next_target = rate_limit(self.target, cand, c.offset_rate, dt)
            if not lo <= next_target <= hi:
                continue
            if not self.rules.path_safe(defender_state, cand, c.defender_speed,
                                        command_target=next_target):
                continue
            if not (predicted_separation_safe(defender_state, delayed, cand, self.line, c)
                    and predicted_separation_safe(defender_state, delayed, next_target,
                                                  self.line, c)):
                continue
            # IDM blocking-value scoring (novel over Defender.decide proximity scoring)
            if react:
                bv = self._blocking_value(cand, delayed.d, delayed.speed, bumper_gap, speed)
                score = bv - c.move_penalty * abs(cand - self.target)
            else:
                score = -abs(cand - nominal) - c.move_penalty * abs(cand - self.target)
            safe.append((score, cand))

        if not safe:
            return self.target, speed, 'stop:no_safe_reachable_target', detail

        best_score, best_cand = max(safe)
        held_entry = next(((s, c_) for s, c_ in safe if abs(c_ - self.target) < 1e-8), None)
        if held_entry is not None and best_score - held_entry[0] < c.usefulness_threshold:
            best_cand = self.target

        self.target = rate_limit(self.target, best_cand, c.offset_rate, dt)
        active = react and abs(best_cand - nominal) > 0.02
        detail['active'] = bool(active)
        reason = ('defend:idm_blocking_shift' if active
                  else ('hold:no_useful_shift' if react
                        else 'nominal:legal_raceline'))
        return self.target, speed, reason, detail
