"""Perfect-state simulation adversary, not hardware perception.

Finite sampled previews do not prove full-maneuver feasibility. Requested
corridor validity, guard interventions and sampled path checks are separate.
"""
from dataclasses import dataclass
import math
import numpy as np
from rival_strategy import CarState, pursuit, rate_limit


def swept_overlap(start, end, center, separation):
    """Closed lateral interval intersects an inflated opponent corridor."""
    return min(start, end) <= center + separation and max(start, end) >= center - separation


@dataclass
class AttackCommand:
    target: float
    speed: float
    reason: str
    requested: bool
    requested_corridor_valid: bool
    guard_intervened: bool
    sampled_path_feasible: bool
    switched: bool
    desired: float
    fallback: str
    unresolved: bool
    selected_path_feasible: bool
    reference_speed_cap: float
    body_band_recovery: bool = False
    initial_body_error: float = 0.
    predicted_max_body_error: float = 0.
    predicted_final_body_error: float = 0.
    partial_body_recovery: bool = False
    passing_window: bool = False
    actual_body_error_delta: float = 0.
    actual_recovery_error_increased: bool = False
    tracking_margin: float = 0.
    predicted_within_tracking_budget: bool = False
    predicted_strict_body_legal: bool = False


@dataclass
class Preview:
    """Checked motion result; a feasible descent repair is NOT legal passing.

    Nonzero body-rule error can be within the explicit tracking budget; that
    is not strict rule compliance. Beyond-budget repairs must descend. The
    road/map footprint and pursuit-target constraints are never expanded.
    """
    feasible: bool
    reason: str
    safe_time: float
    target: float
    speed: float
    progress: float = 0.
    goal_clipped: bool = False
    body_band_recovery: bool = False
    initial_body_error: float = 0.
    max_body_error: float = 0.
    final_body_error: float = 0.
    partial_body_recovery: bool = False
    tracking_margin: float = 0.
    within_tracking_budget: bool = False
    strict_body_legal: bool = False


class Adversary:
    reevaluation_period = .5
    commitment_time = .8
    side_hysteresis = .15
    preview_horizon = .8
    legal_numeric_tolerance = 1e-8
    # Reporting threshold only: never a road expansion or recovery lockout.
    legal_recovery_tolerance = .02
    recovery_error_allowance = 1e-6
    passing_distance = 6.
    passing_headway = 2.

    def __init__(self, initial_target, tracking_margin=0.):
        if not np.isfinite(tracking_margin) or tracking_margin < 0:
            raise ValueError('tracking_margin must be finite and nonnegative')
        # Explicit body-rule tracking budget, shared with the defender by the
        # runner. This never expands footprint or pursuit-target constraints.
        self.tracking_margin = float(tracking_margin)
        self.target = initial_target
        self.speed = 0.
        self.side = None
        self.switched = False
        self._last_evaluation = self._last_switch = None
        self.requested_steps = self.corridor_valid_steps = 0
        self.guard_interventions = self.sampled_path_feasible_steps = 0
        self.fallback_steps = self.unresolved_steps = self.executed_attack_steps = 0
        self.body_recovery_steps = self.body_recovery_feasible_steps = 0
        self.executed_violation_steps = self.recovery_error_growth_steps = 0
        self._last_body_error = None
        self._last_body_recovery = False

    def _select_side(self, t, scenario, gap, defender, lo, hi, cfg):
        utilities = {1: hi - defender.d, -1: defender.d - lo}
        if self.side is None:
            self.side = 1 if utilities[1] >= utilities[-1] else -1
            if scenario in ('left-pass', 'right-pass'):
                self.side = 1 if scenario == 'left-pass' else -1
            self._last_switch = self._last_evaluation = t
            return
        if t - self._last_evaluation < self.reevaluation_period - 1e-9:
            return
        self._last_evaluation = t
        # Do not request a side switch alongside, even when the far side opens.
        if (abs(gap) <= cfg.car_length + cfg.separation_margin
                or t - self._last_switch < self.commitment_time - 1e-9):
            return
        candidate = self.side
        if scenario == 'adaptive':
            if utilities[-self.side] > utilities[self.side] + self.side_hysteresis:
                candidate = -self.side
        elif scenario == 'feint' and not self.switched and t >= cfg.pass_start + cfg.pass_ramp:
            candidate = -self.side
            self.switched = True
        if candidate != self.side:
            self.side = candidate
            self._last_switch = t

    @staticmethod
    def reference_speed_cap(s, envelope, cfg):
        slope = max(abs(envelope.reference_slope(s + ds))
                    for ds in (0., cfg.lookahead, 2 * cfg.lookahead))
        return .8 * cfg.offset_rate / slope if slope > 1e-9 else math.inf

    def preview(self, attacker, defender, line, envelope, track, cfg, goal, speed,
                *, previous_steering=0., measured_forward_speed=None, nominal=False,
                body_recovery=False):
        """Reference XY bicycle rollout using the actual pursuit/command slew.

        This is an approximate kinematic predictor, not Gym physics. Reconstruct
        body-forward speed from the line projection only when unavailable; the
        runner supplies measured body vx. No ideal Frenet offset motion is used.
        """
        dt = cfg.timestep
        state = attacker
        cosine = math.cos(attacker.yaw - line.heading(attacker.s))
        if measured_forward_speed is None:
            measured_forward_speed = attacker.speed / math.copysign(max(abs(cosine), .2), cosine)
        velocity = float(measured_forward_speed)
        target, command_speed, steering = self.target, self.speed, previous_steering
        # Pursuit goals are offsets at the lookahead station. Body checks
        # below remain at the actual projected station, not that future band.
        bias = 0. if nominal else goal - envelope.nominal(attacker.s + cfg.lookahead)
        raw_first_goal = envelope.nominal(attacker.s + cfg.lookahead) + bias
        first_lo, first_hi = envelope.interval(attacker.s + cfg.lookahead)
        first_goal = float(np.clip(raw_first_goal, first_lo, first_hi)) if first_lo <= first_hi else raw_first_goal
        goal_clipped = abs(first_goal - raw_first_goal) > self.legal_numeric_tolerance
        speed = min(speed, self.reference_speed_cap(attacker.s, envelope, cfg))
        first_target = rate_limit(self.target, first_goal, cfg.offset_rate, dt)
        first_speed = rate_limit(self.speed, speed,
            cfg.reference_acceleration if speed >= self.speed else cfg.comfortable_braking, dt)
        progress = 0.
        relative_s = line.signed_distance(attacker.s, defender.s)
        previous_d, previous_gap = state.d, relative_s
        initial_lo, initial_hi = envelope.interval(attacker.s)
        initial_error = max(initial_lo - attacker.d, attacker.d - initial_hi, 0.)
        recovery_active = (body_recovery and
                           initial_error > self.tracking_margin + self.legal_numeric_tolerance)
        max_error = current_error = initial_error

        def result(feasible, reason, tau):
            return Preview(feasible, reason, tau, first_target, first_speed, progress,
                goal_clipped, recovery_active, initial_error, max_error, current_error,
                feasible and recovery_active and current_error > self.legal_numeric_tolerance,
                self.tracking_margin, max_error <= self.tracking_margin + self.legal_numeric_tolerance,
                max_error <= self.legal_numeric_tolerance)

        for index in range(int(math.ceil(self.preview_horizon / dt)) + 1):
            tau = index * dt
            lo, hi = envelope.interval(state.s)
            current_error = max(lo - state.d, state.d - hi, 0.)
            max_error = max(max_error, current_error)
            allowed_error = (initial_error + self.recovery_error_allowance if recovery_active
                             else self.tracking_margin + self.legal_numeric_tolerance)
            if lo > hi or current_error > allowed_error:
                reason = ('body_band_recovery_error_increased' if recovery_active
                          else 'map_or_legal_preview:strict_body_band' if self.tracking_margin == 0.
                          else 'map_or_legal_preview:body_tracking_budget_exceeded')
                return result(False, reason, tau)
            # Footprint/map margins remain hard, including during repair.
            if not track.footprint_safe([state.x, state.y], state.yaw,
                    cfg.car_length, cfg.car_width, cfg.boundary_margin):
                return result(False, 'map_or_legal_preview:footprint', tau)
            gap = relative_s - defender.speed * tau
            if (swept_overlap(previous_d, state.d, defender.d, cfg.car_width + cfg.separation_margin)
                    and min(previous_gap, gap) <= cfg.car_length + cfg.separation_margin
                    and max(previous_gap, gap) >= -cfg.car_length - cfg.separation_margin):
                return result(False, 'sampled_collision_preview', tau)
            if tau >= self.preview_horizon - 1e-9:
                break
            previous_d, previous_gap = state.d, gap
            target_s = state.s + cfg.lookahead
            raw_desired = envelope.nominal(target_s) + bias
            target_lo, target_hi = envelope.interval(target_s)
            if target_lo > target_hi:
                return result(False, 'map_or_legal_preview:lookahead_target', tau)
            desired = float(np.clip(raw_desired, target_lo, target_hi))
            tolerance = self.legal_numeric_tolerance
            goal_clipped |= abs(desired - raw_desired) > tolerance
            # Clip the GOAL only. A lagging slew-limited target is not projected
            # into the band: reject the candidate if it cannot reach legally.
            target = rate_limit(target, desired, cfg.offset_rate, dt)
            if not target_lo - tolerance <= target <= target_hi + tolerance:
                return result(False, 'map_or_legal_preview:lookahead_target', tau)
            desired_speed = min(speed, self.reference_speed_cap(state.s, envelope, cfg))
            command_speed = rate_limit(command_speed, desired_speed,
                cfg.reference_acceleration if desired_speed >= command_speed else cfg.comfortable_braking, dt)
            steering, _, _ = pursuit(state, line, target, command_speed, steering, cfg)

            new_velocity = rate_limit(velocity, command_speed,
                cfg.reference_acceleration if command_speed >= velocity else cfg.comfortable_braking, dt)
            mean_velocity = .5 * (velocity + new_velocity)
            yaw_rate = mean_velocity * math.tan(steering) / cfg.wheelbase
            mid_yaw = state.yaw + .5 * yaw_rate * dt
            x = state.x + mean_velocity * math.cos(mid_yaw) * dt
            y = state.y + mean_velocity * math.sin(mid_yaw) * dt
            yaw = state.yaw + yaw_rate * dt
            s, d = line.project([x, y])
            distance = line.signed_distance(s, state.s)
            relative_s += distance
            progress += distance
            state = CarState(s, d, new_velocity * math.cos(yaw - line.heading(s)), x, y, yaw)
            velocity = new_velocity
        if recovery_active:
            required_descent = min(.005, .2 * initial_error)
            reentered = current_error <= self.legal_numeric_tolerance
            if not reentered and initial_error - current_error < required_descent:
                return result(False, 'body_band_recovery_insufficient_descent', self.preview_horizon)
            return result(True, 'body_band_recovered' if reentered else 'body_band_descent_partial',
                          self.preview_horizon)
        return result(True, 'sampled_xy_path', self.preview_horizon)

    def plan(self, t, scenario, attacker, defender, line, envelope, track, cfg,
             *, previous_steering=0., measured_forward_speed=None):
        target_s = attacker.s + cfg.lookahead
        lo, hi = envelope.interval(target_s)
        gap = line.signed_distance(attacker.s, defender.s)
        self._select_side(t, scenario, gap, defender, lo, hi, cfg)
        separation = cfg.car_width + cfg.separation_margin
        length = cfg.car_length + cfg.separation_margin
        # Use the wrapped relative station only for local passing opportunities,
        # not as a reason to chase an opponent's lateral edge half a lap away.
        behind_window = max(self.passing_distance, self.passing_headway * max(0., attacker.speed))
        passing_window = -behind_window <= gap <= length
        requested = (scenario not in ('no-attack', 'approach') and t >= cfg.pass_start
                     and passing_window)
        nominal = envelope.nominal(target_s)
        defender_bias = defender.d - envelope.nominal(defender.s)
        desired = nominal
        if requested:
            desired += defender_bias + self.side * max(separation, cfg.pass_offset)
        # After fully ahead, return to nominal rather than following the
        # opponent's lateral edge. Scripted surge/braking speeds remain below.
        corridor_valid = lo <= hi and lo - self.legal_numeric_tolerance <= desired <= hi + self.legal_numeric_tolerance
        clipped = float(np.clip(desired, lo, hi)) if lo <= hi else self.target
        target = rate_limit(self.target, clipped, cfg.offset_rate, cfg.timestep)
        speed = cfg.defender_speed if scenario == 'no-attack' else cfg.attacker_speed
        if scenario == 'surge':
            speed = cfg.defender_speed if t < cfg.pass_start else cfg.attacker_speed + .6
        if scenario == 'brake' and gap > length:
            speed = .5 * cfg.defender_speed
        reference_cap = self.reference_speed_cap(attacker.s, envelope, cfg)
        speed = min(speed, reference_cap)
        body_lo, body_hi = envelope.interval(attacker.s)
        initial_body_error = max(body_lo - attacker.d, attacker.d - body_hi, 0.)
        body_error_present = initial_body_error > self.legal_numeric_tolerance
        recovery_required = initial_body_error > self.tracking_margin + self.legal_numeric_tolerance
        actual_error_delta = (0. if self._last_body_error is None
                              else initial_body_error - self._last_body_error)
        actual_error_increased = self._last_body_recovery and actual_error_delta > self.recovery_error_allowance
        self.executed_violation_steps += int(body_error_present)
        self.recovery_error_growth_steps += int(actual_error_increased)
        reasons = []
        if recovery_required:
            reasons.append('body_band_recovery')
        if not corridor_valid:
            reasons.append('clipped_passing_corridor')
        preview_kwargs = dict(previous_steering=previous_steering,
                              measured_forward_speed=measured_forward_speed)
        requested_preview = self.preview(attacker, defender, line, envelope, track, cfg,
                                         desired, speed, **preview_kwargs)
        path_ok = requested_preview.feasible
        if requested and requested_preview.goal_clipped:
            corridor_valid = False
            reasons.append('clipped_future_passing_corridor')
        if not path_ok:
            reasons.append(requested_preview.reason)
        # Include the entire desired sweep: opposite-side endpoints alone can
        # both look clear while the interval between them crosses the defender.
        overlap = (swept_overlap(attacker.d, desired, defender.d, separation)
                   or swept_overlap(attacker.d, target, defender.d, separation))
        if overlap and abs(gap) <= length:
            reasons.append('alongside_cut_in_guard')
            speed = min(speed, max(0., defender.speed - .5))
        elif overlap and gap < 0:
            available = max(0., -gap - length)
            cap = max(0., defender.speed + math.sqrt(2 * cfg.comfortable_braking * available)
                      - cfg.comfortable_braking * cfg.timestep)
            if cfg.time_headway > 0:
                cap = min(cap, available / cfg.time_headway)
            if speed > cap:
                speed = cap
                reasons.append('rear_end_guard')
        guarded = bool(reasons)
        selected, fallback, unresolved = requested_preview, 'none', False
        if guarded:
            # Recovery is never attack credit. Check corrections and crawl
            # speeds before accepting standstill, without weakening any check.
            hold_goal = nominal + attacker.d - envelope.nominal(attacker.s)
            goals = [('hold', hold_goal, False), ('nominal', nominal, True)]
            goals += [(f'correction:{bias:+.1f}', nominal + bias, False)
                      for bias in (-.3, .3, -.6, .6)]
            # Biases may leave a future raw band; preview clips their goals
            # station-by-station and still checks the actual bounded target.
            candidates = []
            # Prefer crawl recovery before faster corrections: give the
            # bounded steering/target commands time to settle near tight bands.
            for fraction in (.2, .1, .35, .7, 0.):
                for name, goal, nominal_fallback in goals:
                    candidate = self.preview(attacker, defender, line, envelope, track, cfg,
                        goal, max(0., speed * fraction), nominal=nominal_fallback,
                        body_recovery=recovery_required, **preview_kwargs)
                    candidates.append((candidate, name + f':{fraction}'))
                    # Signed projected progress excludes backward motion. Keep
                    # safe nonprogress candidates but continue searching lazily.
                    if candidate.feasible and candidate.progress > 1e-6:
                        selected, fallback = candidate, name + f':{fraction}'
                        break
                if fallback != 'none':
                    break
            if fallback == 'none':
                stationary = [(candidate, name) for candidate, name in candidates if candidate.feasible]
                braking = [(candidate, name) for candidate, name in candidates if name.endswith(':0.0')]
                safe_braking = [(candidate, name) for candidate, name in braking if candidate.feasible]
                if stationary:
                    selected, fallback = (safe_braking or stationary)[0]
                else:
                    # No sampled-safe option: longest-preview bounded braking
                    # remains explicitly unresolved, not a feasible attack.
                    selected, fallback = max(braking, key=lambda entry: entry[0].safe_time)
                    fallback = 'unresolved_brake:' + fallback
                    unresolved = True
        self.target, self.speed = selected.target, selected.speed
        self.requested_steps += int(requested)
        self.corridor_valid_steps += int(requested and corridor_valid)
        self.guard_interventions += int(guarded)
        self.sampled_path_feasible_steps += int(requested and path_ok)
        self.executed_attack_steps += int(requested and not guarded and path_ok and corridor_valid)
        self.fallback_steps += int(guarded)
        self.unresolved_steps += int(unresolved)
        self.body_recovery_steps += int(recovery_required)
        self.body_recovery_feasible_steps += int(recovery_required and selected.feasible)
        reason = 'guard:' + ','.join(dict.fromkeys(reasons)) if guarded else ('attack' if requested else 'nominal_catch_up' if scenario != 'no-attack' else 'nominal')
        if initial_body_error > self.legal_numeric_tolerance:
            # Existing reason telemetry is streamed by the runner unchanged.
            reason += (f';body_error_m={initial_body_error:.9g};reporting_threshold_m={self.legal_recovery_tolerance:g}'
                       f';predicted_error_bound_m={(initial_body_error + self.recovery_error_allowance if recovery_required else self.tracking_margin + self.legal_numeric_tolerance):.9g}'
                       f';required_descent_m={(min(.005, .2 * initial_body_error) if recovery_required else 0.):.9g}'
                       f';predicted_max_error_m={selected.max_body_error:.9g}'
                       f';predicted_final_error_m={selected.final_body_error:.9g}'
                       f';body_repair_checked={recovery_required and selected.feasible}'
                       f';partial_descent={selected.partial_body_recovery}'
                       f';actual_error_delta_m={actual_error_delta:.9g}'
                       f';actual_recovery_error_increased={actual_error_increased}'
                       f';rule_violation_present=True')
        reason += (f';tracking_budget_m={self.tracking_margin:g}'
                   f';predicted_initial_error_m={selected.initial_body_error:.9g}'
                   f';predicted_max_body_error_m={selected.max_body_error:.9g}'
                   f';predicted_final_body_error_m={selected.final_body_error:.9g}'
                   f';predicted_within_budget={selected.within_tracking_budget}'
                   f';predicted_strict_body_legal={selected.strict_body_legal}')
        self._last_body_error = initial_body_error
        self._last_body_recovery = recovery_required and selected.feasible
        return AttackCommand(self.target, self.speed, reason, requested, corridor_valid,
                             guarded, path_ok, self.switched, desired, fallback,
                             unresolved, selected.feasible, reference_cap,
                             recovery_required, initial_body_error, selected.max_body_error,
                             selected.final_body_error, selected.partial_body_recovery, passing_window,
                             actual_error_delta, actual_error_increased, self.tracking_margin,
                             selected.within_tracking_budget, selected.strict_body_legal)
