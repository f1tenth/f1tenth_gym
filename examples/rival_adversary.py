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


@dataclass
class Preview:
    feasible: bool
    reason: str
    safe_time: float
    target: float
    speed: float


class Adversary:
    reevaluation_period = .5
    commitment_time = .8
    side_hysteresis = .15
    preview_horizon = .8
    legal_numeric_tolerance = 1e-8

    def __init__(self, initial_target):
        self.target = initial_target
        self.speed = 0.
        self.side = None
        self.switched = False
        self._last_evaluation = self._last_switch = None
        self.requested_steps = self.corridor_valid_steps = 0
        self.guard_interventions = self.sampled_path_feasible_steps = 0
        self.fallback_steps = self.unresolved_steps = self.executed_attack_steps = 0

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

    def preview(self, attacker, defender, line, envelope, track, cfg, goal, speed,
                *, previous_steering=0., measured_forward_speed=None, nominal=False):
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
        first_goal = envelope.nominal(attacker.s) if nominal else goal
        first_target = rate_limit(self.target, first_goal, cfg.offset_rate, dt)
        first_speed = rate_limit(self.speed, speed,
            cfg.reference_acceleration if speed >= self.speed else cfg.comfortable_braking, dt)
        relative_s = line.signed_distance(attacker.s, defender.s)
        previous_d, previous_gap = state.d, relative_s
        for index in range(int(math.ceil(self.preview_horizon / dt)) + 1):
            tau = index * dt
            lo, hi = envelope.interval(state.s)
            if (lo > hi or not lo - self.legal_numeric_tolerance <= state.d <= hi + self.legal_numeric_tolerance) or not track.footprint_safe(
                    [state.x, state.y], state.yaw, cfg.car_length, cfg.car_width, cfg.boundary_margin):
                return Preview(False, 'map_or_legal_preview', tau,
                               self.target if first_target is None else first_target,
                               self.speed if first_speed is None else first_speed)
            gap = relative_s - defender.speed * tau
            if (swept_overlap(previous_d, state.d, defender.d, cfg.car_width + cfg.separation_margin)
                    and min(previous_gap, gap) <= cfg.car_length + cfg.separation_margin
                    and max(previous_gap, gap) >= -cfg.car_length - cfg.separation_margin):
                return Preview(False, 'sampled_collision_preview', tau,
                               self.target if first_target is None else first_target,
                               self.speed if first_speed is None else first_speed)
            if tau >= self.preview_horizon - 1e-9:
                break
            previous_d, previous_gap = state.d, gap
            desired = envelope.nominal(state.s) if nominal else goal
            target = rate_limit(target, desired, cfg.offset_rate, dt)
            command_speed = rate_limit(command_speed, speed,
                cfg.reference_acceleration if speed >= command_speed else cfg.comfortable_braking, dt)
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
            relative_s += line.signed_distance(s, state.s)
            state = CarState(s, d, new_velocity * math.cos(yaw - line.heading(s)), x, y, yaw)
            velocity = new_velocity
        return Preview(True, 'sampled_xy_path', self.preview_horizon, first_target, first_speed)

    def plan(self, t, scenario, attacker, defender, line, envelope, track, cfg,
             *, previous_steering=0., measured_forward_speed=None):
        lo, hi = envelope.interval(attacker.s)
        gap = line.signed_distance(attacker.s, defender.s)
        self._select_side(t, scenario, gap, defender, lo, hi, cfg)
        requested = scenario not in ('no-attack', 'approach') and t >= cfg.pass_start
        separation = cfg.car_width + cfg.separation_margin
        length = cfg.car_length + cfg.separation_margin
        desired = envelope.nominal(attacker.s)
        if requested:
            desired = defender.d + self.side * max(separation, cfg.pass_offset)
        if scenario in ('cut-in', 'brake') and gap > length:
            desired = defender.d
        corridor_valid = lo <= hi and lo - self.legal_numeric_tolerance <= desired <= hi + self.legal_numeric_tolerance
        clipped = float(np.clip(desired, lo, hi)) if lo <= hi else self.target
        target = rate_limit(self.target, clipped, cfg.offset_rate, cfg.timestep)
        speed = cfg.defender_speed if scenario == 'no-attack' else cfg.attacker_speed
        if scenario == 'surge':
            speed = cfg.defender_speed if t < cfg.pass_start else cfg.attacker_speed + .6
        if scenario == 'brake' and gap > length:
            speed = .5 * cfg.defender_speed
        reasons = []
        if not corridor_valid:
            reasons.append('clipped_passing_corridor')
        preview_kwargs = dict(previous_steering=previous_steering,
                              measured_forward_speed=measured_forward_speed)
        requested_preview = self.preview(attacker, defender, line, envelope, track, cfg,
                                         desired, speed, **preview_kwargs)
        path_ok = requested_preview.feasible
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
            # Rejected attack directions never become a fallback goal. Try a
            # measured-lane hold and locally varying nominal at reduced speeds.
            # A fallback is not credited as an executed feasible attack.
            candidates = []
            for fraction in (.7, .35, 0.):
                for name, goal, nominal in (('hold', attacker.d, False),
                                             ('nominal', envelope.nominal(attacker.s), True)):
                    candidate = self.preview(attacker, defender, line, envelope, track, cfg,
                        goal, max(0., speed * fraction), nominal=nominal, **preview_kwargs)
                    candidates.append((candidate, name + f':{fraction}'))
                    if candidate.feasible:
                        selected, fallback = candidate, name + f':{fraction}'
                        break
                if fallback != 'none':
                    break
            if fallback == 'none':
                # No sampled-safe option: retain the longest-preview bounded
                # braking candidate, explicitly unresolved (not a safe attack).
                braking = [(candidate, name) for candidate, name in candidates if name.endswith(':0.0')]
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
        reason = 'guard:' + ','.join(dict.fromkeys(reasons)) if guarded else ('attack' if requested else 'approach')
        return AttackCommand(self.target, self.speed, reason, requested, corridor_valid,
                             guarded, path_ok, self.switched, desired, fallback,
                             unresolved, selected.feasible)
