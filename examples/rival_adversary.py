"""Perfect-state simulation adversary, not hardware perception.

Finite sampled previews do not prove full-maneuver feasibility. Requested
corridor validity, guard interventions and sampled path checks are separate.
"""
import math
import time
from dataclasses import dataclass

import numpy as np

from rival_control import advance_speed_command
from rival_strategy import CarState, pursuit, rate_limit


def swept_overlap(start, end, center, separation):
    """Closed lateral interval intersects an inflated opponent corridor."""
    return min(start, end) <= center + separation and max(start, end) >= center - separation


def advance_speed(speed, acceleration, desired_speed, dt, cfg):
    """Pure command step; opt-in anticipation leaves legacy behavior unchanged."""
    if getattr(cfg, 'corner_tracking', False):
        return advance_speed_command(speed, acceleration, desired_speed, dt,
                                     cfg.max_acceleration, cfg.max_braking, cfg.max_jerk)
    desired_acceleration = float(np.clip((desired_speed - speed) / dt,
                                        -cfg.max_braking, cfg.max_acceleration))
    acceleration = rate_limit(acceleration, desired_acceleration, cfg.max_jerk, dt)
    release = cfg.max_jerk * dt
    braking_limit = max(0., math.sqrt(release ** 2 + 2 * cfg.max_jerk * speed) - release)
    acceleration = max(acceleration, -braking_limit)
    return max(0., speed + acceleration * dt), acceleration


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
    acceleration: float = 0.
    profile_speed_cap: float = math.inf
    defender_prediction_model: str = 'constant_frenet'


@dataclass
class ManeuverPlan:
    """Sampled opportunity, never a safety certificate or completed-pass credit.

    Times are relative to this observation. Defender paths own the legal
    reference; the attacker's lane is not obtained by shifting that reference
    (or today's measured defender bias) over an entire lap. Precheck paths
    contain only s/d/travel; future XY/yaw are NaN until the full XY check.
    """
    feasible: bool
    reason: str
    side: int
    times: np.ndarray
    goals: np.ndarray
    defender_paths: np.ndarray
    entry_time: float = math.inf
    pass_time: float = math.inf
    exit_time: float = math.inf
    failure_time: float = 0.
    minimum_room: float = math.inf
    required_gain: float = 0.

    def goal(self, tau):
        return float(np.interp(tau, self.times, self.goals))


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
    acceleration: float = 0.
    final_speed: float = 0.
    defender_prediction_model: str = 'constant_frenet'


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
    defender_prediction_model = 'sampled_hold_and_reference_bias_constant_speed_and_acceleration'
    defender_hypotheses = ('hold_constant', 'reference_constant',
                           'hold_accelerating', 'reference_accelerating')

    def __init__(self, initial_target, tracking_margin=0., *, speed_profile=None,
                 motion_config=None, maneuver_planning=None):
        if not np.isfinite(tracking_margin) or tracking_margin < 0:
            raise ValueError('tracking_margin must be finite and nonnegative')
        if (speed_profile is None) != (motion_config is None):
            raise ValueError('speed_profile and motion_config must be supplied together')
        if motion_config is not None:
            for name in ('max_acceleration', 'max_braking', 'max_jerk', 'max_lateral_accel'):
                value = getattr(motion_config, name)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
        self.speed_profile, self.motion_config = speed_profile, motion_config
        # Existing track_speed_planning is the runner's explicit opt-in. Custom
        # users can opt in separately; no-profile/legacy behavior is unchanged.
        if maneuver_planning is None:
            maneuver_planning = bool(getattr(motion_config, 'track_speed_planning', False))
        if not isinstance(maneuver_planning, bool):
            raise ValueError('maneuver_planning must be a boolean')
        if maneuver_planning and speed_profile is None:
            raise ValueError('maneuver_planning requires speed_profile and motion_config')
        self.maneuver_planning = maneuver_planning
        self.maneuver_phase = 'idle'
        self.maneuver = None
        self._maneuver_side = None
        self._maneuver_checked_at = -math.inf
        self._maneuver_preview = None
        self._maneuver_defender_profile = None
        self._defender_profile_source = None
        self._defender_reference_snapshot = None
        self.opportunity_reasons = {}
        self.attacker_plan_ms = 0.
        self.maneuver_check_ms = 0.
        self._acceleration = 0.
        self._fast_recovery = None
        self._recovery_until = -math.inf
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
        if self.maneuver_planning and self.maneuver_phase in ('entry', 'pass', 'exit', 'hold_alongside'):
            # A committed lane cannot turn into a feint during overlap.
            return
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

    @staticmethod
    def _envelope_samples(envelope, stations):
        """Batch the unmodified TrackEnvelope interpolation; custom rules stay scalar."""
        from rival_robust import TrackEnvelope
        if type(envelope) is not TrackEnvelope:
            return None
        length = envelope.line.length
        wrapped = np.asarray(stations) % length
        return tuple(np.interp(wrapped, envelope.s, values, period=length)
                     for values in (envelope.lo, envelope.hi, envelope.reference_offsets))

    def _defender_paths(self, defender, line, envelope, cfg):
        """Sample assumptions, NOT future truth or exhaustive defensive maneuvers.

        Hold current offset or follow the reference with fixed measured bias,
        at current projected speed or bounded acceleration. Offset and heading
        transitions are bounded, but this is not a dynamic-model certificate.
        Columns: x, y, yaw, wrapped s, d, signed travel from the current station.
        """
        times = np.arange(int(math.ceil(self.preview_horizon / cfg.timestep)) + 1) * cfg.timestep
        paths = np.empty((4, len(times), 6))
        accelerations = np.array([0., self.motion_config.max_acceleration])
        travels = defender.speed * times + .5 * accelerations[:, None] * times ** 2
        all_stations = (defender.s + travels) % line.length
        queries = np.r_[defender.s, all_stations[:, 1:].ravel()]
        samples = self._envelope_samples(envelope, queries)
        references = None
        if samples is not None:
            lo, hi, reference = samples
            invalid = np.flatnonzero(lo > hi)
            if len(invalid):
                # Match nominal()'s first error, in the original query order.
                envelope.nominal(queries[invalid[0]])
            bias = defender.d - reference[0]
            references = reference[1:].reshape(2, -1)
        else:
            bias = defender.d - envelope.nominal(defender.s)
        lateral_rate = min(cfg.offset_rate, getattr(self.motion_config, 'max_lateral_rate', cfg.offset_rate))
        for case, acceleration in enumerate(accelerations):
            travel, stations = travels[case], all_stations[case]
            for follow in (0, 1):
                path = paths[2 * case + follow]
                path[0] = (defender.x, defender.y, defender.yaw, defender.s, defender.d, 0.)
                anchor = line.sample(defender.s, defender.d)
                for index in range(1, len(times)):
                    s = stations[index]
                    if follow:
                        reference = envelope.nominal(s) if references is None else references[case, index - 1]
                        goal = reference + bias
                    else:
                        goal = defender.d
                    d = rate_limit(path[index - 1, 4], goal, lateral_rate, cfg.timestep)
                    distance = travel[index] - travel[index - 1]
                    heading = line.heading(s) + math.atan2(d - path[index - 1, 4], max(abs(distance), 1e-6))
                    error = (heading - path[index - 1, 2] + math.pi) % (2 * math.pi) - math.pi
                    velocity = abs(defender.speed + acceleration * times[index])
                    yaw_rate = min(velocity * math.tan(cfg.max_steer) / cfg.wheelbase,
                                   self.motion_config.max_lateral_accel / max(velocity, 1e-6))
                    yaw = rate_limit(path[index - 1, 2], path[index - 1, 2] + error,
                                     yaw_rate, cfg.timestep)
                    xy = [defender.x, defender.y] + line.sample(s, d) - anchor
                    path[index] = (*xy, yaw, s, d, travel[index])
        return paths

    @staticmethod
    def _rectangle_gaps(attacker_path, defender_paths, cfg):
        """One batched SAT query, padded for relative translation and rotation.

        Half-step padding on both endpoints bounds the assumed between-node
        motion; sampled hypotheses still do not certify arbitrary maneuvers.
        """
        from rival_robust import sat_clearance
        ego = np.asarray(attacker_path)
        other = defender_paths[:, :len(ego), :3]
        gaps = sat_clearance(ego[None, :, :], other, cfg.car_length, cfg.car_width)
        relative = other[:, :, :2] - ego[None, :, :2]
        translation = np.linalg.norm(np.diff(relative, axis=1), axis=2) / 2
        ego_turn = (np.diff(ego[:, 2]) + np.pi) % (2 * np.pi) - np.pi
        other_turn = (np.diff(other[:, :, 2], axis=1) + np.pi) % (2 * np.pi) - np.pi
        radius = math.hypot(cfg.car_length, cfg.car_width) / 2
        sweep = translation + radius * (np.abs(ego_turn)[None, :] + np.abs(other_turn)) / 2
        padding = np.maximum(np.pad(sweep, ((0, 0), (0, 1))),
                             np.pad(sweep, ((0, 0), (1, 0))))
        return gaps - cfg.separation_margin - padding

    def _following_speed_cap(self, distance, current_speed, lead_speed, cfg):
        """Anticipatory demand, not an instantaneous clamp or safety certificate.

        Reserve one action of bounded acceleration, the positive part of a
        jerk-limited braking ramp, and nonnegative braking release. The lead
        may start braking now; headway feedback also opens a useful steady gap.
        """
        r = self.motion_config
        current_speed, lead_speed = max(0., current_speed), max(0., lead_speed)
        bumper = max(0., distance - cfg.car_length - cfg.separation_margin)
        available = max(0., bumper - cfg.minimum_gap)
        servo_acceleration = min(r.max_acceleration, max(0., (self.speed - current_speed) / cfg.timestep))
        acceleration = min(r.max_acceleration, max(servo_acceleration,
            max(0., self._acceleration) + r.max_jerk * cfg.timestep))
        ramp = (acceleration + r.max_braking) / r.max_jerk
        response = cfg.timestep + ramp
        extra_speed = r.max_acceleration * cfg.timestep + acceleration ** 2 / (2 * r.max_jerk)
        release_time = r.max_braking / r.max_jerk
        release_distance = r.max_braking * (release_time ** 2 / 6
            + release_time * cfg.timestep + cfg.timestep ** 2)
        reaction_distance = (.5 * r.max_acceleration * cfg.timestep ** 2
                             + extra_speed * ramp + release_distance)
        budget = available + (lead_speed / math.sqrt(2 * r.max_braking)) ** 2
        remaining = max(0., budget - reaction_distance - extra_speed ** 2 / (2 * r.max_braking))
        # Stable positive root of v*response + (v+extra_speed)^2/(2*b).
        root = math.sqrt(remaining) * math.sqrt(2 * r.max_braking)
        shift = extra_speed + r.max_braking * response
        braking_cap = root * (root / (math.hypot(shift, root) + shift))
        desired_gap = cfg.minimum_gap + cfg.time_headway * current_speed
        headway_cap = lead_speed + (bumper - desired_gap) / max(.5, cfg.time_headway)
        return max(0., min(braking_cap, headway_cap))

    def _passing_room(self, paths, side, bias, envelope, cfg, *, gap=None,
                      attacker_speed=0., defender_speed=0., track_length=None):
        # Look beyond the alongside pose: fitting today's lane is insufficient
        # if the reference squeezes that side immediately after catching up.
        separation = cfg.car_width + cfg.separation_margin
        # The short collision rollout cannot see a complete lane transition
        # and subsequent braking. Reserve room over that maneuver too, before
        # initiating a pass that would become impossible in the next bend.
        speed = max(0., float(np.max(paths[:, :, 5])) / self.preview_horizon)
        transition = cfg.pass_offset / cfg.offset_rate
        response = self.motion_config.max_acceleration / self.motion_config.max_jerk
        if gap is None:
            reserve = (speed * (transition + response)
                       + speed ** 2 / (2 * self.motion_config.max_braking))
        else:
            # Passing from behind must gain the remaining bumper gap AND both
            # longitudinal envelopes, not merely survive a lane transition.
            # A .5 m/s planning floor avoids an unbounded near-zero division;
            # this is an opportunity estimate, not assumed future acceleration.
            closing = max(.5, cfg.attacker_speed - cfg.defender_speed,
                          attacker_speed - defender_speed)
            length = cfg.car_length + cfg.separation_margin
            gain = max(0., -gap - length) + 2 * length
            speed = max(speed, attacker_speed, cfg.attacker_speed)
            with np.errstate(over='ignore', invalid='ignore'):
                reserve = float(np.float64(speed) * (gain / closing + transition + response)
                    + np.float64(speed) ** 2 / (2 * self.motion_config.max_braking))
            period = track_length
            if period is None:
                period = getattr(getattr(envelope, 'line', None), 'length',
                                 getattr(self.speed_profile, 'length', None))
            if period is not None:
                # One full period covers every future static reference section,
                # even when a vanishing speed advantage implies many laps.
                reserve = min(reserve, period)
            if not math.isfinite(reserve):
                return False
        future = paths[0, 0, 3] + np.linspace(0., reserve, max(2, int(math.ceil(reserve / .25)) + 1))
        stations = np.unique(np.r_[paths[:, :, 3].ravel(), future])
        queries = (stations[:, None] + np.array([0., cfg.lookahead, 2 * cfg.lookahead])).ravel()
        samples = self._envelope_samples(envelope, queries)
        if samples is not None:
            lo, hi, reference = samples
            d = reference + bias + side * separation
            invalid = lo > hi
            fits = ((lo - self.legal_numeric_tolerance <= d)
                    & (d <= hi + self.legal_numeric_tolerance))
            failed = np.flatnonzero(invalid | ~fits)
            if len(failed):
                # Scalar evaluation returns at the first failure: a later empty
                # band must not replace an earlier ordinary rejection with an error.
                index = failed[0]
                if invalid[index]:
                    envelope.nominal(queries[index])
                return False
            return True
        for s in stations:
            for ds in (0., cfg.lookahead, 2 * cfg.lookahead):
                station = s + ds
                lo, hi = envelope.interval(station)
                d = envelope.nominal(station) + bias + side * separation
                if not lo - self.legal_numeric_tolerance <= d <= hi + self.legal_numeric_tolerance:
                    return False
        return True

    @staticmethod
    def passing_opportunities(line, envelope, cfg, spacing=.25):
        """Read-only geometry diagnostics, not a claim of dynamic feasibility.

        Widths are vehicle-center bands, already reduced by map/rule margins.
        Runs are circular (a straight may cross station zero). No cache hides
        an edited envelope. The defender retains its own nominal reference.
        """
        stations = np.linspace(0., line.length, max(16, int(math.ceil(line.length / spacing))), endpoint=False)
        bands = np.array([envelope.interval(s) for s in stations])
        reference = np.array([envelope.nominal(s) for s in stations])
        separation = cfg.car_width + cfg.separation_margin
        result: dict = dict(track_length_m=line.length, required_separation_m=separation,
                      minimum_center_band_m=float(np.min(bands[:, 1] - bands[:, 0])),
                      maximum_center_band_m=float(np.max(bands[:, 1] - bands[:, 0])),
                      sample_spacing_m=line.length / len(stations), sides={})
        for side in (-1, 1):
            room = reference - bands[:, 0] if side == -1 else bands[:, 1] - reference
            usable = room >= max(separation, cfg.pass_offset)
            longest = current = 0
            for value in np.tile(usable, 2):
                current = current + 1 if value else 0
                longest = max(longest, current)
            result['sides'][side] = dict(maximum_reference_room_m=float(np.max(room)),
                alongside_fraction=float(np.mean(room >= separation)),
                configured_lane_fraction=float(np.mean(usable)),
                longest_configured_lane_m=min(longest, len(stations)) * line.length / len(stations))
        return result

    def _reference_defender_samples(self, defender, line, envelope, cfg, horizon, dt, *, poses=True):
        """Reference-owning cruise/accelerate/brake hypotheses at their OWN s.

        Measured lateral error is recovered at the legal rate, not carried as
        a frozen bias into bends. These are assumptions, not exhaustive defense.
        The original short-horizon hold/bias hypotheses remain an independent
        conjunction in plan(). Brake-now is deliberately included here.
        """
        from rival_speed import SpeedProfile
        assert self.motion_config is not None  # constructor-enforced fast opt-in
        profile = None
        if isinstance(self.speed_profile, SpeedProfile):
            source = (id(line), id(envelope), id(cfg), id(self.motion_config), cfg.defender_speed)
            if (source != self._defender_profile_source or self._defender_reference_snapshot is None
                    or not np.array_equal(envelope.reference_offsets, self._defender_reference_snapshot)):
                self._maneuver_defender_profile = SpeedProfile(line, envelope, cfg, self.motion_config,
                                                               cfg.defender_speed)
                self._defender_profile_source = source
                self._defender_reference_snapshot = envelope.reference_offsets.copy()
            profile = self._maneuver_defender_profile
        current = np.tile((defender.x, defender.y, defender.yaw, defender.s, defender.d, 0.), (3, 1))
        velocities = np.full(3, max(0., defender.speed))
        lateral_rate = min(cfg.offset_rate, getattr(self.motion_config, 'max_lateral_rate', cfg.offset_rate))
        anchor = line.sample(defender.s, defender.d) if poses else None
        yield 0., current.copy()
        for index in range(1, int(math.ceil(horizon / dt)) + 1):
            for case in range(3):
                previous, velocity = current[case].copy(), velocities[case]
                if case == 2:
                    new_velocity = max(0., velocity - self.motion_config.max_braking * dt)
                else:
                    ceiling = max(0., defender.speed, cfg.defender_speed) if case == 1 else max(0., defender.speed)
                    demand = min(ceiling, profile.limit(previous[3], velocity)) if profile is not None else ceiling
                    new_velocity = rate_limit(velocity, demand,
                        self.motion_config.max_acceleration if demand >= velocity else self.motion_config.max_braking, dt)
                distance = .5 * (velocity + new_velocity) * dt
                travel = previous[5] + distance
                s = (defender.s + travel) % line.length
                d = rate_limit(previous[4], envelope.nominal(s), min(lateral_rate, new_velocity), dt)
                if poses:
                    heading = line.heading(s) + math.atan2(d - previous[4], max(distance, 1e-6))
                    error = (heading - previous[2] + math.pi) % (2 * math.pi) - math.pi
                    yaw_rate = min(new_velocity * math.tan(cfg.max_steer) / cfg.wheelbase,
                                   self.motion_config.max_lateral_accel / max(new_velocity, 1e-6))
                    yaw = rate_limit(previous[2], previous[2] + error, yaw_rate, dt)
                    xy = [defender.x, defender.y] + line.sample(s, d) - anchor
                    current[case] = (*xy, yaw, s, d, travel)
                else:
                    # Precheck needs only Frenet occupancy. Do not compute XY
                    # or yaw for a future that may be rejected one node later.
                    current[case] = (math.nan, math.nan, math.nan, s, d, travel)
                velocities[case] = new_velocity
            yield index * dt, current.copy()

    def _reference_defender_paths(self, defender, line, envelope, cfg, horizon, dt):
        samples = list(self._reference_defender_samples(defender, line, envelope, cfg, horizon, dt))
        return np.array([tau for tau, _ in samples]), np.stack([path for _, path in samples], axis=1)

    def _exit_clearance(self, speed, follower_speed, cfg):
        """Front return reserve if the attacker brakes just after merging.

        Give the new follower the same response/ramp/release reservation as
        the rear-end guard. Instant full braking by the attacker is the harder
        lead-car assumption; this is not permission to brake-check in practice.
        """
        r = self.motion_config
        assert r is not None
        a, b, j, dt = r.max_acceleration, r.max_braking, r.max_jerk, cfg.timestep
        ramp = (a + b) / j
        extra = a * dt + a ** 2 / (2 * j)
        release = b / j
        reaction = .5 * a * dt ** 2 + extra * ramp + b * (release ** 2 / 6 + release * dt + dt ** 2)
        braking_room = (follower_speed * (dt + ramp) + reaction
                        + ((follower_speed + extra) ** 2 - speed ** 2) / (2 * b))
        return cfg.minimum_gap + max(0., cfg.time_headway * follower_speed, braking_room)

    def _maneuver_room(self, attacker, defender, side, line, envelope, cfg, speed):
        """Cheap bounded space-time precheck before ANY long XY/map rollout.

        Entry and exit need legal occupancy, not a parallel lane all the way
        from today's defender to a full lap later. Only the pass phase needs
        sustained lateral separation. A failure is model/budget-specific, never
        reported as global impossibility or a completed attack.
        """
        assert self.motion_config is not None and self.speed_profile is not None
        horizon, dt = 30., max(cfg.timestep, .1)
        gap = line.signed_distance(attacker.s, defender.s)
        lead_speed = max(0., defender.speed, cfg.defender_speed)
        exit_reserve = self._exit_clearance(max(0., attacker.speed), lead_speed, cfg)
        initial_goal = (envelope.nominal(attacker.s + cfg.lookahead)
                        if gap > cfg.car_length + cfg.separation_margin + exit_reserve
                        else defender.d + side * max(cfg.car_width + cfg.separation_margin, cfg.pass_offset))
        lo, hi = envelope.interval(attacker.s + cfg.lookahead)
        if not lo - self.legal_numeric_tolerance <= initial_goal <= hi + self.legal_numeric_tolerance:
            return ManeuverPlan(False, 'entry_target_corridor', side, np.array([0.]),
                np.array([initial_goal]), np.array([[(defender.x, defender.y, defender.yaw,
                    defender.s, defender.d, 0.)]] * 3), minimum_room=min(initial_goal - lo, hi - initial_goal),
                required_gain=max(0., cfg.car_length + cfg.separation_margin + cfg.minimum_gap - gap))
        times = np.arange(int(math.ceil(horizon / dt)) + 1) * dt
        paths = np.empty((3, len(times), 6))
        paths[:, 0] = (defender.x, defender.y, defender.yaw, defender.s, defender.d, 0.)
        length = cfg.car_length + cfg.separation_margin
        separation = cfg.car_width + cfg.separation_margin
        gain = max(0., length + cfg.minimum_gap - gap)
        goals = np.empty(len(times))
        entry = passed = exited = math.inf
        minimum_room = math.inf
        s, d, velocity = attacker.s, attacker.d, max(0., attacker.speed)
        command, acceleration, travel = self.speed, self._acceleration, 0.
        lateral_rate = min(cfg.offset_rate, getattr(self.motion_config, 'max_lateral_rate', cfg.offset_rate))
        phase = 'entry'
        settling_since = None
        previous_relative_d = d - paths[:, 0, 4]
        previous_gaps = np.full(3, gap)

        def result(feasible, reason, index):
            return ManeuverPlan(feasible, reason, side, times[:index + 1], goals[:index + 1].copy(),
                paths[:, :index + 1].copy(), entry, passed, exited,
                0. if feasible else times[index], minimum_room, gain)

        # No arbitrary .5 m/s division floor, and no all-lap room demand. A
        # genuinely nonclosing cruise is not a full-pass opportunity in this
        # model, even if a brake-now hypothesis would be easy to overtake.
        if speed <= max(0., min(defender.speed, cfg.defender_speed)) + 1e-9 and gap < length:
            goals[0] = envelope.nominal(s + cfg.lookahead)
            return result(False, 'no_predicted_speed_advantage', 0)
        predictions = self._reference_defender_samples(defender, line, envelope, cfg, horizon, dt, poses=False)
        for index, (tau, predicted_paths) in enumerate(predictions):
            paths[:, index] = predicted_paths
            gaps = gap + travel - paths[:, index, 5]
            # Return only with braking/headway room in FRONT of every defender
            # hypothesis. Merely clearing its nose is not a safe exit.
            lead_speed = max(0., defender.speed, cfg.defender_speed)
            exit_reserve = self._exit_clearance(velocity, lead_speed, cfg)
            if np.min(gaps) > length + exit_reserve:
                phase = 'exit'
                if not math.isfinite(passed):
                    passed = tau
            for predicted in paths[:, index]:
                defender_lo, defender_hi = envelope.interval(predicted[3])
                if (defender_lo > defender_hi
                        or not defender_lo - self.tracking_margin - self.legal_numeric_tolerance
                        <= predicted[4] <= defender_hi + self.tracking_margin + self.legal_numeric_tolerance):
                    goals[index] = envelope.nominal(s + cfg.lookahead)
                    return result(False, 'defender_reference_body_corridor', index)
            nominal = envelope.nominal(s + cfg.lookahead)
            if phase == 'exit':
                goal = nominal
            else:
                edge = np.max(paths[:, index, 4]) if side == 1 else np.min(paths[:, index, 4])
                goal = edge + side * max(separation, cfg.pass_offset)
            goals[index] = goal
            lo, hi = envelope.interval(s + cfg.lookahead)
            room = min(goal - lo, hi - goal)
            minimum_room = min(minimum_room, room)
            if lo > hi or room < -self.legal_numeric_tolerance:
                return result(False, phase + '_target_corridor', index)
            if index:
                d = rate_limit(d, goal, lateral_rate, dt)
            lo, hi = envelope.interval(s)
            if lo > hi or not lo - self.tracking_margin <= d <= hi + self.tracking_margin:
                return result(False, phase + '_body_corridor', index)
            relative_d = d - paths[:, index, 4]
            lateral = ((np.minimum(previous_relative_d, relative_d) <= separation)
                       & (np.maximum(previous_relative_d, relative_d) >= -separation))
            longitudinal = ((np.minimum(previous_gaps, gaps) <= length)
                            & (np.maximum(previous_gaps, gaps) >= -length))
            if np.any(lateral & longitudinal):
                case = int(np.flatnonzero(lateral & longitudinal)[0])
                return result(False, 'brake_check_overlap' if case == 2 else phase + '_overlap', index)
            if phase == 'entry' and np.all(side * relative_d > separation):
                entry, phase = tau, 'pass'
            if phase == 'exit' and abs(d - nominal) < .02:
                if settling_since is None:
                    settling_since = tau
                # The ideal lateral-rate precheck has no pursuit lag. Reserve
                # settling time; the XY rollout must actually clear and return.
                settle = self.preview_horizon + 2 * cfg.lookahead / max(velocity, .1)
                if tau - settling_since >= settle:
                    exited = tau
                    return result(True, 'entry_pass_exit_room', index)
            else:
                settling_since = None
            previous_relative_d, previous_gaps = relative_d, gaps
            demand = min(speed, self.speed_profile.limit(s, velocity), self.reference_speed_cap(s, envelope, cfg))
            command, acceleration = advance_speed(command, acceleration, demand, dt, self.motion_config)
            new_velocity = rate_limit(velocity, command,
                self.motion_config.max_acceleration if command >= velocity else self.motion_config.max_braking, dt)
            distance = .5 * (velocity + new_velocity) * dt
            travel += distance
            s = (attacker.s + travel) % line.length
            velocity = new_velocity
        return result(False, 'completion_beyond_planning_budget', len(times) - 1)

    def _check_maneuver(self, t, attacker, defender, line, envelope, track, cfg, speed,
                        previous_steering, measured_forward_speed):
        """Replan at tactical cadence; short safety checks still run every step."""
        # Negative opportunities are cheap to revisit. A positive full rollout
        # is reused at most .5 s, never as a replacement for the current guard.
        if (self.maneuver is not None and t - self._maneuver_checked_at < self.reevaluation_period
                and self._maneuver_side == self.side):
            return self.maneuver
        start = time.perf_counter()
        plan = self._maneuver_room(attacker, defender, self.side, line, envelope, cfg, speed)
        self._maneuver_preview = None
        if plan.feasible:
            _, paths = self._reference_defender_paths(defender, line, envelope, cfg, plan.exit_time, cfg.timestep)
            checked = self.preview(attacker, defender, line, envelope, track, cfg,
                plan.goal(0.), speed, previous_steering=previous_steering,
                measured_forward_speed=measured_forward_speed, defender_paths=paths,
                maneuver=plan, horizon=plan.exit_time)
            self._maneuver_preview = checked
            if not checked.feasible or checked.goal_clipped:
                plan.feasible = False
                plan.reason = 'full_maneuver_' + (checked.reason if not checked.feasible else 'clipped_target')
                plan.failure_time = checked.safe_time
        self.maneuver, self._maneuver_side = plan, self.side
        self._maneuver_checked_at = t
        self.maneuver_check_ms = (time.perf_counter() - start) * 1000
        return plan

    def preview(self, attacker, defender, line, envelope, track, cfg, goal, speed,
                *, previous_steering=0., measured_forward_speed=None, nominal=False,
                body_recovery=False, defender_paths=None, maneuver=None, horizon=None):
        """Reference XY bicycle rollout using the actual pursuit/command slew.

        This is an approximate kinematic predictor, not Gym physics. Reconstruct
        body-forward speed from the line projection only when unavailable; the
        runner supplies measured body vx. No ideal Frenet offset motion is used.
        Fixed-goal compiled accelerators must leave maneuver/horizon rollouts
        on this staged Python path unless they implement the same goal schedule.
        """
        if getattr(self.motion_config, 'compiled_attack_preview', False):
            from rival_attack_rollout import compiled_preview
            result = compiled_preview(self, attacker, defender, line, envelope, track, cfg,
                goal, speed, previous_steering=previous_steering,
                measured_forward_speed=measured_forward_speed, nominal=nominal,
                body_recovery=body_recovery, defender_paths=defender_paths,
                maneuver=maneuver, horizon=horizon)
            if result is not None:
                return result
        dt = cfg.timestep
        horizon = self.preview_horizon if horizon is None else horizon
        state = attacker
        cosine = math.cos(attacker.yaw - line.heading(attacker.s))
        if measured_forward_speed is None:
            measured_forward_speed = attacker.speed / math.copysign(max(abs(cosine), .2), cosine)
        velocity = float(measured_forward_speed)
        target, command_speed, steering = self.target, self.speed, previous_steering
        # Pursuit goals are offsets at the lookahead station. Body checks
        # below remain at the actual projected station, not that future band.
        bias = 0. if nominal else goal - envelope.nominal(attacker.s + cfg.lookahead)
        raw_first_goal = (maneuver.goal(0.) if maneuver is not None
                          else envelope.nominal(attacker.s + cfg.lookahead) + bias)
        first_lo, first_hi = envelope.interval(attacker.s + cfg.lookahead)
        first_goal = float(np.clip(raw_first_goal, first_lo, first_hi)) if first_lo <= first_hi else raw_first_goal
        goal_clipped = abs(first_goal - raw_first_goal) > self.legal_numeric_tolerance
        fast = self.speed_profile is not None
        if fast and defender_paths is None:
            defender_paths = self._defender_paths(defender, line, envelope, cfg)
        attacker_path = []
        acceleration = self._acceleration
        first_demand = min(speed, self.reference_speed_cap(attacker.s, envelope, cfg))
        first_target = rate_limit(self.target, first_goal, cfg.offset_rate, dt)
        if fast:
            first_demand = min(first_demand, self.speed_profile.limit(attacker.s, attacker.speed))
            first_speed, first_acceleration = advance_speed(
                self.speed, acceleration, first_demand, dt, self.motion_config)
        else:
            speed = first_demand
            first_speed = rate_limit(self.speed, speed,
                cfg.reference_acceleration if speed >= self.speed else cfg.comfortable_braking, dt)
            first_acceleration = 0.
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
                max_error <= self.legal_numeric_tolerance, first_acceleration, state.speed,
                self.defender_prediction_model if fast else 'constant_frenet')

        for index in range(int(math.ceil(horizon / dt)) + 1):
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
            if fast:
                attacker_path.append((state.x, state.y, state.yaw))
                gap = relative_s - defender_paths[:, index, 5]
                previous_relative_d = previous_d - defender_paths[:, max(0, index - 1), 4]
                relative_d = state.d - defender_paths[:, index, 4]
                lateral = ((np.minimum(previous_relative_d, relative_d) <= cfg.car_width + cfg.separation_margin)
                           & (np.maximum(previous_relative_d, relative_d) >= -cfg.car_width - cfg.separation_margin))
                longitudinal = ((np.minimum(previous_gap, gap) <= cfg.car_length + cfg.separation_margin)
                                & (np.maximum(previous_gap, gap) >= -cfg.car_length - cfg.separation_margin))
                if np.any(lateral & longitudinal):
                    return result(False, 'sampled_collision_preview:evolving_defender', tau)
            else:
                gap = relative_s - defender.speed * tau
                if (swept_overlap(previous_d, state.d, defender.d, cfg.car_width + cfg.separation_margin)
                        and min(previous_gap, gap) <= cfg.car_length + cfg.separation_margin
                        and max(previous_gap, gap) >= -cfg.car_length - cfg.separation_margin):
                    return result(False, 'sampled_collision_preview', tau)
            waiting_for_exit = (maneuver is not None and tau >= maneuver.pass_time - 1e-9
                and np.min(gap) <= cfg.car_length + cfg.separation_margin
                + self._exit_clearance(max(0., velocity), max(0., defender.speed, cfg.defender_speed), cfg))
            if (fast and velocity ** 2 * abs(math.tan(steering)) / cfg.wheelbase
                    > self.motion_config.max_lateral_accel + 1e-9):
                return result(False, 'lateral_acceleration_preview', tau)
            if tau >= horizon - 1e-9:
                break
            previous_d, previous_gap = state.d, gap
            target_s = state.s + cfg.lookahead
            raw_desired = (maneuver.goal(tau) if maneuver is not None
                           else envelope.nominal(target_s) + bias)
            if maneuver is not None and tau >= maneuver.pass_time - 1e-9 and not waiting_for_exit:
                # Return to the reference at the XY car's actual station, not
                # the faster ideal-station schedule used by the cheap precheck.
                raw_desired = envelope.nominal(target_s)
            if waiting_for_exit:
                # Lateral motion costs projected progress. Never perform a
                # scheduled return until the XY predictor actually clears the
                # brake/headway reserve; hold the SAME passing side instead.
                edge = (np.max(defender_paths[:, index, 4]) if maneuver.side == 1
                        else np.min(defender_paths[:, index, 4]))
                raw_desired = edge + maneuver.side * max(cfg.pass_offset, cfg.car_width + cfg.separation_margin)
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
            if fast:
                desired_speed = min(desired_speed, self.speed_profile.limit(state.s, state.speed))
                command_speed, acceleration = advance_speed(
                    command_speed, acceleration, desired_speed, dt, self.motion_config)
            else:
                command_speed = rate_limit(command_speed, desired_speed,
                    cfg.reference_acceleration if desired_speed >= command_speed else cfg.comfortable_braking, dt)
            steering, _, _ = pursuit(state, line, target, command_speed, steering, cfg)

            motion = self.motion_config
            acceleration_limit = motion.max_acceleration if fast else cfg.reference_acceleration
            braking_limit = motion.max_braking if fast else cfg.comfortable_braking
            new_velocity = rate_limit(velocity, command_speed,
                acceleration_limit if command_speed >= velocity else braking_limit, dt)
            if (fast and max(velocity ** 2, new_velocity ** 2) * abs(math.tan(steering))
                    / cfg.wheelbase > motion.max_lateral_accel + 1e-9):
                return result(False, 'lateral_acceleration_preview', tau + dt)
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
        if fast:
            gaps = self._rectangle_gaps(attacker_path, defender_paths, cfg)
            unsafe = np.argwhere(gaps <= 0.)
            if len(unsafe):
                hypothesis, index = min(unsafe, key=lambda pair: pair[1])
                hypotheses = (('reference_cruise', 'reference_accelerating', 'reference_brake_now')
                              if maneuver is not None else self.defender_hypotheses)
                return result(False, 'sampled_rectangle_collision_preview:' + hypotheses[hypothesis],
                              index * dt)
        if maneuver is not None:
            final_gaps = relative_s - defender_paths[:, len(attacker_path) - 1, 5]
            if (np.min(final_gaps) <= cfg.car_length + cfg.separation_margin
                    + self._exit_clearance(max(0., velocity), max(0., defender.speed, cfg.defender_speed), cfg)
                    or abs(state.d - envelope.nominal(state.s)) > .02):
                return result(False, 'maneuver_not_clear_and_returned', horizon)
        if recovery_active:
            required_descent = min(.005, .2 * initial_error)
            reentered = current_error <= self.legal_numeric_tolerance
            if not reentered and initial_error - current_error < required_descent:
                return result(False, 'body_band_recovery_insufficient_descent', self.preview_horizon)
            return result(True, 'body_band_recovered' if reentered else 'body_band_descent_partial',
                          self.preview_horizon)
        return result(True, 'sampled_xy_path', horizon)

    def _fast_fallback(self, t, attacker, defender, line, envelope, track, cfg,
                       nominal, speed, recovery_required, preview_kwargs, *, prefer_nominal=False,
                       locked_side=None):
        """Search small batches; every continued repair is checked again from now."""
        hold = nominal + attacker.d - envelope.nominal(attacker.s)
        goals = [('nominal', nominal, True), ('hold', hold, False)]
        goals += [(f'correction:{bias:+.1f}', nominal + bias, False)
                  for bias in (-.3, .3, -.6, .6)]
        if locked_side is not None:
            # When already overlapping, an invalid future opportunity means
            # hold the committed side and brake, NOT abandon it through a car.
            goals = [('hold_side', attacker.d, False)] + [entry for entry in goals
                if entry[0] != 'nominal' and locked_side * (entry[1] - defender.d)
                >= cfg.car_width + cfg.separation_margin]
            prefer_nominal = False
        candidates, seen = [], set()

        def moving(candidate):
            # Residual coasting to a stop is not sustained forward recovery.
            return (candidate.feasible and candidate.progress > .05 * self.preview_horizon
                    and candidate.final_speed > .05)

        def check(name, goal, is_nominal, demand):
            demand = max(0., min(speed, demand))
            key = (goal, is_nominal, demand)
            if key in seen:
                return None
            seen.add(key)
            candidate = self.preview(attacker, defender, line, envelope, track, cfg,
                goal, demand, nominal=is_nominal, body_recovery=recovery_required, **preview_kwargs)
            entry = (candidate, name, goal - nominal, is_nominal, demand)
            candidates.append(entry)
            return entry

        def select(entry, unresolved=False):
            candidate, name, bias, is_nominal, demand = entry
            if moving(candidate):
                if (self._fast_recovery is None or t >= self._recovery_until
                        or name != self._fast_recovery[0]):
                    self._recovery_until = t + .2
                self._fast_recovery = (name, bias, is_nominal, demand)
            else:
                self._fast_recovery = None
            label = f'{name}:{demand:g}'
            return candidate, 'unresolved_brake:' + label if unresolved else label, unresolved

        if prefer_nominal and not recovery_required:
            # Invalid passing room is a reason to leave the old passing edge,
            # not maximize tiny progress differences between nominal and hold.
            # An alongside return still has to pass every swept collision check.
            fresh_nominal = check('nominal', nominal, True, speed)
            if fresh_nominal[0].feasible and fresh_nominal[0].progress >= 0.:
                return select(fresh_nominal)

        if locked_side is None and self._fast_recovery is not None and t < self._recovery_until:
            name, bias, is_nominal, demand = self._fast_recovery
            if not recovery_required:
                # Keep line-choice hysteresis, not an obsolete speed ceiling.
                # Both a return to nominal and a faster continued repair must
                # pass the current guard/profile and the entire preview again.
                fresh_nominal = check('nominal', nominal, True, speed)
                if fresh_nominal is not None and moving(fresh_nominal[0]):
                    return select(fresh_nominal)
                if speed > demand:
                    faster = check(name, nominal + bias, is_nominal, speed)
                    if faster is not None and moving(faster[0]):
                        return select(faster)
            continued = check(name, nominal + bias, is_nominal, demand)
            if continued is not None and moving(continued[0]):
                return select(continued)

        measured = preview_kwargs['measured_forward_speed']
        current = min(speed, max(0., self.speed,
                                attacker.speed if measured is None else measured))
        crawl = min(speed, max(current, .5))
        batches = [] if recovery_required else [(speed, goals[:2])]
        if current > 0:
            batches.append((current, goals))
        batches += [(min(crawl, absolute), goals) for absolute in (.5, .25, .1, 0.)]
        for demand, batch_goals in batches:
            batch = [entry for name, goal, is_nominal in batch_goals
                     if (entry := check(name, goal, is_nominal, demand)) is not None]
            progressing = [entry for entry in batch if moving(entry[0])]
            if progressing:
                return select(max(progressing, key=lambda entry: (
                    entry[0].initial_body_error - entry[0].final_body_error if recovery_required else 0.,
                    entry[0].progress)))
        feasible = [entry for entry in candidates if entry[0].feasible]
        if feasible:
            # A cap below the meaningful-progress threshold may still allow
            # honest slow motion. Do not prefer backward motion to a safe stop.
            return select(max(feasible, key=lambda entry: (
                entry[0].progress > 1e-6 and entry[0].final_speed > 1e-6,
                entry[0].progress, -entry[4])))
        braking = [entry for entry in candidates if entry[4] == 0.]
        return select(max(braking, key=lambda entry: entry[0].safe_time), unresolved=True)

    def plan(self, t, scenario, attacker, defender, line, envelope, track, cfg,
             *, previous_steering=0., measured_forward_speed=None):
        plan_started = time.perf_counter()
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
        committed = (self.maneuver_planning and self.maneuver_phase in
                     ('entry', 'pass', 'exit', 'hold_alongside'))
        requested = (scenario not in ('no-attack', 'approach') and t >= cfg.pass_start
                     and (passing_window or committed))
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
        scenario_speed = speed
        reference_cap = self.reference_speed_cap(attacker.s, envelope, cfg)
        profile_cap = (self.speed_profile.limit(attacker.s, attacker.speed)
                       if self.speed_profile is not None else math.inf)
        speed = min(speed, reference_cap, profile_cap)
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
        narrowing = False
        exit_guard = False
        opportunity = None
        if self.speed_profile is not None:
            preview_kwargs['defender_paths'] = self._defender_paths(defender, line, envelope, cfg)
            if self.maneuver_planning and requested:
                opportunity = self._check_maneuver(t, attacker, defender, line, envelope, track, cfg,
                    scenario_speed, previous_steering, measured_forward_speed)
                # Use the measured space-time opportunity's own lane even for
                # rejection telemetry, not an unrelated frozen-bias request.
                desired = opportunity.goal(t - self._maneuver_checked_at)
                corridor_valid = lo <= desired <= hi
                target = rate_limit(self.target, desired, cfg.offset_rate, cfg.timestep)
                reasons = [reason for reason in reasons if reason != 'clipped_passing_corridor']
                if not corridor_valid:
                    reasons.append('clipped_passing_corridor')
                exit_guard = (opportunity.feasible
                    and t - self._maneuver_checked_at >= opportunity.pass_time - 1e-9
                    and gap <= length + self._exit_clearance(max(0., attacker.speed),
                        max(0., defender.speed, cfg.defender_speed), cfg))
                if exit_guard:
                    reasons.append('front_return_braking_guard')
                narrowing = not opportunity.feasible
                if narrowing:
                    self.opportunity_reasons[opportunity.reason] = self.opportunity_reasons.get(opportunity.reason, 0) + 1
                    reasons.append('pass_deferred:' + opportunity.reason)
                    self.maneuver_phase = ('hold_alongside' if abs(gap) <= length
                                           else 'abort_before_overlap' if self.maneuver_phase in ('entry', 'pass')
                                           else 'defer')
            else:
                narrowing = requested and not self._passing_room(
                    preview_kwargs['defender_paths'], self.side, defender_bias, envelope, cfg,
                    gap=gap, attacker_speed=attacker.speed, defender_speed=defender.speed,
                    track_length=line.length)
                if narrowing:
                    reasons.append('narrowing_defender_passing_corridor')
            if narrowing:
                corridor_valid = False
        requested_preview = self.preview(attacker, defender, line, envelope, track, cfg,
                                         desired, scenario_speed if self.speed_profile is not None else speed,
                                         **preview_kwargs)
        path_ok = requested_preview.feasible
        if requested and requested_preview.goal_clipped:
            corridor_valid = False
            reasons.append('clipped_future_passing_corridor')
        if not path_ok:
            reasons.append(requested_preview.reason)
        # Include the entire desired sweep: opposite-side endpoints alone can
        # both look clear while the interval between them crosses the defender.
        overlap = (narrowing or swept_overlap(attacker.d, desired, defender.d, separation)
                   or swept_overlap(attacker.d, target, defender.d, separation))
        if overlap and abs(gap) <= length:
            reasons.append('alongside_cut_in_guard')
            speed = min(speed, max(0., defender.speed - .5))
        elif overlap and gap < 0:
            if self.speed_profile is not None:
                current_speed = max(0., attacker.speed,
                    attacker.speed if measured_forward_speed is None else measured_forward_speed)
                cap = self._following_speed_cap(-gap, current_speed, defender.speed, cfg)
            else:
                available = max(0., -gap - length)
                cap = max(0., defender.speed + math.sqrt(2 * cfg.comfortable_braking * available)
                          - cfg.comfortable_braking * cfg.timestep)
                if cfg.time_headway > 0:
                    cap = min(cap, available / cfg.time_headway)
            if speed > cap:
                speed = cap
                reasons.append('rear_end_guard')
        if (not reasons and self._fast_recovery is not None and t < self._recovery_until):
            reasons.append('continuing_checked_recovery')
        guarded = bool(reasons)
        selected, fallback, unresolved = requested_preview, 'none', False
        if guarded and self.speed_profile is not None:
            selected, fallback, unresolved = self._fast_fallback(t, attacker, defender,
                line, envelope, track, cfg, nominal, speed, recovery_required, preview_kwargs,
                prefer_nominal=narrowing,
                locked_side=(self._maneuver_side if self.maneuver_planning
                             and (exit_guard or abs(gap) <= length
                                  and self.maneuver_phase in ('entry', 'pass', 'hold_alongside')) else None))
        elif guarded:
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
        if self.maneuver_planning:
            if opportunity is not None and opportunity.feasible and not guarded:
                self.maneuver_phase = ('exit' if abs(desired - nominal) < .02 and gap > length
                                      else 'pass' if abs(attacker.d - defender.d) > separation else 'entry')
                if (self.maneuver_phase == 'exit' and gap > length + cfg.minimum_gap
                        and abs(attacker.d - envelope.nominal(attacker.s)) < .02):
                    self.maneuver_phase = 'idle'
            elif gap > length and self.maneuver_phase in ('entry', 'pass', 'hold_alongside'):
                self.maneuver_phase = 'exit'
            elif not requested and abs(attacker.d - envelope.nominal(attacker.s)) < .02:
                self.maneuver_phase = 'idle'
            if guarded and self.maneuver_phase in ('entry', 'pass') and gap < -length:
                self.maneuver_phase = 'abort_before_overlap'
        self.target, self.speed = selected.target, selected.speed
        if self.speed_profile is not None:
            self._acceleration = selected.acceleration
            if not guarded:
                self._fast_recovery = None
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
        if self.speed_profile is not None:
            reason += ';defender_model=' + self.defender_prediction_model + ';not_future_truth_or_exhaustive_defense=True'
        self.attacker_plan_ms = (time.perf_counter() - plan_started) * 1000
        if self.maneuver_planning:
            reason += (f';maneuver_phase={self.maneuver_phase};attacker_plan_ms={self.attacker_plan_ms:.6g}'
                       f';maneuver_check_ms={self.maneuver_check_ms:.6g}')
            if opportunity is not None:
                reason += (f';opportunity={opportunity.reason};entry_s={opportunity.entry_time:.6g}'
                           f';pass_s={opportunity.pass_time:.6g};exit_s={opportunity.exit_time:.6g}'
                           f';opportunity_failure_s={opportunity.failure_time:.6g}'
                           f';opportunity_age_s={t - self._maneuver_checked_at:.6g}'
                           f';opportunity_min_room_m={opportunity.minimum_room:.6g}'
                           f';required_gain_m={opportunity.required_gain:.6g}'
                           ';opportunity_model=legal_reference_cruise_accelerate_brake_now'
                           ';opportunity_is_not_completed_pass=True')
        self._last_body_error = initial_body_error
        self._last_body_recovery = recovery_required and selected.feasible
        return AttackCommand(self.target, self.speed, reason, requested, corridor_valid,
                             guarded, path_ok, self.switched, desired, fallback,
                             unresolved, selected.feasible, reference_cap,
                             recovery_required, initial_body_error, selected.max_body_error,
                             selected.final_body_error, selected.partial_body_recovery, passing_window,
                             actual_error_delta, actual_error_increased, self.tracking_margin,
                             selected.within_tracking_budget, selected.strict_body_legal,
                             self._acceleration, profile_cap,
                             self.defender_prediction_model if self.speed_profile is not None else 'constant_frenet')
