"""Simulation-only receding-horizon defensive controller.

A finite kinematic trajectory lattice with uncertain opponent hypotheses, NOT a
verified safety controller, exhaustive reachable-set proof, or hardware driver.
The independent simulator remains the evaluator; no physics is changed here.
"""
from dataclasses import dataclass, fields
from pathlib import Path
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from rival_strategy import CarState, desired_gap, rate_limit


@dataclass
class RobustConfig:
    horizon: float = 1.5
    prediction_dt: float = .1
    max_physics_dt: float = .01
    control_period: float = .1
    max_acceleration: float = 2.
    max_braking: float = 3.
    max_jerk: float = 6.
    max_lateral_rate: float = .45
    max_lateral_acceleration: float = .8
    max_lateral_accel: float = 3.
    region: str = 'inside'
    envelope_spacing: float = .5
    reference_smoothing_length: float = 3.
    reference_clearance_reserve: float = .12
    commitment_time: float = .8
    tracking_bias: float = .3
    recovery_min_progress: float = .1
    max_opponent_acceleration: float = 2.
    max_opponent_lateral_speed: float = .8
    max_opponent_speed: float = 8.
    position_margin: float = .08
    lateral_tracking_margin: float = .08
    min_defense_ttc: float = 2.
    safe_time_headway: float = .6
    rollout_speed_fractions: tuple = (1., .7, .35, .2, .1, 0.)
    boundaries_path: str | None = None
    uncertainty_multiplier: float = 2.
    escape_speed_fraction: float = 1.2

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown robust settings: {sorted(unknown)}')
        return cls(**values)

    def __post_init__(self):
        if self.region not in ('inside', 'track'):
            raise ValueError('region must be inside or track; track explicitly relaxes inside-half rules')
        optional = {'region', 'boundaries_path', 'rollout_speed_fractions'}
        for f in fields(self):
            if f.name not in optional:
                value = getattr(self, f.name)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'{f.name} must be finite and nonnegative')
        positive = ('horizon', 'prediction_dt', 'max_physics_dt', 'control_period', 'max_acceleration',
                    'max_braking', 'max_jerk', 'max_lateral_rate',
                    'max_lateral_acceleration', 'max_lateral_accel', 'envelope_spacing')
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError(f'{positive} must be positive')
        if self.prediction_dt > self.horizon or self.control_period > self.horizon:
            raise ValueError('prediction_dt and control_period must not exceed horizon')
        speeds = np.asarray(self.rollout_speed_fractions, dtype=float)
        if speeds.ndim != 1 or not len(speeds) or not np.all(np.isfinite(speeds)) or np.any((speeds < 0) | (speeds > 1)):
            raise ValueError('rollout_speed_fractions must be a nonempty sequence in [0,1]')
        self.rollout_speed_fractions = tuple(float(v) for v in speeds)
        if not 1 <= self.escape_speed_fraction <= 1.5:
            raise ValueError('escape_speed_fraction must be in [1,1.5]')


class TrackEnvelope:
    """Precomputed periodic geometric envelope; map footprint tests stay online.

    Optional CSV columns s,lo,hi are measured *boundary* normal offsets from the
    raceline (metres), not usable vehicle-center limits. Sample increasing s in
    [0,L), cover the lap, and do not repeat L. Unknown search-cap endpoints are
    counted in diagnostics. This is an estimated rule region, not certified IAC
    legality; curvature blending deliberately avoids a discontinuous half flip.
    """
    def __init__(self, line, track, vehicle_cfg, robust_cfg, boundaries_path=None):
        self.line, self.track, self.vehicle, self.config = line, track, vehicle_cfg, robust_cfg
        c, r = vehicle_cfg, robust_cfg
        count = max(16, int(np.ceil(line.length / r.envelope_spacing)))
        self.s = np.linspace(0., line.length, count, endpoint=False)
        spacing = line.length / count
        self.truncated_sections = 0
        self.invalid_sections = 0
        path = boundaries_path or r.boundaries_path
        bounds = []
        if path:
            data = np.loadtxt(Path(path), delimiter=',', skiprows=1, ndmin=2)
            if data.shape[1] != 3 or len(data) < 3 or not np.all(np.isfinite(data)):
                raise ValueError('Boundary CSV needs header s,lo,hi and at least three finite rows')
            if np.any(np.diff(data[:, 0]) <= 0) or data[0, 0] != 0 or data[-1, 0] >= line.length or np.any(data[:, 1] >= data[:, 2]):
                raise ValueError('Boundary s must increase from 0 in [0,L), with lo < hi')
            lo = np.interp(self.s, data[:, 0], data[:, 1], period=line.length)
            hi = np.interp(self.s, data[:, 0], data[:, 2], period=line.length)
            body_margin = c.car_width / 2 + c.boundary_margin
            bounds = np.column_stack((lo + body_margin, hi - body_margin))
            self.source = str(path)
        else:
            for s in self.s:
                try:
                    lo, hi = track.corridor(line, s, c.car_width / 2 + c.boundary_margin,
                        max_offset=c.max_offset_search, step=.1)
                    self.truncated_sections += int(lo <= -c.max_offset_search + 1e-5 or hi >= c.max_offset_search - 1e-5)
                except ValueError:
                    lo, hi = 1., -1.
                    self.invalid_sections += 1
                bounds.append((lo, hi))
            bounds = np.asarray(bounds)
            self.source = 'bitmap cross-sections (estimated, search-capped)'
        self.raw_lo, self.raw_hi = bounds[:, 0].copy(), bounds[:, 1].copy()
        lo, hi = self.raw_lo.copy(), self.raw_hi.copy()
        if r.region == 'inside':
            curvature = np.array([line.curvature(s) for s in self.s])
            curvature = gaussian_filter1d(curvature, max(1., 1.5 / spacing), mode='wrap')
            center = (lo + hi) / 2
            # A gradual transition from full corridor on near-straights to the
            # inside half in sustained turns; never silently switch to track mode.
            weight = np.clip((np.abs(curvature) - c.curvature_threshold)
                             / max(2 * c.curvature_threshold, 1e-3), 0., 1.)
            left = curvature >= 0
            inside_lo = np.where(left, center + c.center_margin,
                                 center - c.inner_fraction * (center - lo))
            inside_hi = np.where(left, center + c.inner_fraction * (hi - center),
                                 center - c.center_margin)
            lo = (1 - weight) * lo + weight * inside_lo
            hi = (1 - weight) * hi + weight * inside_hi
        self.lo = np.maximum(lo, c.legal_min)
        self.hi = np.minimum(hi, c.legal_max)
        invalid = self.raw_lo > self.raw_hi
        self.lo[invalid], self.hi[invalid] = 1., -1.
        self.reference_offsets = np.clip(np.zeros_like(self.s), self.lo, self.hi)
        if np.all(self.lo <= self.hi):
            from rival_reference import clearance_reference, smoothed_reference
            from rival_geometry import TrackMap
            if isinstance(track, TrackMap):
                # A center offset inside the legal band need not fit the full
                # oriented car. Prepare a body-clear path, rather than arriving
                # at a margin violation from which every preview rejects t=0.
                self.reference_offsets, self.reference_diagnostics = clearance_reference(
                    line, track, self.s, self.lo, self.hi, c.car_length, c.car_width,
                    c.boundary_margin + r.reference_clearance_reserve,
                    r.reference_smoothing_length)
            else:
                self.reference_offsets, self.reference_diagnostics = smoothed_reference(
                    self.s, self.lo, self.hi, line.length, r.reference_smoothing_length)
        else:
            # Preserve explicit invalid bands; smoothing must never bridge an
            # infeasible road section or manufacture a legal route through it.
            self.reference_diagnostics = dict(solver_success=False,
                reason='empty legal bands; no globally feasible reference')
        self._reference_slopes = (np.roll(self.reference_offsets, -1) - self.reference_offsets) / spacing

    def interval(self, s):
        return (float(np.interp(s % self.line.length, self.s, self.lo, period=self.line.length)),
                float(np.interp(s % self.line.length, self.s, self.hi, period=self.line.length)))

    def nominal(self, s):
        lo, hi = self.interval(s)
        if lo > hi:
            raise ValueError(f'No estimated legal corridor at s={s % self.line.length:.3f}')
        return self.reference(s)

    def reference(self, s):
        return float(np.interp(s % self.line.length, self.s, self.reference_offsets,
                               period=self.line.length))

    def reference_slope(self, s):
        return float(np.interp(s % self.line.length, self.s, self._reference_slopes,
                               period=self.line.length))


def stopping_distance(speed, braking, latency=0.):
    if not all(np.isfinite([speed, braking, latency])) or braking <= 0 or latency < 0:
        raise ValueError('Finite speed, positive braking, and nonnegative latency required')
    return max(0., speed) * latency + max(0., speed) ** 2 / (2 * braking)


def advance_target(target, velocity, goal, dt, cfg):
    """Second-order target filter with bounded offset velocity/acceleration."""
    error = goal - target
    desired_rate = np.sign(error) * min(cfg.max_lateral_rate,
        math.sqrt(max(0., 2 * cfg.max_lateral_acceleration * abs(error))))
    velocity = rate_limit(velocity, desired_rate, cfg.max_lateral_acceleration, dt)
    return float(target + velocity * dt), float(velocity)


def sat_clearance(ego, opponent, length, width):
    """Batched signed separating-axis gap for equal oriented rectangles.

    Positive means separated on at least one axis; nonpositive means overlapping.
    The positive value is a lower bound on Euclidean body clearance. Arrays end
    in x,y,yaw (extra columns ignored); leading dimensions broadcast.
    """
    e, o = np.asarray(ego), np.asarray(opponent)
    delta = o[..., :2] - e[..., :2]
    eh, oh = np.broadcast_arrays(e[..., 2], o[..., 2])
    axes = np.stack((eh, eh + np.pi / 2, oh, oh + np.pi / 2), axis=-1)
    projected = np.abs(delta[..., 0, None] * np.cos(axes) + delta[..., 1, None] * np.sin(axes))
    e_radius = length / 2 * np.abs(np.cos(axes - eh[..., None])) + width / 2 * np.abs(np.sin(axes - eh[..., None]))
    o_radius = length / 2 * np.abs(np.cos(axes - oh[..., None])) + width / 2 * np.abs(np.sin(axes - oh[..., None]))
    return np.max(projected - e_radius - o_radius, axis=-1)


def opponent_hypotheses(estimate, line, times, cfg):
    """Nine bounded acceleration/lateral-rate hypotheses plus age-based inflation.

    This samples plausible intent; it is NOT exhaustive adversarial reachability.
    Unmodelled turns, rate changes, and errors beyond the envelope remain possible.
    """
    if estimate is None or not estimate.valid:
        return np.empty((0, len(times), 3)), np.zeros(len(times))
    q = estimate.state
    speed_uncertainty = cfg.uncertainty_multiplier * estimate.velocity_sigma
    longitudinal_cases = (
        (max(0., q.speed - speed_uncertainty), -cfg.max_braking),
        (max(0., q.speed), float(np.clip(estimate.longitudinal_acceleration,
            -cfg.max_braking, cfg.max_opponent_acceleration))),
        (max(0., q.speed + speed_uncertainty), cfg.max_opponent_acceleration))
    # A measured speed above the configured model bound must not be truncated.
    # step() flags that assumption violation as unresolved/degraded.
    speed_limit = max(cfg.max_opponent_speed, max(0., q.speed))
    lateral_rates = (-cfg.max_opponent_lateral_speed,
        float(np.clip(estimate.lateral_speed, -cfg.max_opponent_lateral_speed,
                      cfg.max_opponent_lateral_speed)), cfg.max_opponent_lateral_speed)
    predictions = []
    for initial_speed, acceleration in longitudinal_cases:
        for lateral in lateral_rates:
            trajectory = []
            for t in times:
                v0 = min(initial_speed, speed_limit)
                bound = 0. if acceleration < 0 else speed_limit
                change_time = t if acceleration == 0 else min(t, max(0., (bound - v0) / acceleration))
                speed = float(np.clip(v0 + acceleration * change_time, 0., speed_limit))
                distance = v0 * change_time + .5 * acceleration * change_time ** 2 + speed * (t - change_time)
                s = q.s + distance
                d = q.d + lateral * t
                xy = line.sample(s, d)
                yaw = line.heading(s) + math.atan2(lateral, max(.2, speed))
                trajectory.append([*xy, yaw])
            predictions.append(trajectory)
    # Initial velocity uncertainty is propagated through the fast/slow saturated
    # hypotheses above, not counted again as an unbounded sigma*t term. Retain
    # a growing tracking/model discrepancy tube; it is not a calibrated bound.
    inflation = (cfg.position_margin + cfg.uncertainty_multiplier * estimate.position_sigma
                 + cfg.lateral_tracking_margin * np.asarray(times))
    return np.asarray(predictions), inflation


@dataclass
class Decision:
    steering: float
    speed: float
    target_offset: float
    mode: str
    reason: str
    feasible: bool
    min_clearance: float
    min_separation: float
    planning_ms: float
    hypothesis_count: int


@dataclass
class _Rollout:
    bias: float
    desired_speed: float
    trajectory: np.ndarray
    map_safe: bool
    legal_safe: bool
    separation_safe: bool
    min_clearance: float
    min_separation: float
    violation: float
    progress: float
    score: float = -math.inf

    @property
    def feasible(self):
        return self.map_safe and self.legal_safe and self.separation_safe


class RobustDefender:
    """Replan a finite lattice; safety rejection precedes defensive usefulness.

    Own localization is supplied externally. No opponent truth is accepted by
    this class; only timestamped estimates from rival_perception enter planning.
    The command-state model is kinematic, not identical to Gym's dynamic model.
    """
    def __init__(self, line, track, vehicle_cfg, robust_cfg, initial_target, envelope=None):
        self.line, self.track, self.vehicle, self.config = line, track, vehicle_cfg, robust_cfg
        self.envelope = envelope or TrackEnvelope(line, track, vehicle_cfg, robust_cfg)
        self.target = float(initial_target)
        self._lateral_rate = self._acceleration = self._steering = self._speed_command = 0.
        self._bias = 0.
        self._desired_speed = vehicle_cfg.defender_speed * .6
        self._last_time = None
        self._next_plan = -math.inf
        self._last_switch = -math.inf
        self._fresh = False
        self._best = None
        self._mode = 'degraded'
        self._reason = 'uninitialized'
        self._planning_ms = 0.
        self._hypothesis_count = 0
        self._defense_enabled = None
        self._assumption_broken = False

    def _goal(self, state, bias):
        # This offset is attached to the pursuit point, not the current body.
        # Clamping it against today's band prevents anticipation of a narrowing
        # or shifting corridor, eventually leaving only a standstill candidate.
        preview_s = state.s + self.vehicle.lookahead
        lo, hi = self.envelope.interval(preview_s)
        if lo > hi:
            return self.target
        nominal = (self.envelope.reference(preview_s) if isinstance(self.envelope, TrackEnvelope)
                   else float(np.clip(0., lo, hi)))
        return float(np.clip(nominal + bias, lo, hi))

    def _command(self, state, target, lateral_rate, acceleration, steering, command_speed,
                 bias, desired_speed, dt):
        r, c = self.config, self.vehicle
        goal = self._goal(state, bias)
        target, lateral_rate = advance_target(target, lateral_rate, goal, dt, r)
        # Curvature-dependent speed rather than fixed speed through every turn.
        curvature = max(abs(self.line.curvature(state.s + ds)) for ds in (0., c.lookahead, 2 * c.lookahead))
        turn_speed = math.sqrt(r.max_lateral_accel / max(curvature, 1e-3))
        desired_speed = min(desired_speed, turn_speed)
        if isinstance(self.envelope, TrackEnvelope):
            slope = max(abs(self.envelope.reference_slope(state.s + ds))
                        for ds in (0., c.lookahead, 2 * c.lookahead))
            desired_speed = min(desired_speed, .8 * r.max_lateral_rate / max(slope, 1e-6))
        desired_accel = np.clip((desired_speed - command_speed) / max(dt, 1e-6), -r.max_braking, r.max_acceleration)
        acceleration = rate_limit(acceleration, desired_accel, r.max_jerk, dt)
        # Reserve enough nonnegative speed to release braking acceleration at
        # the jerk limit. This conservative discrete stopping budget avoids a
        # hard zero-speed clip silently violating emitted-command jerk bounds.
        release = r.max_jerk * dt
        braking_limit = max(0., math.sqrt(release ** 2 + 2 * r.max_jerk * command_speed) - release)
        acceleration = max(acceleration, -braking_limit)
        next_speed = max(0., command_speed + acceleration * dt)
        # Small target overshoots are retained in the prediction and checked;
        # do not clip/teleport an accelerating command to an unmodelled value.
        point = self.line.sample(state.s + c.lookahead, target)
        delta = point - [state.x, state.y]
        lateral = -math.sin(state.yaw) * delta[0] + math.cos(state.yaw) * delta[1]
        desired_steer = math.atan2(2 * c.wheelbase * lateral, max(float(delta @ delta), 1e-6))
        desired_steer = float(np.clip(desired_steer, -c.max_steer, c.max_steer))
        steering = rate_limit(steering, desired_steer, c.max_steer_rate, dt)
        return target, lateral_rate, acceleration, steering, next_speed

    def _trajectory_python(self, ego, bias, desired_speed, times):
        """Readable reference implementation; parity-tested against the JIT kernel."""
        c, r = self.vehicle, self.config
        state = CarState(**vars(ego))
        alignment = math.cos(ego.yaw - self.line.heading(ego.s))
        forward_speed = max(0., ego.speed / alignment) if abs(alignment) > .25 else max(0., self._speed_command, abs(ego.speed))
        target, lateral_rate = self.target, self._lateral_rate
        acceleration, steering, command_speed = self._acceleration, self._steering, self._speed_command
        trajectory = [[state.x, state.y, state.yaw, forward_speed, state.s, state.d, target]]
        start_lo, start_hi = self.envelope.interval(state.s)
        initial_error = max(0., start_lo - state.d, state.d - start_hi)
        violation = 0.
        max_lateral_acceleration = 0.
        for dt, t in zip(np.diff(times), times[1:]):
            target, lateral_rate, acceleration, steering, command_speed = self._command(
                state, target, lateral_rate, acceleration, steering, command_speed, bias, desired_speed, dt)
            # Rate-limited command speed approximates the fast low-level speed
            # servo. It is validated against, not substituted for, Gym physics.
            speed = max(0., forward_speed + np.clip(command_speed - forward_speed,
                -r.max_braking * dt, r.max_acceleration * dt))
            average_speed = (forward_speed + speed) / 2
            yaw_delta = average_speed / c.wheelbase * math.tan(steering) * dt
            x = state.x + average_speed * math.cos(state.yaw + yaw_delta / 2) * dt
            y = state.y + average_speed * math.sin(state.yaw + yaw_delta / 2) * dt
            yaw = state.yaw + yaw_delta
            s, d = self.line.project([x, y])
            projected_speed = speed * math.cos(yaw - self.line.heading(s))
            state = CarState(s, d, projected_speed, x, y, yaw)
            forward_speed = speed
            trajectory.append([x, y, yaw, speed, s, d, target])
            lo, hi = self.envelope.interval(s)
            tolerance = r.lateral_tracking_margin + max(0., initial_error - .1 * t)
            target_lo, target_hi = self.envelope.interval(s + c.lookahead)
            violation = max(violation, lo - d - tolerance, d - hi - tolerance,
                            target_lo - target - tolerance, target - target_hi - tolerance)
            if lo > hi or target_lo > target_hi:
                violation = max(violation, 1.)
            max_lateral_acceleration = max(max_lateral_acceleration,
                speed ** 2 * abs(math.tan(steering)) / c.wheelbase)
        return np.asarray(trajectory), violation, max_lateral_acceleration

    def _rollout(self, ego, bias, desired_speed, times, predictions, inflation):
        c, r = self.vehicle, self.config
        if isinstance(self.envelope, TrackEnvelope):
            from rival_rollout import simulate_trajectory
            controls = (self.target, self._lateral_rate, self._acceleration,
                        self._steering, self._speed_command)
            path, violation, max_lateral_acceleration = simulate_trajectory(
                self.line, self.envelope, ego, controls, bias, desired_speed,
                times, c, r)
        else:
            path, violation, max_lateral_acceleration = self._trajectory_python(
                ego, bias, desired_speed, times)
        # Fast centre-disk certification; only ambiguous poses need the dense
        # orientation-aware footprint grid. Every trajectory pose is checked.
        clearance = self.track.clear_many(path[:, :2])
        disk_radius = math.hypot(c.car_length, c.car_width) / 2
        margin = c.boundary_margin + r.lateral_tracking_margin
        ambiguous = np.flatnonzero(clearance < disk_radius + margin)
        if len(ambiguous) and hasattr(self.track, 'footprints_safe'):
            map_safe = bool(np.all(self.track.footprints_safe(path[ambiguous, :3],
                c.car_length, c.car_width, margin)))
        else:
            map_safe = all(self.track.footprint_safe(path[i, :2], path[i, 2],
                c.car_length, c.car_width, margin) for i in ambiguous)
        min_map = float(np.min(clearance - disk_radius - margin))
        min_sep = math.inf
        if len(predictions):
            gaps = sat_clearance(path[None, :, :3], predictions, c.car_length, c.car_width)
            # Inflate half a step by measured relative translation and rotation;
            # this reduces endpoint tunnelling but isn't a continuous-time proof.
            relative = predictions[:, :, :2] - path[None, :, :2]
            translation = np.linalg.norm(np.diff(relative, axis=1), axis=2) / 2
            opponent_turn = (np.diff(predictions[:, :, 2], axis=1) + np.pi) % (2 * np.pi) - np.pi
            ego_turn = (np.diff(path[:, 2]) + np.pi) % (2 * np.pi) - np.pi
            rotation = disk_radius * (np.abs(opponent_turn) + np.abs(ego_turn)[None, :]) / 2
            sweep = translation + rotation
            inflation_at_nodes = np.maximum(np.pad(sweep, ((0, 0), (0, 1))),
                                           np.pad(sweep, ((0, 0), (1, 0))))
            min_sep = float(np.min(gaps - inflation[None, :] - inflation_at_nodes
                                   - c.separation_margin - r.lateral_tracking_margin))
        progress = sum(self.line.signed_distance(b, a) for a, b in zip(path[:-1, 4], path[1:, 4]))
        legal_safe = violation <= 1e-9 and max_lateral_acceleration <= r.max_lateral_accel + .1
        return _Rollout(bias, desired_speed, path, map_safe, legal_safe, min_sep > 0,
            min_map, min_sep, max(violation, max_lateral_acceleration - r.max_lateral_accel, 0.), progress)

    def _solve(self, t, ego, estimate, defense):
        start = time.perf_counter()
        c, r = self.vehicle, self.config
        fresh = estimate is not None and estimate.valid and estimate.fresh
        # Advance command filters at actual actuation cadence, not a coarser
        # planner cadence. Safety is sampled at least this often as well.
        forecast_dt = min(c.timestep, r.prediction_dt)
        times = np.linspace(0., r.horizon, int(np.ceil(r.horizon / forecast_dt)) + 1)
        predictions, inflation = opponent_hypotheses(estimate, self.line, times, r)
        self._hypothesis_count = len(predictions)
        target_s = ego.s + c.lookahead
        nominal = self.envelope.nominal(target_s) if self.envelope.interval(target_s)[0] <= self.envelope.interval(target_s)[1] else self.target
        gap, closing, ttc = math.inf, 0., math.inf
        opponent_d = ego.d
        if estimate is not None and estimate.valid:
            gap = self.line.signed_distance(estimate.state.s, ego.s)
            closing = estimate.state.speed - ego.speed
            opponent_d = estimate.state.d
            if gap < 0 and closing > .05:
                ttc = max(0., -gap - c.car_length) / closing
        alongside = abs(gap) <= c.car_length + c.separation_margin + r.position_margin
        react = fresh and gap < 0 and closing > c.closing_threshold and (
            -gap - c.car_length < c.reaction_threshold * desired_gap(
                max(0., estimate.state.speed), closing, c))
        may_defend = defense and react and not alongside and ttc >= r.min_defense_ttc
        current_bias = self.target - nominal
        # A geometrically legal reference still has tracking lag. Search small
        # corrections even without a passing threat; otherwise a feasible zero-
        # speed nominal path can hide every forward recovery around a corner.
        biases = [0., current_bias, -r.tracking_bias, r.tracking_bias]
        if abs(ego.speed) < .3:
            biases += [-2 * r.tracking_bias, 2 * r.tracking_bias]
        if may_defend:
            biases += [-.6, -.3, .3, .6, opponent_d + estimate.lateral_speed * .5 - nominal]
        # If lateral overlap is imminent, never chase the opponent's lane.
        if alongside or ttc < r.min_defense_ttc:
            biases += [ego.d - nominal, ego.d - nominal - .3, ego.d - nominal + .3]
        speeds = [c.defender_speed * fraction for fraction in r.rollout_speed_fractions]
        if gap < 0 and -gap < max(6., ego.speed * r.horizon):
            speeds.append(c.defender_speed * r.escape_speed_fraction)
        if not fresh:
            speeds = [min(v, c.defender_speed * .65) for v in speeds]
        if 0 < gap < max(8., ego.speed * r.horizon + 2.):
            # Follow with braking-distance/headway cap rather than binary stop.
            available = max(0., gap - c.car_length - c.minimum_gap)
            lead_speed = max(0., estimate.state.speed)
            follow_cap = min(math.sqrt(lead_speed ** 2 + 2 * r.max_braking * available),
                             lead_speed + available / max(r.safe_time_headway, .1))
            speeds = [min(v, follow_cap) for v in speeds]
        rollouts = []
        for bias in sorted(set(round(float(v), 6) for v in biases)):
            moving_toward = (nominal + bias - ego.d) * (opponent_d - ego.d) > .001
            if (alongside or ttc < r.min_defense_ttc) and moving_toward:
                continue
            switching = current_bias * bias < -.01
            if switching and t - self._last_switch < r.commitment_time:
                continue
            for speed in sorted(set(speeds), reverse=True):
                rollout = self._rollout(ego, bias, speed, times, predictions, inflation)
                terminal_d = rollout.trajectory[-1, 5]
                if alongside:
                    direction = np.sign(opponent_d - ego.d)
                    approach = np.max((rollout.trajectory[:, 5] - ego.d) * direction)
                    # If existing target/steering momentum cannot be arrested,
                    # label it unresolved, never a successful non-encroaching yield.
                    if approach > .02:
                        rollout.separation_safe = False
                defense_value = -abs(terminal_d - opponent_d) if may_defend else -abs(bias)
                # Collision/map/legal rejection dominates utility below. Lap
                # progress matters, so an unnecessary standstill cannot win.
                clearance_reward = min(1., rollout.min_separation) if np.isfinite(rollout.min_separation) else 0.
                rollout.score = rollout.progress + .8 * defense_value + .15 * clearance_reward - .25 * abs(bias - current_bias)
                rollouts.append(rollout)
        if not rollouts:
            rollouts = [self._rollout(ego, current_bias, 0., times, predictions, inflation)]
        feasible = [p for p in rollouts if p.feasible]
        if feasible:
            progressing = [p for p in feasible if p.progress > r.recovery_min_progress]
            # Do not award a permanent safe-looking standstill a better utility
            # than an actually checked forward route merely for small offset.
            best = max(progressing or feasible, key=lambda p: p.score)
            if not fresh:
                mode, reason = 'degraded', 'opponent unobserved/stale; no tactical block; map checks only' if estimate is None else 'stale estimate; inflated predictions and reduced speed'
            elif alongside or ttc < r.min_defense_ttc:
                mode, reason = 'yield', 'yield requested; residual lateral motion checked against opponent hypotheses'
            elif may_defend and abs(best.bias) > .05:
                mode, reason = 'defend', 'useful trajectory passed all sampled hypothesis checks'
            elif 0 < gap < 8.:
                mode, reason = 'follow', 'lead-car braking/headway speed cap'
            elif abs(best.bias) > .05:
                mode, reason = 'recovery', 'checked forward tracking correction; no tactical block'
            else:
                mode, reason = 'cruise', 'nominal progress; no worthwhile legal defensive trajectory'
        else:
            # No claim of safety: retain map-safe escape/progress options when
            # possible, rather than always brake directly in front of a rear car.
            candidates = [p for p in rollouts if p.map_safe and p.legal_safe]
            if not candidates:
                candidates = [p for p in rollouts if p.map_safe] or rollouts
            best = max(candidates, key=lambda p: (-(0. if p.map_safe else 1000.) - p.violation,
                p.min_separation if np.isfinite(p.min_separation) else 0., p.progress))
            mode, reason = 'emergency', 'no modeled-safe trajectory; best sampled map-safe/minimax fallback, NOT certified safe'
        if abs(best.bias - self._bias) > .15:
            self._last_switch = t
        self._bias, self._desired_speed, self._best = best.bias, best.desired_speed, best
        self._mode, self._reason, self._fresh = mode, reason, fresh
        self._defense_enabled = defense
        self._planning_ms = (time.perf_counter() - start) * 1000
        self._next_plan = t + r.control_period

    def step(self, t, ego, estimate, defense=True):
        if not np.isfinite(t) or (self._last_time is not None and t < self._last_time):
            raise ValueError('Control time must be finite and nondecreasing')
        if not np.all(np.isfinite([ego.s, ego.d, ego.speed, ego.x, ego.y, ego.yaw])):
            raise ValueError('Ego state must be finite; external emergency stop required')
        dt = self.vehicle.timestep if self._last_time is None else t - self._last_time
        if dt <= 0:
            raise ValueError('Call step once per positive simulation timestep')
        invalid_estimate = False
        bound_violation = False
        if estimate is not None:
            q = estimate.state
            values = [q.s, q.d, q.speed, q.x, q.y, q.yaw, estimate.age,
                estimate.position_sigma, estimate.velocity_sigma,
                estimate.longitudinal_acceleration, estimate.lateral_speed]
            invalid_estimate = (not np.all(np.isfinite(values)) or
                min(estimate.age, estimate.position_sigma, estimate.velocity_sigma) < 0)
            if invalid_estimate:
                estimate = None
            else:
                bound_violation = (abs(q.speed) > self.config.max_opponent_speed
                                    or abs(estimate.lateral_speed) > self.config.max_opponent_lateral_speed
                                    or estimate.longitudinal_acceleration > self.config.max_opponent_acceleration
                                    or estimate.longitudinal_acceleration < -self.config.max_braking)
        fresh = estimate is not None and estimate.valid and estimate.fresh
        heading_unreliable = abs(math.cos(ego.yaw - self.line.heading(ego.s))) < .5
        assumption_broken = bound_violation or heading_unreliable or invalid_estimate
        effective_defense = defense and not assumption_broken
        replanned = (t + 1e-9 >= self._next_plan or fresh != self._fresh
                     or effective_defense != self._defense_enabled
                     or assumption_broken != self._assumption_broken)
        if replanned:
            self._solve(t, ego, estimate, effective_defense)
            self._assumption_broken = assumption_broken
            if invalid_estimate or bound_violation or heading_unreliable:
                self._mode = 'degraded'
                self._reason = ('invalid opponent packet' if invalid_estimate else
                    'own heading makes speed projection unreliable; recovery only' if heading_unreliable else
                    'opponent exceeds configured speed/acceleration/lateral prediction bounds; no tactical defense')
        self.target, self._lateral_rate, self._acceleration, self._steering, self._speed_command = self._command(
            ego, self.target, self._lateral_rate, self._acceleration, self._steering,
            self._speed_command, self._bias, self._desired_speed, dt)
        self._last_time = t
        best = self._best
        return Decision(float(self._steering), float(self._speed_command), float(self.target),
            self._mode, self._reason, bool(best.feasible and fresh and not bound_violation and not invalid_estimate and not heading_unreliable), best.min_clearance,
            best.min_separation, self._planning_ms if replanned else 0., self._hypothesis_count)
