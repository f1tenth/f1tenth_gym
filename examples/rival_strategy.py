"""Interpretable IDM/MOBIL-inspired heuristics, not either highway model.

Positive d is left of travel. The usable inner half is estimated from a connected
bitmap cross-section, not assumed symmetric around the racing line. Near zero
curvature both sides are permitted. A nominal target is the feasible point
closest to the racing line. No collision-free or rules-compliance guarantee is
implied by the finite-horizon, constant-speed predictions below.
"""
from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass
class Settings:
    timestep: float = .02
    duration: float = 8.
    laps: int = 1
    seed: int = 12345
    start_s: float = 23.
    initial_gap: float = 3.
    defender_speed: float = 2.
    attacker_speed: float = 2.8
    lookahead: float = .85
    wheelbase: float = .3302
    car_length: float = .58
    car_width: float = .31
    boundary_margin: float = .04
    center_margin: float = .02
    legal_min: float = -1.2
    legal_max: float = 1.2
    inner_fraction: float = .95
    curvature_threshold: float = .015
    max_offset_search: float = 3.
    offset_rate: float = .25
    minimum_gap: float = .7
    time_headway: float = .6
    comfortable_braking: float = 2.
    reference_acceleration: float = 2.
    reaction_threshold: float = 1.3
    closing_threshold: float = .05
    prediction_horizon: float = 1.
    separation_margin: float = .12
    usefulness_threshold: float = .015
    move_penalty: float = .08
    pass_offset: float = .75
    pass_start: float = 1.
    pass_ramp: float = 2.
    max_steer: float = .4189
    max_steer_rate: float = 1.5

    @classmethod
    def from_dict(cls, values):
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        config = cls(**values)
        for name in names:
            if not np.isfinite(getattr(config, name)):
                raise ValueError(f'{name} must be finite')
        positive = ('timestep', 'duration', 'laps', 'initial_gap', 'lookahead',
                    'wheelbase', 'car_length', 'car_width', 'max_offset_search',
                    'offset_rate', 'comfortable_braking', 'reference_acceleration',
                    'reaction_threshold', 'prediction_horizon', 'pass_ramp',
                    'max_steer', 'max_steer_rate')
        if any(getattr(config, key) <= 0 for key in positive):
            raise ValueError(f'These settings must be positive: {positive}')
        nonnegative = ('boundary_margin', 'center_margin', 'defender_speed',
                       'attacker_speed', 'minimum_gap', 'time_headway',
                       'closing_threshold', 'separation_margin', 'move_penalty',
                       'usefulness_threshold', 'pass_offset', 'pass_start',
                       'curvature_threshold')
        if any(getattr(config, key) < 0 for key in nonnegative):
            raise ValueError(f'These settings must be nonnegative: {nonnegative}')
        if config.legal_min >= config.legal_max or not 0 < config.inner_fraction <= 1:
            raise ValueError('Invalid legal interval or inner_fraction')
        if int(config.laps) != config.laps or int(config.seed) != config.seed:
            raise ValueError('laps and seed must be integers')
        return config


@dataclass
class CarState:
    s: float
    d: float
    speed: float
    x: float
    y: float
    yaw: float


def observe(obs, index, line):
    x, y = obs['poses_x'][index], obs['poses_y'][index]
    yaw = obs['poses_theta'][index]
    s, d = line.project([x, y])
    # Gym velocities are body-frame values. Project world velocity onto the line.
    vx, vy = obs['linear_vels_x'][index], obs['linear_vels_y'][index]
    relative_yaw = yaw - line.heading(s)
    speed = vx * np.cos(relative_yaw) - vy * np.sin(relative_yaw)
    return CarState(s, d, float(speed), float(x), float(y), float(yaw))


def rate_limit(current, desired, rate, dt):
    return float(current + np.clip(desired - current, -rate * dt, rate * dt))


def desired_gap(speed, closing_speed, cfg):
    """Follower speed and positive closing speed; metres, bumper-to-bumper."""
    dynamic = speed * cfg.time_headway + speed * closing_speed / (
        2 * math.sqrt(cfg.reference_acceleration * cfg.comfortable_braking))
    return cfg.minimum_gap + max(0., dynamic)


def legal_interval(bounds, curvature, cfg):
    """Inside half of a locally estimated *track* cross-section.

    bounds already account for vehicle half-width and boundary margin. Their
    midpoint estimates the track center. inner_fraction shrinks the inside half
    away from its outer boundary. legal_min/max are additional raceline-relative
    limits. Straights have no privileged inside and permit the full corridor.
    """
    lo, hi = bounds
    center = (lo + hi) / 2
    if curvature > cfg.curvature_threshold:
        lo, hi = center + cfg.center_margin, center + cfg.inner_fraction * (hi - center)
    elif curvature < -cfg.curvature_threshold:
        lo, hi = center - cfg.inner_fraction * (center - lo), center - cfg.center_margin
    return max(lo, cfg.legal_min), min(hi, cfg.legal_max)


class TrackRules:
    def __init__(self, line, track, cfg):
        self.line, self.track, self.cfg = line, track, cfg
        self._cache = {}

    def interval(self, s):
        # Cache exact requests (initialization and repeated prediction samples).
        key = float(s % self.line.length)
        if key not in self._cache:
            c = self.cfg
            try:
                bounds = self.track.corridor(self.line, s,
                    c.car_width / 2 + c.boundary_margin,
                    max_offset=c.max_offset_search, step=.1)
                # Smooth the polygon's curvature over a short local neighbourhood.
                curvature = np.mean([self.line.curvature(s + ds) for ds in (-1., 0., 1.)])
                self._cache[key] = legal_interval(bounds, curvature, c)
            except ValueError:
                self._cache[key] = (1., -1.)
        # Bound memory for long runs; no discretization of safety decisions.
        interval = self._cache[key]
        if len(self._cache) > 4000:
            self._cache.clear()
        return interval

    def planning_interval(self, state, speed):
        intervals = [self.interval(state.s + max(0., speed) * t)
                     for t in np.linspace(0., self.cfg.prediction_horizon, 6)]
        return max(lo for lo, _ in intervals), min(hi for _, hi in intervals)

    def footprint(self, s, d, margin=None):
        c = self.cfg
        return self.track.footprint_safe(self.line.sample(s, d), self.line.heading(s),
            c.car_length, c.car_width, c.boundary_margin if margin is None else margin)

    def path_safe(self, state, target, speed, command_target=None):
        """Sample a bounded lateral transition; reject map and legal violations."""
        c = self.cfg
        command_target = target if command_target is None else command_target
        for t in np.linspace(0, c.prediction_horizon, 6):
            s = state.s + max(0., speed) * t
            d = rate_limit(state.d, command_target, c.offset_rate, t)
            lo, hi = self.interval(s)
            # Existing tracking error can be outside the legal band; don't
            # demand instantaneous correction, but require the commanded path
            # to remain map-safe. Target itself must satisfy each local band.
            if not lo <= target <= hi or not self.footprint(s, d):
                return False
        return True


def predicted_separation_safe(defender, attacker, target, line, cfg):
    """Swept Frenet rectangles sampled through the prediction horizon.

    Attacker d is held fixed (constant-velocity prediction); body sizes and a
    margin inflate rectangles. Also check a continuous relative-linear sweep
    within each time slice so fast closings cannot jump over an overlap.
    """
    gap = line.signed_distance(attacker.s, defender.s)
    closing = attacker.speed - defender.speed
    times = np.linspace(0., cfg.prediction_horizon, 11)
    for t0, t1 in zip(times[:-1], times[1:]):
        g0, g1 = gap + closing * t0, gap + closing * t1
        d0 = attacker.d - rate_limit(defender.d, target, cfg.offset_rate, t0)
        d1 = attacker.d - rate_limit(defender.d, target, cfg.offset_rate, t1)
        # Conservative interval boxes can false-reject but cannot skip overlap.
        longitudinal_clear = min(g0, g1) > cfg.car_length + cfg.separation_margin or max(g0, g1) < -cfg.car_length - cfg.separation_margin
        lateral_clear = min(d0, d1) > cfg.car_width + cfg.separation_margin or max(d0, d1) < -cfg.car_width - cfg.separation_margin
        if not (longitudinal_clear or lateral_clear):
            return False
    return True


class Defender:
    def __init__(self, line, rules, cfg, initial_target):
        self.line, self.rules, self.cfg = line, rules, cfg
        self.target = initial_target

    def decide(self, defender, attacker, enabled=True):
        c = self.cfg
        gap = self.line.signed_distance(attacker.s, defender.s)
        closing = attacker.speed - defender.speed
        desired = desired_gap(max(0., attacker.speed), closing, c)
        bumper_gap = -gap - c.car_length
        react = enabled and gap < 0 and closing > c.closing_threshold and bumper_gap < c.reaction_threshold * desired
        lo, hi = self.rules.interval(defender.s)
        details = dict(gap=gap, closing_speed=closing,
            lateral_separation=attacker.d - defender.d, desired_gap=desired,
            legal_lo=lo, legal_hi=hi, reacting=bool(react))
        if lo > hi:
            return self.target, 'stop:no_legal_interval', False, details
        future_lo, future_hi = self.rules.planning_interval(defender, c.defender_speed)
        if future_lo > future_hi:
            return self.target, 'stop:no_preview_legal_interval', False, details
        nominal = float(np.clip(0., future_lo, future_hi))
        candidates = [nominal, float(np.clip(self.target, future_lo, future_hi))]
        if react:
            candidates += list(np.linspace(future_lo, future_hi, 5))
            candidates.append(float(np.clip(attacker.d, future_lo, future_hi)))
        safe = []
        for target in sorted(set(candidates)):
            next_target = rate_limit(self.target, target, c.offset_rate, c.timestep)
            if not lo <= next_target <= hi:
                continue
            if not self.rules.path_safe(defender, target, c.defender_speed, command_target=next_target):
                continue
            if not (predicted_separation_safe(defender, attacker, target, self.line, c)
                    and predicted_separation_safe(defender, attacker, next_target, self.line, c)):
                continue
            # MOBIL-inspired utility: occupy the approaching passing corridor,
            # penalize movement, only change when improvement clears a threshold.
            score = (-abs(target - attacker.d) if react else -abs(target - nominal)) - c.move_penalty * abs(target - self.target)
            safe.append((score, target))
        if not safe:
            return self.target, 'stop:no_safe_reachable_target', False, details
        score, desired_target = max(safe)
        held = next((u for u, d in safe if abs(d - self.target) < 1e-8), None)
        if held is not None and score - held < c.usefulness_threshold:
            desired_target = self.target
        previous = self.target
        self.target = rate_limit(previous, desired_target, c.offset_rate, c.timestep)
        active = react and abs(desired_target - nominal) > .02
        details['active'] = bool(active)
        reason = 'defend:useful_safe_shift' if active else ('hold:no_useful_shift' if react else 'nominal:legal_raceline')
        return self.target, reason, True, details


def pursuit(state, line, offset, speed, previous_steer, cfg):
    """Adapted from waypoint_follow.get_actuation; actual target distance used."""
    point = line.sample(state.s + cfg.lookahead, offset)
    delta = point - [state.x, state.y]
    lateral = -np.sin(state.yaw) * delta[0] + np.cos(state.yaw) * delta[1]
    steer = np.arctan2(2 * cfg.wheelbase * lateral, max(float(delta @ delta), 1e-6))
    steer = np.clip(steer, -cfg.max_steer, cfg.max_steer)
    return rate_limit(previous_steer, steer, cfg.max_steer_rate, cfg.timestep), max(0., speed), point
