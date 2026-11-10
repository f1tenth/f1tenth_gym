"""Simulation-only RIVAL observations; no real-world safety claims.

SimulatedSensor is a truth-derived synthetic detector, NOT real perception.
Tracker uncertainty is a heuristic standard-deviation scale, not a calibrated
confidence bound. Speeds are signed raceline-longitudinal metres/second.
"""
from collections import deque
from dataclasses import dataclass, fields
import math

import numpy as np

from rival_strategy import CarState


@dataclass
class SensorConfig:
    noise_std: float = .03
    speed_noise_std: float = .10
    latency: float = .10
    update_period: float = .10
    dropout: float = .05
    max_age: float = .5
    max_speed: float = 8.
    max_acceleration: float = 3.
    max_lateral_speed: float = .8

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{field.name} must be finite and nonnegative')
        if self.update_period <= 0 or self.max_speed <= 0:
            raise ValueError('update_period and max_speed must be positive')
        if self.dropout > 1:
            raise ValueError('dropout must be in [0, 1]')


@dataclass
class Measurement:
    timestamp: float
    x: float
    y: float
    speed: float | None = None


@dataclass
class OpponentEstimate:
    state: CarState
    age: float
    position_sigma: float
    velocity_sigma: float
    longitudinal_acceleration: float
    lateral_speed: float
    fresh: bool
    valid: bool


def _clock(t, previous):
    if not np.isfinite(t) or (previous is not None and t < previous):
        raise ValueError('Time must be finite and nondecreasing')
    return float(t)


def _clip(value, bound):
    return float(np.clip(value, -bound, bound))


class SimulatedSensor:
    """Sample supplied truth at most once per period, then queue delivery.

    Missed calls are not backfilled with current truth at historical timestamps.
    Repeated times cannot acquire twice. No interpolation or future truth is used.
    """
    def __init__(self, line, config, seed):
        self.line, self.config = line, config
        self._rng = np.random.default_rng(seed)
        self._pending = deque()
        self._time = self._acquired = None

    def update(self, t, opponent_state):
        t = _clock(t, self._time)
        self._time = t
        c = self.config
        if self._acquired is None or t - self._acquired >= c.update_period - 1e-12:
            self._acquired = t
            if self._rng.random() >= c.dropout:
                values = [opponent_state.x, opponent_state.y, opponent_state.speed]
                if not np.all(np.isfinite(values)):
                    raise ValueError('Synthetic truth must be finite')
                xy = np.asarray(values[:2]) + self._rng.normal(0, c.noise_std, 2)
                speed = values[2] + self._rng.normal(0, c.speed_noise_std)
                self._pending.append((t + c.latency,
                                      Measurement(t, float(xy[0]), float(xy[1]), float(speed))))
        result = None
        while self._pending and self._pending[0][0] <= t:
            _, result = self._pending.popleft()
        return result


class OpponentTracker:
    """Bounded alpha-beta-style filter in wrapped raceline coordinates.

    Reject future, duplicate, out-of-order and already stale acquisitions. A
    rejected input never refreshes the track. fresh means age <= max_age, not
    'a packet arrived this call'; valid means an initialized finite prediction.
    Stale predictions remain valid but must not be mistaken for observations.
    Calls must have nondecreasing current time (equal times are supported).
    """
    def __init__(self, line, config):
        self.line, self.config = line, config
        self._time = self._stamp = None
        self._s = self._d = self._v = self._a = self._lateral = 0.
        self._raw = None

    def update(self, t, measurement):
        t = _clock(t, self._time)
        self._time = t
        c = self.config
        m = measurement
        if m is not None:
            values = [m.timestamp, m.x, m.y]
            if m.speed is not None:
                values.append(m.speed)
            acceptable = (np.all(np.isfinite(values)) and m.timestamp <= t
                          and t - m.timestamp <= c.max_age
                          and (self._stamp is None or m.timestamp > self._stamp)
                          and (m.speed is None or abs(m.speed) <= c.max_speed
                               + 6 * c.speed_noise_std))
            if acceptable:
                s, d = self.line.project([m.x, m.y])
                dt = None if self._stamp is None else m.timestamp - self._stamp
                if dt is not None:
                    ds = self.line.signed_distance(s, self._raw[0])
                    dd = d - self._raw[1]
                    # Gates use raw measurements, so filter lag cannot mask jumps.
                    noise = 6 * c.noise_std
                    acceptable = (abs(ds) <= c.max_speed * dt + noise + 1e-9
                                  and abs(dd) <= c.max_lateral_speed * dt + noise + 1e-9
                                  and math.hypot(m.x - self._raw[2], m.y - self._raw[3])
                                  <= math.hypot(c.max_speed, c.max_lateral_speed) * dt
                                  + noise + 1e-9)
                if acceptable:
                    if dt is None:
                        self._s, self._d = s, d
                        self._v = _clip(m.speed or 0., c.max_speed)
                    else:
                        velocity = _clip(ds / dt, c.max_speed)
                        if m.speed is not None:
                            velocity = .5 * (velocity + _clip(m.speed, c.max_speed))
                        old_v = self._v
                        self._v += _clip(.6 * (velocity - self._v), c.max_acceleration * dt)
                        self._v = _clip(self._v, c.max_speed)
                        self._a = _clip(.5 * self._a + .5 * (self._v - old_v) / dt,
                                        c.max_acceleration)
                        self._lateral = _clip(.5 * self._lateral + .5 * dd / dt,
                                              c.max_lateral_speed)
                        predicted_s = self._s + old_v * dt
                        self._s = (predicted_s + .7 * self.line.signed_distance(s, predicted_s)) % self.line.length
                        self._d = .3 * (self._d + self._lateral * dt) + .7 * d
                    self._stamp = float(m.timestamp)
                    self._raw = (s, d, m.x, m.y)
        if self._stamp is None:
            return None
        age = t - self._stamp
        # Integrate bounded acceleration only until the speed limit is reached.
        horizon = age
        if self._a != 0:
            limit = math.copysign(c.max_speed, self._a)
            horizon = min(age, max(0., (limit - self._v) / self._a))
        v = _clip(self._v + self._a * horizon, c.max_speed)
        distance = self._v * horizon + .5 * self._a * horizon ** 2 + v * (age - horizon)
        s = (self._s + distance) % self.line.length
        d = self._d + self._lateral * age
        x, y = self.line.sample(s, d)
        state = CarState(s, d, v, float(x), float(y), self.line.heading(s))
        velocity_sigma = c.speed_noise_std + c.max_acceleration * age
        position_sigma = c.noise_std + c.speed_noise_std * age + .5 * (
            c.max_acceleration + c.max_lateral_speed) * age ** 2
        return OpponentEstimate(state, age, position_sigma, velocity_sigma,
                                self._a, self._lateral, age <= c.max_age, True)


class LidarDetector:
    """Heuristic dynamic-cluster candidate detector, not opponent identification.

    Scan is ordered from -fov/2 to +fov/2 inclusive, in the ego sensor frame,
    with the sensor at ego (x,y,yaw). Default 4.7 rad leaves a rear blind sector.
    Only visible first returns are used: occluded/rear objects cannot be detected.
    Static-map subtraction may miss objects near walls; unmodelled obstacles can
    be false positives. No car truth, synthetic noise, or latency is added here.
    """
    def __init__(self, line, track, config, fov=4.7):
        if not np.isfinite(fov) or not 0 < fov <= 2 * np.pi:
            raise ValueError('fov must be in (0, 2*pi]')
        self.line, self.track, self.config, self.fov = line, track, config, fov
        self._time = None
        self._previous = None

    def detect(self, t, ego: CarState, scan):
        t = _clock(t, self._time)
        if self._time == t:
            return None
        self._time = t
        ranges = np.asarray(scan, dtype=float)
        if ranges.ndim != 1:
            raise ValueError('scan must be one-dimensional')
        if len(ranges) < 2:
            return None
        if not np.all(np.isfinite([ego.x, ego.y, ego.yaw])):
            raise ValueError('ego pose must be finite')
        indices = np.flatnonzero(np.isfinite(ranges) & (ranges > 0))
        angles = ego.yaw + np.linspace(-self.fov / 2, self.fov / 2, len(ranges))[indices]
        points = np.column_stack((np.cos(angles), np.sin(angles))) * ranges[indices, None]
        points += [ego.x, ego.y]
        free = self.track.clear_many(points) > max(.08, 3 * self.config.noise_std)
        indices, points = indices[free], points[free]
        if len(points) < 2:
            return None
        breaks = np.flatnonzero((np.diff(indices) != 1)
                               | (np.linalg.norm(np.diff(points, axis=0), axis=1) > .3)) + 1
        candidates = []
        for cluster in np.split(points, breaks):
            if len(cluster) >= 2 and np.linalg.norm(np.ptp(cluster, axis=0)) <= 1.:
                candidates.append(np.mean(cluster, axis=0))
        if not candidates:
            return None
        if self._previous is not None and t - self._previous[0] <= self.config.max_age:
            dt = t - self._previous[0]
            candidates = [p for p in candidates if np.linalg.norm(p - self._previous[1])
                          <= self.config.max_speed * dt + .5 + 6 * self.config.noise_std]
            reference = self._previous[1]
        else:
            reference = np.array([ego.x, ego.y])
        if not candidates:
            return None
        point = min(candidates, key=lambda p: np.linalg.norm(p - reference))
        self._previous = (t, point)
        return Measurement(t, float(point[0]), float(point[1]))
