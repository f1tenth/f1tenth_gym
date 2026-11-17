"""Optional exact numeric counterparts of rival_robust's defender checks.

No controller dispatch, raster queries, policy state, or reduced sampling here.
Wrappers return None for unsupported inputs so the parent can keep its reference
path. Configuration and geometry arrays are read afresh on every call. All JIT
kernels disable fastmath; warm them before measuring control deadlines.
"""
import math
import sys

import numpy as np
from numba import njit

from rival_geometry import ClosedRaceline
from rival_perception import OpponentEstimate
from rival_rollout import _clip, _segment
from rival_strategy import CarState


@njit(cache=True, fastmath=False)
def _hypotheses(points, edges, lengths, starts, tangents, times, state, limits):
    s0, d0, measured, velocity_sigma, measured_acceleration, measured_lateral = state
    braking, max_acceleration, max_speed, max_lateral, multiplier = limits
    uncertainty = multiplier * velocity_sigma
    speed_limit = max(max_speed, max(0., measured))
    longitudinal = ((max(0., measured - uncertainty), -braking),
                    (max(0., measured), _clip(measured_acceleration, -braking, max_acceleration)),
                    (max(0., measured + uncertainty), max_acceleration))
    lateral_rates = (-max_lateral, _clip(measured_lateral, -max_lateral, max_lateral), max_lateral)
    result = np.empty((9, len(times), 3))
    for case in range(3):
        initial_speed, acceleration = longitudinal[case]
        for side in range(3):
            lateral = lateral_rates[side]
            for index in range(len(times)):
                t = times[index]
                v0 = min(initial_speed, speed_limit)
                bound = 0. if acceleration < 0 else speed_limit
                change_time = t if acceleration == 0 else min(t, max(0., (bound - v0) / acceleration))
                speed = _clip(v0 + acceleration * change_time, 0., speed_limit)
                distance = v0 * change_time + .5 * acceleration * change_time ** 2 + speed * (t - change_time)
                s, d = s0 + distance, d0 + lateral * t
                i, fraction = _segment(s, starts, lengths)
                x = points[i, 0] + fraction * edges[i, 0] + d * (-tangents[i, 1])
                y = points[i, 1] + fraction * edges[i, 1] + d * tangents[i, 0]
                yaw = math.atan2(edges[i, 1], edges[i, 0]) + math.atan2(lateral, max(.2, speed))
                result[3 * case + side, index] = x, y, yaw
    return result


def generate_hypotheses(estimate, line, times, cfg):
    """Return (predictions, inflation), or None for custom geometry/estimates.

    Matches opponent_hypotheses: acceleration-major/lateral-minor ordering, all
    nine cases and supplied times, measured overspeed bound, and unchanged tube.
    Stale *valid* estimates still produce hypotheses. None/invalid estimates
    return the reference's empty (0,N,3) predictions and zero inflation. Times
    must be finite float64 values (a list of Python floats is also accepted).
    """
    if type(line) is not ClosedRaceline:
        return None
    if estimate is not None and type(estimate) is not OpponentEstimate:
        return None
    times = np.asarray(times)
    if times.dtype != np.float64 or times.ndim != 1 or not np.all(np.isfinite(times)):
        return None
    if estimate is None or not estimate.valid:
        return np.empty((0, len(times), 3)), np.zeros(len(times))
    if type(estimate.state) is not CarState:
        return None
    q = estimate.state
    state = tuple(float(v) for v in (q.s, q.d, q.speed, estimate.velocity_sigma,
                                    estimate.longitudinal_acceleration, estimate.lateral_speed))
    limits = tuple(float(v) for v in (cfg.max_braking, cfg.max_opponent_acceleration,
                                     cfg.max_opponent_speed, cfg.max_opponent_lateral_speed,
                                     cfg.uncertainty_multiplier))
    if not all(math.isfinite(v) for v in (*state, *limits, line.length)):
        return None
    inflation = (cfg.position_margin + cfg.uncertainty_multiplier * estimate.position_sigma
                 + cfg.lateral_tracking_margin * times)
    if len(times) == 0:
        # The reference converts nine empty Python trajectories to shape (9,0).
        return np.empty((9, 0)), inflation
    predictions = _hypotheses(line.points, line._edges, line._lengths, line._starts,
                             line._tangents, times, state, limits)
    return predictions, inflation


@njit(cache=True, fastmath=False)
def _sat_gap(x, y, eh, oh, length, width):
    gap = -math.inf
    for angle in (eh, eh + np.pi / 2, oh, oh + np.pi / 2):
        projected = abs(x * math.cos(angle) + y * math.sin(angle))
        ego_radius = length / 2 * abs(math.cos(angle - eh)) + width / 2 * abs(math.sin(angle - eh))
        other_radius = length / 2 * abs(math.cos(angle - oh)) + width / 2 * abs(math.sin(angle - oh))
        gap = max(gap, projected - ego_radius - other_radius)
    return gap


@njit(cache=True, fastmath=False)
def _separation(path, predictions, inflation, length, width, separation_margin, tracking_margin):
    radius = math.hypot(length, width) / 2
    minimum = math.inf
    for h in range(len(predictions)):
        previous_sweep = 0.
        for index in range(len(path)):
            dx = predictions[h, index, 0] - path[index, 0]
            dy = predictions[h, index, 1] - path[index, 1]
            gap = _sat_gap(dx, dy, path[index, 2], predictions[h, index, 2], length, width)
            next_sweep = 0.
            if index + 1 < len(path):
                nx = predictions[h, index + 1, 0] - path[index + 1, 0]
                ny = predictions[h, index + 1, 1] - path[index + 1, 1]
                tx, ty = nx - dx, ny - dy
                translation = math.sqrt(tx * tx + ty * ty) / 2
                opponent_turn = ((predictions[h, index + 1, 2] - predictions[h, index, 2] + np.pi)
                                 % (2 * np.pi) - np.pi)
                ego_turn = (path[index + 1, 2] - path[index, 2] + np.pi) % (2 * np.pi) - np.pi
                rotation = radius * (abs(opponent_turn) + abs(ego_turn)) / 2
                next_sweep = translation + rotation
            padding = max(previous_sweep, next_sweep)
            adjusted = gap - inflation[index] - padding - separation_margin - tracking_margin
            minimum = min(minimum, adjusted)
            previous_sweep = next_sweep
    return minimum


def trajectory_separation(path, predictions, inflation, c, r):
    """Return _rollout's min_separation, or None for unsupported numeric inputs.

    Accepts finite float64 Nx(>=3) path and HxNx(>=3) predictions, with N inflation
    values. Every supplied hypothesis/node is checked. Empty hypotheses return
    +inf. SAT axes, subtraction order, short-angle rotation and both-endpoint
    half-step padding match _rollout; no time-grid assumptions are introduced.
    """
    if (not isinstance(path, np.ndarray) or path.dtype != np.float64
            or path.ndim != 2 or path.shape[1] < 3
            or not isinstance(predictions, np.ndarray) or predictions.dtype != np.float64
            or predictions.ndim != 3 or predictions.shape[1] != len(path) or predictions.shape[2] < 3):
        return None
    if len(predictions) == 0:
        return math.inf
    if (len(path) == 0 or not isinstance(inflation, np.ndarray)
            or inflation.dtype != np.float64 or inflation.shape != (len(path),)
            or not np.all(np.isfinite(path[:, :3])) or not np.all(np.isfinite(predictions[:, :, :3]))
            or not np.all(np.isfinite(inflation))):
        return None
    parameters = tuple(float(v) for v in (c.car_length, c.car_width, c.separation_margin, r.lateral_tracking_margin))
    if not all(math.isfinite(v) for v in parameters):
        return None
    return float(_separation(path, predictions, inflation, *parameters))


@njit(cache=True, fastmath=False)
def _progress(stations, period, compensated):
    total, correction = 0., 0.
    for index in range(1, len(stations)):
        distance = (stations[index] - stations[index - 1] + period / 2) % period - period / 2
        updated = total + distance
        if compensated:
            if abs(total) >= abs(distance):
                correction += (total - updated) + distance
            else:
                correction += (distance - updated) + total
        total = updated
    if compensated and correction != 0. and math.isfinite(correction):
        return total + correction
    return total


def trajectory_progress(path, line):
    """Return sequential signed-distance progress, or None for custom inputs.

    Reads path[:,4] in the same order as _rollout's generator. CPython >=3.12
    uses compensated float sum; retain that rather than a NumPy/tree reduction.
    Empty and single-node paths return 0., as the empty reference sum does.
    """
    if (type(line) is not ClosedRaceline or not isinstance(path, np.ndarray)
            or path.dtype != np.float64 or path.ndim != 2 or path.shape[1] < 5
            or not np.all(np.isfinite(path[:, 4])) or not math.isfinite(line.length)):
        return None
    return float(_progress(path[:, 4], line.length, sys.version_info >= (3, 12)))
