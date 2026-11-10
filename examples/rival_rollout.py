"""Numeric trajectory prediction only; map/opponent checks stay in the caller.

The controls' initial target is also the fixed invalid-envelope goal fallback.
Times are integration nodes (normally the parent's .02 second physics grid).
No fastmath: projection ties and command-state arithmetic matter here.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _clip(x, lo, hi):
    return min(max(x, lo), hi)


@njit(cache=True)
def _rate(current, desired, rate, dt):
    return current + _clip(desired - current, -rate * dt, rate * dt)


@njit(cache=True)
def _segment(s, starts, lengths):
    s = s % starts[-1]
    i = min(np.searchsorted(starts, s, side='right') - 1, len(lengths) - 1)
    return i, (s - starts[i]) / lengths[i]


@njit(cache=True)
def _interval(s, nodes, lo, hi, period):
    s = s % period
    i = np.searchsorted(nodes, s, side='right') - 1
    j = (i + 1) % len(nodes)
    if i < 0:
        i = len(nodes) - 1
        left, right = nodes[i] - period, nodes[j]
    else:
        left = nodes[i]
        right = nodes[j] + (period if j == 0 else 0.)
    fraction = (s - left) / (right - left)
    return (lo[i] + fraction * (lo[j] - lo[i]),
            hi[i] + fraction * (hi[j] - hi[i]))


@njit(cache=True)
def _simulate(points, edges, lengths, starts, tangents, curvatures,
              nodes, lower, upper, reference, slopes, ego, controls, bias, desired_speed, times, c, r):
    s, d, projected_speed, x, y, yaw = ego
    target, lateral_rate, acceleration, steering, command_speed = controls
    initial_target = target
    i, _ = _segment(s, starts, lengths)
    alignment = math.cos(yaw - math.atan2(edges[i, 1], edges[i, 0]))
    if abs(alignment) > .25:
        forward_speed = max(0., projected_speed / alignment)
    else:
        forward_speed = max(0., command_speed, abs(projected_speed))
    path = np.empty((len(times), 7))
    path[0] = x, y, yaw, forward_speed, s, d, target
    lo, hi = _interval(s, nodes, lower, upper, starts[-1])
    initial_error = max(0., lo - d, d - hi)
    violation = maxlat = 0.
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        preview_s = s + c[0]
        plo, phi = _interval(preview_s, nodes, lower, upper, starts[-1])
        nominal, _ = _interval(preview_s, nodes, reference, reference, starts[-1])
        goal = _clip(nominal + bias, plo, phi) if plo <= phi else initial_target
        error = goal - target
        desired_rate = np.sign(error) * min(r[3], math.sqrt(max(0., 2 * r[4] * abs(error))))
        lateral_rate = _rate(lateral_rate, desired_rate, r[4], dt)
        target = target + lateral_rate * dt
        curvature = 0.
        slope = 0.
        for ds in (0., c[0], 2 * c[0]):
            reference_slope, _ = _interval(s + ds, nodes, slopes, slopes, starts[-1])
            slope = max(slope, abs(reference_slope))
            i, f = _segment(s + ds, starts, lengths)
            curvature = max(curvature, abs((1 - f) * curvatures[i] + f * curvatures[(i + 1) % len(points)]))
        turn_speed = math.sqrt(r[5] / max(curvature, 1e-3))
        slope_speed = .8 * r[3] / max(slope, 1e-6)
        desired_accel = _clip((min(desired_speed, turn_speed, slope_speed) - command_speed) / max(dt, 1e-6), -r[1], r[0])
        acceleration = _rate(acceleration, desired_accel, r[2], dt)
        release = r[2] * dt
        braking_limit = max(0., math.sqrt(release ** 2 + 2 * r[2] * command_speed) - release)
        acceleration = max(acceleration, -braking_limit)
        command_speed = max(0., command_speed + acceleration * dt)
        i, f = _segment(s + c[0], starts, lengths)
        px = points[i, 0] + f * edges[i, 0] - target * tangents[i, 1]
        py = points[i, 1] + f * edges[i, 1] + target * tangents[i, 0]
        dx, dy = px - x, py - y
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        desired_steer = _clip(math.atan2(2 * c[1] * lateral, max(dx * dx + dy * dy, 1e-6)), -c[2], c[2])
        steering = _rate(steering, desired_steer, c[3], dt)
        speed = max(0., forward_speed + _clip(command_speed - forward_speed, -r[1] * dt, r[0] * dt))
        average_speed = (forward_speed + speed) / 2
        yaw_delta = average_speed / c[1] * math.tan(steering) * dt
        x = x + average_speed * math.cos(yaw + yaw_delta / 2) * dt
        y = y + average_speed * math.sin(yaw + yaw_delta / 2) * dt
        yaw = yaw + yaw_delta
        best = math.inf
        for j in range(len(points)):
            rx, ry = x - points[j, 0], y - points[j, 1]
            f = _clip((rx * edges[j, 0] + ry * edges[j, 1]) / lengths[j] ** 2, 0., 1.)
            rx, ry = rx - f * edges[j, 0], ry - f * edges[j, 1]
            distance = rx ** 2 + ry ** 2
            if distance < best:
                best = distance
                d = tangents[j, 0] * ry - tangents[j, 1] * rx
                s = (starts[j] + f * lengths[j]) % starts[-1]
        i, _ = _segment(s, starts, lengths)
        projected_speed = speed * math.cos(yaw - math.atan2(edges[i, 1], edges[i, 0]))
        forward_speed = speed
        path[k] = x, y, yaw, speed, s, d, target
        lo, hi = _interval(s, nodes, lower, upper, starts[-1])
        plo, phi = _interval(s + c[0], nodes, lower, upper, starts[-1])
        tolerance = r[6] + max(0., initial_error - .1 * times[k])
        violation = max(violation, lo - d - tolerance, d - hi - tolerance,
                        plo - target - tolerance, target - phi - tolerance)
        if lo > hi or plo > phi:
            violation = max(violation, 1.)
        maxlat = max(maxlat, speed ** 2 * abs(math.tan(steering)) / c[1])
    return path, violation, maxlat


def simulate_trajectory(line, envelope, ego, controls, bias, desired_speed, times,
                        vehicle_cfg, robust_cfg):
    """Return (Nx7 path, legal violation, maximum lateral acceleration).

    Accepts actual ClosedRaceline/TrackEnvelope data; no Python callbacks enter
    the compiled loop. Configuration is packed per call to honor mutable configs.
    """
    c, r = vehicle_cfg, robust_cfg
    return _simulate(line.points, line._edges, line._lengths, line._starts,
                     line._tangents, line._curvatures, envelope.s, envelope.lo,
                     envelope.hi, envelope.reference_offsets, envelope._reference_slopes,
                     (ego.s, ego.d, ego.speed, ego.x, ego.y, ego.yaw),
                     tuple(float(v) for v in controls), float(bias), float(desired_speed),
                     np.asarray(times, dtype=np.float64),
                     (c.lookahead, c.wheelbase, c.max_steer, c.max_steer_rate),
                     (r.max_acceleration, r.max_braking, r.max_jerk,
                      r.max_lateral_rate, r.max_lateral_acceleration,
                      r.max_lateral_accel, r.lateral_tracking_margin))
