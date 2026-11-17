"""Opt-in numeric backend for the *fast* Adversary.preview (not Gym physics).

Call compiled_preview with the same arguments as policy.preview, plus policy as
its first argument. None means unsupported: the caller must use scalar preview.
No policy state is changed. Supply _defender_paths once per plan when possible.
Map certification and deferred swept SAT use the existing geometry routines.
No fastmath, reduced hypotheses, coarser grid, or ideal Frenet motion is used.
"""
import math

import numpy as np
from numba import njit

from rival_control import advance_speed_command
from rival_rollout import _clip, _interval, _profile_limit, _rate, _segment


@njit(cache=True)
def _demand(s, projected_speed, speed, nodes, slopes, period, lookahead,
            offset_rate, speed_nodes, squared, response):
    slope = 0.
    for ds in (0., lookahead, 2 * lookahead):
        value, _ = _interval(s + ds, nodes, slopes, slopes, period)
        slope = max(slope, abs(value))
    cap = .8 * offset_rate / slope if slope > 1e-9 else math.inf
    return min(speed, cap, _profile_limit(s, projected_speed, speed_nodes, squared, response))


@njit(cache=True)
def _advance(speed, acceleration, demand, dt, motion, corner_tracking=False):
    a, b, jerk, _ = motion
    if corner_tracking:
        return advance_speed_command(speed, acceleration, demand, dt, a, b, jerk)
    desired = _clip((demand - speed) / dt, -b, a)
    acceleration = _rate(acceleration, desired, jerk, dt)
    release = jerk * dt
    braking = max(0., math.sqrt(release ** 2 + 2 * jerk * speed) - release)
    acceleration = max(acceleration, -braking)
    return max(0., speed + acceleration * dt), acceleration


@njit(cache=True)
def _numeric(points, edges, lengths, starts, tangents, nodes, lower, upper,
             reference, slopes, speed_nodes, squared, response, ego, controls,
             velocity, bias, speed, horizon, c, motion, rules, initial_error,
             recovery, clipped, defenders, relative_s, corner_tracking=False):
    """Return path summaries and the first non-map failure.

    Columns: x,y,yaw,projected speed,s,d,progress,max error,current error,clipped.
    Codes: 0=end, 1=body, 2=Frenet collision, 3=node lateral acceleration,
    4=target, 5=next-command lateral acceleration, 6=invalid nominal (raises).
    A code-5 result keeps the old state despite reporting tau+dt, like Python.
    """
    dt, lookahead, wheelbase, max_steer, steer_rate, offset_rate, length, width, separation = c
    margin, tolerance, allowance = rules
    s, d, projected, x, y, yaw = ego
    target, command_speed, acceleration, steering = controls
    progress = 0.
    previous_d = d
    previous_gap = np.full(len(defenders), relative_s)
    maximum = initial_error
    count = math.ceil(horizon / dt) + 1
    path = np.empty((count, 10))
    code = 0
    failure_time = horizon
    index = 0
    for index in range(count):
        tau = index * dt
        lo, hi = _interval(s, nodes, lower, upper, starts[-1])
        error = max(lo - d, d - hi, 0.)
        maximum = max(maximum, error)
        path[index] = x, y, yaw, projected, s, d, progress, maximum, error, (1. if clipped else 0.)
        allowed = initial_error + allowance if recovery else margin + tolerance
        if lo > hi or error > allowed:
            code, failure_time = 1, tau
            break
        collision = False
        for h in range(len(defenders)):
            gap = relative_s - defenders[h, index, 5]
            old_d = previous_d - defenders[h, max(0, index - 1), 4]
            new_d = d - defenders[h, index, 4]
            lateral = min(old_d, new_d) <= width + separation and max(old_d, new_d) >= -width - separation
            longitudinal = min(previous_gap[h], gap) <= length + separation and max(previous_gap[h], gap) >= -length - separation
            collision = collision or (lateral and longitudinal)
        if collision:
            code, failure_time = 2, tau
            break
        if velocity ** 2 * abs(math.tan(steering)) / wheelbase > motion[3] + 1e-9:
            code, failure_time = 3, tau
            break
        if tau >= horizon - 1e-9:
            break
        previous_d = d
        for h in range(len(defenders)):
            previous_gap[h] = relative_s - defenders[h, index, 5]
        target_s = s + lookahead
        tlo, thi = _interval(target_s, nodes, lower, upper, starts[-1])
        if tlo > thi:
            # nominal() precedes interval() in the reference and raises here.
            code, failure_time = 6, tau
            break
        nominal, _ = _interval(target_s, nodes, reference, reference, starts[-1])
        raw = nominal + bias
        desired = _clip(raw, tlo, thi)
        clipped = clipped or abs(desired - raw) > tolerance
        target = _rate(target, desired, offset_rate, dt)
        if not tlo - tolerance <= target <= thi + tolerance:
            code, failure_time = 4, tau
            break
        demand = _demand(s, projected, speed, nodes, slopes, starts[-1], lookahead,
                         offset_rate, speed_nodes, squared, response)
        command_speed, acceleration = _advance(command_speed, acceleration, demand, dt, motion, corner_tracking)
        i, fraction = _segment(target_s, starts, lengths)
        px = points[i, 0] + fraction * edges[i, 0] - target * tangents[i, 1]
        py = points[i, 1] + fraction * edges[i, 1] + target * tangents[i, 0]
        dx, dy = px - x, py - y
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        desired_steer = _clip(math.atan2(2 * wheelbase * lateral, max(dx * dx + dy * dy, 1e-6)), -max_steer, max_steer)
        steering = _rate(steering, desired_steer, steer_rate, dt)
        new_velocity = _rate(velocity, command_speed, motion[0] if command_speed >= velocity else motion[1], dt)
        if max(velocity ** 2, new_velocity ** 2) * abs(math.tan(steering)) / wheelbase > motion[3] + 1e-9:
            code, failure_time = 5, tau + dt
            break
        mean_velocity = .5 * (velocity + new_velocity)
        yaw_rate = mean_velocity * math.tan(steering) / wheelbase
        mid_yaw = yaw + .5 * yaw_rate * dt
        x = x + mean_velocity * math.cos(mid_yaw) * dt
        y = y + mean_velocity * math.sin(mid_yaw) * dt
        yaw = yaw + yaw_rate * dt
        old_s = s
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
        distance = (s - old_s + starts[-1] / 2) % starts[-1] - starts[-1] / 2
        relative_s += distance
        progress += distance
        i, _ = _segment(s, starts, lengths)
        projected = new_velocity * math.cos(yaw - math.atan2(edges[i, 1], edges[i, 0]))
        velocity = new_velocity
    return path[:index + 1], code, failure_time, clipped


def compiled_preview(policy, attacker, defender, line, envelope, track, cfg, goal, speed,
                     *, previous_steering=0., measured_forward_speed=None, nominal=False,
                     body_recovery=False, defender_paths=None, maneuver=None, horizon=None):
    """Return Preview, or None for legacy/custom geometry, profile, or policy.

    Exact concrete types intentionally preserve scalar overrides in mocks and
    subclasses. Pass the four full _defender_paths hypotheses, unchanged, or let
    this helper compute them. Arrays/configuration are read afresh per call;
    there is no cross-plan cache, policy mutation, or implicit backend enablement.
    Invalid nominal bands raise ValueError in the same evaluation order as the
    reference. A returned Preview is final, including infeasible outcomes.
        Explicit horizon/maneuver requests use scalar fallback; this backend handles
        only the original four-hypothesis short candidate preview.
    """
    from rival_adversary import Adversary, Preview, advance_speed
    from rival_geometry import ClosedRaceline, TrackMap
    from rival_robust import TrackEnvelope
    from rival_speed import SpeedProfile
    from rival_strategy import rate_limit

    if (maneuver is not None or horizon is not None
            or type(policy) is not Adversary or type(line) is not ClosedRaceline
            or type(envelope) is not TrackEnvelope or type(track) is not TrackMap
            or type(policy.speed_profile) is not SpeedProfile
            or policy.motion_config is None or envelope.line is not line):
        return None
    dt = cfg.timestep
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(policy.preview_horizon) or policy.preview_horizon < 0:
        return None
    count = math.ceil(policy.preview_horizon / dt) + 1
    if defender_paths is not None and (
            not isinstance(defender_paths, np.ndarray) or defender_paths.dtype != np.float64
            or defender_paths.shape != (4, count, 6) or not np.all(np.isfinite(defender_paths))):
        return None
    cosine = math.cos(attacker.yaw - line.heading(attacker.s))
    if measured_forward_speed is None:
        measured_forward_speed = attacker.speed / math.copysign(max(abs(cosine), .2), cosine)
    bias = 0. if nominal else goal - envelope.nominal(attacker.s + cfg.lookahead)
    raw = envelope.nominal(attacker.s + cfg.lookahead) + bias
    lo, hi = envelope.interval(attacker.s + cfg.lookahead)
    first_goal = float(np.clip(raw, lo, hi)) if lo <= hi else raw
    clipped = abs(first_goal - raw) > policy.legal_numeric_tolerance
    if defender_paths is None:
        defender_paths = policy._defender_paths(defender, line, envelope, cfg)
    profile, motion = policy.speed_profile, policy.motion_config
    demand = min(speed, policy.reference_speed_cap(attacker.s, envelope, cfg),
                 profile.limit(attacker.s, attacker.speed))
    first_target = rate_limit(policy.target, first_goal, cfg.offset_rate, dt)
    first_speed, first_acceleration = advance_speed(policy.speed, policy._acceleration, demand, dt, motion)
    lo, hi = envelope.interval(attacker.s)
    initial = max(lo - attacker.d, attacker.d - hi, 0.)
    recovery = body_recovery and initial > policy.tracking_margin + policy.legal_numeric_tolerance
    path, code, tau, final_clipped = _numeric(
        line.points, line._edges, line._lengths, line._starts, line._tangents,
        envelope.s, envelope.lo, envelope.hi, envelope.reference_offsets, envelope._reference_slopes,
        profile._wrapped_s, profile._squared,
        (float(profile.length), float(profile._lookahead), float(profile._response_time),
         float(profile._response_acceleration_distance), float(profile._global_min)),
        tuple(float(getattr(attacker, k)) for k in ('s', 'd', 'speed', 'x', 'y', 'yaw')),
        (float(policy.target), float(policy.speed), float(policy._acceleration), float(previous_steering)),
        float(measured_forward_speed), float(bias), float(speed), float(policy.preview_horizon),
        tuple(float(v) for v in (dt, cfg.lookahead, cfg.wheelbase, cfg.max_steer, cfg.max_steer_rate,
                                cfg.offset_rate, cfg.car_length, cfg.car_width, cfg.separation_margin)),
        tuple(float(v) for v in (motion.max_acceleration, motion.max_braking, motion.max_jerk, motion.max_lateral_accel)),
        (float(policy.tracking_margin), float(policy.legal_numeric_tolerance), float(policy.recovery_error_allowance)),
        float(initial), bool(recovery), bool(clipped), defender_paths,
        line.signed_distance(attacker.s, defender.s), bool(getattr(motion, 'corner_tracking', False)))

    def result(feasible, reason, safe_time, row, goal_clipped):
        maximum, current = float(row[7]), float(row[8])
        return Preview(bool(feasible), reason, float(safe_time), first_target, first_speed,
            float(row[6]), bool(goal_clipped), bool(recovery), initial, maximum, current,
            bool(feasible and recovery and current > policy.legal_numeric_tolerance),
            policy.tracking_margin, maximum <= policy.tracking_margin + policy.legal_numeric_tolerance,
            maximum <= policy.legal_numeric_tolerance, first_acceleration, float(row[3]),
            policy.defender_prediction_model)

    # Body checks precede map checks. Every other numeric failure follows the
    # map check at that node (even a code-5 failure whose reported tau is +dt).
    map_count = len(path) - int(code == 1)
    safe = track.footprints_safe(path[:map_count, :3], cfg.car_length, cfg.car_width, cfg.boundary_margin)
    failed = np.flatnonzero(~safe)
    if len(failed):
        index = int(failed[0])
        return result(False, 'map_or_legal_preview:footprint', index * dt, path[index], path[index, 9])
    if code == 6:
        envelope.nominal(path[-1, 4] + cfg.lookahead)
        raise AssertionError('Invalid nominal band did not raise')
    if code:
        if code == 1:
            reason = ('body_band_recovery_error_increased' if recovery
                      else 'map_or_legal_preview:strict_body_band' if policy.tracking_margin == 0.
                      else 'map_or_legal_preview:body_tracking_budget_exceeded')
        else:
            reason = {2: 'sampled_collision_preview:evolving_defender',
                      3: 'lateral_acceleration_preview', 4: 'map_or_legal_preview:lookahead_target',
                      5: 'lateral_acceleration_preview'}[code]
        return result(False, reason, tau, path[-1], final_clipped)
    gaps = policy._rectangle_gaps(path[:, :3], defender_paths, cfg)
    unsafe = np.argwhere(gaps <= 0.)
    if len(unsafe):
        hypothesis, index = min(unsafe, key=lambda pair: pair[1])
        # SAT is deferred: summaries describe the terminal state, not this tau.
        return result(False, 'sampled_rectangle_collision_preview:' + policy.defender_hypotheses[hypothesis],
                      index * dt, path[-1], final_clipped)
    if recovery:
        current = path[-1, 8]
        reentered = current <= policy.legal_numeric_tolerance
        if not reentered and initial - current < min(.005, .2 * initial):
            return result(False, 'body_band_recovery_insufficient_descent', policy.preview_horizon, path[-1], final_clipped)
        reason = 'body_band_recovered' if reentered else 'body_band_descent_partial'
    else:
        reason = 'sampled_xy_path'
    return result(True, reason, policy.preview_horizon, path[-1], final_clipped)
