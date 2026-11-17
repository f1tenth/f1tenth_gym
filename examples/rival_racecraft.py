"""Interpretable, simulation-only racecraft policy over existing model outputs.

This module changes neither trajectories nor opponent predictions. It grants at
most one early lane choice per encounter; the robust controller still owns every
map, dynamics, region and separation check. These are conservative racecraft
heuristics, not F1/IAC rules certification or a continuous-time safety proof.
"""
import math

import numpy as np

from rival_geometry import ClosedRaceline
from rival_strategy import desired_gap


def trajectory_headings(line, s):
    """Batch the exact outgoing-segment heading lookup, including lap wrap.

    Custom lines and overridden heading methods retain their scalar semantics.
    Each rollout has its own projected s; headings cannot be shared across paths.
    """
    s = np.asarray(s, dtype=float)
    if isinstance(line, ClosedRaceline) and getattr(line.heading, '__func__', None) is ClosedRaceline.heading:
        if not np.all(np.isfinite(s)):
            raise ValueError('s must be finite')
        indices = np.minimum(np.searchsorted(line._starts, s % line.length, side='right') - 1,
                             len(line._points) - 1)
        edges = line._edges[indices]
        return np.arctan2(edges[:, 1], edges[:, 0])
    return np.array([line.heading(float(value)) for value in s])


class RacecraftLayer:
    """Stateful lane permission, utility and model-output rejection.

    ``committed_bias`` is relative to the existing smooth legal reference, not
    an opponent-following target. It is written only by ``selected`` after a
    feasible lane wins. Late/corner/overlap locks survive noisy or missing tracks
    and defense toggles. Only continuously observed, clearly separated dwell
    ends an encounter. Recovery away from the opponent never grants another move.
    """

    def __init__(self, line, envelope, vehicle, config):
        self.line, self.envelope = line, envelope
        self.vehicle, self.config = vehicle, config
        self.encounter = 0
        self.active = False
        self.committed_bias = None
        self.proposed_bias = None
        self.move_used = False
        self.lock = ''
        self.phase = 'idle'
        self.reason = 'no passing encounter'
        self.hold_bias = 0.
        self.opponent_side = 0.
        self.overlap = False
        self.protect_line = False
        self.enabled = False
        self.fresh = False
        self._clear_since = None
        self._clear_kind = None
        self._target = 0.
        self._move_direction = 0.
        self._overlap_anchor = None

    def _reference(self, s):
        reference = (self.envelope.reference if callable(getattr(type(self.envelope), 'reference', None))
                     else self.envelope.nominal)
        return float(reference(s))

    def _reference_path(self, path):
        offsets = getattr(self.envelope, 'reference_offsets', None)
        if isinstance(offsets, np.ndarray):
            return np.interp(path[:, 4] % self.line.length, self.envelope.s, offsets,
                             period=self.line.length)
        return np.array([self._reference(s) for s in path[:, 4]])

    def _move_time(self, target, goal, lateral_rate):
        # Budget a conservative stop-then-translate-and-settle maneuver. Keeping
        # the signed stopping drift accounts for momentum away from the lane.
        r = self.config
        stop_time = abs(lateral_rate) / r.max_lateral_acceleration
        stopped_target = target + lateral_rate * stop_time / 2
        distance = abs(goal - stopped_target)
        if distance <= r.max_lateral_rate ** 2 / r.max_lateral_acceleration:
            translation_time = 2 * math.sqrt(distance / r.max_lateral_acceleration)
        else:
            translation_time = distance / r.max_lateral_rate + r.max_lateral_rate / r.max_lateral_acceleration
        return stop_time + translation_time

    def _lock(self, reason, hold_bias):
        if not self.lock:
            self.lock = reason
            self.hold_bias = float(hold_bias)

    def update(self, t, ego, estimate, enabled, target, nominal, threat=None, reliable=True,
               lateral_rate=0.):
        """Observe validated estimates at actuation cadence; never infer a reset
        from elapsed dropout time. ``threat`` optionally supplies the controller's
        rear approach gate. Return no commands; permissions require rollouts.
        """
        c, r = self.vehicle, self.config
        self._target = float(target)
        self.enabled = bool(enabled)
        self.fresh = reliable and estimate is not None and estimate.valid and estimate.fresh
        if not self.fresh:
            self._clear_since = self._clear_kind = None
            self.phase, self.reason = 'stale', 'no tactical movement; encounter memory retained'
            return
        q = estimate.state
        uncertainty = r.position_margin + r.uncertainty_multiplier * estimate.position_sigma
        gap = self.line.signed_distance(q.s, ego.s)
        measured_closing = q.speed - ego.speed
        if threat is None:
            threat = gap < 0 and measured_closing > c.closing_threshold and (
                -gap - c.car_length < c.reaction_threshold * desired_gap(max(0., q.speed), measured_closing, c))
        closing = measured_closing + r.uncertainty_multiplier * estimate.velocity_sigma
        distance = max(0., -gap - c.car_length - uncertainty)
        acceleration = max(0., estimate.longitudinal_acceleration)
        contact_time = (2 * distance / (closing + math.sqrt(closing ** 2 + 2 * acceleration * distance))
                        if acceleration > 0 and distance > 0 else
                        distance / closing if closing > 0 else math.inf)
        if distance == 0 and gap <= 0:
            contact_time = 0.
        overlap = abs(gap) <= c.car_length + c.separation_margin + uncertainty + r.racecraft_overlap_margin

        # A retreat must be unambiguously non-closing; a pass must have cleared
        # the complete car plus headway. A threshold wobble restarts the dwell.
        clear_kind = ('behind' if gap < -max(r.racecraft_reset_gap, c.car_length + c.minimum_gap) - uncertainty
                      and closing <= c.closing_threshold else
                      'passed' if gap > c.car_length + c.minimum_gap + uncertainty else None)
        if self.active and clear_kind is not None:
            if self._clear_kind != clear_kind:
                self._clear_since, self._clear_kind = t, clear_kind
            if t - self._clear_since >= r.racecraft_reset_dwell:
                self.active = self.move_used = False
                self.committed_bias = self.proposed_bias = None
                self.lock = ''
                self.overlap = False
                self.opponent_side = 0.
                self._overlap_anchor = None
                self._move_direction = 0.
                self._clear_since = self._clear_kind = None
        else:
            self._clear_since = self._clear_kind = None

        if not self.active and (threat or overlap):
            self.active = True
            self.encounter += 1
            self.hold_bias = float(target - nominal)
        if not self.active:
            self.phase, self.reason = ('idle', 'no passing encounter') if enabled else ('disabled', 'defense disabled; recovery only')
            return

        side = float(np.sign(q.d - ego.d))
        # Once overlapping, do not flip the protected side on lateral noise.
        if not self.overlap and abs(q.d - ego.d) > uncertainty:
            self.opponent_side = side
        if overlap:
            self.overlap = True
            self._lock('overlap', ego.d - nominal)
            body_bias = ego.d - self._reference(ego.s)
            if self._overlap_anchor is None or (body_bias - self._overlap_anchor) * self.opponent_side < 0:
                # Follow a checked retreat, never replenish the toward-side
                # tolerance by re-anchoring at an already squeezed position.
                self._overlap_anchor = body_bias

        preview = (max(0., ego.speed) * r.racecraft_corner_time
                   + max(0., ego.speed) ** 2 / (2 * r.max_braking) + c.lookahead)
        curvatures = [self.line.curvature(ego.s + ds) for ds in np.linspace(0., preview, 9)]
        corner = max(abs(k) for k in curvatures) >= r.racecraft_corner_curvature
        if self.proposed_bias is None and not self.lock and enabled and not corner:
            # Pick one lane once: inside a distant bend, otherwise the initially
            # threatened side. Never extrapolate/chase subsequent opponent d.
            bend = self.line.curvature(ego.s + 2 * preview)
            direction = np.sign(bend) if abs(bend) >= r.racecraft_corner_curvature else self.opponent_side
            direction = direction or 1.
            lo, hi = self.envelope.interval(ego.s + c.lookahead)
            lane = float(np.clip(nominal + direction * r.racecraft_lane_offset, lo, hi)) if lo <= hi else nominal
            if abs(lane - target) >= max(r.racecraft_min_move, 1e-6):
                self.proposed_bias = lane - nominal
            else:
                self._lock('hold', target - nominal)

        lane_bias = self.committed_bias if self.move_used else self.proposed_bias
        lo, hi = self.envelope.interval(ego.s + c.lookahead)
        goal = float(np.clip(nominal + lane_bias, lo, hi)) if lane_bias is not None and lo <= hi else target
        move_time = self._move_time(target, goal, lateral_rate) + r.racecraft_reaction_buffer + estimate.age
        late = contact_time < max(r.min_defense_ttc, move_time)
        # Preserve encounter locks but apply line protection only in actual
        # conflict. A distant opponent must not strand ordinary corner recovery.
        self.protect_line = overlap or (gap < 0 and late)
        if corner:
            self._lock('corner', target - nominal)
        elif late:
            self._lock('late', target - nominal)
        if not enabled:
            self.phase, self.reason = 'disabled', 'defense disabled; commitment retained; recovery only'
        elif self.overlap:
            self.phase, self.reason = 'overlap', 'stable line; protect occupied side and leave room'
        elif self.lock:
            self.phase, self.reason = self.lock, 'no new block after ' + self.lock + ' commitment'
        elif self.move_used:
            self.phase, self.reason = 'committed', 'one early lane selected; no opponent-lane chasing'
        else:
            self.phase, self.reason = 'early', 'one early lane available subject to all model checks'

    @property
    def tactical_bias(self):
        if not self.active or not self.enabled or not self.fresh or self.lock:
            return None
        return self.committed_bias if self.move_used else self.proposed_bias

    def utility(self, bias):
        lane = self.tactical_bias
        return -abs(bias - lane) if lane is not None else -abs(bias)

    def accepts(self, rollout, ego, estimate, nominal, target, predictions, inflation):
        """Filter model outputs, including residual motion, on either side.

        Returning False only removes feasibility; it cannot make an unsafe model
        path safe. The caller retains rejected paths for an honest emergency
        minimax fallback if every sampled lane is unresolved.
        """
        c, r = self.vehicle, self.config
        path = rollout.trajectory
        lane = self.tactical_bias
        tactical = lane is not None and abs(rollout.bias - lane) < 1e-5
        lo, hi = self.envelope.interval(ego.s + c.lookahead)
        if lo > hi:
            # There is no legal pursuit goal here. Do not clip against an
            # inverted band or invent a reference that could permit a lane hop.
            return False
        goal = float(np.clip(nominal + rollout.bias, lo, hi))
        if self.active and (not self.lock or self.protect_line) and not tactical:
            # Pursuit goal and body d live at different stations. Classify a
            # lane change relative to each station's reference, not its bend.
            # Use the clamped goal: raw bias need not be physically requested.
            goal_bias = goal - nominal
            current_bias = ego.d - self._reference(ego.s)
            toward = (goal_bias - current_bias) * self.opponent_side
            held = abs(goal - target) <= r.racecraft_line_tolerance
            if toward > r.racecraft_line_tolerance and not held:
                return False
        if self.active and self.protect_line and (self.lock or not self.enabled or not self.fresh):
            if self.opponent_side:
                body_bias = path[:, 5] - self._reference_path(path)
                current_bias = ego.d - self._reference(ego.s)
                anchor = current_bias if self._overlap_anchor is None else self._overlap_anchor
                protected_bias = min(anchor * self.opponent_side, current_bias * self.opponent_side)
                approach = np.max(body_bias * self.opponent_side - protected_bias)
                if approach > r.racecraft_line_tolerance:
                    return False

        if not len(predictions):
            return True
        return self._room_safe(path, predictions, inflation)

    def _room_safe(self, path, predictions, inflation):
        c, r = self.vehicle, self.config
        # Evaluate room at every modeled longitudinal overlap, not just a lane
        # endpoint. Local normal projections use the existing predicted poses.
        headings = trajectory_headings(self.line, path[:, 4])
        tangent = np.column_stack((np.cos(headings), np.sin(headings)))
        normal = np.column_stack((-np.sin(headings), np.cos(headings)))
        relative = predictions[:, :, :2] - path[None, :, :2]
        along = np.sum(relative * tangent[None, :, :], axis=2)
        across = np.sum(relative * normal[None, :, :], axis=2)
        ego_angle = path[:, 2] - headings
        other_angle = predictions[:, :, 2] - headings[None, :]
        extent = (c.car_length * (np.abs(np.cos(ego_angle))[None, :] + np.abs(np.cos(other_angle)))
                  + c.car_width * (np.abs(np.sin(ego_angle))[None, :] + np.abs(np.sin(other_angle)))) / 2
        overlapping = np.abs(along) <= extent + inflation[None, :] + c.separation_margin
        if not np.any(overlapping):
            return True
        # Same exhaustive per-node room checks as the scalar loop. Batch the
        # periodic bands to avoid repeated interpolation/sorting in fast-profile
        # rejection-heavy solves; custom envelopes retain their scalar API.
        batch = callable(getattr(type(self.envelope), 'intervals', None))
        if batch:
            lo, hi = self.envelope.intervals(path[:, 4])
        else:
            bands = np.array([self.envelope.interval(s) for s in path[:, 4]])
            lo, hi = bands[:, 0], bands[:, 1]
        required = c.car_width + c.separation_margin + np.asarray(inflation) + r.racecraft_room_margin
        has_room = hi - lo >= required
        too_close = np.abs(across) < required[None, :]
        no_left_room = (across > 0) & (hi - path[:, 5] < required)[None, :]
        no_right_room = (across < 0) & (path[:, 5] - lo < required)[None, :]
        return not bool(np.any(overlapping & has_room[None, :]
                              & (too_close | no_left_room | no_right_room)))

    def selected(self, rollout, ego, nominal):
        """Commit only a feasible winner; safety recovery consumes no new move."""
        lane = self.tactical_bias
        if not rollout.feasible:
            return
        if lane is not None and abs(rollout.bias - lane) < 1e-5:
            if not self.move_used:
                self._move_direction = float(np.sign(nominal + lane - self._target))
            self.committed_bias = float(lane)
            self.move_used = True
            self.phase, self.reason = 'committed', 'one early model-checked lane committed'
        elif self.move_used and not self.lock:
            lo, hi = self.envelope.interval(ego.s + self.vehicle.lookahead)
            goal = float(np.clip(nominal + rollout.bias, lo, hi)) if lo <= hi else self._target
            changed_command = abs(goal - self._target) > self.config.racecraft_line_tolerance
            body_bias = rollout.trajectory[:, 5] - self._reference_path(rollout.trajectory)
            reversed_path = np.min((body_bias - body_bias[0]) * self._move_direction) < -self.config.racecraft_line_tolerance
            # Goal error may already be zero while a checked correction or
            # inherited steering reverses the body. Neither re-arms the lane.
            if changed_command or reversed_path:
                self._lock('recovery', rollout.bias)
                if self.enabled and self.fresh:
                    self.phase, self.reason = 'recovery', 'checked correction; defensive move remains spent'
