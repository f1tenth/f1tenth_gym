"""Feasibility regressions and optional saved-truth replay, never Gym physics.

Run unit tests with ``python -m unittest discover -s examples -p
 test_rival_feasibility.py``. Add ``--compare --stride 1`` to replay for the
three-way original baseline, previous refinement, and corrected-policy replay. Run a local ignored CSV with
``python examples/test_rival_feasibility.py --replay PATH.csv --stride 4``.
Use ``--compare-coverage --stride 1`` for the current hazard-candidate change,
with ``--all-solves`` to include successful and assumption-gated contexts.
The CSV must contain action-time truth and emitted command/filter derivatives;
its sibling JSON supplies the original settings and sensor seed. Counterfactual
plans never feed back into the saved ego/opponent trajectories. Encounter state
is reconstructed from timestamped estimates plus logged selected commitments.
This is a fixed-state feasibility test, NOT a matched lap or pass-rate test.
"""
import copy
import csv
import json
from collections import Counter
from types import MethodType
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from rival_geometry import ClosedRaceline, TrackMap
from rival_perception import SensorConfig, SimulatedSensor, OpponentTracker, OpponentEstimate
from rival_racecraft import RacecraftLayer, trajectory_headings
from rival_robust import (Decision, RobustConfig, RobustDefender, TrackEnvelope,
                          _Rollout, opponent_hypotheses)
from rival_strategy import CarState, Settings


def replay_states(csv_path, stride=4, emergency_only=True):
    """Reconstruct original observations/filters before each sampled solve.

    Restore the previous row's emitted controls, not this row's (post-step)
    controls. Read original float timestamps without rounding: delivery uses
    exact ``<=`` comparisons. Verify both acquisition age and XY estimate error.
    Only logged solve rows are sampled; skipped rows still update the tracker
    and encounter memory. No simulator is imported or instantiated.
    """
    import yaml

    csv_path = Path(csv_path)
    metadata = json.loads(csv_path.with_suffix('.json').read_text())
    vehicle = Settings(**metadata['settings'])
    config = RobustConfig(**metadata['robust'])
    sensor_config = SensorConfig(**metadata['sensor'])
    source = Path(__file__).parent / 'config_example_map.yaml'
    track_config = yaml.safe_load(source.read_text())
    data = np.loadtxt(source.parent / track_config['wpt_path'],
                      delimiter=track_config['wpt_delim'], skiprows=track_config['wpt_rowskip'])
    line = ClosedRaceline(data[:, [track_config['wpt_xind'], track_config['wpt_yind']]])
    if abs(line.length - metadata['track']['length']) > 1e-9:
        raise ValueError('Replay supports the example-map raceline only')
    track = TrackMap(source.parent / track_config['map_path'], track_config['map_ext'])
    envelope = TrackEnvelope(line, track, vehicle, config)
    controller = RobustDefender(line, track, vehicle, config,
        envelope.nominal(vehicle.start_s + vehicle.lookahead), envelope)
    sensor = SimulatedSensor(line, sensor_config, vehicle.seed)
    tracker = OpponentTracker(line, sensor_config)
    previous = None
    sample = 0
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            t = float(row['time'])
            ego, opponent = [CarState(**{key: float(row[prefix + key])
                for key in CarState.__dataclass_fields__})
                for prefix in ('ego_truth_eval_', 'opponent_truth_eval_')]
            estimate = tracker.update(t, sensor.update(t, opponent))
            assert (estimate is not None) == (row['estimate_available'] == 'True')
            if estimate is not None:
                assert abs(estimate.age - float(row['estimate_age'])) < 1e-12
                error = np.hypot(estimate.state.x - opponent.x, estimate.state.y - opponent.y)
                assert abs(error - float(row['estimate_error'])) < 1e-10
            if previous is not None:
                for attr, key in (('target', 'target'), ('_lateral_rate', 'ego_target_rate'),
                                  ('_acceleration', 'ego_command_acceleration'),
                                  ('_steering', 'steering'), ('_speed_command', 'speed')):
                    setattr(controller, attr, float(previous[key]))
            nominal = envelope.nominal(ego.s + vehicle.lookahead)
            broken = estimate is not None and (
                abs(estimate.state.speed) > config.max_opponent_speed
                or abs(estimate.lateral_speed) > config.max_opponent_lateral_speed
                or estimate.longitudinal_acceleration > config.max_opponent_acceleration
                or estimate.longitudinal_acceleration < -config.max_braking)
            reliable = not broken and abs(np.cos(ego.yaw - line.heading(ego.s))) >= .5
            controller._assumption_broken = not reliable
            controller.racecraft.update(t, ego, estimate, reliable, controller.target,
                nominal, reliable=reliable, lateral_rate=controller._lateral_rate)
            if row['planning_solve'] == 'True' and (not emergency_only or row['mode'] == 'emergency'):
                if sample % stride == 0:
                    trial = copy.copy(controller)
                    trial.racecraft = copy.copy(controller.racecraft)
                    yield trial, t, ego, estimate, row
                sample += 1
            # Restore *original* selected memory, not the counterfactual winner.
            craft = controller.racecraft
            if row['racecraft_committed_bias']:
                if not craft.move_used:
                    craft._move_direction = float(np.sign(nominal
                        + float(row['racecraft_committed_bias']) - controller.target))
                craft.committed_bias = float(row['racecraft_committed_bias'])
                craft.move_used = row['racecraft_move_used'] == 'True'
            if row['racecraft_locked'] == 'True' and not craft.lock:
                craft._lock(row['racecraft_phase'], controller.target - nominal)
            previous = row


def replay_report(csv_path, stride=4, probe=False, emergency_only=True):
    totals = Counter()
    masks = Counter()
    timings = []
    for controller, t, ego, estimate, row in replay_states(csv_path, stride, emergency_only):
        paths = []
        original = controller._rollout

        def capture(*args):
            path = original(*args)
            paths.append(path)
            return path

        controller._rollout = capture
        controller._solve(t, ego, estimate, not controller._assumption_broken,
                          reliable=not controller._assumption_broken)
        totals['states'] += 1
        totals['modeled_feasible'] += int(controller._best.feasible)
        totals['emergency'] += int(controller._mode == 'emergency')
        timings.append(controller._planning_ms)
        for path in paths:
            mask = ''.join(letter for letter, key in (
                ('M', 'map_safe'), ('L', 'legal_safe'), ('S', 'separation_safe'),
                ('R', 'racecraft_safe')) if not getattr(path, key)) or 'accepted'
            masks[mask] += 1
        if probe and not controller._best.feasible:
            r, c = controller.config, controller.vehicle
            nominal = controller.envelope.nominal(ego.s + c.lookahead)
            lo, hi = controller.envelope.interval(ego.s + c.lookahead)
            times = np.linspace(0., r.horizon,
                int(np.ceil(r.horizon / min(c.timestep, r.prediction_dt))) + 1)
            predictions, inflation = opponent_hypotheses(estimate, controller.line, times, r)
            cruise = min(c.defender_speed, controller.speed_profile.limit(ego.s, ego.speed))
            found = []
            for bias in np.linspace(lo - nominal, hi - nominal, 17):
                for fraction in (1., .85, .7, .5, .35, .2, .1, 0.):
                    path = original(ego, float(bias), cruise * fraction, times, predictions, inflation)
                    path.racecraft_safe = controller.racecraft.accepts(path, ego, estimate,
                        nominal, controller.target, predictions, inflation)
                    if estimate is not None and estimate.valid:
                        gap = controller.line.signed_distance(estimate.state.s, ego.s)
                        if abs(gap) <= c.car_length + c.separation_margin + r.position_margin:
                            direction = np.sign(estimate.state.d - ego.d)
                            if np.max((path.trajectory[:, 5] - ego.d) * direction) > .02:
                                path.separation_safe = False
                    if path.feasible:
                        found.append((round(float(bias), 4), round(cruise * fraction, 4)))
            totals['probe_recovered'] += bool(found)
            if found:
                print('probe recovered', t, found[:5], flush=True)
    print(json.dumps(dict(counts=dict(totals), rejection_conjunctions=dict(masks),
        planning_ms=dict(p50=float(np.percentile(timings, 50)),
                         p95=float(np.percentile(timings, 95)))), indent=2))


def scalar_room_safe(self, path, predictions, inflation):
    """Pre-batching room calculation, independent parity/benchmark reference."""
    c, r = self.vehicle, self.config
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
    for index in np.flatnonzero(np.any(overlapping, axis=0)):
        lo, hi = self.envelope.interval(path[index, 4])
        required = c.car_width + c.separation_margin + inflation[index] + r.racecraft_room_margin
        if hi - lo < required:
            continue
        sides = across[overlapping[:, index], index]
        if np.any(np.abs(sides) < required):
            return False
        if np.any(sides > 0) and hi - path[index, 5] < required:
            return False
        if np.any(sides < 0) and path[index, 5] - lo < required:
            return False
    return True


def legacy_accepts(self, rollout, ego, estimate, nominal, target, predictions, inflation):
    """Exact pre-follow-up policy: retain the mixed-station gate for replay."""
    c, r = self.vehicle, self.config
    path = rollout.trajectory
    lane = self.tactical_bias
    tactical = lane is not None and abs(rollout.bias - lane) < 1e-5
    goal = float(np.clip(nominal + rollout.bias, *self.envelope.interval(ego.s + c.lookahead)))
    if self.active and (not self.lock or self.protect_line) and not tactical:
        toward = (goal - ego.d) * self.opponent_side
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


def compare_report(csv_path, stride=1, emergency_only=True):
    """Compare original baseline, previous refinement, and corrected gate.

    ``before`` retains coarse candidates, scalar room, and the old intent gate;
    ``previous`` retains midpoint refinement/batched room with the old gate;
    ``after`` uses the corrected policy. All use the checkout's same rollout
    and hypotheses. Rotate order to limit warm-up/timing bias. Plans never feed
    back into the saved states or the reconstructed encounter history.
    """
    counts = {label: Counter() for label in ('before', 'previous', 'after')}
    timings = {label: [] for label in counts}
    conjunctions = {label: Counter() for label in counts}
    recovered = []
    regressions = []
    baseline_mismatches = []
    causes = {label: Counter() for label in counts}
    policy_changes = Counter()

    def audit_previous(craft, rollout, ego, estimate, nominal, target, predictions, inflation):
        old = legacy_accepts(craft, rollout, ego, estimate, nominal, target, predictions, inflation)
        new = RacecraftLayer.accepts(craft, rollout, ego, estimate, nominal, target, predictions, inflation)
        policy_changes['new_racecraft_passes'] += int(not old and new)
        policy_changes['new_racecraft_rejections'] += int(old and not new)
        lane = craft.tactical_bias
        tactical = lane is not None and abs(rollout.bias - lane) < 1e-5
        lo, hi = craft.envelope.interval(ego.s + craft.vehicle.lookahead)
        if lo <= hi and craft.active and (not craft.lock or craft.protect_line) and not tactical:
            goal = float(np.clip(nominal + rollout.bias, lo, hi))
            tolerance = craft.config.racecraft_line_tolerance
            held = abs(goal - target) <= tolerance
            old_toward = (goal - ego.d) * craft.opponent_side
            new_toward = ((goal - nominal) - (ego.d - craft._reference(ego.s))) * craft.opponent_side
            if old_toward > tolerance and new_toward <= tolerance and not held:
                policy_changes['mixed_station_rejections_removed'] += 1
                policy_changes['nominal_mixed_station_rejections_removed'] += int(abs(rollout.bias) < 1e-8)
        return old

    followup_recovered = []
    followup_regressions = []
    for index, (snapshot, t, ego, estimate, row) in enumerate(
            replay_states(csv_path, stride, emergency_only)):
        results = {}
        labels = tuple(counts)
        order = labels[index % 3:] + labels[:index % 3]
        for label in order:
            controller = copy.copy(snapshot)
            controller.racecraft = copy.copy(snapshot.racecraft)
            audit_inputs = []
            audit_state = []
            if label in ('before', 'previous'):
                controller._hazard_candidates = lambda *args: ()
            if label == 'previous':
                def record_previous(craft, *args):
                    if not audit_state:
                        audit_state.append(copy.copy(craft))
                    audit_inputs.append(args)
                    return legacy_accepts(craft, *args)

                controller.racecraft.accepts = MethodType(record_previous, controller.racecraft)
            elif label == 'before':
                controller.racecraft.accepts = MethodType(legacy_accepts, controller.racecraft)
            if label == 'before':
                controller._refinement_biases = lambda biases: ()
                controller.racecraft._room_safe = MethodType(scalar_room_safe, controller.racecraft)
            controller._solve(t, ego, estimate, not controller._assumption_broken,
                              reliable=not controller._assumption_broken)
            # Audit identical paths against the pre-selection policy state,
            # outside _solve's measured planning interval.
            for args in audit_inputs:
                audit_previous(audit_state[0], *args)
            diagnostics = controller._candidate_diagnostics
            counts[label]['states'] += 1
            counts[label]['modeled_feasible'] += int(controller._best.feasible)
            counts[label]['decision_feasible'] += int(controller._best.feasible
                and estimate is not None and estimate.fresh and not controller._assumption_broken)
            counts[label]['emergency'] += int(controller._mode == 'emergency')
            counts[label]['candidates'] += diagnostics['candidate_count']
            if not controller._best.feasible:
                for field, name in (('separation', 'every_candidate_fails_separation'),
                                    ('map', 'every_candidate_fails_map'),
                                    ('racecraft', 'every_candidate_fails_racecraft')):
                    if diagnostics['candidate_' + field + '_rejections'] == diagnostics['candidate_count']:
                        causes[label][name] += 1
            for mask, count in enumerate(diagnostics['candidate_rejection_masks']):
                conjunctions[label][mask] += count
            timings[label].append(controller._planning_ms)
            results[label] = controller
        before, after = results['before'], results['after']
        baseline_feasible = (before._best.feasible and estimate is not None
                             and estimate.fresh and not before._assumption_broken)
        if baseline_feasible != (row['feasible'] == 'True'):
            baseline_mismatches.append(round(t, 6))
        if not before._best.feasible and after._best.feasible:
            recovered.append(round(t, 6))
        if before._best.feasible and not after._best.feasible:
            regressions.append(round(t, 6))
        previous = results['previous']
        if not previous._best.feasible and after._best.feasible:
            followup_recovered.append(round(t, 6))
        if previous._best.feasible and not after._best.feasible:
            followup_regressions.append(round(t, 6))
    print(json.dumps(dict(
        counts={label: dict(counts[label]) for label in counts},
        rejection_masks={label: dict(conjunctions[label]) for label in counts},
        planning_ms={label: dict(p50=float(np.percentile(timings[label], 50)),
                                p95=float(np.percentile(timings[label], 95))) for label in counts},
        recovered_times=recovered, feasibility_regressions=regressions,
        baseline_feasibility_mismatches=baseline_mismatches,
        followup_recovered_times=followup_recovered,
        followup_feasibility_regressions=followup_regressions,
        same_candidate_policy_changes=dict(policy_changes),
        remaining_causes={label: dict(causes[label]) for label in counts}), indent=2))


def coverage_compare_report(csv_path, stride=1, emergency_only=True):
    """Paired pre-hazard versus hazard coverage, using the same current helpers.

    Unlike --compare (historical lattice/policy ablation), this disables ONLY
    _hazard_candidates in the before arm. Both keep the current racecraft gate,
    midpoint refinement, compiled hypotheses/SAT/progress, and all margins.
    """
    counts = {label: Counter() for label in ('before', 'after')}
    timings = {label: [] for label in counts}
    masks = {label: Counter() for label in counts}
    recovered, regressions, baseline_mismatches = [], [], []
    fallback_changes, feasible_action_changes = [], []
    baseline_count_mismatches = []
    additional = []
    hazard_without_old_trigger = 0
    for index, (snapshot, t, ego, estimate, row) in enumerate(
            replay_states(csv_path, stride, emergency_only)):
        results = {}
        for label in (('before', 'after') if index % 2 == 0 else ('after', 'before')):
            controller = copy.copy(snapshot)
            controller.racecraft = copy.copy(snapshot.racecraft)
            if label == 'before':
                controller._hazard_candidates = lambda *args: ()
            controller._solve(t, ego, estimate, not controller._assumption_broken,
                              reliable=not controller._assumption_broken)
            decision_feasible = bool(controller._best.feasible and estimate is not None
                and estimate.valid and estimate.fresh and not controller._assumption_broken)
            d = controller._candidate_diagnostics
            counts[label]['states'] += 1
            counts[label]['model_feasible'] += int(controller._best.feasible)
            counts[label]['decision_feasible'] += int(decision_feasible)
            counts[label]['unresolved'] += int(not decision_feasible)
            counts[label]['emergency'] += int(controller._mode == 'emergency')
            counts[label]['candidates'] += d['candidate_count']
            for mask, count in enumerate(d['candidate_rejection_masks']):
                masks[label][mask] += count
            timings[label].append(controller._planning_ms)
            results[label] = controller, decision_feasible
        before, before_feasible = results['before']
        after, after_feasible = results['after']
        if before_feasible != (row['feasible'] == 'True'):
            baseline_mismatches.append(round(t, 6))
        if 'planner_candidate_count' in row and before._candidate_diagnostics['candidate_count'] != int(row['planner_candidate_count']):
            baseline_count_mismatches.append(round(t, 6))
        additional.append(after._candidate_diagnostics['candidate_count']
                          - before._candidate_diagnostics['candidate_count'])
        if before.racecraft.protect_line and estimate is not None:
            gap = before.line.signed_distance(estimate.state.s, ego.s)
            closing = estimate.state.speed - ego.speed
            alongside = abs(gap) <= before.vehicle.car_length + before.vehicle.separation_margin + before.config.position_margin
            old_ttc = ((-gap - before.vehicle.car_length) / closing
                       if gap < 0 and closing > .05 else float('inf'))
            hazard_without_old_trigger += not alongside and old_ttc >= before.config.min_defense_ttc
        if not before_feasible and after_feasible:
            best = after._best
            recovered.append(dict(time=round(t, 6), bias=best.bias, desired_speed=best.desired_speed,
                map=best.map_safe, legal=best.legal_safe, separation=best.separation_safe,
                racecraft=best.racecraft_safe, min_separation=best.min_separation))
        if before_feasible and not after_feasible:
            regressions.append(round(t, 6))
        changed_action = (before._best.bias, before._best.desired_speed) != (after._best.bias, after._best.desired_speed)
        if not before_feasible and not after_feasible and changed_action:
            fallback_changes.append(round(t, 6))
        if before_feasible and changed_action:
            feasible_action_changes.append(round(t, 6))
    report = dict(counts={label: dict(counts[label]) for label in counts},
        rejection_masks={label: dict(masks[label]) for label in counts},
        planning_ms={label: dict(p50=float(np.percentile(timings[label], 50)),
                                p95=float(np.percentile(timings[label], 95))) for label in counts},
        recovered=recovered, feasibility_regressions=regressions,
        baseline_feasibility_mismatches=baseline_mismatches,
        baseline_candidate_count_mismatches=baseline_count_mismatches,
        failed_hazard_fallback_action_changes=fallback_changes,
        previously_feasible_action_changes=feasible_action_changes,
        protect_without_old_trigger=int(hazard_without_old_trigger),
        additional_candidates=dict(total=sum(additional), maximum=max(additional, default=0)))
    print(json.dumps(report, indent=2))
    return report


class MovingReference:
    """Synthetic shifting reference; tests policy outputs, not dynamics."""
    def __init__(self, direction):
        self.s = np.array([0., 40., 80., 120.])
        self.reference_offsets = direction * np.array([0., 1.6, 0., -1.6])

    def reference(self, s):
        return float(np.interp(s % 160., self.s, self.reference_offsets, period=160.))

    nominal = reference

    def interval(self, s):
        return -2., 2.


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        from test_rival_robust import BroadTrack
        self.line = ClosedRaceline([[0, 0], [40, 0], [40, 40], [0, 40]])
        self.vehicle = Settings(legal_min=-3., legal_max=3.)
        self.config = RobustConfig(horizon=.4, region='track', envelope_spacing=10.)
        self.envelope = Mock()
        self.envelope.interval.return_value = (-2., 2.)
        self.envelope.nominal.return_value = 0.
        self.track = BroadTrack()

    def controller(self):
        return RobustDefender(self.line, self.track, self.vehicle, self.config,
                              0., envelope=self.envelope)

    def state(self, s=10., d=0., speed=2.):
        x, y = self.line.sample(s, d)
        return CarState(s, d, speed, float(x), float(y), self.line.heading(s))

    def estimate(self, **kwargs):
        values = dict(state=self.state(1., 1., 1.), age=0., position_sigma=.01,
                      velocity_sigma=.02, longitudinal_acceleration=0.,
                      lateral_speed=0., fresh=True, valid=True)
        values.update(kwargs)
        return OpponentEstimate(**values)

    def reference_fixture(self, side, body_bias=0., envelope=None):
        envelope = envelope or MovingReference(side)
        policy = RacecraftLayer(self.line, envelope, self.vehicle, self.config)
        policy.active = policy.move_used = policy.overlap = policy.protect_line = True
        policy.enabled = policy.fresh = True
        policy.encounter = 1
        policy.lock = 'overlap'
        policy.opponent_side = side
        policy.committed_bias = .45 * side
        policy._overlap_anchor = body_bias
        ego = self.state(d=envelope.reference(10.) + body_bias)
        target = policy._target = ego.d
        nominal = envelope.reference(ego.s + self.vehicle.lookahead)
        positions = np.array([ego.s, ego.s + 1., ego.s + 2.])
        path = np.array([[*self.line.sample(s, envelope.reference(s) + body_bias),
                          self.line.heading(s), ego.speed, s, envelope.reference(s) + body_bias,
                          envelope.reference(s + self.vehicle.lookahead) + body_bias]
                         for s in positions])
        rollout = _Rollout(body_bias, 2., path, True, True, True, 1., 1., 0., 2.)
        return policy, ego, nominal, target, rollout

    def policy_accepts(self, policy, ego, nominal, target, rollout, legacy=False):
        method = legacy_accepts if legacy else RacecraftLayer.accepts
        return method(policy, rollout, ego, self.estimate(), nominal, target,
                      np.empty((0, len(rollout.trajectory), 3)), np.zeros(len(rollout.trajectory)))

    def test_bending_reference_is_not_a_second_block_on_either_side(self):
        for side in (-1., 1.):
            for bias in (0., .1 * side):
                with self.subTest(side=side, bias=bias):
                    policy, ego, nominal, target, rollout = self.reference_fixture(side, bias)
                    self.assertGreater((nominal + bias - ego.d) * side,
                                       self.config.racecraft_line_tolerance)
                    self.assertFalse(self.policy_accepts(policy, ego, nominal, target, rollout, legacy=True))
                    self.assertTrue(self.policy_accepts(policy, ego, nominal, target, rollout))
                    np.testing.assert_allclose(rollout.trajectory[:, 5]
                        - policy._reference_path(rollout.trajectory), bias, atol=1e-15)
                    policy.selected(rollout, ego, nominal)
                    self.assertEqual(policy.lock, 'overlap')
                    self.assertTrue(policy.move_used)
                    self.assertEqual(policy.committed_bias, .45 * side)
                    self.assertIsNone(policy.tactical_bias)

    def test_new_toward_bias_is_rejected_even_when_reference_moves_away(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                envelope = MovingReference(side)
                envelope.s = np.array([0., 10., 11., 80., 120.])
                envelope.reference_offsets = side * np.array([0., 0., -.2, 0., 0.])
                policy, ego, nominal, target, rollout = self.reference_fixture(side, envelope=envelope)
                rollout.bias = .1 * side
                self.assertLess((nominal + rollout.bias - ego.d) * side, -.02)
                # Even a not-yet-moving body cannot authorize a new goal toward
                # the occupied side. The old absolute comparison missed this.
                self.assertTrue(self.policy_accepts(policy, ego, nominal, target, rollout, legacy=True))
                self.assertFalse(self.policy_accepts(policy, ego, nominal, target, rollout))

    def test_clamped_goal_not_unrequested_raw_bias_defines_lane_change(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                policy, ego, nominal, target, rollout = self.reference_fixture(side)
                policy.envelope.interval = lambda s: (-2., nominal) if side > 0 else (nominal, 2.)
                rollout.bias = .3 * side
                self.assertTrue(self.policy_accepts(policy, ego, nominal, target, rollout))

    def test_reference_fix_does_not_enlarge_protected_body_tolerance(self):
        for side in (-1., 1.):
            for approach, accepted in ((.019, True), (.021, False)):
                with self.subTest(side=side, approach=approach):
                    policy, ego, nominal, target, rollout = self.reference_fixture(side)
                    rollout.trajectory[-1, 5] += approach * side
                    self.assertEqual(self.policy_accepts(policy, ego, nominal, target, rollout), accepted)
                    self.assertEqual(policy._overlap_anchor, 0.)
            policy, ego, nominal, target, rollout = self.reference_fixture(side)
            policy._overlap_anchor = -.05 * side
            self.assertFalse(self.policy_accepts(policy, ego, nominal, target, rollout))
            self.assertEqual(policy._overlap_anchor, -.05 * side)

    def test_reference_relative_permission_still_checks_room_predictions(self):
        for side in (-1., 1.):
            policy, ego, nominal, target, rollout = self.reference_fixture(side)
            policy._room_safe = Mock(return_value=False)
            predictions = np.broadcast_to(rollout.trajectory[None, :, :3], (9, 3, 3))
            self.assertFalse(policy.accepts(rollout, ego, self.estimate(), nominal, target,
                                            predictions, np.zeros(3)))
            policy._room_safe.assert_called_once()
            self.assertEqual(policy._room_safe.call_args.args[1].shape, (9, 3, 3))

    def test_alongside_guard_still_rejects_absolute_body_approach_on_bending_reference(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                envelope = MovingReference(side)
                ego = self.state(d=envelope.reference(10.))
                controller = RobustDefender(self.line, self.track, self.vehicle, self.config,
                                            ego.d, envelope)
                controller.racecraft._room_safe = Mock(return_value=True)
                paths = []

                def rollout(state, bias, speed, times, predictions, inflation):
                    positions = np.linspace(state.s, state.s + 2., len(times))
                    path = np.array([[*self.line.sample(s, envelope.reference(s)),
                                      self.line.heading(s), state.speed, s,
                                      envelope.reference(s), envelope.reference(s)] for s in positions])
                    candidate = _Rollout(bias, speed, path, True, True, True, 1., 1., 0., 2.)
                    paths.append(candidate)
                    return candidate

                controller._rollout = rollout
                packet = self.estimate(state=self.state(s=ego.s, d=ego.d + side, speed=2.))
                result = controller.step(0., ego, packet)
                # The policy may permit the constant-relative-bias path, but
                # RobustDefender's stricter actual alongside check still fails.
                self.assertTrue(any(path.racecraft_safe for path in paths))
                self.assertTrue(all(not path.separation_safe for path in paths))
                self.assertFalse(result.feasible)
                self.assertEqual(result.mode, 'emergency')
                self.assertEqual(result.hypothesis_count, 9)

    def test_empty_future_band_never_clips_into_a_new_legal_goal(self):
        controller = self.controller()
        policy = controller.racecraft
        policy.active = policy.protect_line = True
        policy.lock = 'late'
        policy.opponent_side = 1.
        policy._target = controller.target = .3
        self.envelope.interval.return_value = (1., -1.)
        self.envelope.nominal.side_effect = AssertionError('empty band has no nominal')
        ego = self.state(d=.3)
        self.assertEqual(controller._goal(ego, 1.), .3)
        path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, .3], (2, 1))
        rollout = _Rollout(1., 0., path, True, False, True, 1., 1., 1., 0.)
        self.assertFalse(self.policy_accepts(policy, ego, .3, .3, rollout))
        self.envelope.nominal.assert_not_called()
        policy.selected(rollout, ego, .3)
        self.assertEqual(policy._target, .3)
        self.assertEqual(policy.lock, 'late')
        self.assertFalse(policy.move_used)

    def test_legacy_decision_construction_and_json_defaults(self):
        from dataclasses import asdict
        result = Decision(0., 0., 0., 'legacy', '', False, 0., 0., 0., 0)
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.candidate_rejection_masks, (0,) * 16)
        self.assertFalse(result.selected_model_feasible)
        json.dumps(asdict(result), allow_nan=False)

    def test_all_rejections_counted_without_short_circuit(self):
        controller = self.controller()
        calls = []

        def rollout(ego, bias, speed, times, predictions, inflation):
            mask = len(calls) % 16
            calls.append(mask)
            path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, bias], (len(times), 1))
            return _Rollout(bias, speed, path, not mask & 1, not mask & 2,
                            not mask & 4, 1., 1., 0., .2)

        controller._rollout = rollout
        controller.racecraft.accepts = Mock(side_effect=lambda *args: not calls[-1] & 8)
        result = controller.step(0., self.state(), self.estimate(), defense=False)
        self.assertEqual(result.candidate_count, len(calls))
        self.assertEqual(controller.racecraft.accepts.call_count, len(calls))
        self.assertEqual(result.candidate_rejection_masks,
                         tuple(calls.count(mask) for mask in range(16)))
        for bit, field in ((1, 'map'), (2, 'legal'), (4, 'separation'), (8, 'racecraft')):
            self.assertEqual(getattr(result, 'candidate_' + field + '_rejections'),
                             sum(bool(mask & bit) for mask in calls))
        self.assertEqual(result.candidate_feasible_count, calls.count(0))
        self.assertTrue(result.feasible)
        self.assertTrue(result.selected_model_feasible)
        self.assertEqual(result.selected_rejections, ())
        self.assertEqual(result.candidate_freshness_rejections, 0)

    def test_emergency_and_freshness_are_separate_exhaustive_gates(self):
        controller = self.controller()
        controller.track.clear_many = lambda points: np.zeros(len(points))
        controller.track.footprint_safe = lambda *args: False
        result = controller.step(0., self.state(), None)
        self.assertEqual(result.mode, 'emergency')
        self.assertFalse(result.feasible)
        self.assertIn('NOT certified safe', result.reason)
        self.assertIn('map', result.selected_rejections)
        self.assertIn('freshness', result.selected_rejections)
        self.assertEqual(result.candidate_map_rejections, result.candidate_count)
        self.assertEqual(result.candidate_freshness_rejections, result.candidate_count)

    def test_stale_and_assumption_failures_do_not_become_model_success(self):
        for packet, gate in ((None, 'freshness'), (self.estimate(fresh=False), 'freshness'),
                             (self.estimate(longitudinal_acceleration=2.5), 'assumptions')):
            with self.subTest(gate=gate, packet=packet):
                result = self.controller().step(0., self.state(), packet, defense=False)
                self.assertTrue(result.selected_model_feasible)
                self.assertFalse(result.feasible)
                self.assertIn(gate, result.selected_rejections)
                field = 'candidate_assumption_rejections' if gate == 'assumptions' else 'candidate_freshness_rejections'
                self.assertEqual(getattr(result, field), result.candidate_count)

    def test_diagnostics_survive_held_plan_without_counting_a_new_solve(self):
        controller = self.controller()
        first = controller.step(0., self.state(), self.estimate(), defense=False)
        second = controller.step(.02, self.state(), self.estimate(), defense=False)
        self.assertEqual(second.planning_ms, 0.)
        self.assertEqual(first.candidate_rejection_masks, second.candidate_rejection_masks)
        self.assertEqual(first.selected_rejections, second.selected_rejections)

    def test_refinement_is_disabled_on_default_profile(self):
        controller = self.controller()
        self.assertEqual(controller._refinement_biases([-.3, 0., .3]), ())
        controller.speed_profile = Mock()
        self.assertEqual(controller._refinement_biases([-.3, 0., .001, .3]), (-.15, .1505))
        controller.racecraft.committed_bias = .45
        controller.racecraft.move_used = True
        controller.racecraft.lock = 'late'
        self.assertEqual(controller._refinement_biases([-.3, 0., .3]), (-.15, .15, .375))
        self.assertIsNone(controller.racecraft.tactical_bias)
        self.assertTrue(controller.racecraft.move_used)

    def test_refinement_only_resolves_fully_checked_midpoint(self):
        controller = self.controller()
        controller.speed_profile = Mock()
        controller.speed_profile.limit.return_value = self.vehicle.defender_speed
        seen = []

        def rollout(ego, bias, speed, times, predictions, inflation):
            seen.append((bias, speed))
            path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, bias], (len(times), 1))
            safe = abs(bias - .15) < 1e-6
            return _Rollout(bias, speed, path, safe, True, True, 1., 1., 0., .2)

        controller._rollout = rollout
        result = controller.step(0., self.state(), self.estimate(), defense=False)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(controller._best.bias, .15)
        self.assertEqual(result.candidate_refinement_count, 2)
        self.assertEqual(result.candidate_count, len(seen))
        self.assertEqual(result.hypothesis_count, 9)
        # A geometric midpoint is still rejected if permission is absent.
        controller = self.controller()
        controller.speed_profile = Mock()
        controller.speed_profile.limit.return_value = self.vehicle.defender_speed
        controller._rollout = rollout
        controller.racecraft.accepts = Mock(return_value=False)
        result = controller.step(0., self.state(), self.estimate(), defense=False)
        self.assertFalse(result.feasible)
        self.assertEqual(result.mode, 'emergency')
        self.assertIn('racecraft', result.selected_rejections)

    def test_successful_fast_solve_does_not_run_refinement(self):
        controller = self.controller()
        controller.speed_profile = Mock()
        controller.speed_profile.limit.return_value = self.vehicle.defender_speed
        controller._refinement_biases = Mock(side_effect=AssertionError('unexpected refinement'))
        controller._hazard_candidates = Mock(side_effect=AssertionError('unexpected hazard probes'))
        result = controller.step(0., self.state(), self.estimate(), defense=False)
        self.assertTrue(result.feasible)
        self.assertEqual(result.candidate_refinement_count, 0)

    def test_hazard_candidates_use_body_station_and_preserve_filter_state(self):
        for side in (-1., 1.):
            for rate in (-.2, .2):
                with self.subTest(side=side, rate=rate):
                    envelope = MovingReference(side)
                    controller = RobustDefender(self.line, self.track, self.vehicle, self.config,
                                                0., envelope)
                    controller.speed_profile = Mock()
                    controller.racecraft.protect_line = True
                    controller.racecraft.opponent_side = side
                    ego = self.state(d=envelope.reference(10.) + .1 * side)
                    nominal = envelope.reference(ego.s + self.vehicle.lookahead)
                    controller.target = nominal + .2
                    controller._lateral_rate = rate
                    controls = (controller.target, controller._lateral_rate, controller._steering,
                                controller._acceleration, controller._speed_command)
                    candidates = controller._hazard_candidates(ego, nominal, False, float('inf'), [2., 1.4, .7, 0.])
                    self.assertEqual(len(candidates), 2)
                    biases = [bias for bias, speed in candidates if speed == 2.]
                    np.testing.assert_allclose(biases, [-.35 * side], atol=1e-15)
                    self.assertEqual({speed for bias, speed in candidates}, {2., 1.4})
                    self.assertEqual(controls, (controller.target, controller._lateral_rate, controller._steering,
                                               controller._acceleration, controller._speed_command))

    def test_hazard_candidates_are_fast_only_and_require_hazard_and_nonempty_band(self):
        controller = self.controller()
        ego = self.state(d=.1)
        controller.racecraft.protect_line = True
        controller.racecraft.opponent_side = 1.
        self.assertEqual(controller._hazard_candidates(ego, 0., True, 0., [2., 1.4]), ())
        controller.speed_profile = Mock()
        controller.racecraft.protect_line = False
        self.assertEqual(controller._hazard_candidates(ego, 0., False, float('inf'), [2., 1.4]), ())
        self.assertTrue(controller._hazard_candidates(ego, 0., True, float('inf'), [2., 1.4]))
        self.assertTrue(controller._hazard_candidates(ego, 0., False, 1., [2., 1.4]))
        controller.racecraft.opponent_side = 0.
        self.assertEqual(controller._hazard_candidates(ego, 0., True, 0., [2., 1.4]), ())
        controller.racecraft.opponent_side = 1.
        self.envelope.interval.return_value = (1., -1.)
        self.assertEqual(controller._hazard_candidates(ego, 0., True, 0., [2., 1.4]), ())

    def hazard_controller(self):
        controller = self.controller()
        controller.speed_profile = Mock()
        controller.speed_profile.limit.return_value = self.vehicle.defender_speed
        controller.racecraft.active = controller.racecraft.protect_line = True
        controller.racecraft.lock = 'late'
        controller.racecraft.opponent_side = 1.
        # Isolate the candidate trigger from encounter acquisition in these
        # synthetic rollout tests. Literal saved states test the real update.
        controller.racecraft.update = Mock()
        return controller

    def test_hazard_recovery_needs_every_check_and_failed_probes_preserve_fallback(self):
        ego = self.state(d=.17)
        for rejection in ('none', 'map', 'legal', 'separation', 'racecraft'):
            with self.subTest(rejection=rejection):
                controller = self.hazard_controller()

                def rollout(state, bias, speed, times, predictions, inflation):
                    repair = abs(bias - (ego.d - 1.5 * self.config.tracking_bias)) < 1e-12
                    path = np.tile([state.x, state.y, state.yaw, 0., state.s, state.d, bias], (len(times), 1))
                    safe = repair and rejection != 'map'
                    return _Rollout(bias, speed, path, safe, rejection != 'legal',
                        rejection != 'separation', 1., 10. if repair else 1., 0., 2. if repair else .2)

                controller._rollout = rollout
                controller.racecraft.accepts = Mock(return_value=rejection != 'racecraft')
                baseline = copy.copy(controller)
                baseline.racecraft = copy.copy(controller.racecraft)
                baseline._hazard_candidates = lambda *args: ()
                before = baseline.step(0., ego, self.estimate())
                result = controller.step(0., ego, self.estimate())
                self.assertFalse(before.feasible)
                self.assertEqual(result.feasible, rejection == 'none')
                self.assertGreater(result.candidate_count, before.candidate_count)
                self.assertEqual(sum(result.candidate_rejection_masks), result.candidate_count)
                self.assertEqual(result.hypothesis_count, 9)
                if rejection == 'none':
                    self.assertAlmostEqual(controller._best.bias, ego.d - 1.5 * self.config.tracking_bias)
                    self.assertTrue(result.selected_map_safe and result.selected_legal_safe
                        and result.selected_separation_safe and result.selected_racecraft_safe)
                else:
                    self.assertEqual((controller._bias, controller._desired_speed),
                                     (baseline._bias, baseline._desired_speed))
                    self.assertEqual(result.mode, 'emergency')
                    self.assertIn('NOT certified safe', result.reason)

    def test_fast_hazard_probes_do_not_bypass_alongside_body_approach(self):
        controller = self.hazard_controller()
        controller.racecraft.accepts = Mock(return_value=True)
        paths = []

        def rollout(ego, bias, speed, times, predictions, inflation):
            path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, bias], (len(times), 1))
            path[:, 5] += np.linspace(0., .05, len(times))
            candidate = _Rollout(bias, speed, path, True, True, True, 1., 1., 0., .2)
            paths.append(candidate)
            return candidate

        controller._rollout = rollout
        ego = self.state()
        packet = self.estimate(state=self.state(s=ego.s, d=1.5, speed=2.))
        result = controller.step(0., ego, packet)
        self.assertTrue(any(abs(path.bias + .45) < 1e-12 for path in paths))
        self.assertTrue(all(path.racecraft_safe for path in paths))
        self.assertTrue(all(not path.separation_safe for path in paths))
        self.assertEqual(result.candidate_separation_rejections, result.candidate_count)
        self.assertFalse(result.feasible)
        self.assertEqual(result.mode, 'emergency')

    def test_missing_or_bound_violating_estimates_do_not_run_hazard_probes(self):
        for packet in (None, self.estimate(fresh=False),
                       self.estimate(longitudinal_acceleration=2.5)):
            with self.subTest(packet=packet):
                controller = self.hazard_controller()
                controller._hazard_candidates = Mock(side_effect=AssertionError('gated estimate'))

                def rollout(ego, bias, speed, times, predictions, inflation):
                    path = np.tile([ego.x, ego.y, ego.yaw, 0., ego.s, ego.d, bias], (len(times), 1))
                    return _Rollout(bias, speed, path, False, True, False, 0., -1., 0., 0.)

                controller._rollout = rollout
                result = controller.step(0., self.state(), packet)
                self.assertFalse(result.feasible)
                controller._hazard_candidates.assert_not_called()

    def test_periodic_batch_bands_equal_scalar_with_wrap_and_invalid_bands(self):
        envelope = TrackEnvelope(self.line, self.track, self.vehicle, self.config)
        envelope.lo[2], envelope.hi[2] = 1., -1.
        positions = np.array([-160.1, -1., 0., 20., 40., 159.99, 160., 320.1])
        scalar = np.array([envelope.interval(s) for s in positions])
        lo, hi = envelope.intervals(positions)
        np.testing.assert_array_equal(np.column_stack((lo, hi)), scalar)

    def test_room_vectorization_matches_scalar_for_both_sides_wrap_and_overlap(self):
        envelope = TrackEnvelope(self.line, self.track, self.vehicle, self.config)
        policy = RacecraftLayer(self.line, envelope, self.vehicle, self.config)
        rng = np.random.default_rng(12345)
        results = Counter()
        for index in range(200):
            s = np.linspace(159.5 if index % 2 else 8., 160.5 if index % 2 else 12., 11)
            path = np.array([[*self.line.sample(pos, d), self.line.heading(pos) + yaw,
                              2., pos, d, d] for pos, d, yaw in
                             zip(s, rng.uniform(-1.8, 1.8, len(s)), rng.uniform(-.3, .3, len(s)))])
            predictions = np.broadcast_to(path[None, :, :3], (9, len(s), 3)).copy()
            predictions[:, :, :2] += rng.uniform(-3., 3., (9, len(s), 2))
            predictions[:, :, 2] += rng.uniform(-.8, .8, (9, len(s)))
            if index % 4 == 0:
                h = trajectory_headings(self.line, s)
                predictions[:, :, :2] += 10 * np.column_stack((np.cos(h), np.sin(h)))[None, :, :]
            inflation = rng.uniform(0., .6, len(s))
            expected = scalar_room_safe(policy, path, predictions, inflation)
            self.assertEqual(policy._room_safe(path, predictions, inflation), expected)
            results[expected] += 1
        self.assertTrue(results[True])
        self.assertTrue(results[False])


    def test_room_scalar_api_and_exact_thresholds_remain_unchanged(self):
        policy = RacecraftLayer(self.line, self.envelope, self.vehicle, self.config)
        s = np.array([10., 10.2])
        path = np.array([[pos, 0., 0., 2., pos, 0., 0.] for pos in s])
        predictions = np.zeros((9, 2, 3))
        predictions[:, :, 0] = s
        inflation = np.full(2, .1)
        required = (self.vehicle.car_width + self.vehicle.separation_margin
                    + .1 + self.config.racecraft_room_margin)
        outcomes = Counter()
        for sign in (-1., 1.):
            for epsilon in (-1e-10, 0., 1e-10):
                predictions[:, :, 1] = sign * (required + epsilon)
                for width in (required - 1e-10, required, required + 1e-10, 4., -1.):
                    self.envelope.interval.return_value = (-width / 2, width / 2)
                    expected = scalar_room_safe(policy, path, predictions, inflation)
                    self.assertEqual(policy._room_safe(path, predictions, inflation), expected)
                    outcomes[expected] += 1
        self.assertTrue(outcomes[True])
        self.assertTrue(outcomes[False])

    def test_limits_full_saved_states_recover_without_relaxation_or_filter_reset(self):
        import yaml
        source = Path(__file__).parent
        raw = yaml.safe_load((source / 'rival_advanced_fast.yaml').read_text())
        vehicle = Settings.from_dict(raw['settings'])
        # These recorded filter states predate corner-reference feedforward.
        config = RobustConfig.from_dict({**raw['robust'], 'corner_tracking': False})
        track_config = yaml.safe_load((source / raw['track_config']).read_text())
        data = np.loadtxt(source / track_config['wpt_path'], delimiter=track_config['wpt_delim'],
                          skiprows=track_config['wpt_rowskip'])
        line = ClosedRaceline(data[:, [track_config['wpt_xind'], track_config['wpt_yind']]])
        track = TrackMap(source / track_config['map_path'], track_config['map_ext'])
        envelope = TrackEnvelope(line, track, vehicle, config)
        # Literal sensor-pipeline/filter snapshots, independent of ignored CSVs.
        # Tuple order: t, ego, tracked opponent, uncertainty/dynamics, controls, side.
        cases = (
            (62.980000000004246,
             (144.40343082766157, .8974938765721555, 1.3232046786732667,
              -5.8718461251018415, -7.144892176785474, 6.140430338013801),
             (139.49578494937919, 1.0738131828097446, 2.0523937861513435,
              -10.380099014109108, -6.2426145943081774, -.2845137381420327),
             (.18000000000002814, .10956000000002206, .6400000000000844,
              -.29707559668708833, .7176825725846938),
             (.7734446482306124, -.15494141007658113, .7999999999999341,
              .030733187534613383, 1.4020329503879123), -1.),
            (64.62000000000428,
             (146.89112905561385, .6314652723258564, 1.0401144626723153,
              -3.5870343704771375, -7.066917538431668, .11075674312137984),
             (141.92552460363177, 1.070991233312573, 1.4285486259651587,
              -8.130750604063714, -6.776079585997375, -.17559778527223607),
             (.11999999999997613, .06935999999998672, .45999999999992836,
              1.210115928481691, .0641754793856884),
             (.41759964333266064, -.4499999999999116, -1.0799999999995702,
              -.017091505150465588, 1.010160215390196), 1.))
        original_config = vars(config).copy()
        for t, ego_values, opponent_values, tracking, controls, side in cases:
            with self.subTest(t=t):
                ego = CarState(*ego_values)
                estimate = OpponentEstimate(CarState(*opponent_values), *tracking, True, True)
                controller = RobustDefender(line, track, vehicle, config, controls[0], envelope)
                (controller.target, controller._lateral_rate, controller._acceleration,
                 controller._steering, controller._speed_command) = controls
                craft = controller.racecraft
                craft.active = craft.move_used = craft.enabled = craft.fresh = True
                craft.encounter = 1
                craft.committed_bias = craft.proposed_bias = .45
                craft.lock = 'late'
                craft.hold_bias = .0003312495563952897
                craft.opponent_side = side
                craft._move_direction = 1.
                baseline = copy.copy(controller)
                baseline.racecraft = copy.copy(craft)
                baseline._hazard_candidates = lambda *args: ()
                before = baseline.step(t, ego, estimate)
                self.assertFalse(before.feasible)
                self.assertEqual(before.mode, 'emergency')
                self.assertTrue(baseline.racecraft.protect_line)
                gap = line.signed_distance(estimate.state.s, ego.s)
                old_ttc = (-gap - vehicle.car_length) / (estimate.state.speed - ego.speed)
                self.assertGreater(old_ttc, config.min_defense_ttc)
                result = controller.step(t, ego, estimate)
                self.assertTrue(result.feasible)
                self.assertTrue(result.selected_map_safe and result.selected_legal_safe
                    and result.selected_separation_safe and result.selected_racecraft_safe)
                self.assertEqual(result.selected_rejections, ())
                self.assertEqual(result.hypothesis_count, 9)
                self.assertGreater(result.min_separation, 0.)
                self.assertLessEqual(result.candidate_count - before.candidate_count, 2)
                self.assertEqual(craft.lock, 'late')
                self.assertTrue(craft.move_used)
                self.assertEqual(craft.committed_bias, .45)
                self.assertNotEqual(result.mode, 'defend')
                self.assertAlmostEqual(controller._best.bias,
                    ego.d - envelope.reference(ego.s) - 1.5 * side * config.tracking_bias)
                dt = vehicle.timestep
                self.assertLessEqual(abs(result.target_offset - controls[0]), config.max_lateral_rate * dt + 1e-12)
                self.assertLessEqual(abs(controller._lateral_rate - controls[1]), config.max_lateral_acceleration * dt + 1e-12)
                self.assertLessEqual(abs(controller._acceleration - controls[2]), config.max_jerk * dt + 1e-12)
                self.assertLessEqual(abs(result.steering - controls[3]), vehicle.max_steer_rate * dt + 1e-12)
                self.assertLessEqual(controller._desired_speed,
                                    controller.speed_profile.limit(ego.s, ego.speed))
        self.assertEqual(vars(config), original_config)

    def test_saved_fast_state_recovers_real_checked_midpoint_without_csv_dependency(self):
        # Literal action-time snapshot at 22.18 s from fast_headway_trial, seed
        # 12345. This remains a regression when the ignored CSV is unavailable.
        import yaml
        source = Path(__file__).parent
        raw = yaml.safe_load((source / 'rival_advanced_fast.yaml').read_text())
        vehicle = Settings.from_dict(raw['settings'])
        # Replay with the controller that generated the historical snapshot.
        config = RobustConfig.from_dict({**raw['robust'], 'corner_tracking': False})
        track_config = yaml.safe_load((source / raw['track_config']).read_text())
        data = np.loadtxt(source / track_config['wpt_path'], delimiter=track_config['wpt_delim'],
                          skiprows=track_config['wpt_rowskip'])
        line = ClosedRaceline(data[:, [track_config['wpt_xind'], track_config['wpt_yind']]])
        track = TrackMap(source / track_config['map_path'], track_config['map_ext'])
        envelope = TrackEnvelope(line, track, vehicle, config)
        controller = RobustDefender(line, track, vehicle, config, -.261959841822828, envelope)
        ego = CarState(61.305863666576144, -.1796706057236318, 2.0326147327676978,
                       -47.68322333322691, 19.892624916420445, 4.518466955496945)
        opponent = CarState(56.063308696561364, 1.182495611307891, 1.7401013756578698,
                            -44.10585068184441, 23.009431180628027, -2.615346395593584)
        estimate = OpponentEstimate(opponent, .27999999999999403, .20695999999999304,
                                    .9399999999999821, .927157562955438,
                                    .31706964643993524, True, True)
        controller._lateral_rate = -.06355124969565085
        controller._acceleration = 1.3254885507596015
        controller._steering = .1294512181763009
        controller._speed_command = 2.3104114874480426
        craft = controller.racecraft
        craft.active = craft.move_used = True
        craft.encounter = 1
        craft.committed_bias = craft.proposed_bias = .45
        craft.lock = 'late'
        craft.hold_bias = .0003312495563952897
        craft.opponent_side = craft._move_direction = 1.
        baseline = copy.copy(controller)
        baseline.racecraft = copy.copy(craft)
        baseline.racecraft.accepts = MethodType(legacy_accepts, baseline.racecraft)
        baseline._refinement_biases = lambda biases: ()
        baseline._hazard_candidates = lambda *args: ()
        t = 22.179999999999616
        baseline._solve(t, ego, estimate, True)
        self.assertEqual(baseline._mode, 'emergency')
        self.assertFalse(baseline._best.feasible)
        decision = controller.step(t, ego, estimate)
        self.assertTrue(decision.feasible)
        self.assertTrue(decision.selected_map_safe and decision.selected_legal_safe
                        and decision.selected_separation_safe and decision.selected_racecraft_safe)
        self.assertEqual(decision.selected_rejections, ())
        self.assertEqual(decision.hypothesis_count, 9)
        self.assertGreater(decision.candidate_refinement_count, 0)
        self.assertAlmostEqual(controller._best.bias, -.155135, delta=2e-6)
        self.assertEqual(craft.lock, 'late')
        self.assertEqual(craft.committed_bias, .45)
        self.assertNotEqual(decision.mode, 'defend')
        self.assertLessEqual(controller._desired_speed, vehicle.defender_speed)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--all-solves', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--compare-coverage', action='store_true')
    args, rest = parser.parse_known_args()
    if args.replay:
        if args.stride < 1:
            parser.error('--stride must be positive')
        if args.compare_coverage:
            coverage_compare_report(args.replay, args.stride, not args.all_solves)
        elif args.compare:
            compare_report(args.replay, args.stride, not args.all_solves)
        else:
            replay_report(args.replay, args.stride, args.probe, not args.all_solves)
    else:
        unittest.main(argv=[__file__, *rest])
