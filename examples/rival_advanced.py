"""Simulation-only evaluation, not hardware perception or a safety guarantee.

Four-tuple Gym, [steer, speed]. Synchronous CPU wall times are diagnostics:
no wall-time delay is added to simulated sensing, tracking or actuation.
"""
import argparse
import csv
from dataclasses import asdict, replace
from types import SimpleNamespace
import json
from pathlib import Path
import time
import numpy as np
import yaml
from rival_baseline import load, Overlay
from rival_strategy import observe, pursuit, Defender, TrackRules
from rival_perception import SensorConfig, SimulatedSensor, OpponentTracker, LidarDetector
from rival_adversary import Adversary

SCENARIOS = ('no-attack', 'approach', 'left-pass', 'right-pass', 'adaptive',
             'feint', 'surge', 'cut-in', 'brake')


def clean(value):
    """Strict JSON: unknown/nonfinite metrics are null, never Infinity."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class PassMetric:
    """Unwrapped relative progress; report lead onset only after confirmation."""
    def __init__(self, initial_gap, threshold):
        self.initial_gap, self.threshold = initial_gap, threshold
        self.since = self.time = None
        self.lead = -initial_gap

    def update(self, t, progress):
        self.lead = -self.initial_gap + progress[1] - progress[0]
        self.since = (t if self.since is None else self.since) if self.lead > self.threshold else None
        if self.time is None and self.since is not None and t - self.since >= .5 - 1e-9:
            self.time = self.since
            return True
        return False


def attack_outcome(progress, elapsed, collisions, offtrack, requested_steps,
                   executed_steps, unresolved_steps):
    """Attacker invalidity is not a defended attack or a defender failure."""
    failures = []
    if collisions[1]:
        failures.append('attacker_collision')
    if offtrack[1]:
        failures.append('attacker_offtrack')
    if requested_steps and progress[1] < .2 * elapsed:
        failures.append('attacker_stalled')
    if requested_steps and not executed_steps:
        failures.append('no_executed_sampled_feasible_attack')
    if unresolved_steps:
        failures.append('attacker_unresolved_preview')
    return dict(invalid_attack=bool(failures), invalid_attack_reasons=failures,
                defender_failure=bool(collisions[0] or offtrack[0]),
                attacker_failure=bool(failures),
                attacker_physical_failure=bool(collisions[1] or offtrack[1]),
                attack_evaluation_eligible=bool(requested_steps and executed_steps and not failures),
                defended_attack_credit=False,
                note='eligibility is not successful-defense credit; attacker fallback/stall/offtrack must not be scored as defended attacks')


class EvaluationBounds:
    """End-of-step legal-band and truth kinematic checks, not collision tests.

    Legal tolerance is 2 cm; duration uses end-sample rectangular integration.
    Sensor bounds are model assumptions, not calibrated measurement error bounds.
    """
    def __init__(self, dt, tolerance=.02):
        self.dt, self.tolerance = dt, tolerance
        self.duration = np.zeros(2)
        self.max_magnitude = np.zeros(2)
        self.excess_integral = np.zeros(2)
        self.invalid_interval_steps = np.zeros(2, dtype=int)
        self.sensor_counts = dict(speed=0, acceleration=0, lateral_speed=0)

    def update(self, states, previous, envelope, sensor):
        row = {}
        for i, (label, state) in enumerate(zip(('ego', 'attacker'), states)):
            lo, hi = envelope.interval(state.s)
            valid = np.isfinite(lo) and np.isfinite(hi) and lo <= hi
            magnitude = max(lo - state.d, state.d - hi, 0.) if valid else None
            violation = not valid or magnitude > self.tolerance
            self.duration[i] += self.dt * int(violation)
            self.invalid_interval_steps[i] += int(not valid)
            if valid:
                self.max_magnitude[i] = max(self.max_magnitude[i], magnitude)
                self.excess_integral[i] += max(0., magnitude - self.tolerance) * self.dt
            row.update({f'{label}_legal_violation': violation,
                        f'{label}_legal_violation_magnitude': magnitude})
        opponent, old = states[1], previous[1]
        acceleration = (opponent.speed - old.speed) / self.dt
        lateral_speed = (opponent.d - old.d) / self.dt
        checks = dict(speed=abs(opponent.speed) > sensor.max_speed + 1e-9,
                      acceleration=abs(acceleration) > sensor.max_acceleration + 1e-9,
                      lateral_speed=abs(lateral_speed) > sensor.max_lateral_speed + 1e-9)
        for key, exceeded in checks.items():
            self.sensor_counts[key] += int(exceeded)
            row[f'sensor_bound_{key}_violation'] = exceeded
        row.update(opponent_truth_acceleration=acceleration, opponent_truth_lateral_speed=lateral_speed)
        return row

    def summary(self):
        return dict(legal_region=dict(tolerance=self.tolerance,
            violation_duration=self.duration, max_magnitude=self.max_magnitude,
            excess_metre_seconds=self.excess_integral,
            invalid_interval_steps=self.invalid_interval_steps,
            note='per-car [defender, attacker]; measured d versus envelope, separate from bitmap offtrack; end-of-step samples'),
            sensor_bound_violation_steps=self.sensor_counts,
            sensor_bound_note='evaluation truth absolute speed, finite-difference longitudinal acceleration and lateral speed versus SensorConfig bounds; not measurement calibration')


class StepMetrics:
    """Record every applied command; solve timing is independent of row cadence.

    Command derivatives include the initial transition from zero commands.
    Late actuation measures synchronous controller wall time, not OS scheduling.
    """
    def __init__(self, dt, period, targets):
        self.dt, self.period = dt, period
        self.commands = np.zeros((2, 2))
        self.targets = np.asarray(targets, dtype=float)
        self.acceleration = np.zeros(2)
        self.planning = []
        self.missed = self.late = 0
        self.max_speed = np.zeros(2)
        self.max_acceleration = np.zeros(2)
        self.max_jerk = np.zeros(2)
        self.max_target_rate = np.zeros(2)
        self.max_steer_rate = np.zeros(2)
        self.pipeline_times = dict(detection_ms=[], tracking_ms=[], perception_tracking_ms=[],
                                   observation_to_command_ms=[])
        self.physics_deadline_misses = 0

    def pipeline(self, detection_ms, tracking_ms, total_ms):
        values = dict(detection_ms=detection_ms, tracking_ms=tracking_ms,
                      perception_tracking_ms=detection_ms + tracking_ms,
                      observation_to_command_ms=total_ms)
        for name, value in values.items():
            self.pipeline_times[name].append(value)
        miss = total_ms > self.dt * 1000
        self.physics_deadline_misses += int(miss)
        return dict(**values, physics_deadline_miss=miss,
                    physics_deadline_miss_count=self.physics_deadline_misses)

    def latency_summary(self):
        return dict(**{name: dict(p50=np.percentile(values, 50) if values else None,
                                  p95=np.percentile(values, 95) if values else None,
                                  max=max(values) if values else None)
                       for name, values in self.pipeline_times.items()},
                    physics_deadline_misses=self.physics_deadline_misses,
                    physics_deadline_ms=self.dt * 1000,
                    note='synchronous wall time from cycle observation read/conversion through perception, tracker, controller, adversary and command telemetry preparation; excludes env.step/render; no added simulated sensing or actuator latency')

    def update(self, commands, targets, planning_ms, controller_ms):
        commands, targets = np.asarray(commands), np.asarray(targets)
        acceleration = (commands[:, 1] - self.commands[:, 1]) / self.dt
        jerk = (acceleration - self.acceleration) / self.dt
        target_rate = (targets - self.targets) / self.dt
        steer_rate = (commands[:, 0] - self.commands[:, 0]) / self.dt
        solve = bool(np.isfinite(planning_ms) and planning_ms > 0)
        miss = solve and planning_ms > self.period * 1000
        self.planning.extend([planning_ms] if solve else [])
        self.missed += int(miss)
        late = controller_ms > self.dt * 1000
        self.late += int(late)
        for name, values in (('speed', commands[:, 1]), ('acceleration', acceleration),
                             ('jerk', jerk), ('target_rate', target_rate), ('steer_rate', steer_rate)):
            setattr(self, 'max_' + name, np.maximum(getattr(self, 'max_' + name), np.abs(values)))
        self.commands, self.targets, self.acceleration = commands.copy(), targets.copy(), acceleration
        row = dict(planning_solve=solve, planning_budget_miss=bool(miss),
                   planning_budget_miss_count=self.missed, controller_wall_ms=controller_ms,
                   late_actuation=late, late_actuation_count=self.late)
        for i, label in enumerate(('ego', 'attacker')):
            row.update({f'{label}_command_acceleration': acceleration[i],
                        f'{label}_command_jerk': jerk[i], f'{label}_target_rate': target_rate[i],
                        f'{label}_steer_rate': steer_rate[i]})
        return row


class LegacyAdapter:
    """Original Defender/pursuit at its original physics-step cadence.

    Same estimate, initial poses, adversary and sensor as robust. Original
    TrackRules and command shaping intentionally remain different. Missing,
    invalid or stale tracks cause a nominal target and zero speed, never truth.
    """
    def __init__(self, line, track, cfg, initial_target, envelope):
        self.line, self.cfg, self.envelope = line, cfg, envelope
        self.rules = TrackRules(line, track, cfg)
        self.defender = Defender(line, self.rules, cfg, initial_target)
        self.steer = 0.

    def step(self, t, ego, estimate, defense=True):
        start = time.perf_counter()
        if estimate is None or not estimate.valid or not estimate.fresh:
            target = self.envelope.nominal(ego.s)
            self.defender.target = target
            reason, feasible, speed = 'legacy:no_valid_fresh_estimate_nominal_stop', False, 0.
        else:
            target, reason, feasible, _ = self.defender.decide(ego, estimate.state, enabled=defense)
            speed = self.cfg.defender_speed if feasible else 0.
            reason = 'legacy:' + reason
        self.steer, speed, _ = pursuit(ego, self.line, target, speed, self.steer, self.cfg)
        return SimpleNamespace(steering=self.steer, speed=speed, target_offset=target,
            mode='legacy', reason=reason, feasible=feasible, min_clearance=None,
            min_separation=None, hypothesis_count=0,
            planning_ms=(time.perf_counter() - start) * 1000)


def suite_cases(kind, cfg, sensor, length):
    """Four/effectively twelve bounded cases, varying geometry and conditions."""
    count = 4 if kind == 'smoke' else 12
    scenarios = ('adaptive', 'feint', 'surge', 'cut-in', 'brake', 'left-pass', 'right-pass')
    return [(scenarios[i % len(scenarios)], replace(cfg,
             start_s=(cfg.start_s + (i % 4) * length / 4) % length,
             initial_gap=(2., 3., 4.)[i % 3],
             attacker_speed=cfg.defender_speed + (.4, .8, 1.2)[i % 3],
             seed=cfg.seed + i), replace(sensor, latency=(0., .1, .25)[i % 3],
             dropout=(0., .05, .2)[(i // 2) % 3])) for i in range(count)]


def run(scenario, defense, cfg, rc, sc, line, track, map_base, map_ext,
        output, sensor_kind='simulated', render=False, controller_kind='robust'):
    from rival_robust import RobustDefender, TrackEnvelope
    import gym
    from f110_gym.envs.base_classes import Integrator
    from f110_gym.envs.f110_env import F110Env
    envelope = TrackEnvelope(line, track, cfg, rc, boundaries_path=rc.boundaries_path)
    poses, offsets = [], []
    for s in (cfg.start_s, cfg.start_s - cfg.initial_gap):
        d = envelope.nominal(s)
        lo, hi = envelope.interval(s)
        if not lo <= d <= hi or not track.footprint_safe(line.sample(s, d),
                line.heading(s), cfg.car_length, cfg.car_width, cfg.boundary_margin):
            raise ValueError(f'Invalid initial footprint/region at s={s}, d={d}')
        offsets.append(d)
        poses.append([*line.sample(s, d), line.heading(s)])
    if np.linalg.norm(np.array(poses[0][:2]) - poses[1][:2]) < cfg.car_length + cfg.separation_margin:
        raise ValueError('Initial footprints too close')
    if controller_kind == 'robust':
        controller = RobustDefender(line, track, cfg, rc, offsets[0], envelope=envelope)
    else:
        controller = LegacyAdapter(line, track, cfg, offsets[0], envelope)
    overlay = Overlay(line) if render else None
    attacker = Adversary(offsets[1])
    sensor = SimulatedSensor(line, sc, cfg.seed)
    detector = LidarDetector(line, track, sc)
    tracker = OpponentTracker(line, sc)
    env = gym.make('f110_gym:f110-v0', map=str(map_base), map_ext=map_ext,
                   num_agents=2, timestep=cfg.timestep, seed=cfg.seed, integrator=Integrator.RK4)
    rows, events = [], []
    metrics = StepMetrics(cfg.timestep, rc.control_period, offsets)
    bounds = EvaluationBounds(cfg.timestep)
    pass_metric = PassMetric(cfg.initial_gap, cfg.car_length + cfg.separation_margin)
    max_actual_speed = np.zeros(2)
    progress = np.zeros(2)
    collision = np.zeros(2, dtype=bool)
    offtrack = np.zeros(2, dtype=bool)
    lap_times = [None, None]
    pass_time = None
    passed = False
    previous_steer = 0.
    previous_target = offsets[0]
    previous_def_steer = 0.
    target_jump = steer_jump = 0.
    minsep = float('inf')
    unavailable = stale = missing = infeasible = 0
    elapsed = 0.
    reason = 'duration_limit'
    wall = time.perf_counter()
    old_callbacks, old_renderer = F110Env.render_callbacks, F110Env.renderer
    F110Env.render_callbacks, F110Env.renderer = [], None
    try:
        if cfg.car_length < env.params['length'] or cfg.car_width < env.params['width']:
            raise ValueError('Configured footprint undersizes simulator vehicle')
        if not np.isclose(cfg.wheelbase, env.params['lf'] + env.params['lr']):
            raise ValueError('Wheelbase mismatch; physics is not overridden')
        if render:
            env.add_render_callback(overlay)
        obs, _, done, _ = env.reset(np.asarray(poses))
        states = [observe(obs, i, line) for i in range(2)]
        minsep = float(np.hypot(states[0].x - states[1].x, states[0].y - states[1].y))
        collision |= np.asarray(obs['collisions'], dtype=bool)
        for i, state in enumerate(states):
            offtrack[i] = not track.footprint_safe([state.x, state.y], state.yaw,
                cfg.car_length, cfg.car_width, 0.)
            max_actual_speed[i] = abs(state.speed)
            if collision[i]:
                events.append(dict(time=0., event='collision', car=i))
            if offtrack[i]:
                events.append(dict(time=0., event='offtrack', car=i))
        if done:
            events.append(dict(time=0., event='gym_done_at_reset'))
        last_render = -1.
        for step in range(int(np.ceil(cfg.duration / cfg.timestep))):
            if done or collision.any() or offtrack.any():
                reason = ('collision_at_reset' if collision.any() else
                          'offtrack_at_reset' if offtrack.any() else
                          'gym_done_at_reset' if step == 0 else 'gym_done')
                break
            t = step * cfg.timestep
            observation_start = time.perf_counter()
            states = [observe(obs, i, line) for i in range(2)]
            ego, opponent = states
            # Truth enters ONLY the synthetic adapter, adversary and evaluator.
            detection_start = time.perf_counter()
            measurement = (sensor.update(t, opponent) if sensor_kind == 'simulated'
                           else detector.detect(t, ego, obs['scans'][0]))
            detection_ms = (time.perf_counter() - detection_start) * 1000
            tracking_start = time.perf_counter()
            estimate = tracker.update(t, measurement)
            tracking_ms = (time.perf_counter() - tracking_start) * 1000
            missing += int(measurement is None)
            unavailable += int(estimate is None)
            stale += int(estimate is not None and not estimate.fresh)
            control_start = time.perf_counter()
            decision = controller.step(t, ego, estimate, defense=defense)
            controller_ms = (time.perf_counter() - control_start) * 1000
            attack = attacker.plan(t, scenario, opponent, ego, line, envelope, track, cfg,
                previous_steering=previous_steer,
                measured_forward_speed=float(obs['linear_vels_x'][1]))
            steer, speed, _ = pursuit(opponent, line, attack.target, attack.speed, previous_steer, cfg)
            previous_steer = steer
            target_jump = max(target_jump, abs(decision.target_offset - previous_target))
            steer_jump = max(steer_jump, abs(decision.steering - previous_def_steer))
            previous_target, previous_def_steer = decision.target_offset, decision.steering
            infeasible += int(not decision.feasible)
            command_metrics = metrics.update([[decision.steering, decision.speed], [steer, speed]],
                [decision.target_offset, attack.target], decision.planning_ms, controller_ms)
            sep = float(np.hypot(ego.x - opponent.x, ego.y - opponent.y))
            row = dict(time=t, mode=decision.mode, reason=decision.reason,
                feasible=decision.feasible, target=decision.target_offset,
                attacker_target=attack.target, attack_reason=attack.reason,
                attack_requested=attack.requested, attack_desired=attack.desired,
                attack_corridor_valid=attack.requested_corridor_valid,
                attack_guard_intervened=attack.guard_intervened,
                attack_sampled_path_feasible=attack.sampled_path_feasible,
                attack_selected_path_feasible=attack.selected_path_feasible,
                attack_fallback=attack.fallback, attack_unresolved=attack.unresolved,
                steering=decision.steering, speed=decision.speed,
                attacker_steering=steer, attacker_speed_command=speed,
                min_predicted_separation=decision.min_separation,
                min_clearance=decision.min_clearance, planning_ms=decision.planning_ms,
                hypothesis_count=decision.hypothesis_count, center_separation=sep,
                relative_progress=pass_metric.lead,
                estimate_available=estimate is not None,
                estimate_age=None if estimate is None else estimate.age,
                estimate_error=None if estimate is None else float(np.hypot(
                    estimate.state.x - opponent.x, estimate.state.y - opponent.y)))
            row.update(command_metrics)
            for label, state in zip(('ego_truth_eval', 'opponent_truth_eval'), states):
                row.update({f'{label}_{k}': v for k, v in asdict(state).items()})
            row.update(metrics.pipeline(detection_ms, tracking_ms,
                                        (time.perf_counter() - observation_start) * 1000))
            rows.append(clean(row))
            obs, _, done, _ = env.step(np.asarray([[decision.steering, decision.speed], [steer, speed]]))
            elapsed = (step + 1) * cfg.timestep
            new = [observe(obs, i, line) for i in range(2)]
            rows[-1].update(clean(bounds.update(new, states, envelope, sc)))
            for i, state in enumerate(new):
                progress[i] += line.signed_distance(state.s, states[i].s)
                max_actual_speed[i] = max(max_actual_speed[i], abs(state.speed))
                hit = bool(obs['collisions'][i])
                outside = not track.footprint_safe([state.x, state.y], state.yaw,
                    cfg.car_length, cfg.car_width, 0.)
                if hit and not collision[i]:
                    events.append(dict(time=elapsed, event='collision', car=i))
                if outside and not offtrack[i]:
                    events.append(dict(time=elapsed, event='offtrack', car=i))
                collision[i] |= hit
                offtrack[i] |= outside
                if progress[i] >= cfg.laps * line.length and lap_times[i] is None:
                    lap_times[i] = elapsed
            minsep = min(minsep, float(np.hypot(new[0].x - new[1].x, new[0].y - new[1].y)))
            if pass_metric.update(elapsed, progress):
                passed, pass_time = True, pass_metric.time
                events.append(dict(time=pass_time, confirmed_at=elapsed, event='sustained_pass',
                    sampled_path_feasible_attack_steps=attacker.sampled_path_feasible_steps))
            rows[-1].update(step_end_time=elapsed, ego_progress=float(progress[0]),
                attacker_progress=float(progress[1]), ego_collision=bool(collision[0]),
                attacker_collision=bool(collision[1]), ego_offtrack=bool(offtrack[0]),
                attacker_offtrack=bool(offtrack[1]))
            states = new
            if render and elapsed - last_render >= .1 - 1e-9:
                overlay.target = line.sample(ego.s + cfg.lookahead, decision.target_offset)
                env.render(mode='human_fast')
                last_render = elapsed
            if collision.any() or offtrack.any():
                reason = 'collision' if collision.any() else 'offtrack'
                break
            if all(t is not None for t in lap_times):
                reason = 'laps_completed'
                break
    finally:
        try:
            if F110Env.renderer is not None:
                F110Env.renderer.close()
        finally:
            F110Env.renderer = old_renderer
            F110Env.render_callbacks = old_callbacks
            env.close()
    planning = metrics.planning
    def span(key):
        values = [r[key] for r in rows if r[key] is not None]
        return [min(values), max(values)] if values else [None, None]
    result = dict(scenario=scenario, defense=defense, sensor_kind=sensor_kind,
        controller=controller_kind,
        controller_notes=('original Defender/TrackRules/pursuit at timestep cadence; same estimates, sensor and adversary; absent/stale estimate nominal stop; no robust command slew' if controller_kind == 'legacy' else 'RobustDefender with internal control-period solves and timestep actuation'),
        physics_steps=len(rows),
        sensor_description='truth-derived synthetic sensor, not hardware perception' if sensor_kind == 'simulated' else 'heuristic lidar clusters, not identified opponent truth',
        settings=asdict(cfg), robust=asdict(rc), sensor=asdict(sc),
        track=dict(map=str(map_base), extension=map_ext, length=line.length),
        stop_reason=reason, elapsed=elapsed, wall_seconds=time.perf_counter()-wall,
        events=events, collisions=collision, offtrack=offtrack,
        completed_laps=np.floor(np.maximum(progress, 0) / line.length).astype(int),
        lap_completion_times=lap_times, completed=all(v is not None for v in lap_times),
        completed_passes=int(passed), pass_time=pass_time, min_center_separation=minsep,
        measured_d_range=span('ego_truth_eval_d'), target_range=span('target'),
        max_target_step=target_jump, max_steering_step=steer_jump,
        command_metrics=dict(max_speed=metrics.max_speed, max_abs_acceleration=metrics.max_acceleration,
            max_abs_jerk=metrics.max_jerk, max_abs_target_rate=metrics.max_target_rate,
            max_abs_steer_rate=metrics.max_steer_rate, max_actual_speed=max_actual_speed,
            late_actuation_steps=metrics.late,
            note='per-car [defender, attacker]; derivatives of actual commands including initial zero transition; late=controller wall time > timestep, not measured simulator/OS actuation'),
        cars={label: dict(elapsed=elapsed, progress=progress[i],
            laps=float(progress[i] / line.length), completed=lap_times[i] is not None,
            finish_time=lap_times[i], collision=collision[i], offtrack=offtrack[i])
            for i, label in enumerate(('defender', 'attacker'))}, 
        progress=progress, stalled=progress < .2 * elapsed,
        unresolved_safety_steps=infeasible, requested_attack_steps=attacker.requested_steps,
        requested_corridor_valid_steps=attacker.corridor_valid_steps,
        guard_intervention_steps=attacker.guard_interventions,
        sampled_path_feasible_attack_steps=attacker.sampled_path_feasible_steps,
        executed_sampled_feasible_attack_steps=attacker.executed_attack_steps,
        attacker_fallback_steps=attacker.fallback_steps,
        attacker_unresolved_steps=attacker.unresolved_steps,
        attack_evaluation=attack_outcome(progress, elapsed, collision, offtrack,
            attacker.requested_steps, attacker.executed_attack_steps, attacker.unresolved_steps),
        adversary_preview=dict(horizon=attacker.preview_horizon,
            reevaluation_period=attacker.reevaluation_period, commitment_time=attacker.commitment_time,
            side_hysteresis=attacker.side_hysteresis,
            actuation_dt=cfg.timestep,
            note='XY kinematic bicycle using measured body-forward speed/yaw, actual pursuit and bounded target/steering/speed slew; approximate speed tracking, constant defender d/speed; not Gym dynamics or full-maneuver proof'),
        pipeline_latency=metrics.latency_summary(), evaluation_bounds=bounds.summary(),
        defended_attack_credit=False,  # A finite run alone does not establish successful defense.
        sensor_unavailable_steps=unavailable, sensor_stale_steps=stale,
        no_measurement_steps=missing,
        no_measurement_note='Includes between-acquisition calls and latency, not just random dropout',
        planning_ms=dict(p50=np.percentile(planning, 50) if planning else None,
            p95=np.percentile(planning, 95) if planning else None,
            max=max(planning) if planning else None,
            solve_count=len(planning), missed_budget=metrics.missed))
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = clean(result)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'{output.name}: {reason}, sim={elapsed:.2f}s wall={result["wall_seconds"]:.2f}s; '
          f'pass={pass_time}; steps={len(rows)} solves={len(planning)} budget_misses={metrics.missed} '
                    f'physics_deadline_misses={metrics.physics_deadline_misses}')
    for label, car in result['cars'].items():
        print(f'  {label}: laps={car["laps"]:.3f} completed={car["completed"]} '
              f'finish={car["finish_time"]} collision={car["collision"]} offtrack={car["offtrack"]}')
    print(f'  attack: invalid={result["attack_evaluation"]["invalid_attack"]} '
          f'reasons={result["attack_evaluation"]["invalid_attack_reasons"]}; '
          f'executed_steps={attacker.executed_attack_steps} fallback={attacker.fallback_steps} '
          f'unresolved={attacker.unresolved_steps}; defender_failure={result["attack_evaluation"]["defender_failure"]}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_suffix('.yaml'))
    parser.add_argument('--scenario', choices=SCENARIOS, default='adaptive')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--controller', choices=('robust', 'legacy'), default='robust')
    parser.add_argument('--sensor', choices=('simulated', 'lidar'), default='simulated')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'rival_results/advanced')
    parser.add_argument('--region', choices=('inside', 'track'))
    parser.add_argument('--suite', choices=('smoke', 'stress'))
    args = parser.parse_args()
    print('R.I.V.A.L. advanced: SIMULATION ONLY — not validated for a real vehicle or arbitrary adversarial behavior.')
    from rival_robust import RobustConfig
    raw = yaml.safe_load(args.config.read_text())
    cfg, line, track, map_base, map_ext = load(args.config)
    if args.duration is not None:
        cfg = Settings_checked(cfg, args.duration)
    robust = dict(raw.get('robust', {}))
    if args.region:
        robust['region'] = args.region
    if robust.get('boundaries_path'):
        robust['boundaries_path'] = str((args.config.parent / robust['boundaries_path']).resolve())
    rc = RobustConfig.from_dict(robust)
    sc = SensorConfig(**raw.get('sensor', {}))
    cases = suite_cases(args.suite, cfg, sc, line.length) if args.suite else [(args.scenario, cfg, sc)]
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (scenario, settings, sensor) in enumerate(cases):
        for defense in ((True, False) if args.compare else (True,)):
            tag = f'{index:02d}_{scenario}_{args.controller}_{"defense" if defense else "no_defense"}'
            try:
                results.append(run(scenario, defense, settings, rc, sensor, line,
                    track, map_base, map_ext, args.output / tag, args.sensor, args.render, args.controller))
            except ValueError as exc:
                print(f'{tag}: invalid_setup: {exc}')
                failure = dict(scenario=scenario, defense=defense, controller=args.controller, status='invalid_setup',
                    error=str(exc), settings=asdict(settings), sensor=asdict(sensor))
                results.append(failure)
                (args.output / f'{tag}.json').write_text(json.dumps(failure, indent=2))
    valid = [r for r in results if r.get('status') != 'invalid_setup']
    aggregate = dict(runs=results, attempted=len(results), valid=len(valid),
        invalid_setup=len(results)-len(valid), completed=sum(r['completed'] for r in valid),
        passes=sum(r['completed_passes'] for r in valid),
        collision_runs=sum(any(r['collisions']) for r in valid),
        offtrack_runs=sum(any(r['offtrack']) for r in valid),
        unresolved_safety_steps=sum(r['unresolved_safety_steps'] for r in valid),
        invalid_attack_runs=sum(r['attack_evaluation']['invalid_attack'] for r in valid),
        attacker_failure_runs=sum(r['attack_evaluation']['attacker_failure'] for r in valid),
        defender_failure_runs=sum(r['attack_evaluation']['defender_failure'] for r in valid),
        eligible_attack_runs=sum(r['attack_evaluation']['attack_evaluation_eligible'] for r in valid),
        defended_attack_credit_runs=0,
        planning_latency=[r['planning_ms'] for r in valid])
    (args.output / 'summary.json').write_text(json.dumps(clean(aggregate), indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in aggregate.items() if k not in ('runs', 'planning_latency')}))


def Settings_checked(cfg, duration):
    from rival_strategy import Settings
    return Settings.from_dict({**asdict(cfg), 'duration': duration})


if __name__ == '__main__':
    main()
