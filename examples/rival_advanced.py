"""Simulation-only evaluation, not hardware perception or a safety guarantee.

Four-tuple Gym, [steer, speed]. Synchronous CPU wall times are diagnostics:
no wall-time delay is added to simulated sensing, tracking or actuation.
Settings.timestep is the action period, not the Gym integration timestep.
Commands/sensors/slew remain at that period; held commands use smaller physics steps.
"""
import argparse
import csv
from collections import deque
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


class RunStream:
    """Incremental action CSV and explicitly nonfinal, atomically replaced checkpoints."""
    def __init__(self, output):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if not self.output.parent.is_dir():
            raise ValueError(f'Output parent is not a directory: {self.output.parent}')
        self.stream = self.output.with_suffix('.csv').open('w', newline='')
        # Reusing an output name starts a new run. A killed rerun must not
        # leave an older successful summary beside a newly partial CSV.
        self.output.with_suffix('.json').unlink(missing_ok=True)
        self.output.with_suffix('.checkpoint.json').unlink(missing_ok=True)
        self.writer = None
        self.row_count = self.last_flush_count = 0
        self.last_flush_time = self.last_row_time = 0.

    def write_row(self, row):
        # The caller supplies a completed action row, including partial-cycle
        # end fields. Do not freeze the header from a pre-physics draft row.
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(clean(row))
        self.row_count += 1
        self.last_row_time = float(row['step_end_time'])
        if (self.row_count == 1 or self.row_count % 50 == 0
                or self.last_row_time - self.last_flush_time >= 1. - 1e-9):
            self.flush()

    def flush(self):
        self.stream.flush()
        self.last_flush_count = self.row_count
        self.last_flush_time = self.last_row_time

    def checkpoint(self, snapshot):
        self.flush()
        payload = clean(dict(snapshot, incomplete=True, completed=False, status='incomplete',
            csv_rows=self.row_count, last_csv_sim_time=self.last_row_time,
            note='diagnostic checkpoint only, not a final result or successful run'))
        path = self.output.with_suffix('.checkpoint.json')
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(path)

    def close(self):
        self.stream.close()  # Also flushes any final buffered rows on exceptions.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class StallMetric:
    """Signed progress over an interpolated five-second window, per car.

    A stall is confirmed once a full window advances less than .25 m. Its
    onset is the start of that qualifying window; reverse travel also stalls.
    No whole-run average can clear an ongoing local stall.
    """
    def __init__(self, window=5., minimum_progress=.25):
        self.window, self.minimum_progress = window, minimum_progress
        self.history = deque([(0., np.zeros(2))])
        self.rolling_progress = np.zeros(2)
        self.active = np.zeros(2, dtype=bool)
        self.start = [None, None]
        self.last_start = [None, None]
        self.max_time = np.zeros(2)
        self.current_time = np.zeros(2)

    def update(self, t, progress):
        values = np.asarray(progress, dtype=float).copy()
        self.history.append((t, values))
        cutoff = max(0., t - self.window)
        while len(self.history) > 2 and self.history[1][0] <= cutoff:
            self.history.popleft()
        ta, a = self.history[0]
        tb, b = self.history[1]
        fraction = (cutoff - ta) / (tb - ta) if tb > ta else 0.
        baseline = a + np.clip(fraction, 0., 1.) * (b - a)
        self.rolling_progress = values - baseline
        active = (t >= self.window - 1e-9) & (self.rolling_progress < self.minimum_progress)
        newly_active = active & ~self.active
        for i in range(2):
            if newly_active[i]:
                self.start[i] = self.last_start[i] = max(0., t - self.window)
            if active[i]:
                self.current_time[i] = t - self.start[i]
                self.max_time[i] = max(self.max_time[i], self.current_time[i])
            else:
                self.start[i] = None
                self.current_time[i] = 0.
        self.active = active
        return newly_active

    def timed_out(self, timeout):
        return timeout is not None and bool(np.any(self.active & (self.current_time >= timeout - 1e-9)))

    def summary(self):
        return dict(window_seconds=self.window, minimum_signed_progress=self.minimum_progress,
            last_rolling_progress=self.rolling_progress, locally_stalled=self.active,
            stall_start=self.start, last_stall_start=self.last_start,
            current_stall_time=self.current_time, max_stall_time=self.max_time,
            note='per-car [defender, attacker]; onset is start of first qualifying full window; signed progress, not distance travelled')


def integration_schedule(action_period, max_physics_dt):
    if not np.isfinite(max_physics_dt) or max_physics_dt <= 0:
        raise ValueError('max_physics_dt must be finite and positive')
    count = int(np.ceil(action_period / max_physics_dt))
    return action_period / count, count


def numerical_failures(obs):
    """Public observations only; no angle wrapping or private-state repair."""
    failures = []
    fields = ('poses_x', 'poses_y', 'poses_theta', 'linear_vels_x',
              'linear_vels_y', 'ang_vels_z')
    for i in range(len(obs['poses_x'])):
        reasons = [f'nonfinite:{key}' for key in fields if not np.isfinite(obs[key][i])]
        if np.isfinite(obs['ang_vels_z'][i]) and abs(obs['ang_vels_z'][i]) > 100.:
            reasons.append('yaw_rate_exceeds_100_rad_s')
        failures.append(reasons)
    return failures


def held_action_steps(env, actions, substeps, integration_dt, start_time, track, cfg):
    """Yield every physics result; never step past an inner failure/done."""
    held = np.asarray(actions, dtype=float).copy()
    for index in range(substeps):
        obs, _, done, _ = env.step(held.copy())
        numerical = numerical_failures(obs)
        collision = np.asarray(obs['collisions'], dtype=bool)
        offtrack = np.array([False if numerical[i] else not track.footprint_safe(
            [obs['poses_x'][i], obs['poses_y'][i]], obs['poses_theta'][i],
            cfg.car_length, cfg.car_width, 0.) for i in range(len(numerical))])
        reason = ('numerical_instability' if any(numerical) else
                  'collision' if collision.any() else 'offtrack' if offtrack.any() else
                  'gym_done' if done else None)
        yield SimpleNamespace(obs=obs, done=done, numerical=numerical,
            collision=collision, offtrack=offtrack, reason=reason, count=index + 1,
            elapsed=start_time + (index + 1) * integration_dt)
        if reason is not None:
            return


def attack_outcome(progress, elapsed, collisions, offtrack, requested_steps,
                   executed_steps, unresolved_steps, numerical=None, local_stalled=None):
    """Attacker invalidity is not a defended attack or a defender failure."""
    numerical = [False, False] if numerical is None else numerical
    failures = []
    if numerical[1]:
        failures.append('attacker_numerical_instability')
    if collisions[1]:
        failures.append('attacker_collision')
    if offtrack[1]:
        failures.append('attacker_offtrack')
    if requested_steps and (progress[1] < .2 * elapsed or
                            (local_stalled is not None and local_stalled[1])):
        failures.append('attacker_stalled')
    if requested_steps and not executed_steps:
        failures.append('no_executed_sampled_feasible_attack')
    if unresolved_steps:
        failures.append('attacker_unresolved_preview')
    return dict(invalid_attack=bool(failures), invalid_attack_reasons=failures,
                defender_failure=bool(collisions[0] or offtrack[0] or numerical[0]),
                attacker_failure=bool(failures),
                attacker_physical_failure=bool(collisions[1] or offtrack[1]),
                attack_evaluation_eligible=bool(requested_steps and executed_steps and not failures and not any(numerical)),
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

    def update(self, states, previous, envelope, sensor, dt=None):
        dt = self.dt if dt is None else dt
        row = {}
        for i, (label, state) in enumerate(zip(('ego', 'attacker'), states)):
            lo, hi = envelope.interval(state.s)
            valid = np.isfinite(lo) and np.isfinite(hi) and lo <= hi
            magnitude = max(lo - state.d, state.d - hi, 0.) if valid else None
            violation = not valid or magnitude > self.tolerance
            self.duration[i] += dt * int(violation)
            self.invalid_interval_steps[i] += int(not valid)
            if valid:
                self.max_magnitude[i] = max(self.max_magnitude[i], magnitude)
                self.excess_integral[i] += max(0., magnitude - self.tolerance) * dt
            row.update({f'{label}_legal_violation': violation,
                        f'{label}_legal_violation_magnitude': magnitude})
        opponent, old = states[1], previous[1]
        acceleration = (opponent.speed - old.speed) / dt
        lateral_speed = (opponent.d - old.d) / dt
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
            sensor_bound_note='evaluation truth absolute speed, finite-difference longitudinal acceleration and lateral speed versus SensorConfig bounds; sampled at integration_dt in the runner, not measurement calibration')


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
                    action_deadline_ms=self.dt * 1000,
                    deadline_note='legacy physics_deadline fields use cfg.timestep action period, not the smaller integration_dt',
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
    """Original Defender/pursuit at the configured action-period cadence.

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
            target = self.envelope.nominal(ego.s + self.cfg.lookahead)
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
        output, sensor_kind='simulated', render=False, controller_kind='robust', stall_timeout=None,
                progress_interval=10.):
    from rival_robust import RobustDefender, TrackEnvelope
    import gym
    from f110_gym.envs.base_classes import Integrator
    from f110_gym.envs.f110_env import F110Env
    if not np.isfinite(progress_interval) or progress_interval <= 0:
        raise ValueError('progress_interval must be finite and positive')
    if stall_timeout is not None and (not np.isfinite(stall_timeout) or stall_timeout < 5.):
        raise ValueError('stall_timeout must be finite and at least 5 seconds')
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
    initial_targets = [envelope.nominal(s + cfg.lookahead)
                       for s in (cfg.start_s, cfg.start_s - cfg.initial_gap)]
    if controller_kind == 'robust':
        controller = RobustDefender(line, track, cfg, rc, initial_targets[0], envelope=envelope)
    else:
        controller = LegacyAdapter(line, track, cfg, offsets[0], envelope)
    overlay = Overlay(line) if render else None
    attacker = Adversary(initial_targets[1], tracking_margin=rc.lateral_tracking_margin)
    sensor = SimulatedSensor(line, sc, cfg.seed)
    detector = LidarDetector(line, track, sc)
    tracker = OpponentTracker(line, sc)
    # Compatibility while the parent adds this documented RobustConfig field.
    max_physics_dt = getattr(rc, 'max_physics_dt', .01)
    integration_dt, substeps = integration_schedule(cfg.timestep, max_physics_dt)
    env = gym.make('f110_gym:f110-v0', map=str(map_base), map_ext=map_ext,
                   num_agents=2, timestep=integration_dt, seed=cfg.seed, integrator=Integrator.RK4)
    rows, events = [], []
    metrics = StepMetrics(cfg.timestep, rc.control_period, initial_targets)
    stalls = StallMetric()
    bounds = EvaluationBounds(cfg.timestep)
    pass_metric = PassMetric(cfg.initial_gap, cfg.car_length + cfg.separation_margin)
    max_actual_speed = np.zeros(2)
    progress = np.zeros(2)
    collision = np.zeros(2, dtype=bool)
    offtrack = np.zeros(2, dtype=bool)
    numerical = np.zeros(2, dtype=bool)
    numerical_counts = np.zeros(2, dtype=int)
    max_abs_yaw_rate = np.zeros(2)
    physics_step_count = 0
    lap_times = [None, None]
    pass_time = None
    passed = False
    previous_steer = 0.
    previous_target = initial_targets[0]
    previous_def_steer = 0.
    target_jump = steer_jump = 0.
    minsep = float('inf')
    unavailable = stale = missing = infeasible = 0
    elapsed = 0.
    reason = 'duration_limit'
    wall = time.perf_counter()
    stream = None
    next_checkpoint, next_progress = 10., progress_interval
    interruption = None

    def snapshot():
        return dict(scenario=scenario, defense=defense, controller=controller_kind,
            elapsed=elapsed, wall_seconds=time.perf_counter() - wall,
            requested_duration=cfg.duration, integration_dt=integration_dt, action_period=cfg.timestep,
            physics_steps=physics_step_count, progress=progress,
            stall_metrics=stalls.summary(), numerical_failures=numerical,
            numerical_failure_counts=numerical_counts, collisions=collision, offtrack=offtrack,
            interruption=interruption,
            cars={label: dict(progress=float(progress[i]),
                last_rolling_progress=float(stalls.rolling_progress[i]),
                locally_stalled=bool(stalls.active[i]), stall_start=stalls.start[i],
                max_stall_time=float(stalls.max_time[i]), numerical_failure=bool(numerical[i]),
                collision=bool(collision[i]), offtrack=bool(offtrack[i]))
                for i, label in enumerate(('defender', 'attacker'))})

    old_callbacks, old_renderer = F110Env.render_callbacks, F110Env.renderer
    F110Env.render_callbacks, F110Env.renderer = [], None
    try:
        stream = RunStream(output)
        stream.checkpoint(snapshot())
        if cfg.car_length < env.params['length'] or cfg.car_width < env.params['width']:
            raise ValueError('Configured footprint undersizes simulator vehicle')
        if not np.isclose(cfg.wheelbase, env.params['lf'] + env.params['lr']):
            raise ValueError('Wheelbase mismatch; physics is not overridden')
        if render:
            env.add_render_callback(overlay)
        obs, _, done, _ = env.reset(np.asarray(poses))
        reset_numerical = numerical_failures(obs)
        numerical |= np.asarray([bool(reasons) for reasons in reset_numerical])
        numerical_counts += numerical.astype(int)
        states = [None if numerical[i] else observe(obs, i, line) for i in range(2)]
        xy = np.array([obs['poses_x'], obs['poses_y']], dtype=float)
        if np.isfinite(xy).all():
            minsep = float(np.linalg.norm(xy[:, 0] - xy[:, 1]))
        collision |= np.asarray(obs['collisions'], dtype=bool)
        for i, state in enumerate(states):
            if numerical[i]:
                events.append(dict(time=0., event='numerical_instability', car=i,
                                   reasons=reset_numerical[i]))
            if np.isfinite(obs['ang_vels_z'][i]):
                max_abs_yaw_rate[i] = abs(obs['ang_vels_z'][i])
            if state is not None:
                offtrack[i] = not track.footprint_safe([state.x, state.y], state.yaw,
                    cfg.car_length, cfg.car_width, 0.)
                max_actual_speed[i] = abs(state.speed)
            if collision[i]:
                events.append(dict(time=0., event='collision', car=i))
            if offtrack[i]:
                events.append(dict(time=0., event='offtrack', car=i))
        if done:
            events.append(dict(time=0., event='gym_done_at_reset'))
        stream.checkpoint(snapshot())
        last_render = -1.
        for step in range(int(np.ceil(cfg.duration / cfg.timestep))):
            if done or collision.any() or offtrack.any() or numerical.any():
                reason = ('numerical_instability' if numerical.any() else
                          'collision_at_reset' if collision.any() else
                          'offtrack_at_reset' if offtrack.any() else
                          'gym_done_at_reset' if step == 0 else 'gym_done')
                break
            t = elapsed
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
                attacker_reference_speed_cap=attack.reference_speed_cap,
                attacker_tracking_margin=attack.tracking_margin,
                attacker_initial_body_error=attack.initial_body_error,
                attacker_predicted_max_body_error=attack.predicted_max_body_error,
                attacker_predicted_final_body_error=attack.predicted_final_body_error,
                attacker_predicted_within_tracking_budget=attack.predicted_within_tracking_budget,
                attacker_predicted_strict_body_legal=attack.predicted_strict_body_legal,
                attacker_body_band_recovery=attack.body_band_recovery,
                attacker_partial_body_recovery=attack.partial_body_recovery,
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
            inner_reason = None
            actions = np.asarray([[decision.steering, decision.speed], [steer, speed]])
            for inner in held_action_steps(env, actions, substeps, integration_dt, t, track, cfg):
                obs, done, elapsed = inner.obs, inner.done, inner.elapsed
                physics_step_count += 1
                new = [None if inner.numerical[i] else observe(obs, i, line) for i in range(2)]
                if all(state is not None for state in new):
                    rows[-1].update(clean(bounds.update(new, states, envelope, sc, dt=integration_dt)))
                for i, state in enumerate(new):
                    if inner.numerical[i]:
                        numerical[i] = True
                        numerical_counts[i] += 1
                        events.append(dict(time=elapsed, event='numerical_instability', car=i,
                            reasons=inner.numerical[i],
                            observation={key: clean(obs[key][i]) for key in
                                ('poses_x', 'poses_y', 'poses_theta', 'linear_vels_x',
                                 'linear_vels_y', 'ang_vels_z')}))
                    if np.isfinite(obs['ang_vels_z'][i]):
                        max_abs_yaw_rate[i] = max(max_abs_yaw_rate[i], abs(obs['ang_vels_z'][i]))
                    if state is not None:
                        progress[i] += line.signed_distance(state.s, states[i].s)
                        max_actual_speed[i] = max(max_actual_speed[i], abs(state.speed))
                    if inner.collision[i] and not collision[i]:
                        events.append(dict(time=elapsed, event='collision', car=i))
                    if inner.offtrack[i] and not offtrack[i]:
                        events.append(dict(time=elapsed, event='offtrack', car=i))
                    collision[i] |= inner.collision[i]
                    offtrack[i] |= inner.offtrack[i]
                    if (not numerical[i] and progress[i] >= cfg.laps * line.length
                            and lap_times[i] is None):
                        lap_times[i] = elapsed
                xy = np.array([obs['poses_x'], obs['poses_y']], dtype=float)
                if np.isfinite(xy).all():
                    minsep = min(minsep, float(np.linalg.norm(xy[:, 0] - xy[:, 1])))
                if not numerical.any() and pass_metric.update(elapsed, progress):
                    passed, pass_time = True, pass_metric.time
                    events.append(dict(time=pass_time, confirmed_at=elapsed, event='sustained_pass',
                        sampled_path_feasible_attack_steps=attacker.sampled_path_feasible_steps))
                states = new
                inner_reason = inner.reason
                if not numerical.any():
                    new_stalls = stalls.update(elapsed, progress)
                    for i in np.flatnonzero(new_stalls):
                        events.append(dict(time=elapsed, event='stall_confirmed', car=int(i),
                            stall_start=stalls.start[i], rolling_progress=float(stalls.rolling_progress[i])))
                    if inner_reason is None and stalls.timed_out(stall_timeout):
                        inner_reason = 'stalled'
                        events.append(dict(time=elapsed, event='stall_timeout',
                            cars=np.flatnonzero(stalls.active).tolist(), timeout=stall_timeout))
                        break
                if done:
                    events.append(dict(time=elapsed, event='gym_done'))
            rows[-1].update(action_time=t, action_period=cfg.timestep,
                integration_dt=integration_dt, physics_step_count=inner.count,
                cumulative_physics_steps=physics_step_count, action_elapsed=elapsed - t,
                step_end_time=elapsed, ego_progress=float(progress[0]),
                attacker_progress=float(progress[1]), ego_collision=bool(collision[0]),
                attacker_collision=bool(collision[1]), ego_offtrack=bool(offtrack[0]),
                attacker_offtrack=bool(offtrack[1]), ego_numerical_failure=bool(numerical[0]),
                attacker_numerical_failure=bool(numerical[1]),
                ego_yaw_rate=clean(obs['ang_vels_z'][0]), attacker_yaw_rate=clean(obs['ang_vels_z'][1]))
            for i, label in enumerate(('ego', 'attacker')):
                rows[-1].update({f'{label}_rolling_progress_5s': float(stalls.rolling_progress[i]),
                    f'{label}_locally_stalled': bool(stalls.active[i]),
                    f'{label}_stall_start': stalls.start[i],
                    f'{label}_max_stall_time': float(stalls.max_time[i])})
            stream.write_row(rows[-1])
            if elapsed >= next_checkpoint - 1e-9:
                stream.checkpoint(snapshot())
                next_checkpoint = (np.floor(elapsed / 10. + 1e-9) + 1) * 10.
            if elapsed >= next_progress - 1e-9:
                print(f'{output.name}: sim={elapsed:.2f}s wall={time.perf_counter()-wall:.1f}s '
                      f'progress=[{progress[0]:.2f},{progress[1]:.2f}]m '
                      f'last5s=[{stalls.rolling_progress[0]:.3f},{stalls.rolling_progress[1]:.3f}]m '
                      f'stalled={stalls.active.tolist()} numerical={numerical.tolist()} '
                      f'collision={collision.tolist()} offtrack={offtrack.tolist()}', flush=True)
                next_progress = (np.floor(elapsed / progress_interval + 1e-9) + 1) * progress_interval
            if inner_reason is not None:
                reason = inner_reason
                break
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
    except BaseException as exc:
        interruption = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        try:
            if stream is not None:
                try:
                    stream.checkpoint(snapshot())
                finally:
                    stream.close()
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
        action_cycles=len(rows), physics_steps=physics_step_count,
        integration_dt=integration_dt, substeps=substeps, action_period=cfg.timestep,
        max_physics_dt=max_physics_dt,
        max_physics_dt_source='RobustConfig' if hasattr(rc, 'max_physics_dt') else 'compatibility default pending parent field',
        integration_note='held [steer,speed] commands; cfg.timestep remains action/sensor/slew period; physics_steps excludes reset initialization; no simulator-state resets or equation changes',
        sensor_description='truth-derived synthetic sensor, not hardware perception' if sensor_kind == 'simulated' else 'heuristic lidar clusters, not identified opponent truth',
        settings=asdict(cfg), robust=asdict(rc), sensor=asdict(sc),
        track=dict(map=str(map_base), extension=map_ext, length=line.length),
        stop_reason=reason, elapsed=elapsed, wall_seconds=time.perf_counter()-wall,
        events=events, collisions=collision, offtrack=offtrack,
        numerical_failures=numerical, numerical_failure_counts=numerical_counts,
        numerical_failure_count=int(numerical_counts.sum()),
        max_abs_observed_yaw_rate=max_abs_yaw_rate,
        completed_laps=np.floor(np.maximum(progress, 0) / line.length).astype(int),
        lap_completion_times=lap_times, completed=reason != 'stalled' and not numerical.any() and all(v is not None for v in lap_times),
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
            finish_time=lap_times[i], collision=collision[i], offtrack=offtrack[i],
                        numerical_failure=bool(numerical[i]), numerical_failure_count=int(numerical_counts[i]),
                                    last_rolling_progress=float(stalls.rolling_progress[i]), locally_stalled=bool(stalls.active[i]),
                                    globally_stationary=bool(progress[i] < .2 * elapsed), stall_start=stalls.start[i],
                                    max_stall_time=float(stalls.max_time[i]))
            for i, label in enumerate(('defender', 'attacker'))}, 
        progress=progress, stalled=stalls.active, globally_stationary=progress < .2 * elapsed,
        stall_metrics=stalls.summary(), stall_timeout=stall_timeout,
        stationary_note='stalled uses the last full 5s window (<.25m signed progress); globally_stationary retains the old whole-run progress < .2 * elapsed criterion',
        unresolved_safety_steps=infeasible, requested_attack_steps=attacker.requested_steps,
        requested_corridor_valid_steps=attacker.corridor_valid_steps,
        guard_intervention_steps=attacker.guard_interventions,
        sampled_path_feasible_attack_steps=attacker.sampled_path_feasible_steps,
        executed_sampled_feasible_attack_steps=attacker.executed_attack_steps,
        attacker_fallback_steps=attacker.fallback_steps,
        attacker_unresolved_steps=attacker.unresolved_steps,
        attack_evaluation=attack_outcome(progress, elapsed, collision, offtrack,
            attacker.requested_steps, attacker.executed_attack_steps, attacker.unresolved_steps,
                        numerical, stalls.active),
        adversary_prediction_budget=dict(tracking_margin=attacker.tracking_margin,
            source='RobustConfig.lateral_tracking_margin, same configured body tracking margin as defender',
            body_recovery_steps=attacker.body_recovery_steps,
            checked_body_recovery_steps=attacker.body_recovery_feasible_steps,
            measured_strict_violation_action_samples=attacker.executed_violation_steps,
            measured_recovery_error_growth_samples=attacker.recovery_error_growth_steps,
            note='prediction tracking budget is not strict rule-band compliance or road expansion; pursuit targets, footprint/map boundary margins and collisions remain hard; measured legal-region monitoring retains its independent 0.02m reporting tolerance'),
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
    result = clean(result)
    result['incomplete'] = False
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'{output.name}: {reason}, sim={elapsed:.2f}s wall={result["wall_seconds"]:.2f}s; '
          f'pass={pass_time}; actions={len(rows)} physics_steps={physics_step_count} '
                    f'numerical_failures={int(numerical_counts.sum())}; solves={len(planning)} budget_misses={metrics.missed} '
                    f'physics_deadline_misses={metrics.physics_deadline_misses}')
    for label, car in result['cars'].items():
        print(f'  {label}: laps={car["laps"]:.3f} completed={car["completed"]} '
              f'finish={car["finish_time"]} collision={car["collision"]} offtrack={car["offtrack"]} '
                            f'numerical_failure={car["numerical_failure"]}; last5s={car["last_rolling_progress"]:.3f}m '
                                          f'local_stall={car["locally_stalled"]} stall_start={car["stall_start"]} '
                                          f'max_stall={car["max_stall_time"]:.2f}s')
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
    parser.add_argument('--progress-interval', type=float, default=10.,
                        help='simulated seconds between progress messages (default: 10)')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--controller', choices=('robust', 'legacy'), default='robust')
    parser.add_argument('--sensor', choices=('simulated', 'lidar'), default='simulated')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'rival_results/advanced')
    parser.add_argument('--region', choices=('inside', 'track'))
    parser.add_argument('--suite', choices=('smoke', 'stress'))
    parser.add_argument('--stall-timeout', type=float, default=None,
                        help='optional explicit stalled termination after at least 5 seconds of low progress; disabled by default')
    args = parser.parse_args()
    if not np.isfinite(args.progress_interval) or args.progress_interval <= 0:
        parser.error('--progress-interval must be finite and positive')
    if args.stall_timeout is not None and (not np.isfinite(args.stall_timeout) or args.stall_timeout < 5.):
        parser.error('--stall-timeout must be finite and at least 5 seconds')
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
    (args.output / 'summary.json').unlink(missing_ok=True)
    for index, (scenario, settings, sensor) in enumerate(cases):
        for defense in ((True, False) if args.compare else (True,)):
            tag = f'{index:02d}_{scenario}_{args.controller}_{"defense" if defense else "no_defense"}'
            try:
                results.append(run(scenario, defense, settings, rc, sensor, line,
                    track, map_base, map_ext, args.output / tag, args.sensor, args.render, args.controller,
                                        args.stall_timeout, args.progress_interval))
            except ValueError as exc:
                checkpoint = (args.output / tag).with_suffix('.checkpoint.json')
                if checkpoint.exists() and json.loads(checkpoint.read_text()).get('elapsed', 0.) > 0:
                    raise  # Runtime errors must not become a completed setup-failure summary.
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
        numerical_failure_runs=sum(any(r['numerical_failures']) for r in valid),
        numerical_failure_count=sum(r['numerical_failure_count'] for r in valid),
        local_stall_runs=sum(any(r['stalled']) for r in valid),
        stall_timeout_runs=sum(r['stop_reason'] == 'stalled' for r in valid),
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
