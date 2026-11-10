"""Run deterministic two-car R.I.V.A.L. experiments (headless by default)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from rival_geometry import ClosedRaceline, TrackMap
from rival_strategy import (Settings, TrackRules, Defender, observe, pursuit,
                            rate_limit, desired_gap)


SCENARIOS = ('no-attack', 'approach', 'left-pass', 'right-pass')


class ScriptedAttacker:
    """Timed side requests with map/region checks and a rear-end braking guard.

    It never reacts tactically to the defender: only a deterministic script plus
    collision avoidance. Infeasible passing requests degrade to following.
    """
    def __init__(self, target):
        self.target = target

    def plan(self, t, scenario, attacker, defender, line, rules, cfg):
        lo, hi = rules.interval(attacker.s)
        if lo > hi:
            return self.target, 0., 'stop:no_legal_interval'
        future_lo, future_hi = rules.planning_interval(attacker, cfg.attacker_speed)
        if future_lo > future_hi:
            return self.target, 0., 'stop:no_preview_legal_interval'
        nominal = float(np.clip(0., future_lo, future_hi))
        direction = {'left-pass': 1., 'right-pass': -1.}.get(scenario, 0.)
        ramp = np.clip((t - cfg.pass_start) / cfg.pass_ramp, 0., 1.)
        requested = nominal + direction * cfg.pass_offset * ramp
        desired = float(np.clip(requested, future_lo, future_hi))
        reachable = rate_limit(self.target, desired, cfg.offset_rate, cfg.timestep)
        feasible_side = direction == 0 or direction * (desired - defender.d) >= cfg.car_width + cfg.separation_margin
        reason = 'line_follow' if direction == 0 else ('pass_lane' if feasible_side else 'pass_infeasible:follow_clipped_lane')
        speed = cfg.defender_speed if scenario == 'no-attack' else cfg.attacker_speed
        if not lo <= reachable <= hi or not rules.path_safe(attacker, desired, speed, command_target=reachable):
            return self.target, 0., 'stop:no_safe_reachable_lane'
        self.target = reachable
        gap = line.signed_distance(defender.s, attacker.s)
        # A scripted vehicle still brakes rather than deliberately ramming a car.
        lateral = abs(attacker.d - defender.d)
        if 0 < gap and lateral < cfg.car_width + cfg.separation_margin:
            buffer = desired_gap(max(attacker.speed, 0.),
                                 attacker.speed - defender.speed, cfg)
            available = gap - cfg.car_length
            speed = min(speed, max(0., defender.speed + (available - buffer) / max(cfg.time_headway, .1)))
            if scenario == 'approach':
                speed = min(speed, max(0., defender.speed + (available - buffer) * .5))
            reason += ':rear_end_guard'
        return self.target, speed, reason


class Overlay:
    """Lazy pyglet callback: never imported in headless experiments."""
    def __init__(self, line):
        self.line = line
        self.target = line.sample(0.)
        self.label = None
        self.point = None
        self.path = None

    def __call__(self, renderer):
        import pyglet
        from pyglet.gl import GL_POINTS, GL_LINE_LOOP
        if self.label is None:
            vertices = np.c_[50 * self.line.points, np.zeros(len(self.line.points))]
            self.path = renderer.batch.add(len(vertices), GL_LINE_LOOP, None,
                ('v3f/static', vertices.ravel().tolist()),
                ('c3B/static', [240, 210, 40] * len(vertices)))
            self.point = renderer.batch.add(1, GL_POINTS, None,
                ('v3f/stream', [0., 0., 0.]), ('c3B/static', [30, 240, 90]))
            self.label = pyglet.text.Label('Defender=blue | Attacker=red | Line=yellow | Target=green',
                font_size=12, batch=renderer.batch)
        self.point.vertices = [50 * self.target[0], 50 * self.target[1], 0.]
        for car, color in zip(renderer.cars, ([40, 100, 245], [240, 65, 50])):
            car.colors = color * 4
        x, y = np.mean(renderer.cars[0].vertices[::2]), np.mean(renderer.cars[0].vertices[1::2])
        renderer.left, renderer.right = x - 600, x + 600
        renderer.bottom, renderer.top = y - 450, y + 450
        self.label.x, self.label.y = x - 560, y + 400
        self.label.font_size = 14


def load(config_path):
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    cfg = Settings.from_dict(config.get('settings', {}))
    source = (config_path.parent / config.get('track_config', 'config_example_map.yaml')).resolve()
    with source.open() as stream:
        track_config = yaml.safe_load(stream)
    data = np.loadtxt(source.parent / track_config['wpt_path'],
                      delimiter=track_config['wpt_delim'], skiprows=track_config['wpt_rowskip'])
    line = ClosedRaceline(data[:, [track_config['wpt_xind'], track_config['wpt_yind']]])
    map_base = (source.parent / track_config['map_path']).resolve()
    if map_base.suffix in ('.yaml', '.yml'):
        map_base = map_base.with_suffix('')
    map_ext = '.' + track_config['map_ext'].lstrip('.')
    track = TrackMap(map_base, map_ext)
    if cfg.initial_gap >= line.length / 2:
        raise ValueError('initial_gap must be less than half the lap length')
    return cfg, line, track, map_base, map_ext


def run(scenario, defense, cfg, line, track, map_base, map_ext, output, render=False):
    import gym
    from f110_gym.envs.base_classes import Integrator
    from f110_gym.envs.f110_env import F110Env

    rules = TrackRules(line, track, cfg)
    initial_s = [cfg.start_s % line.length, (cfg.start_s - cfg.initial_gap) % line.length]
    initial_offsets, poses = [], []
    for s in initial_s:
        lo, hi = rules.interval(s)
        if lo > hi:
            raise ValueError(f'No legal initial interval at s={s:.3f}; change start_s')
        d = float(np.clip(0., lo, hi))
        if not rules.footprint(s, d):
            raise ValueError(f'Initial footprint is unsafe at s={s:.3f}, d={d:.3f}')
        initial_offsets.append(d)
        poses.append([*line.sample(s, d), line.heading(s)])
    if np.linalg.norm(np.array(poses[0][:2]) - poses[1][:2]) < cfg.car_length + cfg.separation_margin:
        raise ValueError('Initial vehicles are too close')
    defender = Defender(line, rules, cfg, initial_offsets[0])
    attacker = ScriptedAttacker(initial_offsets[1])
    env = gym.make('f110_gym:f110-v0', map=str(map_base), map_ext=map_ext,
                   num_agents=2, timestep=cfg.timestep, seed=cfg.seed, integrator=Integrator.RK4)
    if cfg.car_length < env.params['length'] or cfg.car_width < env.params['width']:
        env.close()
        raise ValueError('Configured safety dimensions must not undersize the actual Gym vehicle')
    if not np.isclose(cfg.wheelbase, env.params['lf'] + env.params['lr']):
        env.close()
        raise ValueError('Configured wheelbase must match Gym lf + lr; physics is not overridden')
    # Class-level callback/renderer state belongs to the legacy Gym, so isolate
    # sequential comparison runs here without modifying the Gym implementation.
    old_callbacks = F110Env.render_callbacks
    F110Env.render_callbacks = []
    overlay = Overlay(line) if render else None
    if render:
        env.add_render_callback(overlay)
    tag = f'{scenario}_{"defense" if defense else "no_defense"}'
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f'{tag}.csv'
    progress = np.zeros(2)
    finish_times = [None, None]
    offtrack = np.zeros(2, dtype=bool)
    collisions = np.zeros(2, dtype=bool)
    min_separation = float('inf')
    previous_steer = np.zeros(2)
    target_min = target_max = defender.target
    measured_min = measured_max = initial_offsets[0]
    max_target_step = max_steer_step = 0.
    activations = active_steps = infeasible_steps = 0
    was_active = False
    pass_start = pass_time = None
    stop_reason = 'duration_limit'
    elapsed = 0.
    fields = ['time', 'decision_time', 'defender_s', 'defender_d', 'defender_speed',
        'attacker_s', 'attacker_d', 'attacker_speed', 'decision_gap', 'closing_speed',
        'lateral_separation', 'desired_gap', 'legal_lo', 'legal_hi', 'defender_target',
        'attacker_target', 'defender_reason', 'attacker_reason', 'reacting', 'active',
        'defender_steer', 'attacker_steer', 'defender_command_speed', 'attacker_command_speed',
        'center_separation', 'defender_progress', 'attacker_progress',
        'defender_collision', 'attacker_collision', 'defender_offtrack', 'attacker_offtrack']
    try:
        obs, _, done, _ = env.reset(np.asarray(poses))
        collisions |= np.asarray(obs['collisions'], dtype=bool)
        states = [observe(obs, i, line) for i in range(2)]
        last_s = np.array([state.s for state in states])
        min_separation = float(np.linalg.norm(np.array(poses[0][:2]) - poses[1][:2]))
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(int(np.ceil(cfg.duration / cfg.timestep))):
                if done:
                    stop_reason = 'gym_done_at_reset' if step == 0 else 'gym_done'
                    break
                ds, ats = states
                previous_target = defender.target
                target, reason, safe, detail = defender.decide(ds, ats, enabled=defense)
                attacker_target, attack_speed, attack_reason = attacker.plan(elapsed, scenario, ats, ds, line, rules, cfg)
                speed = cfg.defender_speed if safe else 0.
                infeasible_steps += int(not safe or attack_reason.startswith('stop:') or 'infeasible' in attack_reason)
                active = detail.get('active', False)
                activations += int(active and not was_active)
                active_steps += int(active)
                was_active = active
                actions, target_points = [], []
                for state, offset, velocity, steer in zip(states, (target, attacker_target), (speed, attack_speed), previous_steer):
                    steering, velocity, point = pursuit(state, line, offset, velocity, steer, cfg)
                    actions.append([steering, velocity])
                    target_points.append(point)
                actions = np.asarray(actions)
                max_target_step = max(max_target_step, abs(target - previous_target))
                max_steer_step = max(max_steer_step, float(np.max(np.abs(actions[:, 0] - previous_steer))))
                previous_steer = actions[:, 0].copy()
                obs, _, done, _ = env.step(actions)
                elapsed = (step + 1) * cfg.timestep
                states = [observe(obs, i, line) for i in range(2)]
                for i, state in enumerate(states):
                    progress[i] += line.signed_distance(state.s, last_s[i])
                    last_s[i] = state.s
                    if finish_times[i] is None and progress[i] >= cfg.laps * line.length:
                        finish_times[i] = elapsed
                    # Off-track is a conservative whole-body bitmap test with
                    # zero extra rules margin, distinct from simulator contact.
                    offtrack[i] |= not track.footprint_safe([state.x, state.y], state.yaw,
                        cfg.car_length, cfg.car_width, 0.)
                collisions |= np.asarray(obs['collisions'], dtype=bool)
                separation = float(np.hypot(states[0].x - states[1].x, states[0].y - states[1].y))
                min_separation = min(min_separation, separation)
                relative_progress = -cfg.initial_gap + progress[1] - progress[0]
                if relative_progress > cfg.car_length + cfg.separation_margin:
                    if pass_start is None:
                        pass_start = elapsed
                    if pass_time is None and elapsed - pass_start >= .5:
                        pass_time = pass_start
                else:
                    pass_start = None
                target_min, target_max = min(target_min, target), max(target_max, target)
                measured_min, measured_max = min(measured_min, states[0].d), max(measured_max, states[0].d)
                row = dict(time=elapsed, decision_time=elapsed - cfg.timestep,
                    decision_gap=detail['gap'], closing_speed=detail['closing_speed'],
                    lateral_separation=detail['lateral_separation'], desired_gap=detail['desired_gap'],
                    legal_lo=detail['legal_lo'], legal_hi=detail['legal_hi'], defender_target=target,
                    attacker_target=attacker_target, defender_reason=reason, attacker_reason=attack_reason,
                    reacting=int(detail['reacting']), active=int(active),
                    defender_steer=actions[0, 0], attacker_steer=actions[1, 0],
                    defender_command_speed=actions[0, 1], attacker_command_speed=actions[1, 1], center_separation=separation)
                for i, name in enumerate(('defender', 'attacker')):
                    row.update({f'{name}_s': states[i].s, f'{name}_d': states[i].d,
                        f'{name}_speed': states[i].speed, f'{name}_progress': progress[i],
                        f'{name}_collision': int(collisions[i]), f'{name}_offtrack': int(offtrack[i])})
                writer.writerow(row)
                if render:
                    overlay.target = target_points[0]
                    env.render(mode='human')
                if np.any(collisions) or np.any(offtrack):
                    stop_reason = 'collision' if np.any(collisions) else 'offtrack'
                    break
                if all(value is not None for value in finish_times):
                    stop_reason = 'completed'
                    break
                if done:
                    stop_reason = 'gym_done'
                    break
        summary = dict(scenario=scenario, defense=defense, seed=cfg.seed,
            elapsed=elapsed, stop_reason=stop_reason, passed=pass_time is not None,
            pass_time=pass_time, minimum_center_separation=min_separation,
            defender_target_range=[target_min, target_max],
            defender_offset_range=[measured_min, measured_max], defensive_activations=activations,
            defensive_active_steps=active_steps, infeasible_steps=infeasible_steps,
            max_target_step=max_target_step, max_target_rate=max_target_step / cfg.timestep,
            max_steer_step=max_steer_step, max_steer_rate=max_steer_step / cfg.timestep,
            cars={name: dict(completed=finish_times[i] is not None,
                laps=float(progress[i] / line.length), finish_time=finish_times[i],
                collision=bool(collisions[i]), offtrack=bool(offtrack[i]),
                gym_laps=int(obs['lap_counts'][i]), gym_lap_time=float(obs['lap_times'][i]))
                for i, name in enumerate(('defender', 'attacker'))},
            settings=vars(cfg), csv=str(csv_path), interpretation='baseline behavior; not proof of safety')
        with (output / f'{tag}.json').open('w') as stream:
            json.dump(summary, stream, indent=2)
        print(f'{tag}: {stop_reason}, {elapsed:.2f}s; pass={pass_time}; min centers={min_separation:.3f}m; '
              f'target=[{target_min:.3f},{target_max:.3f}]m; d=[{measured_min:.3f},{measured_max:.3f}]m; '
                            f'activations={activations}; '
              f'max target rate={summary["max_target_rate"]:.3f}m/s; max steer rate={summary["max_steer_rate"]:.3f}rad/s')
        for name, car in summary['cars'].items():
            print(f'  {name}: completed={car["completed"]}, laps={car["laps"]:.3f}, '
                  f'finish={car["finish_time"]}, collision={car["collision"]}, offtrack={car["offtrack"]}')
        print(f'  CSV: {csv_path}')
        return summary
    finally:
        if F110Env.renderer is not None:
            F110Env.renderer.close()
            F110Env.renderer = None
        F110Env.render_callbacks = old_callbacks
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('rival_config.yaml'))
    parser.add_argument('--scenario', choices=(*SCENARIOS, 'all'), default='left-pass')
    parser.add_argument('--duration', type=float, help='Override configured duration in seconds')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-defense', action='store_true', help='Only run the matched no-defense control')
    parser.add_argument('--compare', action='store_true', help='Run defense and matched no-defense sequentially')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('rival_results'))
    args = parser.parse_args()
    if args.compare and args.no_defense:
        parser.error('--compare and --no-defense are mutually exclusive')
    cfg, line, track, map_base, map_ext = load(args.config.resolve())
    if args.duration is not None:
        cfg = Settings.from_dict({**vars(cfg), 'duration': args.duration})
    print('R.I.V.A.L. heuristic baseline — simulation behavior is NOT proof of safety.')
    for scenario in SCENARIOS if args.scenario == 'all' else [args.scenario]:
        for defense in (True, False) if args.compare else [not args.no_defense]:
            run(scenario, defense, cfg, line, track, map_base, map_ext, args.output, args.render)


if __name__ == '__main__':
    main()
