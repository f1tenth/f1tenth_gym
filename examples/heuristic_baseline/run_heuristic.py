"""
Run the IDM-blocking heuristic defender in R.I.V.A.L. — headless only.

Reused from the repo (reported):
  load()            → rival_baseline  (map / config loading)
  ScriptedAttacker  → rival_baseline  (attacker policy for scripted scenarios)
  observe()         → rival_strategy  (Frenet state from raw obs dict)
  pursuit()         → rival_strategy  (pure-pursuit [steer, speed] command)
  TrackRules        → rival_strategy  (legal interval + safety checks)
  Settings          → rival_strategy  (validated config dataclass)

Novel:
  IDMBlockingDefender  — this package (heuristic_defender.py)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from rival_baseline import load, ScriptedAttacker
from rival_strategy import observe, pursuit, TrackRules, Settings

from heuristic_defender import IDMBlockingDefender

SCENARIOS = ('no-attack', 'approach', 'left-pass', 'right-pass')


def run(scenario, defense, cfg, line, track, map_base, map_ext, output, obs_delay=0.0):
    """Run one scenario and write CSV + JSON results to output/.

    Parameters
    ----------
    scenario : str
    defense : bool
    cfg : Settings
    line : ClosedRaceline
    track : TrackMap
    map_base : Path
    map_ext : str
    output : Path
    obs_delay : float
        Seconds of observation delay for the heuristic defender.
    """
    import gym
    from f110_gym.envs.base_classes import Integrator
    from f110_gym.envs.f110_env import F110Env

    rules = TrackRules(line, track, cfg)  # reuse TrackRules from rival_strategy
    initial_s = [cfg.start_s % line.length, (cfg.start_s - cfg.initial_gap) % line.length]
    initial_offsets, poses = [], []
    for s in initial_s:
        lo, hi = rules.interval(s)
        if lo > hi:
            raise ValueError(f'No legal initial interval at s={s:.3f}; change start_s')
        d = float(np.clip(0.0, lo, hi))
        if not rules.footprint(s, d):
            raise ValueError(f'Initial footprint unsafe at s={s:.3f}, d={d:.3f}')
        initial_offsets.append(d)
        poses.append([*line.sample(s, d), line.heading(s)])
    if np.linalg.norm(np.array(poses[0][:2]) - poses[1][:2]) < cfg.car_length + cfg.separation_margin:
        raise ValueError('Initial vehicles are too close')

    # IDMBlockingDefender — novel heuristic (this package)
    defender = IDMBlockingDefender(line, rules, cfg, initial_offsets[0], obs_delay=obs_delay)
    # ScriptedAttacker — reused from rival_baseline (reported reuse)
    attacker = ScriptedAttacker(initial_offsets[1])

    old_callbacks = F110Env.render_callbacks
    F110Env.render_callbacks = []
    output.mkdir(parents=True, exist_ok=True)
    tag = f'{scenario}_{"defense" if defense else "no_defense"}'
    csv_path = output / f'{tag}.csv'

    progress = np.zeros(2)
    finish_times = [None, None]
    offtrack = np.zeros(2, dtype=bool)
    collisions = np.zeros(2, dtype=bool)
    min_separation = float('inf')
    previous_steer = np.zeros(2)

    blocking_time = reacting_time = 0.0
    activations = active_steps = infeasible_steps = 0
    was_active = False
    pass_start = pass_time = None
    stop_reason = 'duration_limit'
    elapsed = 0.0

    fields = [
        'time',
        'defender_s', 'defender_d', 'defender_speed',
        'attacker_s', 'attacker_d', 'attacker_speed',
        'decision_gap', 'bumper_gap', 'closing_speed',
        'lateral_separation', 'desired_gap', 'legal_lo', 'legal_hi',
        'defender_target', 'attacker_target',
        'defender_reason', 'attacker_reason',
        'reacting', 'active',
        'blocking_time', 'reacting_time',
        'defender_steer', 'attacker_steer',
        'defender_command_speed', 'attacker_command_speed',
        'center_separation', 'relative_progress',
        'defender_progress', 'attacker_progress',
        'defender_collision', 'attacker_collision',
        'defender_offtrack', 'attacker_offtrack',
    ]

    env = gym.make('f110_gym:f110-v0', map=str(map_base), map_ext=map_ext,
                   num_agents=2, timestep=cfg.timestep, seed=cfg.seed, integrator=Integrator.RK4)
    if cfg.car_length < env.params['length'] or cfg.car_width < env.params['width']:
        env.close()
        raise ValueError('Configured safety dimensions must not undersize the Gym vehicle')
    if not np.isclose(cfg.wheelbase, env.params['lf'] + env.params['lr']):
        env.close()
        raise ValueError('Configured wheelbase must match Gym lf + lr')

    try:
        obs, _, done, _ = env.reset(np.asarray(poses))
        collisions |= np.asarray(obs['collisions'], dtype=bool)
        # observe() — reused from rival_strategy (reported reuse)
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

                ds, ats = states  # defender, attacker CarState

                # Heuristic defender decision
                target, speed_cmd, reason, detail = defender.decide(
                    elapsed, ds, ats, enabled=defense)

                # ScriptedAttacker — reused from rival_baseline (reported reuse)
                attacker_target, attack_speed, attack_reason = attacker.plan(
                    elapsed, scenario, ats, ds, line, rules, cfg)

                safe = not reason.startswith('stop:')
                infeasible_steps += int(not safe
                                        or attack_reason.startswith('stop:')
                                        or 'infeasible' in attack_reason)
                active = detail.get('active', False)
                activations += int(active and not was_active)
                active_steps += int(active)
                was_active = active
                if detail.get('reacting', False):
                    reacting_time += cfg.timestep
                if active:
                    blocking_time += cfg.timestep

                # pursuit() — reused from rival_strategy (reported reuse)
                def_steer, def_v, _ = pursuit(ds, line, target,
                                              speed_cmd if safe else 0.0,
                                              previous_steer[0], cfg)
                atk_steer, atk_v, _ = pursuit(ats, line, attacker_target,
                                              attack_speed, previous_steer[1], cfg)
                actions = np.asarray([[def_steer, def_v], [atk_steer, atk_v]])
                previous_steer = actions[:, 0].copy()

                obs, _, done, _ = env.step(actions)
                elapsed = (step + 1) * cfg.timestep
                states = [observe(obs, i, line) for i in range(2)]

                for i, state in enumerate(states):
                    progress[i] += line.signed_distance(state.s, last_s[i])
                    last_s[i] = state.s
                    if finish_times[i] is None and progress[i] >= cfg.laps * line.length:
                        finish_times[i] = elapsed
                    offtrack[i] |= not track.footprint_safe(
                        [state.x, state.y], state.yaw,
                        cfg.car_length, cfg.car_width, 0.)
                collisions |= np.asarray(obs['collisions'], dtype=bool)

                separation = float(np.hypot(states[0].x - states[1].x,
                                            states[0].y - states[1].y))
                min_separation = min(min_separation, separation)
                relative_progress = -cfg.initial_gap + progress[1] - progress[0]
                # Pass detection: attacker must hold lead for >= 0.5 s (same as rival_baseline)
                if relative_progress > cfg.car_length + cfg.separation_margin:
                    if pass_start is None:
                        pass_start = elapsed
                    if pass_time is None and elapsed - pass_start >= 0.5:
                        pass_time = pass_start
                else:
                    pass_start = None

                writer.writerow(dict(
                    time=elapsed,
                    defender_s=ds.s, defender_d=ds.d, defender_speed=ds.speed,
                    attacker_s=ats.s, attacker_d=ats.d, attacker_speed=ats.speed,
                    decision_gap=detail['gap'],
                    bumper_gap=detail.get('bumper_gap', 0.0),
                    closing_speed=detail['closing_speed'],
                    lateral_separation=detail['lateral_separation'],
                    desired_gap=detail.get('desired_gap', 0.0),
                    legal_lo=detail['legal_lo'], legal_hi=detail['legal_hi'],
                    defender_target=target, attacker_target=attacker_target,
                    defender_reason=reason, attacker_reason=attack_reason,
                    reacting=int(detail.get('reacting', False)),
                    active=int(active),
                    blocking_time=blocking_time, reacting_time=reacting_time,
                    defender_steer=def_steer, attacker_steer=atk_steer,
                    defender_command_speed=def_v, attacker_command_speed=atk_v,
                    center_separation=separation, relative_progress=relative_progress,
                    defender_progress=progress[0], attacker_progress=progress[1],
                    defender_collision=int(collisions[0]),
                    attacker_collision=int(collisions[1]),
                    defender_offtrack=int(offtrack[0]),
                    attacker_offtrack=int(offtrack[1]),
                ))

                if np.any(collisions) or np.any(offtrack):
                    stop_reason = 'collision' if np.any(collisions) else 'offtrack'
                    break
                if all(v is not None for v in finish_times):
                    stop_reason = 'completed'
                    break
                if done:
                    stop_reason = 'gym_done'
                    break

        summary = dict(
            scenario=scenario, defense=defense, obs_delay=obs_delay,
            seed=cfg.seed, elapsed=elapsed, stop_reason=stop_reason,
            passed=pass_time is not None, pass_time=pass_time,
            minimum_center_separation=min_separation,
            blocking_time=blocking_time, reacting_time=reacting_time,
            defensive_activations=activations, defensive_active_steps=active_steps,
            infeasible_steps=infeasible_steps,
            cars={name: dict(
                completed=finish_times[i] is not None,
                laps=float(progress[i] / line.length),
                finish_time=finish_times[i],
                collision=bool(collisions[i]),
                offtrack=bool(offtrack[i]),
                gym_laps=int(obs['lap_counts'][i]),
                gym_lap_time=float(obs['lap_times'][i]),
            ) for i, name in enumerate(('defender', 'attacker'))},
            settings=vars(cfg),
            csv=str(csv_path),
            interpretation='heuristic IDM/MOBIL defender; not proof of safety',
        )
        with (output / f'{tag}.json').open('w') as stream:
            json.dump(summary, stream, indent=2)

        print(f'{tag}: {stop_reason}, {elapsed:.2f}s; '
              f'pass={pass_time}; min_sep={min_separation:.3f}m; '
              f'blocking_t={blocking_time:.1f}s reacting_t={reacting_time:.1f}s; '
              f'activations={activations}')
        for name, car in summary['cars'].items():
            print(f'  {name}: completed={car["completed"]}, laps={car["laps"]:.3f}, '
                  f'finish={car["finish_time"]}, collision={car["collision"]}, '
                  f'offtrack={car["offtrack"]}')
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
    parser.add_argument('--config', type=Path,
                        default=(Path(__file__).parent.parent / 'rival_config.yaml').resolve())
    parser.add_argument('--scenario', choices=(*SCENARIOS, 'all'), default='left-pass')
    parser.add_argument('--duration', type=float, help='Override configured duration (s)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Observation delay in seconds (default: 0)')
    parser.add_argument('--no-defense', action='store_true',
                        help='Only run without active defense')
    parser.add_argument('--compare', action='store_true',
                        help='Run defense and no-defense sequentially')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    if args.compare and args.no_defense:
        parser.error('--compare and --no-defense are mutually exclusive')

    cfg, line, track, map_base, map_ext = load(args.config.resolve())
    if args.duration is not None:
        cfg = Settings.from_dict({**vars(cfg), 'duration': args.duration})

    print('R.I.V.A.L. IDM/MOBIL heuristic defender — headless simulation only.')
    for scenario in SCENARIOS if args.scenario == 'all' else [args.scenario]:
        for defense in ((True, False) if args.compare else [not args.no_defense]):
            run(scenario, defense, cfg, line, track, map_base, map_ext,
                args.output, obs_delay=args.delay)


if __name__ == '__main__':
    main()
