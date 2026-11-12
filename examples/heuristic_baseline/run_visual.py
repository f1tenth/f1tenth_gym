"""
run_visual.py — Run one episode with the render window open for screenshots.

Usage
-----
    cd examples/heuristic_baseline
    python run_visual.py                        # defaults: scenario=feint, mode=A
    python run_visual.py move_left B --seed 7
    python run_visual.py feint B
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent))

from baseline import (
    run_episode, SCENARIOS,
    MAP_BASE, MAP_EXT,
    _load_wp, _make_line, _precompute_corridors,
)
from idm import TIMESTEP
from track_utils import _track_length

from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.base_classes import Integrator
import gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', nargs='?', default='feint',
                        choices=list(SCENARIOS),
                        help='Attacker scenario (default: feint)')
    parser.add_argument('mode', nargs='?', default='A', choices=['A', 'B'],
                        help='Defender mode A or B (default: A)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Trial seed (default: 0)')
    args = parser.parse_args()

    print(f'Scenario: {args.scenario}  |  Mode: {args.mode}  |  Seed: {args.seed}')
    print('Close the render window or press Ctrl-C to stop.\n')

    wp   = _load_wp()
    line = _make_line(wp)
    tlen = _track_length(wp)
    lo_arr, hi_arr = _precompute_corridors(line)

    # Create env with rendering enabled (don't suppress render_callbacks)
    env = gym.make(
        'f110_gym:f110-v0',
        map=MAP_BASE, map_ext=MAP_EXT,
        num_agents=2, timestep=TIMESTEP,
        seed=0, integrator=Integrator.RK4,
    )
    result = run_episode(
        args.scenario,
        defender_mode=args.mode,
        seed=args.seed,
        _env=env,
        _wp=wp, _line=line, _tlen=tlen,
        _lo=lo_arr, _hi=hi_arr,
        _obs_delay=0.0,
        _render=True,
    )

    env.close()

    print(f"\nResult:")
    print(f"  Overtaken:          {bool(result['overtaken'])}")
    print(f"  Collision:          {bool(result['collision'])}")
    print(f"  Time blocking:      {result['time_blocking']:.2f} s")
    print(f"  Mean atk speed:     {result['mean_attacker_speed']:.3f} m/s")


if __name__ == '__main__':
    main()
