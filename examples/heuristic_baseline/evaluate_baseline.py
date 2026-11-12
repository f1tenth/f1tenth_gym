"""
evaluate_baseline.py — 100-trial heuristic defender baseline evaluation.

Runs 100 trials × 5 scenarios × 2 defender modes (A, B) headless.
Seeds are fixed: seed = trial index (0 … 99) for reproducibility.

Outputs
-------
baseline_results.csv   — one row per trial
baseline_params.json   — all parameters needed to reproduce the run

Usage
-----
    cd examples/heuristic_baseline
    python evaluate_baseline.py
"""
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent))   # examples/ on path

from baseline import (
    run_episode, SCENARIOS,
    MAP_BASE, MAP_EXT,
    ENGAGE_GAP, START_S, START_GAP, FEINT_SWITCH, LATE_CLOSE, SIM_DURATION,
    MOBIL_INTERVAL, B_SAFE, DELTA_A_TH, SWITCH_BIAS, _ALIGN_THRESH,
    _load_wp, _make_line, _precompute_corridors,
)
from idm import (
    IDM_A_MAX, IDM_B, IDM_S0, IDM_T, TIMESTEP,
    V0_DEFENDER, V0_ATTACKER, CAR_LENGTH, LAT_THRESHOLD,
)
from track_utils import LEFT, CENTER, RIGHT, _SAFE_D_MAX, _track_length

# ── Evaluation settings ───────────────────────────────────────────────────────
N_TRIALS  = 100          # trials per (scenario, mode); seed = trial index
MODES     = ['A', 'B']   # defender modes to evaluate
OBS_DELAY = 0.0          # Mode B perception delay (s); 0.0 = perfect sensing

_CSV_PATH    = _HERE / 'baseline_results.csv'
_PARAMS_PATH = _HERE / 'baseline_params.json'


# ── Parameter snapshot ────────────────────────────────────────────────────────

def _build_params():
    return {
        'evaluation': {
            'n_trials':          N_TRIALS,
            'modes':             MODES,
            'scenarios':         list(SCENARIOS),
            'seeds':             f'0 to {N_TRIALS - 1}  (seed = trial index)',
            'obs_delay_mode_b':  OBS_DELAY,
        },
        'map': {
            'name': Path(MAP_BASE).name,
            'base': MAP_BASE,
            'ext':  MAP_EXT,
        },
        'idm': {
            'IDM_A_MAX':     IDM_A_MAX,
            'IDM_B':         IDM_B,
            'IDM_S0':        IDM_S0,
            'IDM_T':         IDM_T,
            'V0_DEFENDER':   V0_DEFENDER,
            'V0_ATTACKER':   V0_ATTACKER,
            'CAR_LENGTH':    CAR_LENGTH,
            'LAT_THRESHOLD': LAT_THRESHOLD,
            'TIMESTEP':      TIMESTEP,
        },
        'mobil': {
            'MOBIL_INTERVAL': MOBIL_INTERVAL,
            'B_SAFE':         B_SAFE,
            'DELTA_A_TH':     DELTA_A_TH,
            'SWITCH_BIAS':    SWITCH_BIAS,
            '_ALIGN_THRESH':  _ALIGN_THRESH,
        },
        'scenario_params': {
            'ENGAGE_GAP':   ENGAGE_GAP,
            'START_S':      START_S,
            'START_GAP':    START_GAP,
            'FEINT_SWITCH': FEINT_SWITCH,
            'LATE_CLOSE':   LATE_CLOSE,
            'SIM_DURATION': SIM_DURATION,
        },
        'lane_offsets': {
            'LEFT':       LEFT,
            'CENTER':     CENTER,
            'RIGHT':      RIGHT,
            '_SAFE_D_MAX': _SAFE_D_MAX,
        },
    }


# ── Headless env context manager ──────────────────────────────────────────────

@contextlib.contextmanager
def _headless_env():
    """Yield a headless 2-car F1TENTH gym env; close on exit."""
    from f110_gym.envs.f110_env import F110Env
    from f110_gym.envs.base_classes import Integrator
    import gym

    old_cbs = F110Env.render_callbacks
    F110Env.render_callbacks = []
    env = gym.make(
        'f110_gym:f110-v0',
        map=MAP_BASE, map_ext=MAP_EXT,
        num_agents=2, timestep=TIMESTEP,
        seed=0, integrator=Integrator.RK4,
    )
    try:
        yield env
    finally:
        env.close()
        F110Env.render_callbacks = old_cbs


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(all_rows):
    """Print mean ± 95% CI for each (scenario, mode) cell."""
    from collections import defaultdict

    groups = defaultdict(list)
    for row in all_rows:
        groups[(row['scenario'], row['mode'])].append(row)

    BAR = '─' * 94
    FMT = '{:<12}  {:>4}  {:>18}  {:>18}  {:>22}  {:>16}'

    print(f'\n{BAR}')
    print('Heuristic Defender Baseline  —  '
          f'{N_TRIALS} trials × {len(SCENARIOS)} scenarios × {len(MODES)} modes  (95% CI)')
    print(BAR)
    print(FMT.format('Scenario', 'Mode',
                     'Overtake rate', 'Collision rate',
                     'Time blocking (s)', 'Atk speed (m/s)'))
    print(BAR)

    for sc in SCENARIOS:
        for mode in MODES:
            rows = groups[(sc, mode)]
            n    = len(rows)
            if n == 0:
                print(FMT.format(sc, mode, 'NO DATA', '', '', ''))
                continue

            ov   = np.array([r['overtaken']           for r in rows], dtype=float)
            col  = np.array([r['collision']            for r in rows], dtype=float)
            tblk = np.array([r['time_blocking']        for r in rows], dtype=float)
            spd  = np.array([r['mean_attacker_speed']  for r in rows], dtype=float)

            ov_m  = ov.mean()
            col_m = col.mean()
            tb_m  = tblk.mean()
            sp_m  = spd.mean()

            # 95% CI: Wilson normal approx for proportions, t-interval for means
            ov_ci  = 1.96 * np.sqrt(ov_m  * (1.0 - ov_m)  / n)
            col_ci = 1.96 * np.sqrt(col_m * (1.0 - col_m) / n)
            tb_ci  = 1.96 * tblk.std(ddof=1) / np.sqrt(n)
            sp_ci  = 1.96 * spd.std(ddof=1)  / np.sqrt(n)

            print(FMT.format(
                sc, mode,
                f'{ov_m:.3f} ± {ov_ci:.3f}',
                f'{col_m:.3f} ± {col_ci:.3f}',
                f'{tb_m:.2f} ± {tb_ci:.2f}',
                f'{sp_m:.3f} ± {sp_ci:.3f}',
            ))
        if sc != SCENARIOS[-1]:
            print(FMT.format('', '', '', '', '', ''))   # blank separator row

    print(BAR)
    print(f'  OBS_DELAY (Mode B) = {OBS_DELAY} s  |  '
          f'map = {Path(MAP_BASE).name}  |  params → baseline_params.json')
    print(BAR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    params = _build_params()
    with open(_PARAMS_PATH, 'w') as fh:
        json.dump(params, fh, indent=2)
    print(f'Parameters saved → {_PARAMS_PATH.name}')

    total_ep = N_TRIALS * len(SCENARIOS) * len(MODES)
    print(f'Running {total_ep} episodes '
          f'({N_TRIALS} trials × {len(SCENARIOS)} scenarios × {len(MODES)} modes) …\n')

    # Precompute shared resources once (~0.3 s for corridor arrays)
    print('Precomputing waypoints and corridor arrays …', end=' ', flush=True)
    t0 = time.time()
    wp             = _load_wp()
    line           = _make_line(wp)
    tlen           = _track_length(wp)
    lo_arr, hi_arr = _precompute_corridors(line)
    print(f'done ({time.time() - t0:.1f} s)\n')

    all_rows   = []
    csv_fields = ['scenario', 'mode', 'trial',
                  'overtaken', 'collision',
                  'time_blocking', 'mean_attacker_speed']

    with open(_CSV_PATH, 'w', newline='') as csvfile:
        writer    = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        t_overall = time.time()
        done_ep   = 0

        with _headless_env() as env:
            for mode in MODES:
                obs_delay = OBS_DELAY if mode == 'B' else 0.0

                for sc in SCENARIOS:
                    t_sc = time.time()

                    for trial in range(N_TRIALS):
                        result = run_episode(
                            sc, defender_mode=mode, seed=trial,
                            _env=env, _wp=wp, _line=line, _tlen=tlen,
                            _lo=lo_arr, _hi=hi_arr, _obs_delay=obs_delay,
                        )

                        row = {
                            'scenario':            sc,
                            'mode':                mode,
                            'trial':               trial,
                            'overtaken':           int(result['overtaken']),
                            'collision':           int(result['collision']),
                            'time_blocking':       round(float(result['time_blocking']),       4),
                            'mean_attacker_speed': round(float(result['mean_attacker_speed']), 4),
                        }
                        # Store with native types for summary computation
                        all_rows.append({
                            'scenario':            sc,
                            'mode':                mode,
                            'trial':               trial,
                            'overtaken':           bool(result['overtaken']),
                            'collision':           bool(result['collision']),
                            'time_blocking':       float(result['time_blocking']),
                            'mean_attacker_speed': float(result['mean_attacker_speed']),
                        })
                        writer.writerow(row)
                        csvfile.flush()   # survive partial runs
                        done_ep += 1

                    elapsed_sc = time.time() - t_sc
                    sc_rows    = [r for r in all_rows
                                  if r['scenario'] == sc and r['mode'] == mode]
                    n_ov  = sum(r['overtaken']  for r in sc_rows)
                    n_col = sum(r['collision']  for r in sc_rows)
                    print(f'  Mode {mode}  {sc:<12}  '
                          f'{n_ov:3}/{N_TRIALS} overtake  '
                          f'{n_col:3}/{N_TRIALS} collision  '
                          f'({elapsed_sc:.1f} s)')
                print()

        total_wall = time.time() - t_overall
        print(f'Total wall time: {total_wall:.1f} s  '
              f'({total_wall / max(done_ep, 1):.2f} s/episode)')

    print(f'\nResults saved → {_CSV_PATH.name}  ({done_ep} rows)')
    _print_summary(all_rows)


if __name__ == '__main__':
    main()
