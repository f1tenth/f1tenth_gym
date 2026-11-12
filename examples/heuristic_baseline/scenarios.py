"""
scenarios.py — Batch evaluation harness for the heuristic baseline.

Public API
----------
run_batch(scenario, mode='A', n_trials=20, obs_delay=0.0) -> dict of aggregated metrics
print_table(batch_results)                   -> single-mode results table
print_side_by_side(results_a, results_b)     -> Mode A vs Mode B comparison table
print_delay_comparison(r0, r_delay)          -> two-row OBS_DELAY comparison

Run the built-in test (20 trials × 5 scenarios, Modes A+B, plus feint OBS_DELAY):
    python scenarios.py
"""

import contextlib
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent))   # examples/ on path

from baseline import (run_episode, SCENARIOS, MAP_BASE, MAP_EXT, TIMESTEP,
                      _load_wp, _make_line, _precompute_corridors)
from track_utils import _track_length


# ── Environment context manager ───────────────────────────────────────────────

@contextlib.contextmanager
def _headless_env():
    """Create and yield a headless 2-car F1TENTH gym env; close on exit."""
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


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(scenario, mode='A', n_trials=20, obs_delay=0.0):
    """
    Run n_trials independent episodes and return aggregated metrics.

    One environment is created for the whole batch and reset between episodes.

    Parameters
    ----------
    scenario  : str   — one of baseline.SCENARIOS
    mode      : str   — defender mode ('A' or 'B')
    n_trials  : int
    obs_delay : float — Mode B perception latency in seconds (default 0.0)

    Returns
    -------
    dict with keys:
        scenario, mode, n_trials, obs_delay
        n_overtake, n_collision
        overtake_rate, collision_rate         (fractions)
        mean_time_blocking, std_time_blocking (s)
        mean_atk_speed,     std_atk_speed     (m/s)
        mean_def_progress,  std_def_progress  (m)
        raw                                   (list of per-trial dicts)
    """
    wp   = _load_wp()
    line = _make_line(wp)
    tlen = _track_length(wp)
    lo_arr, hi_arr = _precompute_corridors(line)   # ~0.3 s, once per batch

    raw = []
    with _headless_env() as env:
        for trial in range(n_trials):
            r = run_episode(scenario, mode, seed=trial,
                            _env=env, _wp=wp, _line=line, _tlen=tlen,
                            _lo=lo_arr, _hi=hi_arr, _obs_delay=obs_delay)
            raw.append(r)

    n_overtake  = sum(r['overtaken'] for r in raw)
    n_collision = sum(r['collision'] for r in raw)

    block_t  = np.array([r['time_blocking']       for r in raw])
    atk_spd  = np.array([r['mean_attacker_speed']  for r in raw])
    def_prog = np.array([r['defender_progress']    for r in raw])

    return {
        'scenario':            scenario,
        'mode':                mode,
        'n_trials':            n_trials,
        'obs_delay':           obs_delay,
        'n_overtake':          n_overtake,
        'n_collision':         n_collision,
        'overtake_rate':       n_overtake  / n_trials,
        'collision_rate':      n_collision / n_trials,
        'mean_time_blocking':  float(np.mean(block_t)),
        'std_time_blocking':   float(np.std(block_t)),
        'mean_atk_speed':      float(np.mean(atk_spd)),
        'std_atk_speed':       float(np.std(atk_spd)),
        'mean_def_progress':   float(np.mean(def_prog)),
        'std_def_progress':    float(np.std(def_prog)),
        'raw':                 raw,
    }


# ── Results tables ────────────────────────────────────────────────────────────

def _row(r):
    """Format one run_batch result as column strings."""
    return (
        f"{r['n_overtake']}/{r['n_trials']}",
        f"{r['n_collision']}/{r['n_trials']}",
        f"{r['mean_time_blocking']:4.1f}±{r['std_time_blocking']:3.1f}s",
        f"{r['mean_atk_speed']:4.2f}m/s",
    )


def print_table(results):
    """
    Print a formatted results table from a list of run_batch() outputs (one mode).
    """
    if not results:
        print("(no results)")
        return

    n  = results[0]['n_trials']
    md = results[0]['mode']
    od = results[0].get('obs_delay', 0.0)
    od_str = f", OBS_DELAY={od:.1f}s" if od else ""
    bar = "─" * 72

    print(f"\nDefender Mode {md}{od_str}  —  {n} trials per scenario")
    print(bar)
    print(f"{'Scenario':<12}  {'Overtake':>8}  {'Collision':>9}  "
          f"{'Block_t':>12}  {'Atk_spd':>8}")
    print(bar)
    for r in results:
        ov, col, blk, spd = _row(r)
        print(f"{r['scenario']:<12}  {ov:>8}  {col:>9}  {blk:>12}  {spd:>8}")
    print(bar)


def print_side_by_side(results_a, results_b):
    """
    Print a side-by-side comparison of Mode A and Mode B results.

    Parameters
    ----------
    results_a : list[dict] — run_batch outputs for Mode A (one per scenario)
    results_b : list[dict] — run_batch outputs for Mode B (one per scenario)
    """
    if not results_a or not results_b:
        print("(no results)")
        return

    n   = results_a[0]['n_trials']
    bar = "─" * 94

    print(f"\n{'':12}  {'── Mode A ──':^38}  {'── Mode B ──':^38}")
    print(f"{'Scenario':<12}  {'Overtake':>8}  {'Coll':>5}  {'Block_t':>12}  {'Spd':>6}  "
          f"  {'Overtake':>8}  {'Coll':>5}  {'Block_t':>12}  {'Spd':>6}")
    print(bar)

    for ra, rb in zip(results_a, results_b):
        sc = ra['scenario']
        ov_a, col_a, blk_a, spd_a = _row(ra)
        ov_b, col_b, blk_b, spd_b = _row(rb)
        print(f"{sc:<12}  {ov_a:>8}  {col_a:>5}  {blk_a:>12}  {spd_a:>6}  "
              f"  {ov_b:>8}  {col_b:>5}  {blk_b:>12}  {spd_b:>6}")

    print(bar)
    print(f"({n} trials per scenario per mode)")


def print_delay_comparison(r_nodelay, r_delay):
    """
    Print a two-column comparison showing the effect of OBS_DELAY.

    Parameters
    ----------
    r_nodelay : dict — run_batch result with obs_delay=0.0
    r_delay   : dict — run_batch result with obs_delay > 0
    """
    sc  = r_nodelay['scenario']
    n   = r_nodelay['n_trials']
    d   = r_delay['obs_delay']
    bar = "─" * 50

    print(f"\nScenario: {sc}  —  Mode {r_nodelay['mode']}, {n} trials")
    print(f"{'Metric':<22}  {'delay=0.0':>12}  {f'delay={d:.1f}s':>12}")
    print(bar)

    rows = [
        ('Overtake',     f"{r_nodelay['n_overtake']}/{n}",
                         f"{r_delay['n_overtake']}/{n}"),
        ('Collision',    f"{r_nodelay['n_collision']}/{n}",
                         f"{r_delay['n_collision']}/{n}"),
        ('Block_t (s)',  f"{r_nodelay['mean_time_blocking']:.1f}±{r_nodelay['std_time_blocking']:.1f}",
                         f"{r_delay['mean_time_blocking']:.1f}±{r_delay['std_time_blocking']:.1f}"),
        ('Atk_spd (m/s)',f"{r_nodelay['mean_atk_speed']:.2f}±{r_nodelay['std_atk_speed']:.2f}",
                         f"{r_delay['mean_atk_speed']:.2f}±{r_delay['std_atk_speed']:.2f}"),
        ('Def_prog (m)', f"{r_nodelay['mean_def_progress']:.1f}",
                         f"{r_delay['mean_def_progress']:.1f}"),
    ]
    for label, v0, v1 in rows:
        print(f"{label:<22}  {v0:>12}  {v1:>12}")
    print(bar)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N_TRIALS = 20

    # ── Part 1: Mode A vs Mode B, all scenarios ───────────────────────────────
    print(f"=== Mode A vs Mode B — {N_TRIALS} trials per scenario ===")
    print(f"Episodes up to 30 s each at dt={TIMESTEP} s — please wait.\n")

    res_a, res_b = [], []
    t_start = time.time()

    for sc in SCENARIOS:
        t0  = time.time()
        ra  = run_batch(sc, 'A', N_TRIALS)
        rb  = run_batch(sc, 'B', N_TRIALS)
        dt  = time.time() - t0
        print(f"  {sc:<12}  A: {ra['n_overtake']}/{N_TRIALS} ov  "
              f"B: {rb['n_overtake']}/{N_TRIALS} ov  ({dt:.1f}s)")
        res_a.append(ra)
        res_b.append(rb)

    print_side_by_side(res_a, res_b)

    # ── Part 2: feint, Mode B, OBS_DELAY effect ───────────────────────────────
    print(f"\n=== Feint — Mode B, OBS_DELAY effect ({N_TRIALS} trials) ===\n")

    r_feint_b0  = next(r for r in res_b if r['scenario'] == 'feint')
    t0          = time.time()
    r_feint_b03 = run_batch('feint', 'B', N_TRIALS, obs_delay=0.3)
    print(f"  feint B delay=0.3: {r_feint_b03['n_overtake']}/{N_TRIALS} ov  "
          f"({time.time()-t0:.1f}s)")

    print_delay_comparison(r_feint_b0, r_feint_b03)

    total = time.time() - t_start
    print(f"\nTotal wall time: {total:.1f}s  "
          f"({total / (N_TRIALS * (len(SCENARIOS)*2 + 1)):.2f}s per episode)")
