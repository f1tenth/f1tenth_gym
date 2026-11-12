# Heuristic Defender Baseline

This directory contains the heuristic blocking defender baseline for the RIVAL F1TENTH racing scenario.
`evaluate_baseline.py` runs the full evaluation and produces reproducible results that any team member
can replicate or compare their own method against.

---

## How to run

```bash
# From the repo root:
source .venv/bin/activate
cd examples/heuristic_baseline
python evaluate_baseline.py
```

The script suppresses the render window automatically.
Expected wall time: **3–6 minutes** (1 000 episodes, ~0.2–0.4 s each).

Outputs written to this directory:

| File | Contents |
|---|---|
| `baseline_results.csv` | One row per trial — 1 000 rows total |
| `baseline_params.json` | All parameters used to produce the results |

---

## Metrics

| Column | Type | Meaning |
|---|---|---|
| `overtaken` | 0/1 | Attacker centre passed defender centre before the episode ended |
| `collision` | 0/1 | A car-to-car collision was detected (`obs['collisions']` flag) |
| `time_blocking` (s) | float | Total seconds the attacker was within `ENGAGE_GAP = 5 m` behind the defender |
| `mean_attacker_speed` (m/s) | float | Per-step mean of `obs['linear_vels_x'][1]` over the episode |

An overtake ends the episode immediately (the trial is counted as a failure for the defender).
A collision also ends the episode.
Episodes that reach `SIM_DURATION = 30 s` without either event are a defender success.

---

## Defender modes

| Mode | Description |
|---|---|
| **A** | Mirror defender — copies the attacker's lateral offset `d` when within `ENGAGE_GAP`, otherwise holds the centreline. |
| **B** | MOBIL lane selector — re-evaluates the best blocking lane every `MOBIL_INTERVAL = 0.2 s` using IDM blocking values and a safety/incentive criterion. `OBS_DELAY = 0.0 s` (perfect sensing) in the baseline run. |

---

## Fixed scenarios

Each scenario scripts the attacker's lateral behaviour; both cars follow pure-pursuit over offset waypoints.

| Scenario | Attacker behaviour |
|---|---|
| `straight` | Holds the centreline for the entire run |
| `move_left` | Approaches to within `ENGAGE_GAP`, then shifts permanently LEFT |
| `move_right` | Approaches to within `ENGAGE_GAP`, then shifts permanently RIGHT |
| `feint` | Approaches, ducks one direction, reverses after `FEINT_SWITCH ≈ 1 s` |
| `late_move` | Approaches to within `LATE_CLOSE ≈ 1.5 m` bumper gap, then shifts LEFT |

---

## Seeds and reproducibility

**Seed = trial index.**  Trial 0 uses seed 0, trial 1 uses seed 1, …, trial 99 uses seed 99.
The seed initialises a `numpy.default_rng` that jitters the initial gap, start speeds, feint
switch time, and late-move threshold.  The gym env is created with `seed=0` (physics engine
only); all scenario randomness comes from the per-trial numpy RNG.

All IDM constants, MOBIL thresholds, lane offsets, `OBS_DELAY`, and the map name are saved to
`baseline_params.json` alongside the CSV.

---

## Loading and comparing results

```python
import pandas as pd
import numpy as np

baseline = pd.read_csv('baseline_results.csv')

# Mean overtake rate per (scenario, mode) with 95% CI
def ci95_proportion(rate, n):
    return 1.96 * np.sqrt(rate * (1 - rate) / n)

def ci95_mean(series):
    return 1.96 * series.std(ddof=1) / np.sqrt(len(series))

summary = (baseline
    .groupby(['scenario', 'mode'])
    .agg(
        overtake_rate=('overtaken',           'mean'),
        collision_rate=('collision',           'mean'),
        mean_time_blocking=('time_blocking',   'mean'),
        mean_atk_speed=('mean_attacker_speed', 'mean'),
        n=('trial', 'count'),
    )
    .reset_index()
)
print(summary)
```

### Comparing your method against the baseline

Run your own evaluation using **the same 100 seeds (0–99), the same 5 scenarios, and the same
`SIM_DURATION`**, then join on `(scenario, seed)` to compare trial-matched outcomes.

```python
yours    = pd.read_csv('your_results.csv')     # same schema as baseline_results.csv
baseline = pd.read_csv('baseline_results.csv')

merged = yours.merge(
    baseline[['scenario', 'trial', 'overtaken', 'mean_attacker_speed']],
    on=['scenario', 'trial'],
    suffixes=('_yours', '_baseline'),
)

delta = merged.groupby('scenario').apply(
    lambda g: pd.Series({
        'overtake_delta': g['overtaken_yours'].mean() - g['overtaken_baseline'].mean(),
        'speed_delta':    g['mean_attacker_speed_yours'].mean()
                          - g['mean_attacker_speed_baseline'].mean(),
    })
)
print(delta)
```

A negative `overtake_delta` means your method defends better than the baseline on that scenario.
A negative `speed_delta` means your method slows the attacker more on average.

---

## Key parameters (from `baseline_params.json`)

| Parameter | Value | Description |
|---|---|---|
| `V0_DEFENDER` | 4.0 m/s | Defender free-road target speed |
| `V0_ATTACKER` | 4.5 m/s | Attacker free-road target speed |
| `IDM_A_MAX` | 2.0 m/s² | IDM maximum acceleration |
| `IDM_B` | 3.0 m/s² | IDM comfortable deceleration |
| `IDM_S0` | 0.5 m | IDM minimum standstill gap |
| `IDM_T` | 0.6 s | IDM desired time headway |
| `MOBIL_INTERVAL` | 0.2 s | Mode B lane re-evaluation period |
| `B_SAFE` | 4.0 m/s² | Mode B safety braking threshold |
| `DELTA_A_TH` | 0.3 m/s² | Mode B minimum incentive to switch |
| `ENGAGE_GAP` | 5.0 m | Defender activates within this gap |
| `SIM_DURATION` | 30.0 s | Maximum episode length |
| `OBS_DELAY` (Mode B) | 0.0 s | Perception latency (0 = perfect sensing) |
| Map | `example_map` | RIVAL example map, 156 m closed loop |
