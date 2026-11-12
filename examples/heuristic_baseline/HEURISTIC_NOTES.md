# Heuristic Baseline — Integration Notes

Working branch: `heuristic-baseline`  
All new code lives in `examples/heuristic_baseline/`.  
No existing repo files are modified.

---

## 1. Exact Env API

### Creating the environment

```python
import gym
from f110_gym.envs.base_classes import Integrator

env = gym.make(
    'f110_gym:f110-v0',
    map=str(map_base),     # absolute path to map WITHOUT extension
    map_ext=map_ext,        # e.g. '.png'
    num_agents=2,
    timestep=cfg.timestep,  # physics integration dt in seconds (0.02 s typical)
    seed=cfg.seed,
    integrator=Integrator.RK4,
)
```

Vehicle params default to the 1:10 F1TENTH car:
`lf=0.15875 m, lr=0.17145 m → wheelbase=0.3302 m, width=0.31 m, length=0.58 m,
s_min/max=±0.4189 rad, sv_min/max=±3.2 rad/s, a_max=9.51 m/s², v_min=-5.0, v_max=20.0 m/s`

Warn: `car_length` and `car_width` in `Settings` are safety-envelope values used for footprint
checks; they must NOT be smaller than the actual Gym vehicle dimensions above.
`cfg.wheelbase` must equal `env.params['lf'] + env.params['lr']` exactly.

### reset(poses)

```python
obs, reward, done, info = env.reset(np.asarray(poses))
# poses: ndarray shape (num_agents, 3), each row [x_world, y_world, theta_world]
# theta in radians, 0 = east (+x), CCW positive
```

`reset` internally fires one zero-action `step`, so the first obs already has
valid LiDAR and physics state. Both cars start at **rest** (zero velocity).

### step(actions)

```python
obs, reward, done, info = env.step(actions)
# actions: ndarray shape (num_agents, 2)
#   col 0 → steering ANGLE  (rad)   clamped to [s_min, s_max] = [−0.4189, 0.4189]
#   col 1 → target SPEED    (m/s)   clamped to [v_min, v_max] = [−5.0, 20.0]
```

**Important:** these are NOT steering velocity and NOT acceleration.
The simulator's internal `pid()` converts the requested (speed, steer_angle) pair into
(longitudinal acceleration, steering velocity) for the single-track vehicle dynamics.
The RIVAL scripts use `[steering_angle, speed]` rows, defender at index 0, attacker at index 1.

`reward` = `timestep` (not meaningful here).  
`done` = True when ego (index 0) collides OR all agents complete two lap-zone crossings.  
`info = {'checkpoint_done': toggle_list}` where toggle_list is a per-agent crossing count.

### Observation dict (all per-agent lists unless noted)

| Key | Shape/Type | Description |
|---|---|---|
| `ego_idx` | int | Always 0 |
| `scans` | list of `(1080,)` float arrays | LiDAR ranges in metres, 4.7 rad FOV, **rear blind sector** |
| `poses_x` | list of float | World x (m) |
| `poses_y` | list of float | World y (m) |
| `poses_theta` | list of float | Heading θ (rad, 0=east, CCW+) |
| `linear_vels_x` | list of float | Body-frame **forward speed** (m/s) ← use this as speed |
| `linear_vels_y` | list of float | **Always 0.0** — slip angle not exposed by this API |
| `ang_vels_z` | list of float | Yaw rate (rad/s) |
| `collisions` | `(num_agents,)` bool array | True when in contact |
| `lap_times` | `(num_agents,)` float array | Added by F110Env — time of last lap start |
| `lap_counts` | `(num_agents,)` float array | Added by F110Env — full laps completed |

**Speed note:** `linear_vels_x` is body-frame longitudinal speed. To get Frenet
longitudinal speed, `rival_strategy.observe()` projects it onto the raceline tangent:
`speed = vx * cos(θ − heading(s)) − vy * sin(θ − heading(s))` where `vy=0` always.

**Gym lap-count note:** Gym uses a start-zone heuristic (position relative to reset pose)
that need not equal the RIVAL Frenet lap count. Use RIVAL's accumulated `progress / line.length`
for a consistent lap metric.

---

## 2. What the existing RIVAL code already provides

### rival_geometry.py — map and raceline

**`ClosedRaceline(points)`** — periodic piecewise-linear raceline over Nx2 xy waypoints.
Closing duplicate is stripped; self-arc-length computed from xy (not from CSV's `s` column).

| Method | Returns | Notes |
|---|---|---|
| `project([x,y])` | `(s, d)` | Nearest arc-length + signed lateral offset |
| `sample(s, d=0)` | `[x, y]` | World point at arc-length s, offset d metres left |
| `heading(s)` | radians | Tangent angle at s (0=east) |
| `curvature(s)` | 1/m | Signed; positive = left turn |
| `signed_distance(s_a, s_b)` | metres | Wrapped gap in [−L/2, L/2); negative = b is ahead |
| `.length` | metres | Total lap length (156.356 m on example map) |
| `.points` | Nx2 array | Read-only vertices |

**`TrackMap(map_base, map_ext)`** — raster bitmap with EDT-based conservative clearance.

| Method | Returns | Notes |
|---|---|---|
| `clear(xy)` | metres | Conservative clearance to obstacle; 0 if occupied/outside |
| `clear_many(Nx2)` | (N,) | Batched version |
| `safe(xy, clearance)` | bool | Is clearance > required? |
| `corridor(line, s, clearance)` | `(lo, hi)` | Connected free offsets either side of raceline point |
| `footprint_safe(xy, yaw, L, W, margin)` | bool | Full oriented-rectangle bitmap check |
| `footprints_safe(Nx3, L, W, margin)` | (N,) bool | Batched version |

Map convention: pixels > 128 are free; image is flipped top-to-bottom before use; outside = occupied.
Map origin is the lower-left outer corner of the image; origin[2] is rotation (usually 0).

### rival_strategy.py — state, rules, controller primitives

**`Settings`** — validated dataclass; `Settings.from_dict(yaml_dict)` enforces types.
Key fields: `timestep`, `defender_speed`, `attacker_speed`, `lookahead`, `wheelbase`,
`car_length`, `car_width`, `boundary_margin`, `legal_min/max`, `curvature_threshold`,
`offset_rate`, `minimum_gap`, `time_headway`, `comfortable_braking`, `reference_acceleration`,
`reaction_threshold`, `closing_threshold`, `prediction_horizon`, `separation_margin`,
`usefulness_threshold`, `move_penalty`, `max_steer`, `max_steer_rate`.

**`CarState(s, d, speed, x, y, yaw)`** — Frenet + world pose dataclass.

**`observe(obs, index, line) → CarState`** — extracts Frenet state from raw obs dict.
Projects body-frame velocity onto raceline tangent for Frenet speed.

**`rate_limit(current, desired, rate, dt) → float`** — clamps change per step.

**`desired_gap(speed, closing_speed, cfg) → float`** — IDM bumper-to-bumper gap:
`s* = minimum_gap + max(0, v·T + v·Δv / (2√(a_ref·b)))`

**`legal_interval(bounds, curvature, cfg) → (lo, hi)`** — computes the inside-half
of the estimated track cross-section from bitmap bounds. Near-straight → full corridor;
left curve → right half; right curve → left half. Intersected with `cfg.legal_min/max`.

**`TrackRules(line, track, cfg)`** — cached interval queries and path safety checks.
- `.interval(s)` → `(lo, hi)` — legal d-interval at arc-length s (cached)
- `.planning_interval(state, speed)` → `(lo, hi)` — intersection over prediction horizon
- `.footprint(s, d, margin=None) → bool` — bitmap footprint check at raceline offset
- `.path_safe(state, target, speed, command_target=None) → bool` — samples predicted path

**`predicted_separation_safe(defender, attacker, target, line, cfg) → bool`** — swept
Frenet-rectangle overlap check through prediction horizon.

**`Defender`** — IDM/MOBIL-inspired blocking controller:
- Trigger: `gap < 0 AND closing > closing_threshold AND bumper_gap < reaction_threshold · s*`
- When triggered, candidates = {nominal, held, 5 linspace across future legal, attacker.d};
  scored by `-|target − attacker.d|` (react) or `-|target − nominal|` (cruise) minus `move_penalty · |shift|`;
  safety-rejected by map/path/separation checks; held if improvement < `usefulness_threshold`.
- `decide(defender_state, attacker_state, enabled=True)` → `(target_d, reason, safe, detail_dict)`

**`pursuit(state, line, offset, speed, previous_steer, cfg) → (steer, speed, target_xy)`** —
pure pursuit returning `[steering_angle, speed]`-ready values + the lookahead point.
Formula: `atan2(2 · wheelbase · lateral_error, ‖δ‖²)`. Steering is rate-limited.

### rival_baseline.py — map loading, scripted attacker, run loop

**`load(config_path) → (cfg, line, track, map_base, map_ext)`** — single call loads
YAML config → Settings, waypoint CSV → ClosedRaceline, map image → TrackMap.
Config-relative paths work regardless of shell working directory.

**`ScriptedAttacker`** — timed side-offset requests with:
- Scenario-driven direction (`left-pass` → positive d, `right-pass` → negative d)
- Linear ramp over `pass_ramp` seconds starting at `pass_start`
- Legal-interval clipping + path safety check; degrades to follow if infeasible
- Rear-end speed guard using `desired_gap()`

**`run(scenario, defense, cfg, line, track, ...)`** — main headless simulation loop:
creates env, runs for `cfg.duration` seconds, streams CSV, computes all metrics.

### rival_adversary.py — adaptive/feint/surge/cut-in/brake attacker

**`Adversary`** — advanced attacker used by `rival_advanced.py`. Has path preview
using `rival_robust.py`-style checks, swept collision guards, body-rule tracking budget,
and multiple scenario modes. Can be dropped in as the attacker policy.

Scenarios available: `no-attack`, `approach`, `left-pass`, `right-pass`, `adaptive`,
`feint`, `surge`, `cut-in`, `brake`.

### rival_perception.py — noise/delay/dropout sensing

**`SensorConfig`** — `noise_std`, `speed_noise_std`, `latency`, `update_period`, `dropout`, `max_age`.

**`SimulatedSensor`** — truth-derived synthetic measurements with configurable delay,
noise, and packet dropout. Not real perception.

**`OpponentTracker`** — bounded Frenet-coordinate filter; `age`, `position_sigma`,
`velocity_sigma`, `lateral_speed`, `fresh/valid` flags on the estimate.

**`LidarDetector`** — extracts dynamic objects from raw LiDAR by subtracting map
returns. Heuristic only; subject to wall false-positives and blind-sector misses.

### rival_robust.py / rival_rollout.py — trajectory lattice planner

Replanning at 10 Hz, commands at 20 ms. Nine opponent hypotheses (slow/est/fast ×
left/est/right). Numba-compiled numeric kernel in `rival_rollout.py` with Python
reference in `rival_reference.py`. Curvature-dependent speed limits, braking reserves,
substepped physics at `max_physics_dt=0.01 s`. Used by `rival_advanced.py` only.

### Maps, waypoints, and lane conventions

| Item | File/Value |
|---|---|
| Map image | `examples/example_map.png` |
| Map metadata | `examples/example_map.yaml` (resolution, origin) |
| Map config | `examples/config_example_map.yaml` |
| Waypoints | `examples/example_waypoints.csv` (delimiter `;`, skip 3 rows, x=col1, y=col2) |
| Track length | 156.356 m closed loop |
| Lateral convention | `d > 0` = left of travel; `d < 0` = right |
| Curvature convention | `κ > 0` = left turn |
| Legal region | Inside half of estimated bitmap cross-section in turns; full corridor on straights |
| `legal_min/max` | ±1.2 m (raceline-relative hard limits from config) |
| Initial poses | Computed from `start_s` on the raceline — not hardcoded map coordinates |

---

## 3. Where my heuristic defender plugs in

My heuristic (`reactive_blocker.py`) has three components that map onto RIVAL:

### A. IDM longitudinal control

My `idm_accel()` computes a **continuous acceleration** command.
The RIVAL env takes a **target speed** (not acceleration).
Integration pattern: `v_new = clip(v + idm_accel(...) * dt, 0, v_max)`, then pass `v_new` as
`actions[i, 1]`. The internal PID will drive toward that speed each step.

**Can reuse:** `desired_gap()` from `rival_strategy.py` — identical formula to my IDM gap.
Reuse it instead of my inline version.

### B. MOBIL-inspired lane-change selector

My `defender_mobil()` evaluates which lane maximises blocking value.
To wire into RIVAL:
- Replace `self.target` in my code with the raceline-relative d offset (same coordinate as RIVAL).
- Use `TrackRules.planning_interval()` for the candidate clipping (instead of my raw `LANE_CENTERS`).
- Use `predicted_separation_safe()` for the safety criterion (instead of my `atk_accel_after()`).
- Use `TrackRules.path_safe()` for map/footprint rejection.
- Scoring: replace my `b_val()` (IDM blocking value) with the same function, but supply
  attacker speed from `observe()` rather than from my internal state.

**Can reuse:** `TrackRules`, `rate_limit`, `observe`, `Settings`, `ClosedRaceline`, `TrackMap`
from `rival_strategy.py` and `rival_geometry.py`.

### C. Pure Pursuit tracking

My `pure_pursuit()` → body-frame lateral error → heading update.
RIVAL's `pursuit()` already does the same in raceline offset space and returns
`[steering_angle, speed]` directly compatible with `env.step()`.

**Can reuse:** `pursuit()` from `rival_strategy.py` exactly as-is; drop my `pure_pursuit()`.

### D. Observation delay buffer

My `obs_buf` deque with `OBS_DELAY` is structurally equivalent to
`SimulatedSensor` / `OpponentTracker` in `rival_perception.py`, which supports
configurable `latency`, `update_period`, and `dropout`.

**Can reuse:** `SimulatedSensor` + `OpponentTracker` for a richer (noise+dropout+delay) model,
or keep my lightweight deque for a minimal-dependency baseline.

### Minimal plug-in sketch

```python
# examples/heuristic_baseline/heuristic_defender.py
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from rival_baseline import load
from rival_strategy import observe, pursuit, rate_limit, desired_gap, TrackRules, Settings
from rival_geometry import ClosedRaceline, TrackMap
# ... my IDM/MOBIL logic operating on CarState, outputting (d_target, speed_target)
# ... call pursuit(state, line, d_target, speed_target, prev_steer, cfg) for the action
```

The run loop from `rival_baseline.run()` can be adapted (not modified) by copying
its structure into `examples/heuristic_baseline/run_heuristic.py`, swapping
`Defender.decide()` for my MOBIL lane selector.

---

## 4. Overlap with existing code — check with teammate before building

| My planned component | Existing equivalent | Overlap risk |
|---|---|---|
| IDM gap + speed control | `desired_gap()` in `rival_strategy.py` | **High** — formula is identical; just reuse |
| MOBIL lane selector | `Defender.decide()` in `rival_strategy.py` | **Medium** — same inspiration; different scoring; mine uses explicit IDM blocking value rather than `attacker.d` proximity score |
| Pure pursuit | `pursuit()` in `rival_strategy.py` | **High** — identical formula; just reuse |
| OBS_DELAY deque | `SimulatedSensor` + `OpponentTracker` in `rival_perception.py` | **Medium** — mine is simpler (delay-only, no noise/dropout); use theirs if full sensing model is needed |
| Run loop / metrics | `rival_baseline.run()`, `PassMetric` | **High** — reuse the loop; don't duplicate CSV/JSON metric logic |
| Scripted attacker | `ScriptedAttacker` in `rival_baseline.py` | **High** — reuse for `left-pass`/`right-pass` baselines |
| Adaptive attacker | `Adversary` in `rival_adversary.py` | **High** — reuse for stress tests |
| Map loading | `load()` in `rival_baseline.py` | **High** — reuse exactly |
| Legal interval | `TrackRules.interval()` / `legal_interval()` | **High** — reuse; don't reimplement |
| Footprint checks | `TrackMap.footprint_safe()`, `TrackRules.footprint()` | **High** — reuse |

**Novel contributions from my heuristic (not in existing code):**
1. **IDM blocking-value scoring** — `b_val()` quantifies how much a lane choice
   reduces the attacker's IDM acceleration, rather than scoring by proximity to `attacker.d`.
2. **Explicit IDM speed command** — the existing `Defender` uses fixed `defender_speed`;
   mine computes a speed each step from IDM dynamics.
3. **Observation delay as a first-class parameter** — existing `SimulatedSensor` does this
   properly, but the baseline `rival_baseline.py` uses perfect observations. My `OBS_DELAY`
   deque makes the effect visible in the simplest possible runner.

---

## 5. Quick-start checklist for new code

```bash
# From repo root with venv active:
source .venv/bin/activate        # (Fish: source .venv/bin/activate.fish)

# Smoke-test imports from heuristic_baseline folder:
cd examples/heuristic_baseline
python -c "
import sys; sys.path.insert(0, '..')
from rival_baseline import load
from rival_strategy import observe, pursuit, rate_limit, desired_gap, Defender, TrackRules
from rival_geometry import ClosedRaceline, TrackMap
from pathlib import Path
cfg, line, track, map_base, map_ext = load(Path('../rival_config.yaml').resolve())
print('map_base:', map_base, '  line length:', round(line.length, 3), 'm')
print('OK')
"

# Full test suite (from repo root):
python -m unittest discover -s examples -p 'test_rival_*.py' -v
```

All new scripts must set `matplotlib.use('Agg')` before importing pyplot and must NOT
call `env.render()` or `plt.show()`. Pass `render=False` to any RIVAL `run()` calls.
