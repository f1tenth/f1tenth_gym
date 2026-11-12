# GYM_NOTES — F1TENTH Gym (RIVAL Fork) Reference

Working branch: `heuristic-baseline`  
All new code lives in `examples/heuristic_baseline/`.  
No existing repo files are modified.

This document records every gym detail discovered while building `gym_setup.py`.
Companion to `HEURISTIC_NOTES.md` (which covers the RIVAL heuristic layer on top).

---

## 1. Installation and import

The gym is installed in editable mode from `gym/`:

```bash
source .venv/bin/activate
# Already installed: pip install -e gym/
```

```python
import gym                                      # gym 0.26.2 (legacy API)
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.f110_env import F110Env
```

---

## 2. Creating the environment

```python
env = gym.make(
    'f110_gym:f110-v0',
    map=str(map_base),     # absolute path WITHOUT extension, e.g. '/abs/path/example_map'
    map_ext='.png',        # extension including dot; '.pgm' also accepted
    num_agents=2,          # 1 or 2
    timestep=0.01,         # physics dt in seconds (0.01 s = 100 Hz recommended)
    seed=0,
    integrator=Integrator.RK4,   # RK4 (recommended) or Euler
)
```

**Always suppress the render window before `gym.make`:**

```python
old_callbacks = F110Env.render_callbacks
F110Env.render_callbacks = []          # prevents any pyglet window
env = gym.make(...)
...
finally:
    env.close()
    F110Env.render_callbacks = old_callbacks
```

---

## 3. Vehicle parameters (`env.params`)

All values are for the default 1:10 F1TENTH car.

| Key | Value | Description |
|---|---|---|
| `lf` | 0.15875 m | Front axle to CoM |
| `lr` | 0.17145 m | Rear axle to CoM |
| `length` | 0.58 m | Full body length |
| `width` | 0.31 m | Full body width |
| `s_min` / `s_max` | ±0.4189 rad | Steering angle limits |
| `sv_min` / `sv_max` | ±3.2 rad/s | Steering velocity limits |
| `a_max` | 9.51 m/s² | Maximum longitudinal acceleration |
| `v_min` / `v_max` | −5.0 / 20.0 m/s | Speed limits |
| `v_switch` | 7.319 m/s | Speed where dynamics model switches |
| `mu` | 1.0489 | Friction coefficient |
| `m` | 3.74 kg | Mass |

**Derived constants (use these in your code):**
```python
WHEELBASE  = env.params['lf'] + env.params['lr']   # 0.3302 m
STEER_MAX  = env.params['s_max']                    # 0.4189 rad
```

Always verify at runtime:
```python
assert np.isclose(WHEELBASE, env.params['lf'] + env.params['lr'], rtol=1e-3)
```

---

## 4. reset(poses) — legacy 4-tuple API

```python
obs, reward, done, info = env.reset(np.asarray(poses))
# poses: float64 array, shape (num_agents, 3)
# Each row: [x_world_m, y_world_m, theta_world_rad]
# theta = 0 → east (+x), CCW positive  (standard 2-D convention)
```

- Both cars start at **rest** (zero velocity).
- `reward` = `timestep` (not useful for RL-free sims).
- `done` = `False` at reset unless a collision is detected immediately.
- `info = {'checkpoint_done': array([False, False])}` — lap-zone crossing flags.

---

## 5. step(actions) — legacy 4-tuple API

```python
obs, reward, done, info = env.step(actions)
# actions: float64 array, shape (num_agents, 2)
# actions[i] = [steering_angle_rad, target_speed_m_s]
#   col 0 → steering ANGLE (rad), clamped to [s_min, s_max] = [−0.4189, 0.4189]
#   col 1 → target SPEED  (m/s), clamped to [v_min, v_max] = [−5.0, 20.0]
```

**Important:** these are NOT steering velocity and NOT acceleration.  
An internal PID converts (speed, steer_angle) commands to (longitudinal
acceleration, steering velocity) for the single-track vehicle model.

`done = True` when **ego (index 0) collides** OR all agents cross the lap
start-zone twice.

---

## 6. Observation dictionary

All per-agent fields have `len = num_agents`.  
`obs['collisions']` and `obs['scans']` are `ndarray`; most others are Python `list`.

| Key | Type / Shape | Description |
|---|---|---|
| `poses_x` | `list[float]` | World x (m) |
| `poses_y` | `list[float]` | World y (m) |
| `poses_theta` | `list[float]` | Heading θ (rad, 0 = east, CCW+) |
| `linear_vels_x` | `list[float]` | Body-frame **forward speed** (m/s) ← use for speed |
| `linear_vels_y` | `list[float]` | Lateral body speed — **always 0.0** (slip not exposed) |
| `ang_vels_z` | `list[float]` | Yaw rate (rad/s); check `abs > 100` for instability |
| `collisions` | `ndarray (N,)` | `True` while in contact |
| `scans` | `list[ndarray(1080,)]` | LiDAR ranges (m); 4.7 rad FOV, 1080 beams, ~0.5 °/beam |
| `lap_times` | `ndarray (N,)` | Time of last lap-zone crossing |
| `lap_counts` | `ndarray (N,)` | Full laps completed |
| `ego_idx` | `int` | Always 0 |

**Speed note:** `linear_vels_x` is the body-frame longitudinal speed.
To project onto the raceline tangent (Frenet s-speed):

```python
# from rival_strategy.observe():
relative_yaw = yaw - line.heading(s)
speed = vx * cos(relative_yaw) - vy * sin(relative_yaw)   # vy = 0 always
```

---

## 7. Available maps

Built-in maps live in `gym/f110_gym/envs/maps/`.  
Map YAML fields: `image`, `resolution` (m/px), `origin` ([x, y, yaw] of lower-left corner).

| Map | Image | Size | Coverage |
|---|---|---|---|
| `berlin` | `berlin.png` | 600 × 600 px | 30 × 30 m — small racing circuit |
| `vegas` | `vegas.png` | 3000 × 2248 px | 150 × 112 m — large circuit |
| `levine` | `levine.pgm` | 2048 × 2048 px | 102 × 102 m — indoor building |
| `skirk` | `skirk.png` | 431 × 602 px | 22 × 30 m — indoor |
| `stata_basement` | `stata_basement.png` | 1730 × 1300 px | 87 × 66 m — indoor |

**RIVAL demo map** (outside `gym/`):

| Map | Path | Coverage | Waypoints |
|---|---|---|---|
| `example_map` | `examples/example_map.png` | ~52 × 40 m | `examples/example_waypoints.csv` |

Track length of `example_map`: **156.4 m** closed loop.

**Image loading convention** (matches the gym's laser model):

```python
from PIL import Image
import numpy as np
raw  = np.array(Image.open(map_path).convert('L'))[::-1]  # flip top-to-bottom
free = raw > 128                                            # True = free space
```

Pixel → world coordinates:

```python
x_world = origin[0] + col * resolution
y_world = origin[1] + row * resolution
```

---

## 8. Waypoints — format and critical pitfall

### CSV format (`example_waypoints.csv`)

```
# comment lines
# comment
# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
0.0;  1.0748956; -0.1528569; 0.0349893; 0.0799358; 6.4310783; 0.4791257
...
```

```python
data = np.loadtxt('example_waypoints.csv', delimiter=';', skiprows=3)
# Column indices from config_example_map.yaml:
#   wpt_xind=1, wpt_yind=2, wpt_thind=3
xs  = data[:, 1]   # world x
ys  = data[:, 2]   # world y
psi = data[:, 3]   # ← DO NOT USE AS WORLD HEADING (see pitfall below)
```

### ⚠ Critical pitfall: `psi_rad` is NOT the world-frame heading

The `psi_rad` column is a **raceline-local** angle (related to track curvature
parameterisation).  It is **not** the world-frame tangent angle θ used by the gym.

`ClosedRaceline` in `rival_geometry.py` ignores `psi_rad` entirely and recomputes
heading from the xy tangent.  You must do the same:

```python
# WRONG — heading from CSV column:
wpts = data[:, [1, 2, 3]]          # psi ≈ 0.035 rad for waypoints[0] → ~2°

# CORRECT — heading from geometric tangent:
xs, ys = data[:, 1], data[:, 2]
dx = np.append(np.diff(xs), xs[0] - xs[-1])
dy = np.append(np.diff(ys), ys[0] - ys[-1])
headings = np.arctan2(dy, dx)      # ≈ 1.526 rad ≈ 87° (mostly north) for waypoints[0]
wpts = np.column_stack([xs, ys, headings])
```

Using the CSV `psi_rad` as world heading will give wrong initial poses, causing
cars to steer at ~90° to the actual track direction and hit walls immediately.

### Companion config YAML

`config_example_map.yaml` maps the waypoint columns:

```yaml
wpt_path: './example_waypoints.csv'
wpt_delim: ';'
wpt_rowskip: 3
wpt_xind: 1
wpt_yind: 2
wpt_thind: 3   # ignored — recompute heading from xy tangent
```

---

## 9. Pure pursuit — exact formula

From `examples/waypoint_follow.py::get_actuation()`:

```python
def pure_pursuit(car_x, car_y, car_theta, waypoints_xy, lookahead, wheelbase):
    """Return steering angle (rad) for one step."""
    # 1. Find nearest waypoint
    dists = np.linalg.norm(waypoints_xy - [car_x, car_y], axis=1)
    nearest = int(np.argmin(dists))

    # 2. Walk forward (wrapping) to first waypoint >= lookahead away
    n = len(waypoints_xy)
    lookahead_pt = None
    for k in range(n):
        idx = (nearest + k) % n
        if np.linalg.norm(waypoints_xy[idx] - [car_x, car_y]) >= lookahead:
            lookahead_pt = waypoints_xy[idx]
            break
    if lookahead_pt is None:
        return 0.0

    # 3. Transform to car frame
    dx = lookahead_pt[0] - car_x
    dy = lookahead_pt[1] - car_y
    y_car = np.sin(-car_theta) * dx + np.cos(-car_theta) * dy

    # 4. Pure-pursuit curvature → steering angle (Coulter 1992)
    # radius   = lookahead^2 / (2 * y_car)
    # steering = arctan(wheelbase / radius) = arctan(2*wheelbase*y_car / lookahead^2)
    steer = np.arctan2(2.0 * wheelbase * y_car, lookahead ** 2)
    return float(np.clip(steer, -STEER_MAX, STEER_MAX))
```

Action format for `env.step`:
```python
actions = np.array([[steer0, speed0],   # car 0 (defender)
                    [steer1, speed1]])   # car 1 (attacker)
# col 0 = steering angle (rad), col 1 = target speed (m/s)
```

---

## 10. Multi-agent notes

- **Index 0** = ego/defender; **index 1** = attacker.
- Both cars start from **rest** regardless of the speed command.
- The `done` flag triggers only when **car 0 collides** or all cars complete
  two lap-zone crossings.  A collision on car 1 alone does NOT set `done`.
- Both cars are physically simulated independently; there is no "leader/follower"
  coupling in the gym.

**Rear-end collision on shared centreline:**  
Two cars on the same path at different speeds will collide when the faster car
closes the gap. With DEFENDER_SPEED = 3.0 m/s, ATTACKER_SPEED = 3.5 m/s and
a 2 m initial gap, collision occurs after ~4 s.  Options:

1. **Rear-end guard** (used in `gym_setup.py`): when centre-to-centre
   separation < `FOLLOWING_GAP` (must be > `car_length = 0.58 m`), clamp
   attacker speed to defender's measured speed.

2. **Lateral lane offset**: give the attacker its own parallel waypoint path
   offset by ≥ `car_width = 0.31 m`.  Check offset path clearance against the
   EDT first — tight inside corners can put the offset path in a wall.

3. **Larger initial gap**: `ATTACKER_GAP ≥ speed_diff × SIM_DURATION` ensures
   no catch-up within the run window.

---

## 11. Auto-generating a centreline (no waypoint file)

For maps without a companion CSV, `gym_setup.py` uses **polar EDT sampling**:

```
1. Load image, flip top-to-bottom, compute EDT (scipy distance_transform_edt)
2. Find the inner-wall component (wall pixel cluster not touching image boundary)
   → its centroid is inside the enclosed circuit area
3. Cast N rays from centroid; each ray's max-EDT pixel = medial-axis sample
4. Sort samples by polar angle → closed CCW loop
5. Resample to uniform spacing; compute headings from xy tangent
6. Cache as <map_base>.centerline.npy
```

**Works well for:** maps with a clear ring topology (outer wall + inner obstacle).  
**Known limitation:** linear interpolation during resampling can cross a wall at
the loop-closure seam if the first and last polar samples are far apart.
If a wall hit occurs on the first lap, delete the cache and try `example_map`.

---

## 12. Gotchas and debugging checklist

| Symptom | Root cause | Fix |
|---|---|---|
| Car steers into wall at t=0 | Used CSV `psi_rad` as heading; actual heading ≈90° different | Compute heading from `arctan2(dy, dx)` of xy tangent |
| Collision at t<5 s, two-car run | Faster car catches slower on shared centreline | Add rear-end guard or use `ATTACKER_GAP ≥ speed_diff × duration` |
| Collision at ~t, attacker only | Offset path hits wall at tight corner | Reduce offset; check EDT at offset waypoints |
| Stray float printed in output | Variable name collision: `sep` used for both a `─────` bar string and a float | Rename the separator string (e.g., `bar`) |
| `gym_done` immediately at step 0 | Starting pose is in a wall or on top of the other car | Check EDT at starting positions; ensure Euclidean gap > `car_length` |
| Numerical instability (yaw rate > 100 rad/s) | Speed command too high for the curvature, or starting in a wall | Reduce speed; verify poses; use RK4 integrator |
| `lap_counts` doesn't match arc-length progress | Gym uses a start-zone heuristic, not Frenet arc length | Use accumulated `line.signed_distance(s_new, s_old)` for consistent lap counting |

---

## 13. Quick-start snippet

```python
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))   # examples/

import gym
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.f110_env import F110Env

MAP_BASE = Path(__file__).parent.parent / 'example_map'
MAP_EXT  = '.png'

# Suppress render window
old_cbs = F110Env.render_callbacks
F110Env.render_callbacks = []

env = gym.make(
    'f110_gym:f110-v0',
    map=str(MAP_BASE), map_ext=MAP_EXT,
    num_agents=2, timestep=0.01, seed=0, integrator=Integrator.RK4,
)
# lf + lr must equal WHEELBASE constant
assert np.isclose(0.3302, env.params['lf'] + env.params['lr'], rtol=1e-3)

try:
    obs, _, done, _ = env.reset(np.array([[x0, y0, th0], [x1, y1, th1]]))
    for _ in range(int(20.0 / 0.01)):     # 20 s
        if done: break
        actions = np.array([[steer0, 3.0], [steer1, 3.5]])   # [steer, speed]
        obs, _, done, _ = env.step(actions)
        if np.any(obs['collisions']): break
finally:
    env.close()
    F110Env.render_callbacks = old_cbs
```
