"""
gym_setup.py — headless 2-agent F1TENTH pure-pursuit baseline.

NOTE: GYM_NOTES.md was not found in the repo.  API details are sourced from
HEURISTIC_NOTES.md (examples/heuristic_baseline/) and direct gym inspection.

Map selection (MAP_NAME constant):
  'berlin'      — built-in racing circuit; waypoints are auto-generated from
                  the bitmap if no .centerline.npy cache exists yet.
  'example_map' — RIVAL demo map; loads from companion config_example_map.yaml
                  → example_waypoints.csv; no generation needed.

No render window is opened at any point.  Set MAP_NAME at the top to switch maps.

Named constants (all in one place — edit to tune):
  LOOKAHEAD             pure-pursuit look-ahead radius (m)
  DEFENDER_SPEED        constant speed command for car 0 (m/s)
  ATTACKER_SPEED        constant speed command for car 1 (m/s)
  FOLLOWING_GAP         minimum centre-to-centre distance before the rear-end
                        guard clamps the attacker's speed to the defender's (m)
  TIMESTEP              physics integration dt (s)
  WHEELBASE             bicycle model lf + lr (m)

Both cars follow the same centreline.  The attacker runs at ATTACKER_SPEED until
it closes to within FOLLOWING_GAP of the defender, then matches the defender's
measured speed — keeping a safe (> car_length = 0.58 m) bumper-to-bumper gap.
"""

import sys
from pathlib import Path

import numpy as np
import yaml

# ── Repo-relative path setup ──────────────────────────────────────────────────
_HERE     = Path(__file__).parent.resolve()
_REPO     = _HERE.parent.parent           # f1tenth_gym_RIVAL/
_MAPS     = _REPO / 'gym' / 'f110_gym' / 'envs' / 'maps'
_EXAMPLES = _REPO / 'examples'
sys.path.insert(0, str(_EXAMPLES))

# ── Named constants ───────────────────────────────────────────────────────────
LOOKAHEAD      = 1.0    # m  — pure-pursuit look-ahead radius
DEFENDER_SPEED = 3.0    # m/s — car 0 (index 0) constant speed command
ATTACKER_SPEED = 3.5    # m/s — car 1 (index 1) constant speed command
TIMESTEP       = 0.01   # s  — physics integration dt (100 Hz)
WHEELBASE      = 0.3302 # m  — lf + lr for the 1:10 F1TENTH car
STEER_MAX      = 0.4189 # rad — physical steering limit (s_min/s_max in gym)
ATTACKER_GAP   = 2.0    # m  — arc distance behind defender at start
FOLLOWING_GAP  = 0.75   # m  — rear-end guard: attacker matches defender speed below this
SIM_DURATION   = 20.0   # s
PRINT_INTERVAL = 2.0    # s  — status line cadence

# Which map to use.
#   'example_map'  — RIVAL demo circuit; loads waypoints from pre-existing CSV.
#                    Recommended: reliable loop closure, wide corridor. ← default
#   'berlin'       — gym built-in racing circuit; waypoints are auto-generated
#                    from the bitmap. The polar-EDT generation can produce a rough
#                    seam near loop closure; use 'example_map' if you see a wall hit.
MAP_NAME = 'example_map'

# ── Map path resolution ───────────────────────────────────────────────────────

def _map_paths(name: str):
    """Return (map_base: Path, map_ext: str) for a known map name."""
    if name == 'example_map':
        return _EXAMPLES / 'example_map', '.png'
    return _MAPS / name, '.png'


def _xy_to_waypoints(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return (N, 3) [x, y, heading_rad] with headings from geometric tangent.

    The psi column in RIVAL waypoint CSVs is a local raceline angle, NOT a
    world-frame heading.  ClosedRaceline always recomputes heading from the xy
    tangent; this helper does the same so pure-pursuit headings are consistent.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    # Closed-loop tangent: last edge wraps back to first vertex
    dx = np.append(np.diff(xs), xs[0] - xs[-1])
    dy = np.append(np.diff(ys), ys[0] - ys[-1])
    return np.column_stack([xs, ys, np.arctan2(dy, dx)])


# ── Waypoint loading / generation ─────────────────────────────────────────────

def load_or_generate_waypoints(map_base: Path, map_ext: str) -> np.ndarray:
    """Return (N, 3) float array [x_m, y_m, heading_rad].

    Loading order:
      1. Cached <map_base>.centerline.npy   — fastest; written on first generation.
      2. Companion CSV at <map_base>.csv or <map_base>_waypoints.csv
         — expects semicolon-delimited, 3 header rows, columns s;x;y;psi;…
      3. RIVAL-style config: config_<name>.yaml with wpt_path / wpt_delim / … keys
         — used automatically for example_map (config_example_map.yaml).
      4. Auto-generate from map bitmap using EDT polar-ray sampling.
         — caches result to .centerline.npy for subsequent runs.
    """
    cache = Path(str(map_base) + '.centerline.npy')

    if cache.exists():
        wpts = np.load(cache)
        print(f'  cached  : {len(wpts)} pts  ({cache.name})')
        return wpts

    # CSV companions (s;x;y;psi;… format, 3-row header)
    for csv in (map_base.with_suffix('.csv'),
                Path(str(map_base) + '_waypoints.csv')):
        if csv.exists():
            data = np.loadtxt(str(csv), delimiter=';', skiprows=3)
            wpts = _xy_to_waypoints(data[:, 1], data[:, 2])
            np.save(cache, wpts)
            print(f'  CSV     : {len(wpts)} pts  ({csv.name})')
            return wpts

    # RIVAL-style companion config  config_<name>.yaml → wpt_path
    rival_cfg = map_base.parent / f'config_{map_base.name}.yaml'
    if rival_cfg.exists():
        with rival_cfg.open() as fh:
            cfg = yaml.safe_load(fh)
        if 'wpt_path' in cfg:
            wpt_file = (rival_cfg.parent / cfg['wpt_path']).resolve()
            if wpt_file.exists():
                delim = cfg.get('wpt_delim', ';')
                skip  = int(cfg.get('wpt_rowskip', 3))
                xi    = int(cfg.get('wpt_xind',  1))
                yi    = int(cfg.get('wpt_yind',  2))
                data  = np.loadtxt(str(wpt_file), delimiter=delim, skiprows=skip)
                wpts  = _xy_to_waypoints(data[:, xi], data[:, yi])
                np.save(cache, wpts)
                print(f'  config  : {len(wpts)} pts  ({wpt_file.name})')
                return wpts

    # Generate from map image (EDT polar-ray sampling)
    img_path  = map_base.with_suffix(map_ext)
    yaml_path = map_base.with_suffix('.yaml')
    if not img_path.exists():
        raise FileNotFoundError(f'Map image not found: {img_path}')
    if not yaml_path.exists():
        raise FileNotFoundError(f'Map YAML not found: {yaml_path}')
    print(f'  Generating centerline for {map_base.name} '
          f'(polar EDT, cached after first run)…')
    wpts = _generate_polar_edt(img_path, yaml_path, cache)
    print(f'  generated: {len(wpts)} pts  → {cache.name}')
    return wpts


def _generate_polar_edt(img_path: Path, yaml_path: Path,
                         cache_path: Path, n_angles: int = 360) -> np.ndarray:
    """Generate a closed-loop centerline using EDT polar-ray sampling.

    For maps whose free space forms a ring around an inner wall obstacle
    (the typical F1TENTH racing-circuit topology), rays cast from a point
    inside the enclosed area each cross the medial axis exactly once.
    The max-EDT sample on each ray is a medial-axis waypoint.

    Centroid selection:
      - If the wall image has a connected component NOT touching the image
        boundary (i.e., an inner wall encircled by the track), its centroid
        is used — this point is guaranteed to be inside the enclosed area.
      - Otherwise fall back to the bounding-box centre of the free space.
    """
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, label

    with yaml_path.open() as fh:
        meta = yaml.safe_load(fh)
    res = float(meta['resolution'])             # m/px
    ox  = float(meta['origin'][0])
    oy  = float(meta['origin'][1])

    # Gym convention: flip top-to-bottom before use
    raw  = np.array(Image.open(str(img_path)).convert('L'))[::-1]
    free = raw > 128
    edt  = distance_transform_edt(free)         # px
    H, W = edt.shape

    # Minimum EDT clearance for a valid waypoint pose (car half-width + margin)
    min_clr = max(2.0, 0.155 / res + 1.0)

    # --- Find a seed point inside the enclosed area ---
    # Look for a wall component that is NOT connected to the image boundary.
    # Such a component is the inner obstacle; its centroid lies inside the circuit.
    wall = ~free
    labeled_wall, n_wall = label(wall)

    # Collect labels that touch the image boundary
    edge_labels: set = set()
    edge_labels.update(labeled_wall[0,  :].ravel())
    edge_labels.update(labeled_wall[-1, :].ravel())
    edge_labels.update(labeled_wall[:,  0].ravel())
    edge_labels.update(labeled_wall[:, -1].ravel())
    edge_labels.discard(0)

    cr, cc = None, None
    best_inner_size = 0
    for cid in range(1, n_wall + 1):
        if cid in edge_labels:
            continue
        size = int((labeled_wall == cid).sum())
        if size > best_inner_size:
            best_inner_size = size
            rr, rc = np.where(labeled_wall == cid)
            cr, cc = float(rr.mean()), float(rc.mean())

    if cr is None:
        # No inner wall found — use bounding-box centre of the free space
        rf, cf = np.where(free)
        cr = float((rf.min() + rf.max()) / 2)
        cc = float((cf.min() + cf.max()) / 2)

    # --- Cast n_angles rays from (cr, cc); sample max-EDT point on each ---
    angles    = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    pts_r: list = []
    pts_c: list = []

    max_ray = int(np.hypot(H, W)) + 1
    ts       = np.arange(1, max_ray, dtype=float)

    for ang in angles:
        ray_r = np.round(cr + ts * np.sin(ang)).astype(int)
        ray_c = np.round(cc + ts * np.cos(ang)).astype(int)
        ok    = (ray_r >= 0) & (ray_r < H) & (ray_c >= 0) & (ray_c < W)
        rr, rc = ray_r[ok], ray_c[ok]
        if rr.size == 0:
            continue
        vals = edt[rr, rc]
        mx   = int(np.argmax(vals))
        if vals[mx] >= min_clr:
            pts_r.append(int(rr[mx]))
            pts_c.append(int(rc[mx]))

    if len(pts_r) < 20:
        raise RuntimeError(
            f'Centerline generation found only {len(pts_r)} points for '
            f'{img_path.name}. The map may not have a closed-ring topology.\n'
            'Fix: set MAP_NAME = "example_map" to use the RIVAL demo map.')

    # World coordinates
    xs = ox + np.array(pts_c, dtype=float) * res
    ys = oy + np.array(pts_r, dtype=float) * res

    # Sort by polar angle from seed centroid (gives CCW ordering)
    cx       = ox + cc * res
    cy       = oy + cr * res
    ang_sort = np.arctan2(ys - cy, xs - cx)

    # Deduplicate: round to 1-cm grid, keep unique, re-sort by angle
    keys      = np.round(np.column_stack([xs, ys]) * 100).astype(int)
    _, unique = np.unique(keys, axis=0, return_index=True)
    unique    = unique[np.argsort(ang_sort[unique])]
    xs, ys    = xs[unique], ys[unique]

    # Resample to uniform ~0.2 m spacing so large polar gaps don't cause jerky steering
    pts = np.column_stack([xs, ys])
    pts = _uniform_resample(pts, spacing=0.2)
    xs, ys = pts[:, 0], pts[:, 1]

    # Headings from consecutive tangent vectors (closed loop)
    dx = np.append(np.diff(xs), xs[0] - xs[-1])
    dy = np.append(np.diff(ys), ys[0] - ys[-1])
    headings = np.arctan2(dy, dx)

    wpts = np.column_stack([xs, ys, headings])
    np.save(cache_path, wpts)
    return wpts


def _uniform_resample(pts: np.ndarray, spacing: float = 0.2) -> np.ndarray:
    """Resample a closed polygon to approximately uniform arc-length spacing."""
    # Close the loop for arc-length calculation
    pts_c = np.vstack([pts, pts[0]])
    segs  = np.linalg.norm(np.diff(pts_c, axis=0), axis=1)
    total = float(segs.sum())
    n_new = max(10, int(total / spacing))
    s_old = np.concatenate([[0.], np.cumsum(segs)])
    s_new = np.linspace(0., total, n_new, endpoint=False)
    new_x = np.interp(s_new, s_old, pts_c[:, 0])
    new_y = np.interp(s_new, s_old, pts_c[:, 1])
    return np.column_stack([new_x, new_y])


# ── Pure pursuit ──────────────────────────────────────────────────────────────

def pure_pursuit(car_x: float, car_y: float, car_theta: float,
                 waypoints: np.ndarray, lookahead: float) -> float:
    """Return steering angle (rad) for one bicycle-model step.

    Parameters
    ----------
    car_x, car_y, car_theta  World pose; theta=0 → east (+x), CCW positive.
    waypoints                (N, ≥2) array; column 0 = x, column 1 = y.
    lookahead                Look-ahead distance in metres (LOOKAHEAD constant).

    Algorithm
    ---------
    1. Find the nearest waypoint index by Euclidean distance.
    2. Walk forward (wrapping around the closed path) to the first waypoint
       whose distance from the car is ≥ lookahead.
    3. Transform that point into the car frame:
           y_car = sin(−θ)·Δx + cos(−θ)·Δy
       (matches get_actuation() in examples/waypoint_follow.py)
    4. Curvature: κ = 2·y_car / lookahead²
    5. Steer:  δ = arctan(WHEELBASE·κ) = arctan(2·WHEELBASE·y_car / lookahead²)
       Clipped to ±STEER_MAX.
    """
    n      = len(waypoints)
    car_xy = np.array([car_x, car_y], dtype=float)
    wxy    = waypoints[:, :2]

    # 1. Nearest waypoint
    nearest = int(np.argmin(np.sum((wxy - car_xy) ** 2, axis=1)))

    # 2. First waypoint at or beyond lookahead (wrap-around)
    lookahead_pt = None
    for k in range(n):
        idx = (nearest + k) % n
        if np.sqrt(np.sum((wxy[idx] - car_xy) ** 2)) >= lookahead:
            lookahead_pt = wxy[idx]
            break

    if lookahead_pt is None:
        return 0.0  # degenerate: waypoints too sparse or car lost

    # 3. Transform to car frame
    dx    = lookahead_pt[0] - car_x
    dy    = lookahead_pt[1] - car_y
    y_car = np.sin(-car_theta) * dx + np.cos(-car_theta) * dy

    # 4–5. Pure-pursuit steering (Coulter 1992)
    steer = np.arctan2(2.0 * WHEELBASE * y_car, lookahead ** 2)
    return float(np.clip(steer, -STEER_MAX, STEER_MAX))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arc_offset(waypoints: np.ndarray, start_idx: int,
                arc: float, direction: int = 1):
    """Walk arc metres from start_idx along the path; return (x, y, theta, idx)."""
    n            = len(waypoints)
    accumulated  = 0.0
    idx          = start_idx
    for _ in range(n):
        nxt  = (idx + direction) % n
        step = float(np.linalg.norm(waypoints[nxt, :2] - waypoints[idx, :2]))
        if accumulated + step >= arc:
            break
        accumulated += step
        idx = nxt
        if idx == start_idx:   # full loop without reaching arc
            break
    return (float(waypoints[idx, 0]), float(waypoints[idx, 1]),
            float(waypoints[idx, 2]), idx)


def _offset_path(waypoints: np.ndarray, offset_m: float) -> np.ndarray:
    """Return waypoints shifted offset_m metres left of the heading direction.

    Left normal in a right-hand frame: n = (−sin θ, cos θ).
    Positive offset moves the path to the left of travel; negative to the right.
    Headings are unchanged (the shifted path is parallel to the original).
    """
    if offset_m == 0.0:
        return waypoints
    h  = waypoints[:, 2]
    dx = -np.sin(h) * offset_m
    dy =  np.cos(h) * offset_m
    return np.column_stack([waypoints[:, 0] + dx,
                             waypoints[:, 1] + dy,
                             h])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import gym
    from f110_gym.envs.base_classes import Integrator
    from f110_gym.envs.f110_env import F110Env

    # ── Map / waypoints ──────────────────────────────────────────────────
    map_base, map_ext = _map_paths(MAP_NAME)
    print(f'Map  : {map_base.name}{map_ext}')
    print('Waypoints:')
    try:
        waypoints = load_or_generate_waypoints(map_base, map_ext)
        arc = float(np.sum(np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)))
        if len(waypoints) < 20 or arc < 5.0:
            raise RuntimeError(f'Too few waypoints or circuit too short '
                               f'({len(waypoints)} pts, arc={arc:.1f} m)')
    except Exception as exc:
        print(f'  {MAP_NAME} generation failed: {exc}')
        print('  Falling back to example_map (uses pre-existing waypoints CSV).')
        map_base, map_ext = _map_paths('example_map')
        waypoints = load_or_generate_waypoints(map_base, map_ext)
        arc = float(np.sum(np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)))

    arc = float(np.sum(np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)))
    print(f'  {len(waypoints)} waypoints  •  circuit ≈ {arc:.1f} m')

    # ── Initial poses ────────────────────────────────────────────────────
    # Both cars follow the same centreline.  Attacker starts ATTACKER_GAP m
    # behind.  A rear-end guard (see step loop) prevents collision when
    # the faster attacker closes in.
    start_idx = 0
    def_x, def_y, def_th = (float(waypoints[start_idx, 0]),
                             float(waypoints[start_idx, 1]),
                             float(waypoints[start_idx, 2]))
    atk_x, atk_y, atk_th, _ = _arc_offset(waypoints, start_idx,
                                            ATTACKER_GAP, direction=-1)
    poses = np.array([[def_x, def_y, def_th],
                      [atk_x, atk_y, atk_th]])
    sep = float(np.hypot(def_x - atk_x, def_y - atk_y))
    print(f'\nInitial poses:')
    print(f'  Defender  ({MAP_NAME}): '
          f'({def_x:.3f}, {def_y:.3f})  θ={np.degrees(def_th):.1f}°')
    print(f'  Attacker  (~{ATTACKER_GAP:.1f} m behind): '
          f'({atk_x:.3f}, {atk_y:.3f})  θ={np.degrees(atk_th):.1f}°')
    print(f'  Euclidean gap at reset: {sep:.3f} m')

    # ── Environment (headless) ───────────────────────────────────────────
    old_callbacks = F110Env.render_callbacks
    F110Env.render_callbacks = []          # never open a render window
    env = gym.make(
        'f110_gym:f110-v0',
        map=str(map_base),
        map_ext=map_ext,
        num_agents=2,
        timestep=TIMESTEP,
        seed=0,
        integrator=Integrator.RK4,
    )
    actual_wb = env.params['lf'] + env.params['lr']
    if not np.isclose(WHEELBASE, actual_wb, rtol=1e-3):
        print(f'Warning: WHEELBASE={WHEELBASE} vs gym lf+lr={actual_wb:.6f}')

    try:
        obs, _, done, _ = env.reset(poses)

        # Abort early if the starting positions are invalid
        if np.any(obs['collisions']):
            # Delete the bad cache so the next run regenerates
            cache = Path(str(map_base) + '.centerline.npy')
            if cache.exists():
                cache.unlink()
                print('Deleted stale centerline cache.')
            raise RuntimeError(
                'Collision at reset — waypoints land in a wall.\n'
                'The cache has been deleted; re-run to regenerate.\n'
                'If the problem persists set MAP_NAME = "example_map".')

        # ── Simulation loop ──────────────────────────────────────────────
        dist        = np.zeros(2)
        prev_x      = np.array([obs['poses_x'][i] for i in range(2)])
        prev_y      = np.array([obs['poses_y'][i] for i in range(2)])
        next_print  = PRINT_INTERVAL
        total_steps = int(SIM_DURATION / TIMESTEP)
        crash       = False

        hdr = (f'{"t(s)":>5}  '
               f'{"def_x":>8} {"def_y":>8} {"def_spd":>8} {"def_dist":>9}  '
               f'{"atk_x":>8} {"atk_y":>8} {"atk_spd":>8} {"atk_dist":>9}')
        bar = '─' * len(hdr)
        print(f'\n{hdr}\n{bar}')

        for step in range(total_steps):
            if done:
                print(f'Gym done at step {step} (t={step * TIMESTEP:.2f} s)')
                break

            # Pure-pursuit steering — both cars track the same centreline
            steer0 = pure_pursuit(
                obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                waypoints, LOOKAHEAD)
            steer1 = pure_pursuit(
                obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],
                waypoints, LOOKAHEAD)

            # Rear-end guard: once attacker closes to FOLLOWING_GAP, match
            # the defender's measured speed instead of commanding ATTACKER_SPEED.
            sep = float(np.hypot(obs['poses_x'][0] - obs['poses_x'][1],
                                  obs['poses_y'][0] - obs['poses_y'][1]))
            atk_v = (abs(obs['linear_vels_x'][0])   # match defender
                     if sep < FOLLOWING_GAP
                     else ATTACKER_SPEED)

            # Actions: [[steer_angle, speed], ...]  (legacy gym API)
            actions = np.array([[steer0, DEFENDER_SPEED],
                                [steer1, atk_v]])
            obs, _, done, _ = env.step(actions)

            t    = (step + 1) * TIMESTEP
            cx   = np.array([obs['poses_x'][i] for i in range(2)])
            cy   = np.array([obs['poses_y'][i] for i in range(2)])
            dist += np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
            prev_x, prev_y = cx.copy(), cy.copy()

            if np.any(obs['collisions']):
                print(f'\n  COLLISION at t={t:.2f} s — '
                      f'cars {list(np.where(np.asarray(obs["collisions"]))[0])}')
                crash = True
                break

            if t >= next_print - TIMESTEP * 0.5:
                spd = [abs(obs['linear_vels_x'][i]) for i in range(2)]
                print(f'{t:>5.1f}  '
                      f'{cx[0]:>8.3f} {cy[0]:>8.3f} {spd[0]:>8.3f} {dist[0]:>9.3f}  '
                      f'{cx[1]:>8.3f} {cy[1]:>8.3f} {spd[1]:>8.3f} {dist[1]:>9.3f}')
                next_print += PRINT_INTERVAL

        print(f'\n{bar}')
        print(f'Defender  : {dist[0]:.2f} m  ({dist[0] / SIM_DURATION:.2f} m/s avg)')
        print(f'Attacker  : {dist[1]:.2f} m  ({dist[1] / SIM_DURATION:.2f} m/s avg)')
        result = 'FAIL — collision' if crash else 'OK — neither car crashed'
        print(f'Result    : {result}')

    finally:
        env.close()
        F110Env.render_callbacks = old_callbacks


if __name__ == '__main__':
    main()
