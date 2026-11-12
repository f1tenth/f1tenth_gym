"""
baseline.py — Mode A mirror-blocking defender, scripted attacker, and episode runner.

Defender Mode A
---------------
When the attacker is within ENGAGE_GAP metres behind, the defender copies the
attacker's lateral offset d (clamped to the safe track limits) and follows the
resulting offset path via pure pursuit.  Outside that range it returns to CENTER.

Scripted attacker scenarios
----------------------------
straight   — holds CENTER for the entire run
move_left  — approaches then shifts to LEFT and holds
move_right — approaches then shifts to RIGHT and holds
feint      — approaches, shifts one way, reverses after FEINT_SWITCH seconds
late_move  — approaches until within LATE_CLOSE m (bumper), then shifts to LEFT

Wall-safety
-----------
offset paths are corridor-clamped per waypoint using TrackMap.corridor().
run_batch precomputes corridor arrays once; run_episode accepts them as _lo/_hi
and also computes them on first use when called standalone.

run_episode API
---------------
run_episode(scenario, defender_mode='A', seed=0) -> dict with keys:
    overtaken, collision, time_blocking, mean_attacker_speed, defender_progress
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

_HERE   = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent))   # examples/ on path for rival_geometry

from rival_geometry import ClosedRaceline
from track_utils import LEFT, CENTER, RIGHT, _SAFE_D_MAX, _track_length, offset_waypoints
from idm import (idm_accel, gym_speed_command,
                 V0_DEFENDER, V0_ATTACKER, TIMESTEP, CAR_LENGTH, LAT_THRESHOLD,
                 IDM_A_MAX, IDM_B, IDM_S0)

# ── Simulation constants ──────────────────────────────────────────────────────
LOOKAHEAD    = 1.0      # pure-pursuit lookahead radius (m)
WHEELBASE    = 0.3302   # m (lf + lr for the 1:10 F1TENTH car)
STEER_MAX    = 0.4189   # rad (s_max from gym params)
START_S      = 23.0     # starting arc-length on the centreline (m)
START_GAP    = 5.0      # nominal initial centre-to-centre gap (m)
ENGAGE_GAP   = 5.0      # defender activates when attacker is within this m behind
SIM_DURATION = 30.0     # s — maximum episode length
FEINT_SWITCH = 1.0      # s — feint reversal delay after phase-1 starts

# LATE_CLOSE is the bumper gap at which late_move shifts AND the safety-brake
# threshold for the attacker: the attacker uses free-road speed until it is
# within this distance in the same lane, so it arrives at the maneuver point
# at full speed.  Tight enough to allow last-moment lane changes.
LATE_CLOSE   = 1.5      # m

# Tight IDM for the safety-brake case (same headway as _ATK_T).
_ATK_T = 0.3

# ── Mode B (MOBIL) constants ──────────────────────────────────────────────────
MOBIL_INTERVAL = 0.2    # s  — lane re-evaluation period
B_SAFE         = 4.0    # m/s² — attacker braking threshold: reject if accel < -B_SAFE
DELTA_A_TH     = 0.3    # m/s² — min blocking-value gain to justify a lane switch
SWITCH_BIAS    = 0.1    # m/s² — hysteresis: extra cost on any switch
LANE_OPTIONS   = (LEFT, CENTER, RIGHT)   # discrete candidate lanes
# Lateral alignment threshold: defender affects attacker IDM only within this distance
_ALIGN_THRESH  = 0.31 + 0.12   # car_width + separation_margin (m)

SCENARIOS = ('straight', 'move_left', 'move_right', 'feint', 'late_move')

MAP_BASE = str(_HERE.parent / 'example_map')
MAP_EXT  = '.png'
_WP_FILE = _HERE.parent / 'example_waypoints.csv'

# Conservative car-centre clearance from the wall used for corridor queries.
_CORRIDOR_CLEARANCE = 0.155 + 0.04 + 0.08  # car_half_width + margin + overshoot buffer (m)


# ── IDM for attacker (tight headway) ─────────────────────────────────────────

def _atk_idm(v, v_leader, gap, v0):
    """IDM with tight time headway T=_ATK_T for the attacker."""
    v   = max(float(v),   0.)
    gap = max(float(gap), 0.1)
    dv  = v - float(v_leader)
    s_star = IDM_S0 + v * _ATK_T + v * dv / (2.0 * np.sqrt(IDM_A_MAX * IDM_B))
    return float(IDM_A_MAX * (1.0 - (v / max(float(v0), 1e-6)) ** 4 - (s_star / gap) ** 2))


# ── Static helpers ────────────────────────────────────────────────────────────

def _load_wp():
    return np.loadtxt(_WP_FILE, delimiter=';', skiprows=3)


def _make_line(wp):
    return ClosedRaceline(wp[:, 1:3])


def _precompute_corridors(line):
    """
    Return per-waypoint safe corridor arrays (lo_arr, hi_arr).

    Queries TrackMap.corridor at each waypoint arc-length using
    _CORRIDOR_CLEARANCE.  Called once per batch; takes ~0.3 s on this map.
    """
    from rival_geometry import TrackMap
    track   = TrackMap(MAP_BASE)
    n       = len(line.points)
    lo_arr  = np.full(n, -0.5)
    hi_arr  = np.full(n,  0.5)
    for i in range(n):
        s = float(line._starts[i])
        try:
            lo, hi = track.corridor(line, s, clearance=_CORRIDOR_CLEARANCE,
                                    max_offset=0.5)
            lo_arr[i] = lo
            hi_arr[i] = hi
        except Exception:
            pass   # keep ±0.5 fallback if EDT query fails
    return lo_arr, hi_arr


def _safe_path(wp, d, lo_arr, hi_arr):
    """
    Offset path clamped per-waypoint to the safe corridor.

    Like offset_waypoints(wp, d)[:, 1:3] but clips d[i] to [lo_arr[i], hi_arr[i]]
    so the path never enters a wall.  Returns an Nx2 xy array.
    """
    pts = np.array(wp[:, 1:3], dtype=float)
    cx, cy = pts[:, 0], pts[:, 1]
    ex = np.roll(cx, -1) - cx
    ey = np.roll(cy, -1) - cy
    seg_len = np.hypot(ex, ey)
    valid   = seg_len > 1e-9
    safe_sl = np.where(valid, seg_len, 1.)
    tx = np.where(valid, ex / safe_sl, 0.)
    ty = np.where(valid, ey / safe_sl, 0.)
    # lo_arr/hi_arr have len(line.points) entries; wp may have one extra closing row.
    n_wp = len(pts)
    n_co = len(lo_arr)
    if n_co < n_wp:
        # Pad corridor arrays by repeating the last entry for the duplicate closing wp.
        lo_full = np.append(lo_arr, lo_arr[-1])
        hi_full = np.append(hi_arr, hi_arr[-1])
    else:
        lo_full, hi_full = lo_arr, hi_arr
    d_raw = np.clip(float(d), lo_full[:n_wp], hi_full[:n_wp])
    # Smooth over ~3 m (15 waypoints × 0.2 m) to remove kinks that cause
    # pure-pursuit lookahead overshoot at tight corners.
    _K = 15
    d_arr = np.convolve(d_raw, np.ones(_K) / _K, mode='same')
    d_arr = np.clip(d_arr, lo_full[:n_wp], hi_full[:n_wp])  # re-clamp after smooth
    pts[:, 0] += d_arr * (-ty)
    pts[:, 1] += d_arr * tx
    return pts


def pure_pursuit(car_x, car_y, car_theta, path_xy,
                 lookahead=LOOKAHEAD, wheelbase=WHEELBASE):
    """
    Pure-pursuit steering angle (rad) for a waypoint path.

    Finds the nearest point on path_xy, then walks forward (wrapping) to the
    first point at least `lookahead` metres away.  Formula: Coulter (1992).
    """
    pos   = np.array([car_x, car_y], dtype=float)
    dists = np.linalg.norm(path_xy - pos, axis=1)
    near  = int(np.argmin(dists))
    n     = len(path_xy)
    target = path_xy[(near + 1) % n]   # fallback: next waypoint
    for k in range(1, n):
        idx = (near + k) % n
        if np.linalg.norm(path_xy[idx] - pos) >= lookahead:
            target = path_xy[idx]
            break
    dx    = target[0] - car_x
    dy    = target[1] - car_y
    y_car = np.sin(-car_theta) * dx + np.cos(-car_theta) * dy
    steer = np.arctan2(2.0 * wheelbase * y_car, lookahead ** 2)
    return float(np.clip(steer, -STEER_MAX, STEER_MAX))


# ── Defender ──────────────────────────────────────────────────────────────────

class ModeADefender:
    """
    Mode A mirror defender.

    Copies the attacker's lateral offset when within ENGAGE_GAP, otherwise
    holds CENTER.  Offset paths are corridor-clamped to avoid walls.
    """

    def __init__(self, lo_arr=None, hi_arr=None):
        self._lo = lo_arr
        self._hi = hi_arr

    def reset(self):
        pass   # no mutable state; corridor arrays are fixed per episode

    def step(self, x_def, y_def, th_def, d_atk, gap, wp):
        """
        Parameters
        ----------
        d_atk : float — attacker's measured lateral offset
        gap   : float — signed longitudinal gap (positive = defender ahead)

        Returns
        -------
        steer    : float — steering angle command (rad)
        d_target : float — lateral offset the defender is targeting
        """
        if 0.0 < gap < ENGAGE_GAP:
            d_target = float(np.clip(d_atk, -_SAFE_D_MAX, _SAFE_D_MAX))
        else:
            d_target = CENTER

        if self._lo is not None:
            path = _safe_path(wp, d_target, self._lo, self._hi)
        else:
            path = offset_waypoints(wp, d_target)[:, 1:3]

        steer = pure_pursuit(x_def, y_def, th_def, path)
        return steer, d_target


# ── Mode B defender ───────────────────────────────────────────────────────────

class ModeBDefender:
    """
    Mode B MOBIL-inspired lane-selector.

    Re-evaluates the optimal blocking lane every MOBIL_INTERVAL seconds using
    IDM-based blocking values.  An observation-delay buffer (obs_delay > 0)
    simulates perception latency; ~0.3 s makes the feint scenario effective
    because a quick direction reversal fools the defender reacting to stale data.

    MOBIL logic (Kesting et al. 2007, adapted for blocking):
      1. Predict the attacker's target lane from observed d + lat_vel × 0.5 s.
      2. Blocking value B(candidate) = max(0, free_accel − blocked_accel);
         non-zero only when candidate equals the predicted attacker lane.
      3. Safety criterion: reject the switch if attacker accel after the move
         would drop below −B_SAFE.
      4. Incentive criterion: switch to the best adjacent lane only when
         B(best) − B(current) > DELTA_A_TH + SWITCH_BIAS.
    """

    def __init__(self, lo_arr=None, hi_arr=None, obs_delay=0.0):
        self._lo          = lo_arr
        self._hi          = hi_arr
        self.obs_delay    = float(obs_delay)
        self._obs_buf     = deque()   # (t, d_atk, v_atk)
        self._d_target    = CENTER
        self._mobil_timer = 0.0       # counts down to next MOBIL evaluation
        self._debug       = ''

    def reset(self):
        self._obs_buf.clear()
        self._d_target    = CENTER
        self._mobil_timer = 0.0
        self._debug       = ''

    # ── Observation delay buffer ─────────────────────────────────────────────

    def _push(self, t, d_atk, v_atk):
        """Buffer the attacker's true state; trim entries older than obs_delay+1 s."""
        self._obs_buf.append((t, d_atk, v_atk))
        cutoff = t - max(self.obs_delay, 0.0) - 1.0
        while len(self._obs_buf) > 1 and self._obs_buf[0][0] < cutoff:
            self._obs_buf.popleft()

    def _get_delayed(self, t):
        """
        Return (d_atk, v_atk, lat_vel) from obs_delay seconds ago.

        Lateral velocity is estimated from the entry just before the returned one.
        With obs_delay=0 this is essentially the current state; with obs_delay=0.3 s
        the defender sees the attacker as it was 0.3 s ago — feints become effective.
        """
        if not self._obs_buf:
            return CENTER, V0_ATTACKER, 0.0
        target_t = t - self.obs_delay
        result     = self._obs_buf[0]
        result_idx = 0
        for i, entry in enumerate(self._obs_buf):
            if entry[0] <= target_t:
                result     = entry
                result_idx = i
            else:
                break
        t_r, d_r, v_r = result
        lat_vel = 0.0
        if result_idx > 0:
            t_p, d_p, _ = self._obs_buf[result_idx - 1]
            dt = t_r - t_p
            if dt > 1e-9:
                lat_vel = (d_r - d_p) / dt
        return d_r, v_r, lat_vel

    # ── MOBIL lane selection ─────────────────────────────────────────────────

    def _mobil(self, obs_d_atk, obs_lat_vel, obs_v_atk, v_def, gap):
        """
        Evaluate which lane maximises the blocking value.

        Returns (best_lane_d, debug_string).
        """
        bumper_gap = max(gap - CAR_LENGTH, 0.1)

        # Predict the attacker's target lane
        d_pred    = obs_d_atk + obs_lat_vel * 0.5
        pred_lane = min(LANE_OPTIONS, key=lambda d: abs(d - d_pred))

        # Index of current lane in LANE_OPTIONS
        cur_lane = self._d_target
        cur_idx  = min(range(len(LANE_OPTIONS)),
                       key=lambda i: abs(LANE_OPTIONS[i] - cur_lane))

        # Attacker free-road acceleration (no leader)
        atk_free = idm_accel(obs_v_atk, obs_v_atk, 1e9, V0_ATTACKER)

        def _bv(cand_d):
            """Blocking value: reduction in attacker accel if defender is at cand_d."""
            if cand_d != pred_lane:
                return 0.0                           # only block predicted lane
            if abs(obs_d_atk - cand_d) >= _ALIGN_THRESH:
                return 0.0                           # attacker not laterally aligned
            atk_blk = idm_accel(obs_v_atk, v_def, bumper_gap, V0_ATTACKER)
            return max(0.0, atk_free - atk_blk)

        def _atk_a_after(cand_d):
            """Attacker IDM accel if defender were at cand_d (for safety check)."""
            if abs(obs_d_atk - cand_d) >= _ALIGN_THRESH:
                return atk_free
            return idm_accel(obs_v_atk, v_def, bumper_gap, V0_ATTACKER)

        # Only consider adjacent lanes (no skipping)
        candidates = []
        if cur_idx > 0:
            candidates.append(LANE_OPTIONS[cur_idx - 1])
        if cur_idx < len(LANE_OPTIONS) - 1:
            candidates.append(LANE_OPTIONS[cur_idx + 1])

        B_cur     = _bv(cur_lane)
        best_lane = cur_lane
        best_B    = -1e9
        found     = False
        rejected  = False

        for cand in candidates:
            B_cand = _bv(cand)
            # Safety criterion: reject if the move forces attacker into hard braking
            if _atk_a_after(cand) < -B_SAFE:
                rejected = True
                continue
            # Incentive criterion: improvement must exceed threshold + hysteresis
            if B_cand - B_cur > DELTA_A_TH + SWITCH_BIAS and B_cand > best_B:
                best_B    = B_cand
                best_lane = cand
                found     = True

        if found:
            direction = 'LEFT' if best_lane > cur_lane else 'RIGHT'
            debug = f'MOBIL: switch {direction} (B={best_B:.1f})'
        elif rejected:
            debug = 'MOBIL: rejected (unsafe)'
        else:
            debug = 'MOBIL: stay'

        return best_lane, debug

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, t, x_def, y_def, th_def, d_atk, v_atk, gap, v_def, wp):
        """
        Parameters
        ----------
        t     : float — current elapsed time (s)
        d_atk : float — attacker's true lateral offset (buffered for OBS_DELAY)
        v_atk : float — attacker's true speed
        gap   : float — signed longitudinal gap (positive = defender ahead)
        v_def : float — defender's current speed (used in MOBIL blocking value)

        Returns
        -------
        steer    : float — steering angle command (rad)
        d_target : float — current target lane offset
        """
        # Buffer the attacker's true state; MOBIL reads the delayed version
        self._push(t, d_atk, v_atk)

        # Re-evaluate lane choice when engaged and timer expires
        self._mobil_timer -= TIMESTEP
        if 0.0 < gap < ENGAGE_GAP:
            if self._mobil_timer <= 0:
                self._mobil_timer = MOBIL_INTERVAL
                obs_d, obs_v, obs_lat_v = self._get_delayed(t)
                self._d_target, self._debug = self._mobil(
                    obs_d, obs_lat_v, obs_v, v_def, gap)
        else:
            self._d_target = CENTER
            self._debug    = ''

        if self._lo is not None:
            path = _safe_path(wp, self._d_target, self._lo, self._hi)
        else:
            path = offset_waypoints(wp, self._d_target)[:, 1:3]

        steer = pure_pursuit(x_def, y_def, th_def, path)
        return steer, self._d_target


# ── Attacker ──────────────────────────────────────────────────────────────────

class ScriptedAttacker:
    """
    Scripted lateral manoeuvre attacker.

    Phase logic per scenario:
      straight   — always CENTER
      move_left  — CENTER until gap < ENGAGE_GAP, then LEFT
      move_right — CENTER until gap < ENGAGE_GAP, then RIGHT
      feint      — CENTER → feint_dir*LEFT → -feint_dir*LEFT  (after FEINT_SWITCH s)
      late_move  — CENTER until bumper_gap < late_close, then LEFT

    Speed is controlled by _atk_idm (in run_episode); only lateral target is set here.
    """

    def __init__(self, scenario, rng, feint_switch_t=FEINT_SWITCH,
                 late_close=LATE_CLOSE, lo_arr=None, hi_arr=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'; choose from {SCENARIOS}")
        self.scenario       = scenario
        self.feint_switch_t = feint_switch_t
        self.late_close     = late_close
        self._phase         = 0
        self._phase_start   = 0.0
        self._feint_sign    = 1 if rng.random() < 0.5 else -1
        self.d_target       = CENTER
        self._lo            = lo_arr
        self._hi            = hi_arr
        self._path_cache    = {}   # d_target → Nx2 xy (small discrete set)

    def reset(self):
        self._phase       = 0
        self._phase_start = 0.0
        self.d_target     = CENTER
        self._path_cache  = {}

    def step(self, t, x_atk, y_atk, th_atk, gap, bumper_gap, wp):
        """
        Parameters
        ----------
        gap        : float — signed longitudinal gap (positive = defender ahead)
        bumper_gap : float — gap − CAR_LENGTH

        Returns
        -------
        steer    : float — steering angle command (rad)
        d_target : float — current lateral target
        """
        sc = self.scenario

        if sc == 'straight':
            self.d_target = CENTER

        elif sc == 'move_left':
            if self._phase == 0 and 0.0 < gap < ENGAGE_GAP:
                self._phase = 1
            self.d_target = LEFT if self._phase >= 1 else CENTER

        elif sc == 'move_right':
            if self._phase == 0 and 0.0 < gap < ENGAGE_GAP:
                self._phase = 1
            self.d_target = RIGHT if self._phase >= 1 else CENTER

        elif sc == 'feint':
            if self._phase == 0 and 0.0 < gap < ENGAGE_GAP:
                self._phase       = 1
                self._phase_start = t
            elif self._phase == 1 and t - self._phase_start >= self.feint_switch_t:
                self._phase = 2
            if self._phase == 1:
                self.d_target = self._feint_sign * abs(LEFT)
            elif self._phase == 2:
                self.d_target = -self._feint_sign * abs(LEFT)
            else:
                self.d_target = CENTER

        elif sc == 'late_move':
            if self._phase == 0 and bumper_gap > 0.0 and bumper_gap < self.late_close:
                self._phase = 1
            self.d_target = LEFT if self._phase >= 1 else CENTER

        # Cache corridor-clamped paths (d_target is one of ~3 discrete values)
        if self.d_target not in self._path_cache:
            if self._lo is not None:
                self._path_cache[self.d_target] = _safe_path(wp, self.d_target,
                                                              self._lo, self._hi)
            else:
                self._path_cache[self.d_target] = offset_waypoints(
                    wp, self.d_target)[:, 1:3]

        steer = pure_pursuit(x_atk, y_atk, th_atk, self._path_cache[self.d_target])
        return steer, self.d_target


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(scenario, defender_mode='A', seed=0,
                *, _env=None, _wp=None, _line=None, _tlen=None,
                _lo=None, _hi=None, _obs_delay=0.0, _render=False):
    """
    Run one episode and return metrics.

    Parameters
    ----------
    scenario      : str — one of SCENARIOS
    defender_mode : str — 'A' (mirror) or 'B' (MOBIL lane selector)
    seed          : int — RNG seed for start-condition noise
    _obs_delay    : float — Mode B perception latency in seconds (default 0.0);
                    ~0.3 s makes the feint scenario effective for Mode B

    Returns
    -------
    dict with keys:
        overtaken           bool
        collision           bool
        time_blocking       float (s) — time defender was actively blocking
        mean_attacker_speed float (m/s)
        defender_progress   float (m) — arc-length covered by the defender
    """
    if defender_mode not in ('A', 'B'):
        raise NotImplementedError(f"Defender mode '{defender_mode}' not implemented")

    rng = np.random.default_rng(seed)

    # ── Jitter start conditions ───────────────────────────────────────────────
    gap_0      = START_GAP    + rng.uniform(-0.5,  0.5)
    v_def_0    = V0_DEFENDER  + rng.uniform(-0.1,  0.1)
    v_atk_0    = V0_ATTACKER  + rng.uniform(-0.1,  0.1)
    t_feint    = FEINT_SWITCH + rng.uniform(-0.1,  0.1)
    late_close = LATE_CLOSE   + rng.uniform(-0.1,  0.1)

    # ── Shared resources ─────────────────────────────────────────────────────
    wp   = _wp   if _wp   is not None else _load_wp()
    line = _line if _line is not None else _make_line(wp)
    tlen = _tlen if _tlen is not None else _track_length(wp)

    # ── Corridor arrays (precomputed in run_batch; computed here if standalone)
    lo_arr = _lo if _lo is not None else _precompute_corridors(line)[0]
    hi_arr = _hi if _hi is not None else (
        _precompute_corridors(line)[1] if _lo is None else _hi)
    # Compact: if _lo was None, we computed both; grab properly.
    if _lo is None:
        lo_arr, hi_arr = _precompute_corridors(line)

    # ── Gym environment ───────────────────────────────────────────────────────
    own_env = (_env is None)
    if own_env:
        from f110_gym.envs.f110_env import F110Env
        from f110_gym.envs.base_classes import Integrator
        import gym
        _old_cbs = F110Env.render_callbacks
        F110Env.render_callbacks = []
        env = gym.make('f110_gym:f110-v0',
                       map=MAP_BASE, map_ext=MAP_EXT,
                       num_agents=2, timestep=TIMESTEP,
                       seed=0, integrator=Integrator.RK4)
    else:
        env = _env

    # ── Initial poses ─────────────────────────────────────────────────────────
    s_def0 = START_S
    s_atk0 = (START_S - gap_0) % tlen
    p_def  = [*line.sample(s_def0, 0.0), line.heading(s_def0)]
    p_atk  = [*line.sample(s_atk0, 0.0), line.heading(s_atk0)]

    # ── Agents ────────────────────────────────────────────────────────────────
    if defender_mode == 'A':
        defender = ModeADefender(lo_arr, hi_arr)
    else:
        defender = ModeBDefender(lo_arr, hi_arr, obs_delay=_obs_delay)
    attacker = ScriptedAttacker(scenario, rng, t_feint, late_close, lo_arr, hi_arr)

    # ── Episode loop ──────────────────────────────────────────────────────────
    try:
        obs, _, done, _ = env.reset(np.array([p_def, p_atk]))

        # Speed state variables: initialised at desired speed so the gym's PID
        # immediately drives the car to cruise speed.  IDM is integrated on
        # these; observed speed is used only for bumper-gap physics.
        v_def_state = v_def_0
        v_atk_state = v_atk_0

        time_blocking   = 0.0
        atk_speed_sum   = 0.0
        atk_speed_count = 0
        def_progress    = 0.0
        overtaken       = False
        collision       = False
        elapsed         = 0.0
        n_steps         = int(round(SIM_DURATION / TIMESTEP))

        for _step in range(n_steps):
            if done:
                break

            x_def  = obs['poses_x'][0];     y_def  = obs['poses_y'][0]
            th_def = obs['poses_theta'][0]
            x_atk  = obs['poses_x'][1];     y_atk  = obs['poses_y'][1]
            th_atk = obs['poses_theta'][1]
            v_def_obs = float(obs['linear_vels_x'][0])
            v_atk_obs = float(obs['linear_vels_x'][1])

            s_def, d_def = line.project([x_def, y_def])
            s_atk, d_atk = line.project([x_atk, y_atk])

            gap        = line.signed_distance(s_def, s_atk)   # > 0 = defender ahead
            bumper_gap = gap - CAR_LENGTH

            if defender_mode == 'A':
                def_steer, _ = defender.step(x_def, y_def, th_def, d_atk, gap, wp)
            else:
                def_steer, _ = defender.step(
                    elapsed, x_def, y_def, th_def,
                    d_atk, v_atk_obs, gap, v_def_state, wp)
            atk_steer, _ = attacker.step(elapsed, x_atk, y_atk, th_atk,
                                          gap, bumper_gap, wp)

            # Defender: free-road IDM
            a_def_v     = idm_accel(v_def_state, v_def_state, 1e9, v0=v_def_0)
            v_def_state = gym_speed_command(v_def_state, a_def_v, TIMESTEP, v_max=v_def_0)

            # Attacker speed strategy: free-road approach until LATE_CLOSE m
            # bumper gap in the same lane; safety brake only inside that window.
            # This lets the attacker arrive at the manoeuvre point at full speed
            # so feint / late_move can exploit the lateral opening.
            lat_sep = abs(d_atk - d_def)
            if bumper_gap > 0.0 and bumper_gap < LATE_CLOSE and lat_sep < LAT_THRESHOLD:
                # Safety brake: close in same lane
                a_atk_v = _atk_idm(v_atk_state, v_def_state, bumper_gap, v_atk_0)
            else:
                # Free-road: far away OR on a different lateral track
                a_atk_v = idm_accel(v_atk_state, v_atk_state, 1e9, v0=v_atk_0)
            v_atk_state = gym_speed_command(v_atk_state, a_atk_v, TIMESTEP, v_max=v_atk_0)

            obs, _, done, _ = env.step(
                np.array([[def_steer, v_def_state], [atk_steer, v_atk_state]]))
            if _render:
                env.render(mode='human')
            elapsed += TIMESTEP

            if np.any(obs['collisions']):
                collision = True
                break

            if 0.0 < gap < ENGAGE_GAP:
                time_blocking += TIMESTEP

            atk_speed_sum   += v_atk_obs
            atk_speed_count += 1
            def_progress    += v_def_obs * TIMESTEP

            # Overtake: attacker centre has passed defender centre.
            # Threshold 0 (not -CAR_LENGTH) catches the pass before the two
            # cars physically overlap during the transition.
            if not overtaken and gap < 0:
                overtaken = True
                break

    finally:
        if own_env:
            env.close()
            from f110_gym.envs.f110_env import F110Env
            F110Env.render_callbacks = _old_cbs

    return {
        'overtaken':           overtaken,
        'collision':           collision,
        'time_blocking':       time_blocking,
        'mean_attacker_speed': atk_speed_sum / max(atk_speed_count, 1),
        'defender_progress':   def_progress,
    }
