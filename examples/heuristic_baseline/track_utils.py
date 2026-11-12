"""
track_utils.py — Track-relative (Frenet) coordinate utilities.

Conventions
-----------
s   arc-length progress along the centreline, in [0, track_length)   (m)
d   signed lateral offset; positive = LEFT of the direction of travel  (m)

Waypoint array layout (N × >=4):
    col 0  s_m       arc-length from start
    col 1  x_m       world x
    col 2  y_m       world y
    col 3  psi_rad   heading (radians)
    …      (extra columns preserved by offset_waypoints)

Matches example_waypoints.csv after np.loadtxt(..., delimiter=';', skiprows=3).

Run as a script for a headless self-test (no render window):
    python track_utils.py
"""

import warnings
from pathlib import Path

import numpy as np

# ── Lane constants (m from centreline; positive = left of travel) ─────────────
# Track boundary from rival_config.yaml: legal_min/max = ±1.2 m.
# Safe car-centre limit: ±(1.2 − car_half_width − boundary_margin)
#   = ±(1.2 − 0.155 − 0.04) ≈ ±1.005 m.
# ±0.4 m is well within that bound.
LEFT   = +0.4   # (m)
CENTER =  0.0   # (m)
RIGHT  = -0.4   # (m)

_TRACK_HALF_WIDTH = 1.2    # rival_config.yaml legal_max (m)
_CAR_HALF_WIDTH   = 0.155  # half of car_width 0.31 m
_BOUNDARY_MARGIN  = 0.04   # rival_config.yaml boundary_margin (m)
_SAFE_D_MAX = _TRACK_HALF_WIDTH - _CAR_HALF_WIDTH - _BOUNDARY_MARGIN  # ≈ 1.005 m

# Warn at import time if any named lane constant violates the safe bound.
for _d in (LEFT, CENTER, RIGHT):
    if abs(_d) > _SAFE_D_MAX:
        warnings.warn(
            f"Lane constant d={_d:+.3f} m falls outside safe track bound "
            f"±{_SAFE_D_MAX:.3f} m (legal ±{_TRACK_HALF_WIDTH:.3f} m).",
            stacklevel=1,
        )
del _d


# ── Internal helpers ──────────────────────────────────────────────────────────

def _track_length(waypoints):
    """Total circumference including the wrap-around closing segment (m)."""
    pts = np.asarray(waypoints, dtype=float)
    closing = float(np.hypot(pts[0, 1] - pts[-1, 1], pts[0, 2] - pts[-1, 2]))
    return float(pts[-1, 0] + closing)


# ── Public API ────────────────────────────────────────────────────────────────

def to_track_frame(x, y, waypoints):
    """
    Project world position (x, y) onto the centreline and return (s, d).

    Considers all N segments of the closed centreline polyline, including the
    wrap-around closing segment from the last waypoint back to the first.
    Ties are broken in favour of the first segment encountered.

    Parameters
    ----------
    x, y : float
        World-frame position (m).
    waypoints : array-like, shape (N, >=4)
        Columns: s_m, x_m, y_m, psi_rad, ...

    Returns
    -------
    s : float
        Arc-length progress in [0, track_length)  (m).
    d : float
        Signed lateral offset; positive = left of travel  (m).
    """
    pts = np.asarray(waypoints, dtype=float)
    n = len(pts)
    s_ref = pts[:, 0]
    cx, cy = pts[:, 1], pts[:, 2]
    tlen = _track_length(pts)

    # Edge vectors: segment i → waypoint (i+1) % n
    cx_next = np.roll(cx, -1)
    cy_next = np.roll(cy, -1)

    # s at the far end of each segment; closing segment ends at tlen (wraps to 0)
    s_next = np.roll(s_ref, -1).copy()
    s_next[-1] = tlen

    ex = cx_next - cx
    ey = cy_next - cy
    seg_len = np.hypot(ex, ey)

    valid = seg_len > 1e-9
    safe_len = np.where(valid, seg_len, 1.)
    tx = np.where(valid, ex / safe_len, 0.)
    ty = np.where(valid, ey / safe_len, 0.)

    # Signed projection of (x, y) along each tangent, clamped to [0, seg_len]
    rel_x = x - cx
    rel_y = y - cy
    proj_cl = np.clip(rel_x * tx + rel_y * ty, 0., seg_len)

    # Closest point on each segment and squared distance to it
    foot_x = cx + proj_cl * tx
    foot_y = cy + proj_cl * ty
    i = int(np.argmin((x - foot_x) ** 2 + (y - foot_y) ** 2))

    # Arc-length at the foot-point
    frac = proj_cl[i] / seg_len[i] if seg_len[i] > 1e-9 else 0.
    s = float((s_ref[i] + frac * (s_next[i] - s_ref[i])) % tlen)

    # Signed lateral offset: cross product tangent × residual (z-component).
    # Positive when (x, y) is to the LEFT of the travel direction.
    d = float(tx[i] * (y - foot_y[i]) - ty[i] * (x - foot_x[i]))

    return s, d


def offset_waypoints(waypoints, d):
    """
    Return the centreline shifted laterally by d metres.

    Each point is displaced along its left normal derived from the outgoing
    edge direction (not psi_rad), ensuring exact round-trip consistency with
    to_track_frame.  Left normal = rotate tangent 90 deg CCW = (−ty, tx).

    Warns if |d| exceeds the safe car-centre limit for this track.

    Parameters
    ----------
    waypoints : array-like, shape (N, >=4)
        Columns: s_m, x_m, y_m, psi_rad, ...  Extra columns are preserved.
    d : float
        Lateral shift (m).  Positive = left; negative = right.

    Returns
    -------
    ndarray, same shape as input
    """
    if abs(d) > _SAFE_D_MAX:
        warnings.warn(
            f"offset_waypoints: d={d:+.3f} m places the car outside the safe "
            f"track bound ±{_SAFE_D_MAX:.3f} m.",
            stacklevel=2,
        )
    pts = np.array(waypoints, dtype=float, copy=True)
    cx, cy = pts[:, 1], pts[:, 2]

    # Outgoing-edge tangent at each waypoint (including wrap-around closing edge).
    # Using geometry avoids heading-convention ambiguity in the psi_rad column.
    ex = np.roll(cx, -1) - cx
    ey = np.roll(cy, -1) - cy
    seg_len = np.hypot(ex, ey)
    valid = seg_len > 1e-9
    safe_len = np.where(valid, seg_len, 1.)
    tx = np.where(valid, ex / safe_len, 0.)
    ty = np.where(valid, ey / safe_len, 0.)

    # Left normal = rotate tangent 90° CCW = (−ty, tx)
    pts[:, 1] += d * (-ty)
    pts[:, 2] += d * tx
    return pts


def get_gap(attacker, defender, track_length=None, waypoints=None):
    """
    Longitudinal gap and lateral difference between two cars.

    Parameters
    ----------
    attacker, defender : object with .s and .d attributes, or (s, d) sequence
    track_length : float, optional
        Total track circumference (m).
    waypoints : array-like, optional
        Used to compute track_length when not given directly.

    Returns
    -------
    gap : float
        Signed longitudinal gap in [−track_length/2, +track_length/2).
        Positive = defender is AHEAD of attacker.
    d_diff : float
        Lateral difference: attacker.d − defender.d (m).
        Positive = attacker is further LEFT than defender.
    """
    if track_length is None:
        if waypoints is None:
            raise ValueError("Provide either track_length or waypoints.")
        track_length = _track_length(waypoints)

    s_atk = attacker.s if hasattr(attacker, 's') else attacker[0]
    d_atk = attacker.d if hasattr(attacker, 'd') else attacker[1]
    s_def = defender.s if hasattr(defender, 's') else defender[0]
    d_def = defender.d if hasattr(defender, 'd') else defender[1]

    half = track_length / 2.0
    gap   = float((s_def - s_atk + half) % track_length - half)
    d_diff = float(d_atk - d_def)
    return gap, d_diff


# ── Headless self-test ────────────────────────────────────────────────────────

def _interp_xy(path, s_car):
    """Return (x, y) on a waypoint path at arc-length s_car (wraps around)."""
    tlen = _track_length(path)
    s_ref = path[:, 0]
    s_car = float(s_car) % tlen
    idx = int(np.clip(np.searchsorted(s_ref, s_car, side='right') - 1,
                      0, len(path) - 2))
    ds = s_ref[idx + 1] - s_ref[idx]
    frac = (s_car - s_ref[idx]) / ds if ds > 1e-9 else 0.
    x = path[idx, 1] + frac * (path[idx + 1, 1] - path[idx, 1])
    y = path[idx, 2] + frac * (path[idx + 1, 2] - path[idx, 2])
    return float(x), float(y)


if __name__ == '__main__':
    # Waypoints are one directory above this file (examples/example_waypoints.csv)
    wp_file = Path(__file__).parent.parent / 'example_waypoints.csv'
    waypoints = np.loadtxt(wp_file, delimiter=';', skiprows=3)
    tlen = _track_length(waypoints)

    print(f"Track length : {tlen:.3f} m  ({len(waypoints)} waypoints)")
    print(f"Lane offsets : LEFT={LEFT:+.3f} m  CENTER={CENTER:.3f} m  RIGHT={RIGHT:+.3f} m")
    print(f"Safe bound   : ±{_SAFE_D_MAX:.3f} m  (legal ±{_TRACK_HALF_WIDTH:.3f} m)")
    print()

    # Build offset paths for defender (LEFT) and attacker (RIGHT)
    def_path = offset_waypoints(waypoints, LEFT)
    atk_path = offset_waypoints(waypoints, RIGHT)

    # Simulate: both cars at the same constant speed; defender starts 10 m ahead.
    DT          = 0.1   # s
    SPEED       = 2.0   # m/s
    N_STEPS     = 50
    PRINT_EVERY = 5
    DEF_S0      = 10.0  # defender start arc-length (m)
    ATK_S0      =  0.0  # attacker start arc-length (m)

    def_s = DEF_S0
    atk_s = ATK_S0

    print(f"{'t':>5}  {'def_s':>7}  {'def_d':>7}  {'atk_s':>7}  {'atk_d':>7}  "
          f"{'gap':>9}  {'d_diff':>8}")
    print("─" * 66)

    for step in range(N_STEPS + 1):
        t = step * DT

        # World positions on each offset path
        def_x, def_y = _interp_xy(def_path, def_s)
        atk_x, atk_y = _interp_xy(atk_path, atk_s)

        # Frenet coordinates relative to the original centreline
        s_def, d_def = to_track_frame(def_x, def_y, waypoints)
        s_atk, d_atk = to_track_frame(atk_x, atk_y, waypoints)

        gap, d_diff = get_gap((s_atk, d_atk), (s_def, d_def), track_length=tlen)

        if step % PRINT_EVERY == 0:
            print(f"{t:>5.1f}  {s_def:>7.3f}  {d_def:>+7.3f}  {s_atk:>7.3f}  "
                  f"{d_atk:>+7.3f}  {gap:>+9.4f}  {d_diff:>+8.4f}")

        def_s = (def_s + SPEED * DT) % tlen
        atk_s = (atk_s + SPEED * DT) % tlen

    print()
    print("Expected: def_d ≈ +0.400 m (LEFT lane),  atk_d ≈ −0.400 m (RIGHT lane),")
    print("          gap ≈ +10.000 m (constant — same speed),  d_diff ≈ −0.800 m")
