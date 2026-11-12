"""
idm.py — Intelligent Driver Model (IDM) speed control for the 1/10-scale F1TENTH car.

Reference: Treiber, Hennecke & Helbing (2000), Phys. Rev. E 62(2).

Gym integration
---------------
The F1TENTH gym accepts a TARGET SPEED, not acceleration, as actions[i, 1].
Convert each sim step:

    a             = idm_accel(v, v_leader, gap, v0)
    v             = gym_speed_command(v, a, dt=TIMESTEP, v_max=v0)
    actions[i, 1] = v          # pass to env.step()

Defender is always in free-road mode (gap=1e9).
Attacker switches to following mode when the defender is ahead AND within
LAT_THRESHOLD laterally (d values from track_utils.to_track_frame):

    bumper_gap = s_def - s_atk - CAR_LENGTH
    if bumper_gap > 0 and abs(d_atk - d_def) < LAT_THRESHOLD:
        a_atk = idm_accel(v_atk, v_def,  bumper_gap,  V0_ATTACKER)
    else:
        a_atk = idm_accel(v_atk, v_atk,  1e9,         V0_ATTACKER)

Run headless self-test (no render window, no gym):
    python idm.py
"""

import numpy as np

# ── IDM parameters — edit here to tune ───────────────────────────────────────
IDM_A_MAX = 2.0    # maximum acceleration         (m/s²)
IDM_B     = 3.0    # comfortable deceleration      (m/s²)
IDM_S0    = 0.5    # minimum standstill bumper gap (m)
IDM_T     = 0.6    # desired time headway          (s)

# ── Desired free-road speeds ──────────────────────────────────────────────────
V0_DEFENDER = 4.0  # defender target speed  (m/s)
V0_ATTACKER = 4.5  # attacker target speed  (m/s) — faster so it naturally closes

# ── Integration / gym interface ───────────────────────────────────────────────
TIMESTEP      = 0.01  # physics dt (s) — match the gym's timestep= kwarg
CAR_LENGTH    = 0.58  # (m) full body length; bumper_gap = centre_gap − CAR_LENGTH
LAT_THRESHOLD = 0.3   # (m) |d_atk − d_def| below which defender is treated as leader


# ── Public API ────────────────────────────────────────────────────────────────

def idm_accel(v, v_leader, gap, v0=V0_DEFENDER):
    """
    IDM longitudinal acceleration.

    a = A_MAX * (1 − (v/v0)^4 − (s*/gap)^2)

    where  s* = s0 + v*T + v*(v − v_leader) / (2*sqrt(A_MAX*B))

    Parameters
    ----------
    v        : float — ego forward speed (m/s); internally clamped to ≥ 0
    v_leader : float — leader forward speed (m/s)
    gap      : float — bumper-to-bumper gap to leader (m); clamped to ≥ 0.1 m
    v0       : float — desired free-road speed (m/s); defaults to V0_DEFENDER

    Returns
    -------
    float — acceleration (m/s²); can be negative (braking)

    Free-road mode: pass gap=1e9. The interaction term (s*/gap)^2 → 0 and
    a → A_MAX * (1 − (v/v0)^4), which drives v toward v0.
    """
    v   = max(float(v),   0.)
    gap = max(float(gap), 0.1)
    dv  = v - float(v_leader)                           # positive = closing
    v0_ = max(float(v0),  1e-6)
    s_star = IDM_S0 + v * IDM_T + v * dv / (2.0 * np.sqrt(IDM_A_MAX * IDM_B))
    return float(IDM_A_MAX * (1.0 - (v / v0_) ** 4 - (s_star / gap) ** 2))


def gym_speed_command(v, a, dt=TIMESTEP, v_max=V0_DEFENDER):
    """
    Integrate one IDM step and return a gym-compatible target-speed command.

    v_new = clip(v + a * dt, 0, v_max)

    Pass the return value as  actions[i, 1]  in env.step().

    Parameters
    ----------
    v     : float — current speed (m/s)
    a     : float — IDM acceleration from idm_accel() (m/s²)
    dt    : float — timestep (s); default = TIMESTEP
    v_max : float — upper clamp; use V0_DEFENDER or V0_ATTACKER as appropriate
    """
    return float(np.clip(v + a * dt, 0., v_max))


# ── Headless self-test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Scenario: attacker already at desired speed (4.5 m/s) placed 3 m (bumper gap)
    # directly behind a defender at steady cruise (4.0 m/s).
    # Expected: IDM immediately brakes the attacker; it slows to ~4.0 m/s and
    # follows at gap ≈ s*(4.0) / sqrt(1 − (4.0/4.5)^4) ≈ 4.73 m.

    DT          = TIMESTEP   # 0.01 s
    SIM_TIME    = 20.0       # s
    PRINT_EVERY = 1.0        # s — print interval

    # Initial conditions
    v_def  = V0_DEFENDER      # 4.0 m/s — defender already at cruise
    v_atk  = V0_ATTACKER      # 4.5 m/s — attacker at its desired speed
    s_def  = CAR_LENGTH + 3.0 # centre position (m); bumper gap = 3.0 m
    s_atk  = 0.0

    d_def  = 0.0              # both in the same lane (d = 0)
    d_atk  = 0.0

    n_steps      = int(round(SIM_TIME / DT))
    print_stride = int(round(PRINT_EVERY / DT))

    # Theoretical steady-state following gap (Δv=0):
    # a=0 → (v/v0)^4 + (s*/gap)^2 = 1  → gap = s*(v) / sqrt(1 − (v/v0_atk)^4)
    s_star_eq = IDM_S0 + V0_DEFENDER * IDM_T
    gap_eq    = s_star_eq / np.sqrt(1.0 - (V0_DEFENDER / V0_ATTACKER) ** 4)

    print(f"IDM self-test  —  defender v0={V0_DEFENDER} m/s  |  attacker v0={V0_ATTACKER} m/s")
    print(f"Initial bumper gap = {s_def - s_atk - CAR_LENGTH:.2f} m  "
          f"|  dt={DT} s  |  sim={SIM_TIME:.0f} s")
    print(f"IDM params: A={IDM_A_MAX} m/s²  B={IDM_B} m/s²  s0={IDM_S0} m  T={IDM_T} s")
    print(f"Expected settle: v_atk ≈ {V0_DEFENDER:.2f} m/s  "
          f"gap ≈ {gap_eq:.2f} m  (s*/{np.sqrt(1-(V0_DEFENDER/V0_ATTACKER)**4):.3f})\n")

    hdr = f"{'t':>6}  {'v_def':>7}  {'v_atk':>7}  {'gap':>8}  mode"
    print(hdr)
    print("─" * len(hdr))

    for step in range(n_steps + 1):
        t = step * DT

        bumper_gap = s_def - s_atk - CAR_LENGTH
        lat_sep    = abs(d_atk - d_def)

        following = bumper_gap > 0 and lat_sep < LAT_THRESHOLD
        mode = "FOLLOW" if following else "FREE  "

        if step % print_stride == 0:
            print(f"{t:>6.1f}  {v_def:>7.4f}  {v_atk:>7.4f}  {bumper_gap:>8.4f}  {mode}")

        # Defender: always free-road IDM
        a_def = idm_accel(v_def, v_def, 1e9, v0=V0_DEFENDER)

        # Attacker: following when defender is ahead and laterally aligned
        if following:
            a_atk = idm_accel(v_atk, v_def, bumper_gap, v0=V0_ATTACKER)
        else:
            a_atk = idm_accel(v_atk, v_atk, 1e9, v0=V0_ATTACKER)

        # Integrate: gym_speed_command pattern (clamp to [0, v0])
        v_def = gym_speed_command(v_def, a_def, DT, v_max=V0_DEFENDER)
        v_atk = gym_speed_command(v_atk, a_atk, DT, v_max=V0_ATTACKER)

        s_def += v_def * DT
        s_atk += v_atk * DT

    final_gap = s_def - s_atk - CAR_LENGTH
    print(f"\nFinal: v_def={v_def:.4f} m/s  v_atk={v_atk:.4f} m/s  gap={final_gap:.4f} m")
    print(f"Theory: v_atk → {V0_DEFENDER:.4f} m/s  gap → {gap_eq:.4f} m")

    v_tol   = 0.05  # attacker speed must reach within 0.05 m/s of defender
    gap_tol = 0.5   # gap must be within 0.5 m of theoretical equilibrium
    assert abs(v_atk - V0_DEFENDER) < v_tol, (
        f"Attacker did not slow to follow defender: v_atk={v_atk:.3f} m/s "
        f"(expected ≈{V0_DEFENDER:.1f} ± {v_tol})")
    assert final_gap > IDM_S0, (
        f"Gap collapsed below minimum: gap={final_gap:.3f} m < IDM_S0={IDM_S0}")
    assert abs(final_gap - gap_eq) < gap_tol, (
        f"Gap {final_gap:.3f} m not close to theoretical equilibrium {gap_eq:.3f} m")
    print("Self-test PASSED")
