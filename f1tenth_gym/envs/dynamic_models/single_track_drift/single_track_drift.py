import numpy as np
from numba import njit
from numba.typed import Dict

from ..utils import steering_constraint, accl_constraints
from ..tire_model import formula_longitudinal, formula_lateral, formula_longitudinal_comb, formula_lateral_comb
from ..kinematic import vehicle_dynamics_ks_cog


def vehicle_dynamics_std(x: np.ndarray, u_init: np.ndarray, params: dict):
    """
    Single Track Drift model.
    From: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/PYTHON/vehiclemodels/vehicle_dynamics_std.py?ref_type=heads

    Syntax:
        f = vehicle_dynamics_std(x,u,p)

    Inputs:
        :param x: (numpy.ndarray (9,)): vehicle state vector (x0, x1, x2, x3, x4, x5, x6, x7, x8)
                x[0]: x position in global coordinates
                x[1]: y position in global coordinates
                x[2]: steering angle of front wheels
                x[3]: velocity in x direction
                x[4]:yaw angle
                x[5]: yaw rate
                x[6]: slip angle at vehicle center
                x[7]: angular speed of the front wheel
                x[8]: angular speed of the rear wheel
        :param u_init: (numpy.ndarray (2,)): control input vector (u1, u2)
                u_init[0]: steering angle velocity of front wheels
                u_init[1]: longitudinal acceleration
        :param params: (dict): dictionary containing necessary parameters:

    Outputs:
        :return f: (numpy.ndarray): right hand side of differential equations

    Author: Teodor Ilie
    Written: 18-October-2025
    """
    # Get states from state vector
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    PSI_DOT = x[5]
    BETA = x[6]
    ANG_FRONT = x[7]
    ANG_REAR = x[8]

    # set gravity constant
    g = 9.81  # [m/s^2]

    # wheelbase
    lwb = params["lf"] + params["lr"]

    # mix models parameters
    v_s = 0.2
    v_b = 0.05
    v_min = v_s / 2  # note this is different from the v_min defined in params, and used for steering constraints

    # apply steering, accl constraints same as single_track.py
    u = np.array(
        [
            steering_constraint(
                DELTA,
                u_init[0],
                params["s_min"],
                params["s_max"],
                params["sv_min"],
                params["sv_max"],
            ),
            accl_constraints(
                V,
                u_init[1],
                params["v_switch"],
                params["a_max"],
                params["v_min"],
                params["v_max"],
            ),
        ]
    )
    # Control commands
    STEER_VEL = u[0]
    ACCL = u[1]

    # compute lateral tire slip angles
    alpha_f = np.arctan((V * np.sin(BETA) + PSI_DOT * params["lf"]) / (V * np.cos(BETA))) - DELTA if V > v_min else 0
    alpha_r = np.arctan((V * np.sin(BETA) - PSI_DOT * params["lr"]) / (V * np.cos(BETA))) if V > v_min else 0

    # compute vertical tire forces
    F_zf = params["m"] * (-ACCL * params["h_s"] + g * params["lr"]) / (params["lr"] + params["lf"])
    F_zr = params["m"] * (ACCL * params["h_s"] + g * params["lf"]) / (params["lr"] + params["lf"])

    # compute front and rear tire speeds
    u_wf = max(0, V * np.cos(BETA) * np.cos(DELTA) + (V * np.sin(BETA) + params["lf"] * PSI_DOT) * np.sin(DELTA))
    u_wr = max(0, V * np.cos(BETA))

    # compute longitudinal tire slip
    s_f = 1 - params["R_w"] * ANG_FRONT / max(u_wf, v_min)
    s_r = 1 - params["R_w"] * ANG_REAR / max(u_wr, v_min)

    # compute tire forces (Pacejka)
    # pure slip longitudinal forces
    F0_xf = formula_longitudinal(s_f, 0, F_zf, params)
    F0_xr = formula_longitudinal(s_r, 0, F_zr, params)

    # pure slip lateral forces
    res = formula_lateral(alpha_f, 0, F_zf, params)
    F0_yf = res[0]
    mu_yf = res[1]
    res = formula_lateral(alpha_r, 0, F_zr, params)
    F0_yr = res[0]
    mu_yr = res[1]

    # combined slip longitudinal forces
    F_xf = formula_longitudinal_comb(s_f, alpha_f, F0_xf, params)
    F_xr = formula_longitudinal_comb(s_r, alpha_r, F0_xr, params)

    # combined slip lateral forces
    F_yf = formula_lateral_comb(s_f, alpha_f, 0, mu_yf, F_zf, F0_yf, params)
    F_yr = formula_lateral_comb(s_r, alpha_r, 0, mu_yr, F_zr, F0_yr, params)

    # convert acceleration input to brake and engine torque
    if ACCL > 0:
        T_B = 0.0
        T_E = params["m"] * params["R_w"] * ACCL
    else:
        T_B = params["m"] * params["R_w"] * ACCL
        T_E = 0.0

    # system dynamics
    d_v = (
        1
        / params["m"]
        * (-F_yf * np.sin(DELTA - BETA) + F_yr * np.sin(BETA) + F_xr * np.cos(BETA) + F_xf * np.cos(DELTA - BETA))
    )
    dd_psi = (
        1
        / params["I_z"]
        * (F_yf * np.cos(DELTA) * params["lf"] - F_yr * params["lr"] + F_xf * np.sin(DELTA) * params["lf"])
    )
    d_beta = (
        -PSI_DOT
        + 1
        / (params["m"] * V)
        * (F_yf * np.cos(DELTA - BETA) + F_yr * np.cos(BETA) - F_xr * np.sin(BETA) + F_xf * np.sin(DELTA - BETA))
        if V > v_min
        else 0
    )

    # wheel dynamics (negative wheel spin forbidden)
    d_omega_f = (
        1 / params["I_y_w"] * (-params["R_w"] * F_xf + params["T_sb"] * T_B + params["T_se"] * T_E)
        if ANG_FRONT >= 0
        else 0
    )
    ANG_FRONT = max(0, ANG_FRONT)
    d_omega_r = (
        1 / params["I_y_w"] * (-params["R_w"] * F_xr + (1 - params["T_sb"]) * T_B + (1 - params["T_se"]) * T_E)
        if ANG_REAR >= 0
        else 0
    )
    ANG_REAR = max(0, ANG_REAR)

    # *** Mix with kinematic model at low speeds ***
    # Due to errors when using the scipy.odeint with a "hard" switch to the kinematic model, we overblend both models
    # around the switching velocity to achieve a "smoother" transition between both models.
    # kinematic system dynamics
    x_ks = [X, Y, DELTA, V, PSI]
    f_ks = vehicle_dynamics_ks_cog(np.array(x_ks), u, params)
    # derivative of slip angle and yaw rate (kinematic)
    d_beta_ks = (params["lr"] * STEER_VEL) / (
        lwb * np.cos(DELTA) ** 2 * (1 + (np.tan(DELTA) ** 2 * params["lr"] / lwb) ** 2)
    )
    dd_psi_ks = (
        1
        / lwb
        * (
            ACCL * np.cos(BETA) * np.tan(DELTA)
            - V * np.sin(BETA) * d_beta_ks * np.tan(DELTA)
            + V * np.cos(BETA) * STEER_VEL / np.cos(DELTA) ** 2
        )
    )
    # derivative of angular speeds (kinematic)
    d_omega_f_ks = (1 / 0.02) * (u_wf / params["R_w"] - ANG_FRONT)
    d_omega_r_ks = (1 / 0.02) * (u_wr / params["R_w"] - ANG_REAR)

    # weights for mixing both models
    w_std = 0.5 * (np.tanh((V - v_s) / v_b) + 1)
    w_ks = 1 - w_std

    # output vector: mix results of dynamic and kinematic model
    f = np.array(
        [
            V * np.cos(BETA + PSI),
            V * np.sin(BETA + PSI),
            STEER_VEL,
            w_std * d_v + w_ks * f_ks[3],
            w_std * PSI_DOT + w_ks * f_ks[4],
            w_std * dd_psi + w_ks * dd_psi_ks,
            w_std * d_beta + w_ks * d_beta_ks,
            w_std * d_omega_f + w_ks * d_omega_f_ks,
            w_std * d_omega_r + w_ks * d_omega_r_ks,
        ]
    )

    # return f, the time derivatives of the input state x
    return f


@njit(cache=True)
def get_standardized_state_std(x: np.ndarray) -> dict:
    """
    The standard state is fetched to use for calculating observations.
        Args:
            x: current state vector
        Returns:
            Return the standard state for the STD model
            [
                x - global coords
                y - global coords
                delt - steering angle
                v_x - velocity in x
                v_y - velocity in y
                yaw - yaw phi
                yaw_rate - phi dot
                slip - beta
            ]
    """
    d = dict()
    d["x"] = x[0]
    d["y"] = x[1]
    d["delta"] = x[2]
    d["v_x"] = x[3] * np.cos(x[6])
    d["v_y"] = x[3] * np.sin(x[6])
    d["yaw"] = x[4]
    d["yaw_rate"] = x[5]
    d["slip"] = x[6]
    return d
