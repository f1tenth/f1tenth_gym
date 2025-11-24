"""
Multi-body model initialization functions
"""

import numpy as np
from numba import njit

from .multi_body import vehicle_dynamics_mb, get_standardized_state_mb


def init_mb(init_state, params: dict) -> np.ndarray:
    # init_MB - generates the initial state vector for the multi-body model
    #
    # Syntax:
    #     x0 = init_MB(init_state, p)
    #
    # Inputs:
    #     init_state - core initial states
    #     p - parameter vector
    #
    # Outputs:
    #     x0 - initial state vector
    #
    # Example:
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      11-January-2017
    # Last update:  ---
    # Last revision:---

    ### Parameters
    ## steering constraints
    s_min = params["s_min"]  # minimum steering angle [rad]
    s_max = params["s_max"]  # maximum steering angle [rad]
    ## longitudinal constraints
    v_min = params["v_min"]  # minimum velocity [m/s]
    v_max = params["v_max"]  # minimum velocity [m/s]
    ## masses
    m_s = params["m_s"]  # sprung mass [kg]  SMASS
    m_uf = params["m_uf"]  # unsprung mass front [kg]  UMASSF
    m_ur = params["m_ur"]  # unsprung mass rear [kg]  UMASSR
    ## axes distances
    lf = params["lf"]
    # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params["lr"]
    # distance from spring mass center of gravity to rear axle [m]  LENB

    ## geometric parameters
    K_zt = params["K_zt"]  # vertical spring rate of tire [N/m]  TSPRINGR
    R_w = params["R_w"]
    # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]
    # create equivalent bicycle parameters
    g = 9.81  # [m/s^2]

    # obtain initial states from vector
    sx0 = init_state[0]  # x-position in a global coordinate system
    sy0 = init_state[1]  # y-position in a global coordinate system
    delta0 = init_state[2]  # steering angle of front wheels
    vel0 = init_state[3]  # speed of the car
    Psi0 = init_state[4]  # yaw angle
    dotPsi0 = init_state[5]  # yaw rate
    beta0 = init_state[6]  # slip angle

    if delta0 > s_max:
        delta0 = s_max

    if delta0 < s_min:
        delta0 = s_min

    if vel0 > v_max:
        vel0 = v_max

    if vel0 < v_min:
        vel0 = v_min

    # auxiliary initial states
    F0_z_f = m_s * g * lr / (lf + lr) + m_uf * g
    F0_z_r = m_s * g * lf / (lf + lr) + m_ur * g

    # sprung mass states
    x0 = np.zeros((29,))  # init initial state vector
    x0[0] = sx0  # x-position in a global coordinate system
    x0[1] = sy0  # y-position in a global coordinate system
    x0[2] = delta0  # steering angle of front wheels
    x0[3] = np.cos(beta0) * vel0  # velocity in x-direction
    x0[4] = Psi0  # yaw angle
    x0[5] = dotPsi0  # yaw rate
    x0[6] = 0  # roll angle
    x0[7] = 0  # roll rate
    x0[8] = 0  # pitch angle
    x0[9] = 0  # pitch rate
    x0[10] = np.sin(beta0) * vel0  # velocity in y-direction
    x0[11] = 0  # z-position (zero height corresponds to steady state solution)
    x0[12] = 0  # velocity in z-direction

    # unsprung mass states (front)
    x0[13] = 0  # roll angle front
    x0[14] = 0  # roll rate front
    x0[15] = np.sin(beta0) * vel0 + lf * dotPsi0  # velocity in y-direction front
    x0[16] = (F0_z_f) / (2 * K_zt)  # z-position front
    x0[17] = 0  # velocity in z-direction front

    # unsprung mass states (rear)
    x0[18] = 0  # roll angle rear
    x0[19] = 0  # roll rate rear
    x0[20] = np.sin(beta0) * vel0 - lr * dotPsi0  # velocity in y-direction rear
    x0[21] = (F0_z_r) / (2 * K_zt)  # z-position rear
    x0[22] = 0  # velocity in z-direction rear

    # wheel states
    x0[23] = x0[3] / (R_w)  # left front wheel angular speed
    x0[24] = x0[3] / (R_w)  # right front wheel angular speed
    x0[25] = x0[3] / (R_w)  # left rear wheel angular speed
    x0[26] = x0[3] / (R_w)  # right rear wheel angular speed

    x0[27] = 0  # delta_y_f
    x0[28] = 0  # delta_y_r

    return x0
