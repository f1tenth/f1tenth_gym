"""
Single-track drift model initialization function
"""
import numpy as np
from numba import njit

from .single_track_drift import vehicle_dynamics_std, get_standardized_state_std


def init_std(init_state, params: dict) -> np.ndarray:
    """
    init_std generates the initial state vector for the drift single track model

    Syntax:
        x0 = init_std(init_state, p)

    Inputs:
        :param init_state: core initial states
        :param p: parameter vector

    Outputs:
        :return x0: initial state vector

    Author: Teodor Ilie
    Written: 18-October-2025
    """
    x0 = np.zeros((9,))

    # Steering, vel constraints

    delta0 = init_state[2]  # steering angle of front wheels
    vel0 = init_state[3]  # speed of the car

    ## steering constraints
    s_min = params["s_min"]  # minimum steering angle [rad]
    s_max = params["s_max"]  # maximum steering angle [rad]

    ## longitudinal constraints
    v_min = params["v_min"]  # minimum velocity [m/s]
    v_max = params["v_max"]  # minimum velocity [m/s]

    if delta0 > s_max:
        delta0 = s_max

    if delta0 < s_min:
        delta0 = s_min

    if vel0 > v_max:
        vel0 = v_max

    if vel0 < v_min:
        vel0 = v_min

    # Copy first 7 states as-is with constraints on steering, velocity
    x0[0] = init_state[0]
    x0[1] = init_state[1]
    x0[2] = delta0
    x0[3] = vel0
    x0[4] = init_state[4]
    x0[5] = init_state[5]
    x0[6] = init_state[6]

    # Additional 2 states calculations
    x0[7] = x0[3] * np.cos(x0[6]) * np.cos(x0[2]) / params["R_w"]  # init front wheel angular speed
    x0[8] = x0[3] * np.cos(x0[6]) / params["R_w"]  # init rear wheel angular speed

    return x0
