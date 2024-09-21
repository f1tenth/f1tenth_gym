import numpy as np
from numba import njit

from .utils import steering_constraint, accl_constraints

@njit(cache=True)
def vehicle_dynamics_st(
    x,
    u_init,
    mu,
    C_Sf,
    C_Sr,
    lf,
    lr,
    h,
    m,
    I,
    s_min,
    s_max,
    sv_min,
    sv_max,
    v_switch,
    a_max,
    v_min,
    v_max,
):
    """
    Single Track Vehicle Dynamics.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x0, x1, x2, x3, x4, x5, x6)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4:yaw angle
                x5: yaw rate
                x6: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration
            mu (float): friction coefficient
            C_Sf (float): cornering stiffness of front wheels
            C_Sr (float): cornering stiffness of rear wheels
            lf (float): distance from center of gravity to front axle
            lr (float): distance from center of gravity to rear axle
            h (float): height of center of gravity
            m (float): mass of vehicle
            I (float): moment of inertia of vehicle, about Z axis
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity
            v_switch (float): velocity above which the acceleration is no longer able to create wheel spin
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # States
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    PSI_DOT = x[5]
    BETA = x[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = np.arctan2(np.sin(BETA), np.cos(BETA))

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array(
        [
            steering_constraint(DELTA, u_init[0], s_min, s_max, sv_min, sv_max),
            accl_constraints(V, u_init[1], v_switch, a_max, v_min, v_max),
        ]
    )
    # Controls
    STEER_VEL = u[0]
    ACCL = u[1]

    # switch to kinematic model for small velocities
    if V < 0.1:
        # wheelbase
        lwb = lf + lr
        BETA_HAT = np.arctan(np.tan(DELTA) * lr / lwb)
        BETA_DOT = (
            (1 / (1 + (np.tan(DELTA) * (lr / lwb)) ** 2))
            * (lr / (lwb * np.cos(DELTA) ** 2))
            * STEER_VEL
        )
        f = np.array(
            [
                V * np.cos(PSI + BETA_HAT),  # X_DOT
                V * np.sin(PSI + BETA_HAT),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                V * np.cos(BETA_HAT) * np.tan(DELTA) / lwb,  # PSI_DOT
                (1 / lwb)
                * (
                    ACCL * np.cos(BETA) * np.tan(DELTA)
                    - V * np.sin(BETA) * np.tan(DELTA) * BETA_DOT
                    + ((V * np.cos(BETA) * STEER_VEL) / (np.cos(DELTA) ** 2))
                ),  # PSI_DOT_DOT
                BETA_DOT,  # BETA_DOT
            ]
        )
    else:
        # system dynamics
        glr = (g * lr - ACCL * h)
        glf = (g * lf + ACCL * h)
        f = np.array(
            [
                V * np.cos(PSI + BETA),  # X_DOT
                V * np.sin(PSI + BETA),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                PSI_DOT,  # PSI_DOT
                ((mu * m) / (I * (lf + lr))) * (
                lf * C_Sf * (g * lr - ACCL * h) * DELTA
                + (
                    lr * C_Sr * (g * lf + ACCL * h) - lf * C_Sf * (g * lr - ACCL * h)
                ) * BETA
                - (
                    lf * lf * C_Sf * (g * lr - ACCL * h) + lr * lr * C_Sr * (g * lf + ACCL * h)
                ) * (PSI_DOT / V)
            ),  # PSI_DOT_DOT
                (mu / (V * (lr + lf))) * (
                C_Sf * (g * lr - ACCL * h) * DELTA \
                - (C_Sr * (g * lf + ACCL * h) + C_Sf * (g * lr - ACCL * h)) * BETA
                + (
                    C_Sr * (g * lf + ACCL * h) * lr - C_Sf * (g * lr - ACCL * h) * lf
                ) * (PSI_DOT / V)
            ) - PSI_DOT,  # BETA_DOT
            ]
        )

    return f
