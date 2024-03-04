import warnings
from enum import Enum

import numpy as np
from numba import njit


class DynamicModel(Enum):
    """Enum for specifying dynamic models

    KS: 1 (kinematic single track)

    ST: 2 (single track)

    """

    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track

    @staticmethod
    def from_string(model: str):
        """Set dynamic models from string.

        Parameters
        ----------
        model : str
            dynamic model type

        Returns
        -------
        int
            dynamic model type


        Raises
        ------
        ValueError
            Unknown model type
        """
        if model == "ks":
            warnings.warn(
                "Chosen model is KS. This is different from previous versions of the gym."
            )
            return DynamicModel.KS
        elif model == "st":
            return DynamicModel.ST
        else:
            raise ValueError(f"Unknown model type {model}")

    def get_initial_state(self, pose=None):
        """Set initial dynamic state based on model

        Parameters
        ----------
        pose : np.ndarray, optional
            initial pose, by default None

        Returns
        -------
        np.ndarray
            initial state

        Raises
        ------
        ValueError
            Unknown model type
        """
        # initialize zero state
        if self == DynamicModel.KS:
            # state is [x, y, steer_angle, vel, yaw_angle]
            state = np.zeros(5)
        elif self == DynamicModel.ST:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            state = np.zeros(7)
        else:
            raise ValueError(f"Unknown model type {self}")

        # set initial pose if provided
        if pose is not None:
            state[0:2] = pose[0:2]
            state[4] = pose[2]

        return state

    @property
    def f_dynamics(self):
        """property, returns dynamics function

        Returns
        -------
        Callable
            dynamic function

        Raises
        ------
        ValueError
            Unknown model type
        """
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        else:
            raise ValueError(f"Unknown model type {self}")


@njit(cache=True)
def upper_accel_limit(vel, a_max, v_switch):
    """Upper acceleration limit, adjusts the acceleration based on constraints

    Parameters
    ----------
    vel : float
        current velocity of the vehicle
    a_max : float
        maximum allowed acceleration, symmetrical
    v_switch : float
        switching velocity (velocity at which the acceleration is no longer able to create wheel spin)

    Returns
    -------
    float
        adjusted acceleration
    """
    if vel > v_switch:
        pos_limit = a_max * (v_switch / vel)
    else:
        pos_limit = a_max

    return pos_limit


@njit(cache=True)
def accl_constraints(vel, a_long_d, v_switch, a_max, v_min, v_max):
    """Acceleration constraints, adjusts the acceleration based on constraints

    Parameters
    ----------
    vel : float
        current velocity of the vehicle
    a_long_d : float
        unconstrained desired acceleration in the direction of travel.
    v_switch : float
        switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
    a_max : float
        maximum allowed acceleration, symmetrical
    v_min : float
        minimum allowed velocity
    v_max : float
        maximum allowed velocity

    Returns
    -------
    float
        adjusted acceleration
    """
    uac = upper_accel_limit(vel, a_max, v_switch)

    if (vel <= v_min and a_long_d <= 0) or (vel >= v_max and a_long_d >= 0):
        a_long = 0.0
    elif a_long_d <= -a_max:
        a_long = -a_max
    elif a_long_d >= uac:
        a_long = uac
    else:
        a_long = a_long_d

    return a_long


@njit(cache=True)
def steering_constraint(
    steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max
):
    """Steering constraints, adjusts the steering velocity based on constraints

    Parameters
    ----------
    steering_angle : float
        current steering_angle of the vehicle
    steering_velocity : float
        unconstraint desired steering_velocity
    s_min : float
        minimum steering angle
    s_max : float
        maximum steering angle
    sv_min : float
        minimum steering velocity
    sv_max : float
        maximum steering velocity

    Returns
    -------
    float
        adjusted steering velocity
    """
    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (
        steering_angle >= s_max and steering_velocity >= 0
    ):
        steering_velocity = 0.0
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(
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
    """Kinematic Single Track Vehicle Dynamics.
    Follows https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 5

    Parameters
    ----------
    x : np.ndarray
        vehicle state vector (x0, x1, x2, x3, x4)
            x0: x position in global coordinates
            x1: y position in global coordinates
            x2: steering angle of front wheels
            x3: velocity in x direction
            x4: yaw angle
    u_init : np.ndarray
        control input vector (u1, u2)
            u1: steering angle velocity of front wheels
            u2: longitudinal acceleration
    mu : float
        friction coefficient
    C_Sf : float
        cornering stiffness of front wheels
    C_Sr : float
        cornering stiffness of rear wheels
    lf : float
        distance from center of gravity to front axle
    lr : float
        distance from center of gravity to rear axle
    h : float
        height of center of gravity
    m : float
        mass of vehicle
    I : float
        moment of inertia of vehicle, about Z axis
    s_min : float
        minimum steering angle
    s_max : float
        maximum steering angle
    sv_min : float
        minimum steering velocity
    sv_max : float
        maximum steering velocity
    v_switch : float
        velocity above which the acceleration is no longer able to create wheel slip
    a_max : float
        maximum allowed acceleration
    v_min : float
        minimum allowed velocity
    v_max : float
        maximum allowed velocity

    Returns
    -------
    np.ndarray
        right hand side of differential equations
    """
    # Controls
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    # Raw Actions
    RAW_STEER_VEL = u_init[0]
    RAW_ACCL = u_init[1]
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array(
        [
            steering_constraint(DELTA, RAW_STEER_VEL, s_min, s_max, sv_min, sv_max),
            accl_constraints(V, RAW_ACCL, v_switch, a_max, v_min, v_max),
        ]
    )
    # Corrected Actions
    STEER_VEL = u[0]
    ACCL = u[1]

    # system dynamics
    f = np.array(
        [
            V * np.cos(PSI),  # X_DOT
            V * np.sin(PSI),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            (V / lwb) * np.tan(DELTA),  # PSI_DOT
        ]
    )
    return f


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
    """Single Track Vehicle Dynamics.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

    Parameters
    ----------
    x : np.ndarray
        vehicle state vector (x0, x1, x2, x3, x4, x5, x6)
            x0: x position in global coordinates
            x1: y position in global coordinates
            x2: steering angle of front wheels
            x3: velocity in x direction
            x4:yaw angle
            x5: yaw rate
            x6: slip angle at vehicle center
    u_init : np.ndarray
        control input vector (u1, u2)
            u1: steering angle velocity of front wheels
            u2: longitudinal acceleration
    mu : float
        friction coefficient
    C_Sf : float
        cornering stiffness of front wheels
    C_Sr : float
        cornering stiffness of rear wheels
    lf : float
        distance from center of gravity to front axle
    lr : float
        distance from center of gravity to rear axle
    h : float
        height of center of gravity
    m : float
        mass of vehicle
    I : float
        moment of inertia of vehicle, about Z axis
    s_min : float
        minimum steering angle
    s_max : float
        maximum steering angle
    sv_min : float
        minimum steering velocity
    sv_max : float
        maximum steering velocity
    v_switch : float
        velocity above which the acceleration is no longer able to create wheel slip
    a_max : float
        maximum allowed acceleration
    v_min : float
        minimum allowed velocity
    v_max : float
        maximum allowed velocity

    Returns
    -------
    np.ndarray
        right hand side of differential equations
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
    if V < 0.5:
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
        f = np.array(
            [
                V * np.cos(PSI + BETA),  # X_DOT
                V * np.sin(PSI + BETA),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                PSI_DOT,  # PSI_DOT
                ((mu * m) / (I * (lf + lr)))
                * (
                    lf * C_Sf * (g * lr - ACCL * h) * DELTA
                    + (
                        lr * C_Sr * (g * lf + ACCL * h)
                        - lf * C_Sf * (g * lr - ACCL * h)
                    )
                    * BETA
                    - (
                        lf * lf * C_Sf * (g * lr - ACCL * h)
                        + lr * lr * C_Sr * (g * lf + ACCL * h)
                    )
                    * (PSI_DOT / V)
                ),  # PSI_DOT_DOT
                (mu / (V * (lr + lf)))
                * (
                    C_Sf * (g * lr - ACCL * h) * DELTA
                    - (C_Sr * (g * lf + ACCL * h) + C_Sf * (g * lr - ACCL * h)) * BETA
                    + (
                        C_Sr * (g * lf + ACCL * h) * lr
                        - C_Sf * (g * lr - ACCL * h) * lf
                    )
                    * (PSI_DOT / V)
                )
                - PSI_DOT,  # BETA_DOT
            ]
        )

    return f


@njit(cache=True)
def pid_steer(steer, current_steer, max_sv):
    """PID control for steering angle to steering velocity

    Parameters
    ----------
    steer : float
        requested steering angle
    current_steer : float
        current steering angle
    max_sv : float
        maximum steering velocity

    Returns
    -------
    float
        steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    return sv


@njit(cache=True)
def pid_accl(speed, current_speed, max_a, max_v, min_v):
    """PID control for speed to acceleration

    Parameters
    ----------
    speed : float
        requested speed
    current_speed : float
        current speed
    max_a : float
        maximum acceleration
    max_v : float
        maximum velocity
    min_v : float
        minimum velocity

    Returns
    -------
    float
        acceleration
    """
    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.0:
        if vel_diff > 0:
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if vel_diff > 0:
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl


def func_KS(
    x,
    t,
    u,
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
    f = vehicle_dynamics_ks(
        x,
        u,
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
    )
    return f


def func_ST(
    x,
    t,
    u,
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
    f = vehicle_dynamics_st(
        x,
        u,
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
    )
    return f
