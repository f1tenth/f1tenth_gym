import numpy as np
from numba import njit

@njit(cache=True)
def upper_accel_limit(vel, a_max, v_switch):
    """
    Upper acceleration limit, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            a_max (float): maximum allowed acceleration, symmetrical
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)

        Returns:
            positive_accel_limit (float): adjusted acceleration
    """
    if vel > v_switch:
        pos_limit = a_max * (v_switch / vel)
    else:
        pos_limit = a_max

    return pos_limit

@njit(cache=True)
def accl_constraints(vel, a_long_d, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            a_long_d (float): unconstrained desired acceleration in the direction of travel.
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration, symmetrical
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
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
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
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
def pid_steer(steer, current_steer, max_sv):
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    return sv


@njit(cache=True)
def pid_accl(speed, current_speed, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
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
