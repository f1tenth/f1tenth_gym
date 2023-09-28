# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""
import warnings
from enum import Enum

import numpy as np
from numba import njit


class DynamicModel(Enum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track

    @staticmethod
    def from_string(model: str):
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
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        else:
            raise ValueError(f"Unknown model type {self}")


@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max * v_switch / vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.0
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl


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
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array(
        [
            steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
            accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max),
        ]
    )

    # system dynamics
    f = np.array(
        [
            x[3] * np.cos(x[4]),
            x[3] * np.sin(x[4]),
            u[0],
            u[1],
            x[3] / lwb * np.tan(x[2]),
        ]
    )
    return f


@njit(cache=True)
def vehicle_dynamics_ks_cog(
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
    Single Track Kinematic Vehicle Dynamics at CoG.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array(
        [
            steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
            accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max),
        ]
    )

    # system dynamics
    beta = np.arctan(np.tan(x[2]) * lr / lwb)
    f = np.array(
        [
            x[3] * np.cos(beta + x[4]),
            x[3] * np.sin(beta + x[4]),
            u[0],
            u[1],
            x[3] * np.cos(beta) * np.tan(x[2]) / lwb,
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
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array(
        [
            steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
            accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max),
        ]
    )

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks_cog(
            x_ks,
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
        d_beta = (lr * u[0]) / (
            lwb * np.cos(x[2]) ** 2 * (1 + (np.tan(x[2]) ** 2 * lr / lwb) ** 2)
        )
        dd_psi = (
            1
            / lwb
            * (
                u[1] * np.cos(x[6]) * np.tan(x[2])
                - x[3] * np.sin(x[6]) * d_beta * np.tan(x[2])
                + x[3] * np.cos(x[6]) * u[0] / np.cos(x[2]) ** 2
            )
        )
        f = np.hstack(
            (
                f_ks,
                np.array(
                    [
                        dd_psi,
                        d_beta,
                    ]
                ),
            )
        )

    else:
        # system dynamics
        f = np.array(
            [
                x[3] * np.cos(x[6] + x[4]),
                x[3] * np.sin(x[6] + x[4]),
                u[0],
                u[1],
                x[5],
                -mu
                * m
                / (x[3] * I * (lr + lf))
                * (
                    lf**2 * C_Sf * (g * lr - u[1] * h)
                    + lr**2 * C_Sr * (g * lf + u[1] * h)
                )
                * x[5]
                + mu
                * m
                / (I * (lr + lf))
                * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h))
                * x[6]
                + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
                (
                    mu
                    / (x[3] ** 2 * (lr + lf))
                    * (
                        C_Sr * (g * lf + u[1] * h) * lr
                        - C_Sf * (g * lr - u[1] * h) * lf
                    )
                    - 1
                )
                * x[5]
                - mu
                / (x[3] * (lr + lf))
                * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h))
                * x[6]
                + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2],
            ]
        )

    return f


@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

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

    return accl, sv


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
