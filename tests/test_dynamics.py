# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""

import time
import unittest

import numpy as np
from f1tenth_gym.envs.dynamic_models import (
    vehicle_dynamics_ks,
    vehicle_dynamics_st,
)


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
    params = np.array([
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
    ])
    f = vehicle_dynamics_ks(
        x,
        u,
        params=params,
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
    params = np.array([
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
    ])
    f = vehicle_dynamics_st(
        x,
        u,
        params=params,
    )
    return f


class DynamicsTest(unittest.TestCase):
    def setUp(self):
        # test params
        self.mu = 1.0489
        self.C_Sf = 21.92 / 1.0489
        self.C_Sr = 21.92 / 1.0489
        self.lf = 0.3048 * 3.793293
        self.lr = 0.3048 * 4.667707
        self.h = 0.3048 * 2.01355
        self.m = 4.4482216152605 / 0.3048 * 74.91452
        self.I = 4.4482216152605 * 0.3048 * 1321.416  # noqa: E741

        # steering constraints
        self.s_min = -1.066  # minimum steering angle [rad]
        self.s_max = 1.066  # maximum steering angle [rad]
        self.sv_min = -0.4  # minimum steering velocity [rad/s]
        self.sv_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.v_min = -13.6  # minimum velocity [m/s]
        self.v_max = 50.8  # minimum velocity [m/s]
        self.v_switch = 7.319  # switching velocity [m/s]
        self.a_max = 11.5  # maximum absolute acceleration [m/s^2]

    def test_derivatives(self):
        # ground truth derivatives
        f_ks_gt = [
            16.3475935934250209,
            0.4819314886013121,
            0.1500000000000000,
            5.1464424102339752,
            0.2401426578627629,
        ]
        f_st_gt = [
            15.7213512030862397,
            0.0925527979719355,
            0.1500000000000000,
            5.3536773276413925,
            0.0529001056654038,
            0.6435589397748606,
            0.0313297971641291,
        ]

        # system dynamics
        g = 9.81
        x_ks = np.array(
            [
                3.9579422297936526,
                0.0391650102771405,
                0.0378491427211811,
                16.3546957860883566,
                0.0294717351052816,
            ]
        )
        x_st = np.array(
            [
                2.0233348142065677,
                0.0041907137716636,
                0.0197545248559617,
                15.7216236334290116,
                0.0025857914776859,
                0.0529001056654038,
                0.0033012170610298,
            ]
        )
        v_delta = 0.15
        acc = 0.63 * g
        u = np.array([v_delta, acc])

        f_ks = func_KS(
            x_ks,
            0,
            u,
            self.mu,
            self.C_Sf,
            self.C_Sr,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
            self.v_switch,
            self.a_max,
            self.v_min,
            self.v_max,
        )
        f_st = func_ST(
            x_st,
            0,
            u,
            self.mu,
            self.C_Sf,
            self.C_Sr,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
            self.v_switch,
            self.a_max,
            self.v_min,
            self.v_max,
        )

        start = time.time()
        for i in range(10000):
            f_st = func_ST(
                x_st,
                0,
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            )
        duration = time.time() - start
        avg_fps = 10000 / duration

        self.assertAlmostEqual(np.max(np.abs(f_ks_gt - f_ks)), 0.0)
        self.assertAlmostEqual(np.max(np.abs(f_st_gt - f_st)), 0.0)
        self.assertGreater(avg_fps, 5000)

    def test_zeroinit_roll(self):
        from scipy.integrate import odeint

        # testing for zero initial state, zero input singularities
        # g = 9.81
        t_start = 0.0
        t_final = 1.0
        delta0 = 0.0
        vel0 = 0.0
        Psi0 = 0.0
        dotPsi0 = 0.0
        beta0 = 0.0
        sy0 = 0.0
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set input: rolling car (velocity should stay constant)
        u = np.array([0.0, 0.0])

        # simulate single-track model
        x_roll_st = odeint(
            func_ST,
            x0_ST,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )
        # simulate kinematic single-track model
        x_roll_ks = odeint(
            func_KS,
            x0_KS,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )

        self.assertTrue(all(x_roll_st[-1] == x0_ST))
        self.assertTrue(all(x_roll_ks[-1] == x0_KS))

    def test_zeroinit_dec(self):
        from scipy.integrate import odeint

        # testing for zero initial state, decelerating input singularities
        g = 9.81
        t_start = 0.0
        t_final = 1.0
        delta0 = 0.0
        vel0 = 0.0
        Psi0 = 0.0
        dotPsi0 = 0.0
        beta0 = 0.0
        sy0 = 0.0
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.0, -0.7 * g])

        # simulate single-track model
        x_dec_st = odeint(
            func_ST,
            x0_ST,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )
        # simulate kinematic single-track model
        x_dec_ks = odeint(
            func_KS,
            x0_KS,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )

        # ground truth for single-track model
        x_dec_st_gt = [
            -3.4335000000000013,
            0.0000000000000000,
            0.0000000000000000,
            -6.8670000000000018,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
        ]
        # ground truth for kinematic single-track model
        x_dec_ks_gt = [
            -3.4335000000000013,
            0.0000000000000000,
            0.0000000000000000,
            -6.8670000000000018,
            0.0000000000000000,
        ]

        self.assertTrue(all(abs(x_dec_st[-1] - x_dec_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_dec_ks[-1] - x_dec_ks_gt) < 1e-2))

    def test_zeroinit_acc(self):
        from scipy.integrate import odeint

        # testing for zero initial state, accelerating with left steer input singularities
        # wheel spin and velocity should increase more wheel spin at rear
        g = 9.81
        t_start = 0.0
        t_final = 1.0
        delta0 = 0.0
        vel0 = 0.0
        Psi0 = 0.0
        dotPsi0 = 0.0
        beta0 = 0.0
        sy0 = 0.0
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.63 * g])

        # simulate single-track model
        x_acc_st = odeint(
            func_ST,
            x0_ST,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )
        # simulate kinematic single-track model
        x_acc_ks = odeint(
            func_KS,
            x0_KS,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )

        # ground truth for single-track model
        x_acc_st_gt = [
            3.0731976046859715,
            0.2869835398304389,
            0.1500000000000000,
            6.1802999999999999,
            0.1097747074946325,
            0.3248268063223301,
            0.0697547542798040,
        ]
        # ground truth for kinematic single-track model
        x_acc_ks_gt = [
            3.0845676868494927,
            0.1484249221523042,
            0.1500000000000000,
            6.1803000000000017,
            0.1203664469224163,
        ]

        self.assertTrue(all(abs(x_acc_st[-1] - x_acc_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_acc_ks[-1] - x_acc_ks_gt) < 1e-2))

    def test_zeroinit_rollleft_kinematic(self):
        from scipy.integrate import odeint

        # testing for zero initial state, rolling and steering left input singularities
        # g = 9.81
        t_start = 0.0
        t_final = 1.0
        delta0 = 0.0
        vel0 = 0.0
        Psi0 = 0.0
        dotPsi0 = 0.0
        beta0 = 0.0
        sy0 = 0.0
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = np.array(initial_state[0:5])

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.0])

        # simulate kinematic single-track model
        x_left_ks = odeint(
            func_KS,
            x0_KS,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )
        # ground truth for kinematic single-track model
        x_left_ks_gt = [
            0.0000000000000000,
            0.0000000000000000,
            0.1500000000000000,
            0.0000000000000000,
            0.0000000000000000,
        ]
        np.testing.assert_array_almost_equal(x_left_ks[-1], x_left_ks_gt, decimal=2)

    def test_zeroinit_rollleft_singletrack(self):
        from scipy.integrate import odeint

        # testing for zero initial state, rolling and steering left input singularities
        # g = 9.81
        t_start = 0.0
        t_final = 1.0
        delta0 = 0.0
        vel0 = 0.0
        Psi0 = 0.0
        dotPsi0 = 0.0
        beta0 = 0.0
        sy0 = 0.0
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.0])

        # simulate single-track model
        x_left_st = odeint(
            func_ST,
            x0_ST,
            t,
            args=(
                u,
                self.mu,
                self.C_Sf,
                self.C_Sr,
                self.lf,
                self.lr,
                self.h,
                self.m,
                self.I,
                self.s_min,
                self.s_max,
                self.sv_min,
                self.sv_max,
                self.v_switch,
                self.a_max,
                self.v_min,
                self.v_max,
            ),
        )

        # ground truth for single-track model
        x_left_st_gt = [
            0.0000000000000000,
            0.0000000000000000,
            0.1500000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0833661500000000,
        ]

        np.testing.assert_array_almost_equal(x_left_st[-1], x_left_st_gt, decimal=2)
