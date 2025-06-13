import numpy as np
from numba import njit
from numba.typed import Dict

from f1tenth_gym.envs.dynamic_models.utils import steering_constraint, accl_constraints

from .tire_model import (
    formula_lateral,
    formula_lateral_comb,
    formula_longitudinal,
    formula_longitudinal_comb,
)
from ..kinematic import vehicle_dynamics_ks_cog


def vehicle_dynamics_mb(x: np.ndarray, u_init: np.ndarray, params: dict):
    """
    vehicleDynamics_mb - multi-body vehicle dynamics based on the DoT (department of transportation) vehicle dynamics
    reference point: center of mass

    Syntax:
        f = vehicleDynamics_mb(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param uInit: vehicle input vector
        :param params: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations
    """

    # ------------- BEGIN CODE --------------

    # set gravity constant
    g = 9.81  # [m/s^2]

    # states
    # x0 = x-position in a global coordinate system
    # x1 = y-position in a global coordinate system
    # x2 = steering angle of front wheels
    # x3 = velocity in x-direction
    # x4 = yaw angle
    # x5 = yaw rate

    # x6 = roll angle
    # x7 = roll rate
    # x8 = pitch angle
    # x9 = pitch rate
    # x10 = velocity in y-direction
    # x11 = z-position
    # x12 = velocity in z-direction

    # x13 = roll angle front
    # x14 = roll rate front
    # x15 = velocity in y-direction front
    # x16 = z-position front
    # x17 = velocity in z-direction front

    # x18 = roll angle rear
    # x19 = roll rate rear
    # x20 = velocity in y-direction rear
    # x21 = z-position rear
    # x22 = velocity in z-direction rear

    # x23 = left front wheel angular speed
    # x24 = right front wheel angular speed
    # x25 = left rear wheel angular speed
    # x26 = right rear wheel angular speed

    # x27 = delta_y_f
    # x28 = delta_y_r

    # u0 = steering angle velocity of front wheels
    # u1 = acceleration

    u = np.zeros_like(u_init)

    # vehicle body dimensions
    length = params["length"]  # vehicle length [m]
    width = params["width"]  # vehicle width [m]

    # steering constraints
    # s_min = params["s_min"]  # minimum steering angle [rad]
    # s_max = params["s_max"]  # maximum steering angle [rad]
    # sv_min = params["sv_min"]  # minimum steering velocity [rad/s]
    # sv_max = params["sv_max"]  # maximum steering velocity [rad/s]

    # longitudinal constraints
    # v_min = params["v_min"]  # minimum velocity [m/s]
    # v_max = params["v_max"]  # minimum velocity [m/s]
    # v_switch = params["v_switch"]  # switching velocity [m/s]
    # a_max = params["a_max"]  # maximum absolute acceleration [m/s^2]

    # # masses
    # m = params["m"]  # vehicle mass [kg]  MASS
    # m_s = params["m_s"]  # sprung mass [kg]  SMASS
    # m_uf = params["m_uf"]  # unsprung mass front [kg]  UMASSF
    # m_ur = params["m_ur"]  # unsprung mass rear [kg]  UMASSR

    # # axes distances
    # lf = params["lf"]
    # # distance from spring mass center of gravity to front axle [m]  LENA
    # lr = params["lf"]
    # # distance from spring mass center of gravity to rear axle [m]  LENB

    # # moments of inertia of sprung mass
    # I_Phi_s = params["I_Phi_s"]
    # # moment of inertia for sprung mass in roll [kg m^2]  IXS
    # I_y_s = params["I_y_s"]  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
    # I_z = params["I_z"]  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    # I_xz_s = params["I_xz_s"]  # moment of inertia cross product [kg m^2]  IXZ

    # # suspension parameters
    # K_sf = params["K_sf"]  # suspension spring rate (front) [N/m]  KSF
    # K_sdf = params["K_sdf"]  # suspension damping rate (front) [N s/m]  KSDF
    # K_sr = params["K_sr"]  # suspension spring rate (rear) [N/m]  KSR
    # K_sdr = params["K_sdr"]  # suspension damping rate (rear) [N s/m]  KSDR

    # # geometric parameters
    # T_f = params["T_f"]  # track width front [m]  TRWF
    # T_r = params["T_r"]  # track width rear [m]  TRWB
    # K_ras = params["K_ras"]
    # # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

    # K_tsf = params["K_tsf"]
    # # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
    # K_tsr = params["K_tsr"]
    # # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
    # K_rad = params["K_rad"]
    # # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
    # K_zt = params["K_zt"]  # vertical spring rate of tire [N/m]  TSPRINGR

    # h_cg = params["h_cg"]
    # # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
    # h_raf = params["h_raf"]  # height of roll axis above ground (front) [m]  HRAF
    # h_rar = params["h_rar"]  # height of roll axis above ground (rear) [m]  HRAR

    # h_s = params["h_s"]  # M_s center of gravity above ground [m]  HS

    # I_uf = params["I_uf"]
    # # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
    # I_ur = params["I_ur"]
    # # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
    # I_y_w = params["I_y_w"]
    # # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

    # K_lt = params["K_lt"]
    # # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
    # R_w = params["R_w"]
    # # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

    # # split of brake and engine torque
    # T_sb = params["T_sb"]
    # T_se = params["T_se"]

    # # suspension parameters
    # D_f = params["D_f"]  # [rad/m]  DF
    # D_r = params["D_r"]  # [rad/m]  DR
    # E_f = params["E_f"]  # [needs conversion if nonzero]  EF
    # E_r = params["E_r"]  # [needs conversion if nonzero]  ER

    KIN_THRESH = 0.5

    # consider steering and acceleration constraints - outside of the integration
    u[0] = steering_constraint(
        x[2],
        u_init[0],
        params["s_min"],
        params["s_max"],
        params["sv_min"],
        params["sv_max"],
    )
    u[1] = accl_constraints(
        x[3],
        u_init[1],
        params["v_switch"],
        params["a_max"],
        params["v_min"],
        params["v_max"],
    )

    if abs(x[3]) < KIN_THRESH:
        beta = 0.0
    else:
        beta = np.arctan(x[10] / x[3])
    vel = np.sqrt(x[3] ** 2 + x[10] ** 2)

    # vertical tire forces
    F_z_LF = (
        x[16]
        + params["R_w"] * (np.cos(x[13]) - 1)
        - 0.5 * params["T_f"] * np.sin(x[13])
    ) * params["K_zt"]
    F_z_RF = (
        x[16]
        + params["R_w"] * (np.cos(x[13]) - 1)
        + 0.5 * params["T_f"] * np.sin(x[13])
    ) * params["K_zt"]
    F_z_LR = (
        x[21]
        + params["R_w"] * (np.cos(x[18]) - 1)
        - 0.5 * params["T_r"] * np.sin(x[18])
    ) * params["K_zt"]
    F_z_RR = (
        x[21]
        + params["R_w"] * (np.cos(x[18]) - 1)
        + 0.5 * params["T_r"] * np.sin(x[18])
    ) * params["K_zt"]

    # obtain individual tire speeds
    u_w_lf = (x[3] + 0.5 * params["T_f"] * x[5]) * np.cos(x[2]) + (
        x[10] + params["lf"] * x[5]
    ) * np.sin(x[2])
    u_w_rf = (x[3] - 0.5 * params["T_f"] * x[5]) * np.cos(x[2]) + (
        x[10] + params["lf"] * x[5]
    ) * np.sin(x[2])
    u_w_lr = x[3] + 0.5 * params["T_r"] * x[5]
    u_w_rr = x[3] - 0.5 * params["T_r"] * x[5]

    # negative wheel spin forbidden
    if u_w_lf < 0.0:
        u_w_lf *= 0

    if u_w_rf < 0.0:
        u_w_rf *= 0

    if u_w_lr < 0.0:
        u_w_lr *= 0

    if u_w_rr < 0.0:
        u_w_rr *= 0
    # compute longitudinal slip
    # switch to kinematic model for small velocities
    if abs(x[3]) < KIN_THRESH:
        s_lf = 0.0
        s_rf = 0.0
        s_lr = 0.0
        s_rr = 0.0
    else:
        s_lf = 1 - params["R_w"] * x[23] / u_w_lf
        s_rf = 1 - params["R_w"] * x[24] / u_w_rf
        s_lr = 1 - params["R_w"] * x[25] / u_w_lr
        s_rr = 1 - params["R_w"] * x[26] / u_w_rr

    # lateral slip angles
    # switch to kinematic model for small velocities
    if abs(x[3]) < KIN_THRESH:
        alpha_LF = 0.0
        alpha_RF = 0.0
        alpha_LR = 0.0
        alpha_RR = 0.0
    else:
        alpha_LF = (
            np.arctan(
                (x[10] + params["lf"] * x[5] - x[14] * (params["R_w"] - x[16]))
                / (x[3] + 0.5 * params["T_f"] * x[5])
            )
            - x[2]
        )
        alpha_RF = (
            np.arctan(
                (x[10] + params["lf"] * x[5] - x[14] * (params["R_w"] - x[16]))
                / (x[3] - 0.5 * params["T_f"] * x[5])
            )
            - x[2]
        )
        alpha_LR = np.arctan(
            (x[10] - params["lr"] * x[5] - x[19] * (params["R_w"] - x[21]))
            / (x[3] + 0.5 * params["T_r"] * x[5])
        )
        alpha_RR = np.arctan(
            (x[10] - params["lr"] * x[5] - x[19] * (params["R_w"] - x[21]))
            / (x[3] - 0.5 * params["T_r"] * x[5])
        )

    # auxiliary suspension movement
    z_SLF = (
        (params["h_s"] - params["R_w"] + x[16] - x[11]) / np.cos(x[6])
        - params["h_s"]
        + params["R_w"]
        + params["lf"] * x[8]
        + 0.5 * (x[6] - x[13]) * params["T_f"]
    )
    z_SRF = (
        (params["h_s"] - params["R_w"] + x[16] - x[11]) / np.cos(x[6])
        - params["h_s"]
        + params["R_w"]
        + params["lf"] * x[8]
        - 0.5 * (x[6] - x[13]) * params["T_f"]
    )
    z_SLR = (
        (params["h_s"] - params["R_w"] + x[21] - x[11]) / np.cos(x[6])
        - params["h_s"]
        + params["R_w"]
        - params["lr"] * x[8]
        + 0.5 * (x[6] - x[18]) * params["T_r"]
    )
    z_SRR = (
        (params["h_s"] - params["R_w"] + x[21] - x[11]) / np.cos(x[6])
        - params["h_s"]
        + params["R_w"]
        - params["lr"] * x[8]
        - 0.5 * (x[6] - x[18]) * params["T_r"]
    )

    dz_SLF = x[17] - x[12] + params["lf"] * x[9] + 0.5 * (x[7] - x[14]) * params["T_f"]
    dz_SRF = x[17] - x[12] + params["lf"] * x[9] - 0.5 * (x[7] - x[14]) * params["T_f"]
    dz_SLR = x[22] - x[12] - params["lr"] * x[9] + 0.5 * (x[7] - x[19]) * params["T_r"]
    dz_SRR = x[22] - x[12] - params["lr"] * x[9] - 0.5 * (x[7] - x[19]) * params["T_r"]

    # camber angles
    gamma_LF = x[6] + params["D_f"] * z_SLF + params["E_f"] * (z_SLF) ** 2
    gamma_RF = x[6] - params["D_f"] * z_SRF - params["E_f"] * (z_SRF) ** 2
    gamma_LR = x[6] + params["D_r"] * z_SLR + params["E_r"] * (z_SLR) ** 2
    gamma_RR = x[6] - params["D_r"] * z_SRR - params["E_r"] * (z_SRR) ** 2

    # compute longitudinal tire forces using the magic formula for pure slip
    F0_x_LF = formula_longitudinal(s_lf, gamma_LF, F_z_LF, params)
    F0_x_RF = formula_longitudinal(s_rf, gamma_RF, F_z_RF, params)
    F0_x_LR = formula_longitudinal(s_lr, gamma_LR, F_z_LR, params)
    F0_x_RR = formula_longitudinal(s_rr, gamma_RR, F_z_RR, params)

    # compute lateral tire forces using the magic formula for pure slip
    res = formula_lateral(alpha_LF, gamma_LF, F_z_LF, params)
    F0_y_LF = res[0]
    mu_y_LF = res[1]
    res = formula_lateral(alpha_RF, gamma_RF, F_z_RF, params)
    F0_y_RF = res[0]
    mu_y_RF = res[1]
    res = formula_lateral(alpha_LR, gamma_LR, F_z_LR, params)
    F0_y_LR = res[0]
    mu_y_LR = res[1]
    res = formula_lateral(alpha_RR, gamma_RR, F_z_RR, params)
    F0_y_RR = res[0]
    mu_y_RR = res[1]

    # compute longitudinal tire forces using the magic formula for combined slip
    F_x_LF = formula_longitudinal_comb(s_lf, alpha_LF, F0_x_LF, params)
    F_x_RF = formula_longitudinal_comb(s_rf, alpha_RF, F0_x_RF, params)
    F_x_LR = formula_longitudinal_comb(s_lr, alpha_LR, F0_x_LR, params)
    F_x_RR = formula_longitudinal_comb(s_rr, alpha_RR, F0_x_RR, params)

    # compute lateral tire forces using the magic formula for combined slip
    F_y_LF = formula_lateral_comb(
        s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, params
    )
    F_y_RF = formula_lateral_comb(
        s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, params
    )
    F_y_LR = formula_lateral_comb(
        s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, params
    )
    F_y_RR = formula_lateral_comb(
        s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, params
    )

    # auxiliary movements for compliant joint equations
    delta_z_f = params["h_s"] - params["R_w"] + x[16] - x[11]
    delta_z_r = params["h_s"] - params["R_w"] + x[21] - x[11]

    delta_phi_f = x[6] - x[13]
    delta_phi_r = x[6] - x[18]

    dot_delta_phi_f = x[7] - x[14]
    dot_delta_phi_r = x[7] - x[19]

    dot_delta_z_f = x[17] - x[12]
    dot_delta_z_r = x[22] - x[12]

    dot_delta_y_f = x[10] + params["lf"] * x[5] - x[15]
    dot_delta_y_r = x[10] - params["lr"] * x[5] - x[20]

    delta_f = (
        delta_z_f * np.sin(x[6])
        - x[27] * np.cos(x[6])
        - (params["h_raf"] - params["R_w"]) * np.sin(delta_phi_f)
    )
    delta_r = (
        delta_z_r * np.sin(x[6])
        - x[28] * np.cos(x[6])
        - (params["h_rar"] - params["R_w"]) * np.sin(delta_phi_r)
    )

    dot_delta_f = (
        (delta_z_f * np.cos(x[6]) + x[27] * np.sin(x[6])) * x[7]
        + dot_delta_z_f * np.sin(x[6])
        - dot_delta_y_f * np.cos(x[6])
        - (params["h_raf"] - params["R_w"]) * np.cos(delta_phi_f) * dot_delta_phi_f
    )
    dot_delta_r = (
        (delta_z_r * np.cos(x[6]) + x[28] * np.sin(x[6])) * x[7]
        + dot_delta_z_r * np.sin(x[6])
        - dot_delta_y_r * np.cos(x[6])
        - (params["h_rar"] - params["R_w"]) * np.cos(delta_phi_r) * dot_delta_phi_r
    )

    # compliant joint forces
    F_RAF = delta_f * params["K_ras"] + dot_delta_f * params["K_rad"]
    F_RAR = delta_r * params["K_ras"] + dot_delta_r * params["K_rad"]

    # auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    F_SLF = (
        params["m_s"] * g * params["lr"] / (2 * (params["lf"] + params["lr"]))
        - z_SLF * params["K_sf"]
        - dz_SLF * params["K_sdf"]
        + (x[6] - x[13]) * params["K_tsf"] / params["T_f"]
    )

    F_SRF = (
        params["m_s"] * g * params["lr"] / (2 * (params["lf"] + params["lr"]))
        - z_SRF * params["K_sf"]
        - dz_SRF * params["K_sdf"]
        - (x[6] - x[13]) * params["K_tsf"] / params["T_f"]
    )

    F_SLR = (
        params["m_s"] * g * params["lf"] / (2 * (params["lf"] + params["lr"]))
        - z_SLR * params["K_sr"]
        - dz_SLR * params["K_sdr"]
        + (x[6] - x[18]) * params["K_tsr"] / params["T_r"]
    )

    F_SRR = (
        params["m_s"] * g * params["lf"] / (2 * (params["lf"] + params["lr"]))
        - z_SRR * params["K_sr"]
        - dz_SRR * params["K_sdr"]
        - (x[6] - x[18]) * params["K_tsr"] / params["T_r"]
    )

    # auxiliary variables sprung mass
    sumX = (
        F_x_LR
        + F_x_RR
        + (F_x_LF + F_x_RF) * np.cos(x[2])
        - (F_y_LF + F_y_RF) * np.sin(x[2])
    )

    sumN = (
        (F_y_LF + F_y_RF) * params["lf"] * np.cos(x[2])
        + (F_x_LF + F_x_RF) * params["lf"] * np.sin(x[2])
        + (F_y_RF - F_y_LF) * 0.5 * params["T_f"] * np.sin(x[2])
        + (F_x_LF - F_x_RF) * 0.5 * params["T_f"] * np.cos(x[2])
        + (F_x_LR - F_x_RR) * 0.5 * params["T_r"]
        - (F_y_LR + F_y_RR) * params["lr"]
    )

    sumY_s = (F_RAF + F_RAR) * np.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR) * np.sin(
        x[6]
    )

    sumL = (
        0.5 * F_SLF * params["T_f"]
        + 0.5 * F_SLR * params["T_r"]
        - 0.5 * F_SRF * params["T_f"]
        - 0.5 * F_SRR * params["T_r"]
        - F_RAF
        / np.cos(x[6])
        * (
            params["h_s"]
            - x[11]
            - params["R_w"]
            + x[16]
            - (params["h_raf"] - params["R_w"]) * np.cos(x[13])
        )
        - F_RAR
        / np.cos(x[6])
        * (
            params["h_s"]
            - x[11]
            - params["R_w"]
            + x[21]
            - (params["h_rar"] - params["R_w"]) * np.cos(x[18])
        )
    )

    sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR) * np.cos(x[6]) - (F_RAF + F_RAR) * np.sin(
        x[6]
    )

    sumM_s = (
        params["lf"] * (F_SLF + F_SRF)
        - params["lr"] * (F_SLR + F_SRR)
        + (
            (F_x_LF + F_x_RF) * np.cos(x[2])
            - (F_y_LF + F_y_RF) * np.sin(x[2])
            + F_x_LR
            + F_x_RR
        )
        * (params["h_s"] - x[11])
    )

    # auxiliary variables unsprung mass
    sumL_uf = (
        0.5 * F_SRF * params["T_f"]
        - 0.5 * F_SLF * params["T_f"]
        - F_RAF * (params["h_raf"] - params["R_w"])
        + F_z_LF
        * (
            params["R_w"] * np.sin(x[13])
            + 0.5 * params["T_f"] * np.cos(x[13])
            - params["K_lt"] * F_y_LF
        )
        - F_z_RF
        * (
            -params["R_w"] * np.sin(x[13])
            + 0.5 * params["T_f"] * np.cos(x[13])
            + params["K_lt"] * F_y_RF
        )
        - ((F_y_LF + F_y_RF) * np.cos(x[2]) + (F_x_LF + F_x_RF) * np.sin(x[2]))
        * (params["R_w"] - x[16])
    )

    sumL_ur = (
        0.5 * F_SRR * params["T_r"]
        - 0.5 * F_SLR * params["T_r"]
        - F_RAR * (params["h_rar"] - params["R_w"])
        + F_z_LR
        * (
            params["R_w"] * np.sin(x[18])
            + 0.5 * params["T_r"] * np.cos(x[18])
            - params["K_lt"] * F_y_LR
        )
        - F_z_RR
        * (
            -params["R_w"] * np.sin(x[18])
            + 0.5 * params["T_r"] * np.cos(x[18])
            + params["K_lt"] * F_y_RR
        )
        - (F_y_LR + F_y_RR) * (params["R_w"] - x[21])
    )

    sumZ_uf = F_z_LF + F_z_RF + F_RAF * np.sin(x[6]) - (F_SLF + F_SRF) * np.cos(x[6])

    sumZ_ur = F_z_LR + F_z_RR + F_RAR * np.sin(x[6]) - (F_SLR + F_SRR) * np.cos(x[6])

    sumY_uf = (
        (F_y_LF + F_y_RF) * np.cos(x[2])
        + (F_x_LF + F_x_RF) * np.sin(x[2])
        - F_RAF * np.cos(x[6])
        - (F_SLF + F_SRF) * np.sin(x[6])
    )

    sumY_ur = (F_y_LR + F_y_RR) - F_RAR * np.cos(x[6]) - (F_SLR + F_SRR) * np.sin(x[6])

    # dynamics common with single-track model
    f = np.zeros(29)  # init 'right hand side'
    # switch to kinematic model for small velocities
    if abs(x[3]) < KIN_THRESH:
        # wheelbase
        # lwb = lf + lr

        # system dynamics
        # x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        # f_ks = vehicle_dynamics_ks(x_ks, u, p)
        # f.extend(f_ks)
        # f.append(u[1]*lwb*np.tan(x[2]) + x[3]/(lwb*np.cos(x[2])**2)*u[0])

        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = params["lf"] + params["lr"]
        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        # kinematic model
        f_ks = vehicle_dynamics_ks_cog(np.array(x_ks), u, params)
        f[0:5] = np.array([f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]])
        # derivative of slip angle and yaw rate
        d_beta = (params["lr"] * u[0]) / (
            lwb
            * np.cos(x[2]) ** 2
            * (1 + (np.tan(x[2]) ** 2 * params["lr"] / lwb) ** 2)
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
        f[5] = dd_psi

    else:
        f[0] = np.cos(beta + x[4]) * vel
        f[1] = np.sin(beta + x[4]) * vel
        f[2] = u[0]
        f[3] = 1 / params["m"] * sumX + x[5] * x[10]
        f[4] = x[5]
        f[5] = (
            1
            / (params["I_z"] - (params["I_xz_s"]) ** 2 / params["I_Phi_s"])
            * (sumN + params["I_xz_s"] / params["I_Phi_s"] * sumL)
        )

    # remaining sprung mass dynamics
    f[6] = x[7]
    f[7] = (
        1
        / (params["I_Phi_s"] - (params["I_xz_s"]) ** 2 / params["I_z"])
        * (params["I_xz_s"] / params["I_z"] * sumN + sumL)
    )
    f[8] = x[9]
    f[9] = 1 / params["I_y_s"] * sumM_s
    f[10] = 1 / params["m_s"] * sumY_s - x[5] * x[3]
    f[11] = x[12]
    f[12] = g - 1 / params["m_s"] * sumZ_s

    # unsprung mass dynamics (front)
    f[13] = x[14]
    f[14] = 1 / params["I_uf"] * sumL_uf
    f[15] = 1 / params["m_uf"] * sumY_uf - x[5] * x[3]
    f[16] = x[17]
    f[17] = g - 1 / params["m_uf"] * sumZ_uf

    # unsprung mass dynamics (rear)
    f[18] = x[19]
    f[19] = 1 / params["I_ur"] * sumL_ur
    f[20] = 1 / params["m_ur"] * sumY_ur - x[5] * x[3]
    f[21] = x[22]
    f[22] = g - 1 / params["m_ur"] * sumZ_ur

    # convert acceleration input to brake and engine torque
    if u[1] > 0:
        T_B = 0.0
        T_E = params["m"] * params["R_w"] * u[1]
    else:
        T_B = params["m"] * params["R_w"] * u[1]
        T_E = 0.0

    # wheel dynamics  (T  new parameter for torque splitting)
    f[23] = (
        1
        / params["I_y_w"]
        * (
            -params["R_w"] * F_x_LF
            + 0.5 * params["T_sb"] * T_B
            + 0.5 * params["T_se"] * T_E
        )
    )
    f[24] = (
        1
        / params["I_y_w"]
        * (
            -params["R_w"] * F_x_RF
            + 0.5 * params["T_sb"] * T_B
            + 0.5 * params["T_se"] * T_E
        )
    )
    f[25] = (
        1
        / params["I_y_w"]
        * (
            -params["R_w"] * F_x_LR
            + 0.5 * (1 - params["T_sb"]) * T_B
            + 0.5 * (1 - params["T_se"]) * T_E
        )
    )
    f[26] = (
        1
        / params["I_y_w"]
        * (
            -params["R_w"] * F_x_RR
            + 0.5 * (1 - params["T_sb"]) * T_B
            + 0.5 * (1 - params["T_se"]) * T_E
        )
    )

    # negative wheel spin forbidden
    for iState in range(23, 27):
        if x[iState] < 0.0:
            x[iState] = 0.0
            f[iState] = 0.0

    # compliant joint equations
    f[27] = dot_delta_y_f
    f[28] = dot_delta_y_r

    return f


@njit(cache=True)
def get_standardized_state_mb(x: np.ndarray) -> dict:
    """[X,Y,DELTA,V_X, V_Y,YAW,YAW_RATE,SLIP]"""
    d = dict()
    d["x"] = x[0]
    d["y"] = x[1]
    d["delta"] = x[2]
    d["v_x"] = x[3]
    d["v_y"] = x[10]
    d["yaw"] = x[4]
    d["yaw_rate"] = x[5]
    d["slip"] = np.arctan2(x[10], x[3])
    return d
