import numpy as np
from numba import njit

from .tire_model import (
    formula_lateral,
    formula_lateral_comb,
    formula_longitudinal,
    formula_longitudinal_comb,
)
from ..kinematic import vehicle_dynamics_ks


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
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity in x-direction
    # x5 = yaw angle
    # x6 = yaw rate

    # x7 = roll angle
    # x8 = roll rate
    # x9 = pitch angle
    # x10 = pitch rate
    # x11 = velocity in y-direction
    # x12 = z-position
    # x13 = velocity in z-direction

    # x14 = roll angle front
    # x15 = roll rate front
    # x16 = velocity in y-direction front
    # x17 = z-position front
    # x18 = velocity in z-direction front

    # x19 = roll angle rear
    # x20 = roll rate rear
    # x21 = velocity in y-direction rear
    # x22 = z-position rear
    # x23 = velocity in z-direction rear

    # x24 = left front wheel angular speed
    # x25 = right front wheel angular speed
    # x26 = left rear wheel angular speed
    # x27 = right rear wheel angular speed

    # x28 = delta_y_f
    # x29 = delta_y_r

    # u1 = steering angle velocity of front wheels
    # u2 = acceleration

    u = u_init

    # vehicle body dimensions
    length = params["length"]  # vehicle length [m]
    width = params["width"]  # vehicle width [m]

    # steering constraints
    s_min = params["s_min"]  # minimum steering angle [rad]
    s_max = params["s_max"]  # maximum steering angle [rad]
    sv_min = params["sv_min"]  # minimum steering velocity [rad/s]
    sv_max = params["sv_max"]  # maximum steering velocity [rad/s]

    # longitudinal constraints
    v_min = params["s_min"]  # minimum velocity [m/s]
    v_max = params["s_max"]  # minimum velocity [m/s]
    v_switch = params["sv_min"]  # switching velocity [m/s]
    a_max = params["sv_max"]  # maximum absolute acceleration [m/s^2]

    # masses
    m = params["m"]  # vehicle mass [kg]  MASS
    m_s = params["m_s"]  # sprung mass [kg]  SMASS
    m_uf = params["m_uf"]  # unsprung mass front [kg]  UMASSF
    m_ur = params["m_ur"]  # unsprung mass rear [kg]  UMASSR

    # axes distances
    lf = params["lf"]
    # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params["lf"]
    # distance from spring mass center of gravity to rear axle [m]  LENB

    # moments of inertia of sprung mass
    I_Phi_s = params["I_Phi_s"]
    # moment of inertia for sprung mass in roll [kg m^2]  IXS
    I_y_s = params["I_y_s"]  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
    I_z = params["I_z"]  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    I_xz_s = params["I_xz_s"]  # moment of inertia cross product [kg m^2]  IXZ

    # suspension parameters
    K_sf = params["K_sf"]  # suspension spring rate (front) [N/m]  KSF
    K_sdf = params["K_sdf"]  # suspension damping rate (front) [N s/m]  KSDF
    K_sr = params["K_sr"]  # suspension spring rate (rear) [N/m]  KSR
    K_sdr = params["K_sdr"]  # suspension damping rate (rear) [N s/m]  KSDR

    # geometric parameters
    T_f = params["T_f"]  # track width front [m]  TRWF
    T_r = params["T_r"]  # track width rear [m]  TRWB
    K_ras = params["K_ras"]
    # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

    K_tsf = params["K_tsf"]
    # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
    K_tsr = params["K_tsr"]
    # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
    K_rad = params["K_rad"]
    # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
    K_zt = params["K_zt"]  # vertical spring rate of tire [N/m]  TSPRINGR

    h_cg = params["h_cg"]
    # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
    h_raf = params["h_raf"]  # height of roll axis above ground (front) [m]  HRAF
    h_rar = params["h_rar"]  # height of roll axis above ground (rear) [m]  HRAR

    h_s = params["h_s"]  # M_s center of gravity above ground [m]  HS

    I_uf = params["I_uf"]
    # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
    I_ur = params["I_ur"]
    # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
    I_y_w = params["I_y_w"]
    # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

    K_lt = params["K_lt"]
    # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
    R_w = params["R_w"]
    # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

    # split of brake and engine torque
    T_sb = params["T_sb"]
    T_se = params["T_se"]

    # suspension parameters
    D_f = params["D_f"]  # [rad/m]  DF
    D_r = params["D_r"]  # [rad/m]  DR
    E_f = params["E_f"]  # [needs conversion if nonzero]  EF
    E_r = params["E_r"]  # [needs conversion if nonzero]  ER

    use_kinematic = True if x[3] < 0.1 else False

    # consider steering and acceleration constraints - outside of the integration
    # u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # compute slip angle at cg - outside of the integration
    # switch to kinematic model for small velocities handeled outside
    if use_kinematic:
        beta = 0.0
    else:
        beta = np.arctan(x[10] / x[3])
        vel = np.sqrt(x[3] ** 2 + x[10] ** 2)

    # vertical tire forces
    F_z_LF = (x[16] + R_w * (np.cos(x[13]) - 1) - 0.5 * T_f * np.sin(x[13])) * K_zt
    F_z_RF = (x[16] + R_w * (np.cos(x[13]) - 1) + 0.5 * T_f * np.sin(x[13])) * K_zt
    F_z_LR = (x[21] + R_w * (np.cos(x[18]) - 1) - 0.5 * T_r * np.sin(x[18])) * K_zt
    F_z_RR = (x[21] + R_w * (np.cos(x[18]) - 1) + 0.5 * T_r * np.sin(x[18])) * K_zt

    # obtain individual tire speeds
    u_w_lf = (x[3] + 0.5 * T_f * x[5]) * np.cos(x[2]) + (x[10] + lf * x[5]) * np.sin(
        x[2]
    )
    u_w_rf = (x[3] - 0.5 * T_f * x[5]) * np.cos(x[2]) + (x[10] + lf * x[5]) * np.sin(
        x[2]
    )
    u_w_lr = x[3] + 0.5 * T_r * x[5]
    u_w_rr = x[3] - 0.5 * T_r * x[5]

    # negative wheel spin forbidden
    u_w_lf = np.maximum(u_w_lf, 0.0)
    u_w_rf = np.maximum(u_w_rf, 0.0)
    u_w_lr = np.maximum(u_w_lr, 0.0)
    u_w_rr = np.maximum(u_w_rr, 0.0)
    # if u_w_lf < 0.0:
    #     u_w_lf *= 0
    #
    # if u_w_rf < 0.0:
    #     u_w_rf *= 0
    #
    # if u_w_lr < 0.0:
    #     u_w_lr *= 0
    #
    # if u_w_rr < 0.0:
    #     u_w_rr *= 0

    # compute longitudinal slip
    # switch to kinematic model for small velocities
    if use_kinematic:
        s_lf = 0.0
        s_rf = 0.0
        s_lr = 0.0
        s_rr = 0.0
    else:
        s_lf = 1 - R_w * x[23] / u_w_lf
        s_rf = 1 - R_w * x[24] / u_w_rf
        s_lr = 1 - R_w * x[25] / u_w_lr
        s_rr = 1 - R_w * x[26] / u_w_rr

    # lateral slip angles
    # switch to kinematic model for small velocities
    if use_kinematic:
        alpha_LF = 0.0
        alpha_RF = 0.0
        alpha_LR = 0.0
        alpha_RR = 0.0
    else:
        alpha_LF = (
            np.arctan(
                (x[10] + lf * x[5] - x[14] * (R_w - x[16])) / (x[3] + 0.5 * T_f * x[5])
            )
            - x[2]
        )
        alpha_RF = (
            np.arctan(
                (x[10] + lf * x[5] - x[14] * (R_w - x[16])) / (x[3] - 0.5 * T_f * x[5])
            )
            - x[2]
        )
        alpha_LR = np.arctan(
            (x[10] - lr * x[5] - x[19] * (R_w - x[21])) / (x[3] + 0.5 * T_r * x[5])
        )
        alpha_RR = np.arctan(
            (x[10] - lr * x[5] - x[19] * (R_w - x[21])) / (x[3] - 0.5 * T_r * x[5])
        )

    # auxiliary suspension movement
    z_SLF = (
        (h_s - R_w + x[16] - x[11]) / np.cos(x[6])
        - h_s
        + R_w
        + lf * x[8]
        + 0.5 * (x[6] - x[13]) * T_f
    )
    z_SRF = (
        (h_s - R_w + x[16] - x[11]) / np.cos(x[6])
        - h_s
        + R_w
        + lf * x[8]
        - 0.5 * (x[6] - x[13]) * T_f
    )
    z_SLR = (
        (h_s - R_w + x[21] - x[11]) / np.cos(x[6])
        - h_s
        + R_w
        - lr * x[8]
        + 0.5 * (x[6] - x[18]) * T_r
    )
    z_SRR = (
        (h_s - R_w + x[21] - x[11]) / np.cos(x[6])
        - h_s
        + R_w
        - lr * x[8]
        - 0.5 * (x[6] - x[18]) * T_r
    )

    dz_SLF = x[17] - x[12] + lf * x[9] + 0.5 * (x[7] - x[14]) * T_f
    dz_SRF = x[17] - x[12] + lf * x[9] - 0.5 * (x[7] - x[14]) * T_f
    dz_SLR = x[22] - x[12] - lr * x[9] + 0.5 * (x[7] - x[19]) * T_r
    dz_SRR = x[22] - x[12] - lr * x[9] - 0.5 * (x[7] - x[19]) * T_r

    # camber angles
    gamma_LF = x[6] + D_f * z_SLF + E_f * (z_SLF) ** 2
    gamma_RF = x[6] - D_f * z_SRF - E_f * (z_SRF) ** 2
    gamma_LR = x[6] + D_r * z_SLR + E_r * (z_SLR) ** 2
    gamma_RR = x[6] - D_r * z_SRR - E_r * (z_SRR) ** 2

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
    delta_z_f = h_s - R_w + x[16] - x[11]
    delta_z_r = h_s - R_w + x[21] - x[11]

    delta_phi_f = x[6] - x[13]
    delta_phi_r = x[6] - x[18]

    dot_delta_phi_f = x[7] - x[14]
    dot_delta_phi_r = x[7] - x[19]

    dot_delta_z_f = x[17] - x[12]
    dot_delta_z_r = x[22] - x[12]

    dot_delta_y_f = x[10] + lf * x[5] - x[15]
    dot_delta_y_r = x[10] - lr * x[5] - x[20]

    delta_f = (
        delta_z_f * np.sin(x[6])
        - x[27] * np.cos(x[6])
        - (h_raf - R_w) * np.sin(delta_phi_f)
    )
    delta_r = (
        delta_z_r * np.sin(x[6])
        - x[28] * np.cos(x[6])
        - (h_rar - R_w) * np.sin(delta_phi_r)
    )

    dot_delta_f = (
        (delta_z_f * np.cos(x[6]) + x[27] * np.sin(x[6])) * x[7]
        + dot_delta_z_f * np.sin(x[6])
        - dot_delta_y_f * np.cos(x[6])
        - (h_raf - R_w) * np.cos(delta_phi_f) * dot_delta_phi_f
    )
    dot_delta_r = (
        (delta_z_r * np.cos(x[6]) + x[28] * np.sin(x[6])) * x[7]
        + dot_delta_z_r * np.sin(x[6])
        - dot_delta_y_r * np.cos(x[6])
        - (h_rar - R_w) * np.cos(delta_phi_r) * dot_delta_phi_r
    )

    # compliant joint forces
    F_RAF = delta_f * K_ras + dot_delta_f * K_rad
    F_RAR = delta_r * K_ras + dot_delta_r * K_rad

    # auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    F_SLF = (
        m_s * g * lr / (2 * (lf + lr))
        - z_SLF * K_sf
        - dz_SLF * K_sdf
        + (x[6] - x[13]) * K_tsf / T_f
    )

    F_SRF = (
        m_s * g * lr / (2 * (lf + lr))
        - z_SRF * K_sf
        - dz_SRF * K_sdf
        - (x[6] - x[13]) * K_tsf / T_f
    )

    F_SLR = (
        m_s * g * lf / (2 * (lf + lr))
        - z_SLR * K_sr
        - dz_SLR * K_sdr
        + (x[6] - x[18]) * K_tsr / T_r
    )

    F_SRR = (
        m_s * g * lf / (2 * (lf + lr))
        - z_SRR * K_sr
        - dz_SRR * K_sdr
        - (x[6] - x[18]) * K_tsr / T_r
    )

    # auxiliary variables sprung mass
    sumX = (
        F_x_LR
        + F_x_RR
        + (F_x_LF + F_x_RF) * np.cos(x[2])
        - (F_y_LF + F_y_RF) * np.sin(x[2])
    )

    sumN = (
        (F_y_LF + F_y_RF) * lf * np.cos(x[2])
        + (F_x_LF + F_x_RF) * lf * np.sin(x[2])
        + (F_y_RF - F_y_LF) * 0.5 * T_f * np.sin(x[2])
        + (F_x_LF - F_x_RF) * 0.5 * T_f * np.cos(x[2])
        + (F_x_LR - F_x_RR) * 0.5 * T_r
        - (F_y_LR + F_y_RR) * lr
    )

    sumY_s = (F_RAF + F_RAR) * np.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR) * np.sin(
        x[6]
    )

    sumL = (
        0.5 * F_SLF * T_f
        + 0.5 * F_SLR * T_r
        - 0.5 * F_SRF * T_f
        - 0.5 * F_SRR * T_r
        - F_RAF
        / np.cos(x[6])
        * (h_s - x[11] - R_w + x[16] - (h_raf - R_w) * np.cos(x[13]))
        - F_RAR
        / np.cos(x[6])
        * (h_s - x[11] - R_w + x[21] - (h_rar - R_w) * np.cos(x[18]))
    )

    sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR) * np.cos(x[6]) - (F_RAF + F_RAR) * np.sin(
        x[6]
    )

    sumM_s = (
        lf * (F_SLF + F_SRF)
        - lr * (F_SLR + F_SRR)
        + (
            (F_x_LF + F_x_RF) * np.cos(x[2])
            - (F_y_LF + F_y_RF) * np.sin(x[2])
            + F_x_LR
            + F_x_RR
        )
        * (h_s - x[11])
    )

    # auxiliary variables unsprung mass
    sumL_uf = (
        0.5 * F_SRF * T_f
        - 0.5 * F_SLF * T_f
        - F_RAF * (h_raf - R_w)
        + F_z_LF * (R_w * np.sin(x[13]) + 0.5 * T_f * np.cos(x[13]) - K_lt * F_y_LF)
        - F_z_RF * (-R_w * np.sin(x[13]) + 0.5 * T_f * np.cos(x[13]) + K_lt * F_y_RF)
        - ((F_y_LF + F_y_RF) * np.cos(x[2]) + (F_x_LF + F_x_RF) * np.sin(x[2]))
        * (R_w - x[16])
    )

    sumL_ur = (
        0.5 * F_SRR * T_r
        - 0.5 * F_SLR * T_r
        - F_RAR * (h_rar - R_w)
        + F_z_LR * (R_w * np.sin(x[18]) + 0.5 * T_r * np.cos(x[18]) - K_lt * F_y_LR)
        - F_z_RR * (-R_w * np.sin(x[18]) + 0.5 * T_r * np.cos(x[18]) + K_lt * F_y_RR)
        - (F_y_LR + F_y_RR) * (R_w - x[21])
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
    f = np.zeros((29,))  # init 'right hand side'
    # switch to kinematic model for small velocities
    if use_kinematic:
        # wheelbase
        # lwb = lf + lr

        # system dynamics
        # x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        # f_ks = vehicle_dynamics_ks(x_ks, u, p)
        # f.extend(f_ks)
        # f.append(u[1]*lwb*np.tan(x[2]) + x[3]/(lwb*np.cos(x[2])**2)*u[0])

        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        # kinematic model
        # f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        # f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]] # 1 2 3 4 5
        f_ks = vehicle_dynamics_ks(x_ks, u, params)
        f[0:5] = f_ks
        # derivative of slip angle and yaw rate
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
        # f.append(dd_psi) # 6
        f[5] = dd_psi

    else:
        f[0] = np.cos(beta + x[4]) * vel  # 1
        f[1] = np.sin(beta + x[4]) * vel  # 2
        f[2] = u[0]  # 3
        f[3] = 1 / m * sumX + x[5] * x[10]  # 4
        f[4] = x[5]  # 5
        f[5] = (
            1 / (I_z - (I_xz_s) ** 2 / I_Phi_s) * (sumN + I_xz_s / I_Phi_s * sumL)
        )  # 6

    # remaining sprung mass dynamics
    f[6] = x[7]  # 7
    f[7] = 1 / (I_Phi_s - (I_xz_s) ** 2 / I_z) * (I_xz_s / I_z * sumN + sumL)  # 8
    f[8] = x[9]  # 9
    f[9] = 1 / I_y_s * sumM_s  # 10
    f[10] = 1 / m_s * sumY_s - x[5] * x[3]  # 11
    f[11] = x[12]  # 12
    f[12] = g - 1 / m_s * sumZ_s  # 13

    # unsprung mass dynamics (front)
    f[13] = x[14]  # 14
    f[14] = 1 / I_uf * sumL_uf  # 15
    f[15] = 1 / m_uf * sumY_uf - x[5] * x[3]  # 16
    f[16] = x[17]  # 17
    f[17] = g - 1 / m_uf * sumZ_uf  # 18

    # unsprung mass dynamics (rear)
    f[18] = x[19]  # 19
    f[19] = 1 / I_ur * sumL_ur  # 20
    f[20] = 1 / m_ur * sumY_ur - x[5] * x[3]  # 21
    f[21] = x[22]  # 22
    f[22] = g - 1 / m_ur * sumZ_ur  # 23

    # convert acceleration input to brake and engine torque - splitting for brake/drive torque is outside of the integration
    if u[0] > 0:
        T_B = 0.0
        T_E = m * R_w * u[0]
    else:
        T_B = m * R_w * u[0]
        T_E = 0.0

    # wheel dynamics (p.T  new parameter for torque splitting) - splitting for brake/drive torque is outside of the integration
    f[23] = 1 / I_y_w * (-R_w * F_x_LF + 0.5 * T_sb * T_B + 0.5 * T_se * T_E)  # 24
    f[24] = 1 / I_y_w * (-R_w * F_x_RF + 0.5 * T_sb * T_B + 0.5 * T_se * T_E)  # 25
    f[25] = (
        1 / I_y_w * (-R_w * F_x_LR + 0.5 * (1 - T_sb) * T_B + 0.5 * (1 - T_se) * T_E)
    )  # 26
    f[26] = (
        1 / I_y_w * (-R_w * F_x_RR + 0.5 * (1 - T_sb) * T_B + 0.5 * (1 - T_se) * T_E)
    )  # 27

    # negative wheel spin forbidden handeled outside of this function
    for iState in range(23, 27):
        if x[iState] < 0.0:
            x[iState] = 0.0
            f[iState] = 0.0

    # compliant joint equations
    f[27] = dot_delta_y_f  # 28
    f[28] = dot_delta_y_r  # 29

    # longitudinal slip
    # s_lf
    # s_rf
    # s_lr
    # s_rr

    # lateral slip
    # alpha_LF
    # alpha_RF
    # alpha_LR
    # alpha_RR

    return f
