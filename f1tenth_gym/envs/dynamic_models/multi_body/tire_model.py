import numpy as np
from numba import njit


# longitudinal tire forces
@njit(cache=True)
def formula_longitudinal(kappa, gamma, F_z, params):
    # longitudinal coefficients
    # tire_p_cx1 = params["tire_p_cx1"]  # Shape factor Cfx for longitudinal force
    # tire_p_dx1 = params["tire_p_dx1"]  # Longitudinal friction Mux at Fznom
    # tire_p_dx3 = params["tire_p_dx3"]  # Variation of friction Mux with camber
    # tire_p_ex1 = params["tire_p_ex1"]  # Longitudinal curvature Efx at Fznom
    # tire_p_kx1 = params["tire_p_kx1"]  # Longitudinal slip stiffness Kfx/Fz at Fznom
    # tire_p_hx1 = params["tire_p_hx1"]  # Horizontal shift Shx at Fznom
    # tire_p_vx1 = params["tire_p_vx1"]  # Vertical shift Svx/Fz at Fznom
    tire_p_cx1 = params[55]  # shape factor Cfx
    tire_p_dx1 = params[56]  # peak longitudinal friction
    tire_p_dx3 = params[57]  # friction variation with camber
    tire_p_ex1 = params[58]  # curvature factor Efx
    tire_p_kx1 = params[59]  # slip stiffness Kfx/Fz
    tire_p_hx1 = params[60]  # horizontal shift Shx
    tire_p_vx1 = params[61]  # vertical shift Svx/Fz

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    kappa = -kappa

    S_hx = tire_p_hx1
    S_vx = F_z * tire_p_vx1

    kappa_x = kappa + S_hx
    mu_x = tire_p_dx1 * (1 - tire_p_dx3 * gamma**2)

    C_x = tire_p_cx1
    D_x = mu_x * F_z
    E_x = tire_p_ex1
    K_x = F_z * tire_p_kx1
    B_x = K_x / (C_x * D_x)

    # magic tire formula
    return D_x * np.sin(
        C_x
        * np.arctan(B_x * kappa_x - E_x * (B_x * kappa_x - np.arctan(B_x * kappa_x)))
        + S_vx
    )


# lateral tire forces
@njit(cache=True)
def formula_lateral(alpha, gamma, F_z, params):
    # lateral coefficients
    # tire_p_cy1 = params["tire_p_cy1"]  # Shape factor Cfy for lateral forces
    # tire_p_dy1 = params["tire_p_dy1"]  # Lateral friction Muy
    # tire_p_dy3 = params["tire_p_dy3"]  # Variation of friction Muy with squared camber
    # tire_p_ey1 = params["tire_p_ey1"]  # Lateral curvature Efy at Fznom
    # tire_p_ky1 = params["tire_p_ky1"]  # Maximum value of stiffness Kfy/Fznom
    # tire_p_hy1 = params["tire_p_hy1"]  # Horizontal shift Shy at Fznom
    # tire_p_hy3 = params["tire_p_hy3"]  # Variation of shift Shy with camber
    # tire_p_vy1 = params["tire_p_vy1"]  # Vertical shift in Svy/Fz at Fznom
    # tire_p_vy3 = params["tire_p_vy3"]  # Variation of shift Svy/Fz with camber
    tire_p_cy1 = params[67]  # shape factor Cfy
    tire_p_dy1 = params[68]  # lateral friction Muy
    tire_p_dy3 = params[69]  # friction variation with camber²
    tire_p_ey1 = params[70]  # curvature factor Efy
    tire_p_ky1 = params[71]  # stiffness Kfy/Fz
    tire_p_hy1 = params[72]  # horizontal shift Shy
    tire_p_hy3 = params[73]  # shift variation with camber
    tire_p_vy1 = params[74]  # vertical shift Svy/Fz
    tire_p_vy3 = params[75]  # Svy/Fz variation with camber

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    # alpha = -alpha

    S_hy = np.sign(gamma) * (tire_p_hy1 + tire_p_hy3 * np.fabs(gamma))
    S_vy = np.sign(gamma) * F_z * (tire_p_vy1 + tire_p_vy3 * np.fabs(gamma))

    alpha_y = alpha + S_hy
    mu_y = tire_p_dy1 * (1 - tire_p_dy3 * gamma**2)

    C_y = tire_p_cy1
    D_y = mu_y * F_z
    E_y = tire_p_ey1
    K_y = F_z * tire_p_ky1  # simplify K_y0 to tire_p_ky1*F_z
    B_y = K_y / (C_y * D_y)

    # magic tire formula
    F_y = (
        D_y
        * np.sin(
            C_y
            * np.arctan(
                B_y * alpha_y - E_y * (B_y * alpha_y - np.arctan(B_y * alpha_y))
            )
        )
        + S_vy
    )

    res = []
    res.append(F_y)
    res.append(mu_y)
    return res


# longitudinal tire forces for combined slip
@njit(cache=True)
def formula_longitudinal_comb(kappa, alpha, F0_x, params):
    # longitudinal coefficients
    # tire_r_bx1 = params["tire_r_bx1"]  # Slope factor for combined slip Fx reduction
    # tire_r_bx2 = params["tire_r_bx2"]  # Variation of slope Fx reduction with kappa
    # tire_r_cx1 = params["tire_r_cx1"]  # Shape factor for combined slip Fx reduction
    # tire_r_ex1 = params["tire_r_ex1"]  # Curvature factor of combined Fx
    # tire_r_hx1 = params["tire_r_hx1"]  # Shift factor for combined slip Fx reduction
    tire_r_bx1 = params[62]  # slope for Fx reduction
    tire_r_bx2 = params[63]  # Fx slope variation w/ kappa
    tire_r_cx1 = params[64]  # shape factor for Fx reduction
    tire_r_ex1 = params[65]  # curvature for Fx reduction
    tire_r_hx1 = params[66]  # shift for Fx reduction

    # turn slip '' neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hxalpha = tire_r_hx1

    alpha_s = alpha + S_hxalpha

    B_xalpha = tire_r_bx1 * np.cos(np.arctan(tire_r_bx2 * kappa))
    C_xalpha = tire_r_cx1
    E_xalpha = tire_r_ex1
    D_xalpha = F0_x / (
        np.cos(
            C_xalpha
            * np.arctan(
                B_xalpha * S_hxalpha
                - E_xalpha * (B_xalpha * S_hxalpha - np.arctan(B_xalpha * S_hxalpha))
            )
        )
    )

    # magic tire formula
    return D_xalpha * np.cos(
        C_xalpha
        * np.arctan(
            B_xalpha * alpha_s
            - E_xalpha * (B_xalpha * alpha_s - np.arctan(B_xalpha * alpha_s))
        )
    )


# lateral tire forces for combined slip
@njit(cache=True)
def formula_lateral_comb(kappa, alpha, gamma, mu_y, F_z, F0_y, params):
    # lateral coefficients
    # tire_r_by1 = params["tire_r_by1"]  # Slope factor for combined Fy reduction
    # tire_r_by2 = params["tire_r_by2"]  # Variation of slope Fy reduction with alpha
    # tire_r_by3 = params["tire_r_by3"]  # Shift term for alpha in slope Fy reduction
    # tire_r_cy1 = params["tire_r_cy1"]  # Shape factor for combined Fy reduction
    # tire_r_ey1 = params["tire_r_ey1"]  # Curvature factor of combined Fy
    # tire_r_hy1 = params["tire_r_hy1"]  # Shift factor for combined Fy reduction
    # tire_r_vy1 = params["tire_r_vy1"]  # Kappa induced side force Svyk/Muy*Fz at Fznom
    # tire_r_vy3 = params["tire_r_vy3"]  # Variation of Svyk/Muy*Fz with camber
    # tire_r_vy4 = params["tire_r_vy4"]  # Variation of Svyk/Muy*Fz with alpha
    # tire_r_vy5 = params["tire_r_vy5"]  # Variation of Svyk/Muy*Fz with kappa
    # tire_r_vy6 = params["tire_r_vy6"]  # Variation of Svyk/Muy*Fz with atan(kappa)
    tire_r_by1 = params[76]  # slope for Fy reduction
    tire_r_by2 = params[77]  # slope variation with alpha
    tire_r_by3 = params[78]  # alpha shift in Fy slope
    tire_r_cy1 = params[79]  # shape factor for Fy reduction
    tire_r_ey1 = params[80]  # curvature for Fy reduction
    tire_r_hy1 = params[81]  # shift for Fy reduction
    tire_r_vy1 = params[82]  # kappa-induced side force
    tire_r_vy3 = params[83]  # side force variation w/ camber
    tire_r_vy4 = params[84]  # variation w/ alpha
    tire_r_vy5 = params[85]  # variation w/ kappa
    tire_r_vy6 = params[86]  # variation w/ atan(kappa)

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hykappa = tire_r_hy1

    kappa_s = kappa + S_hykappa

    B_ykappa = tire_r_by1 * np.cos(np.arctan(tire_r_by2 * (alpha - tire_r_by3)))
    C_ykappa = tire_r_cy1
    E_ykappa = tire_r_ey1
    D_ykappa = F0_y / (
        np.cos(
            C_ykappa
            * np.arctan(
                B_ykappa * S_hykappa
                - E_ykappa * (B_ykappa * S_hykappa - np.arctan(B_ykappa * S_hykappa))
            )
        )
    )

    D_vykappa = (
        mu_y
        * F_z
        * (tire_r_vy1 + tire_r_vy3 * gamma)
        * np.cos(np.arctan(tire_r_vy4 * alpha))
    )
    S_vykappa = D_vykappa * np.sin(tire_r_vy5 * np.arctan(tire_r_vy6 * kappa))

    # magic tire formula
    return (
        D_ykappa
        * np.cos(
            C_ykappa
            * np.arctan(
                B_ykappa * kappa_s
                - E_ykappa * (B_ykappa * kappa_s - np.arctan(B_ykappa * kappa_s))
            )
        )
        + S_vykappa
    )
