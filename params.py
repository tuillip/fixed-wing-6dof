"""
params.py -- Aerosonde UAV constants.

Source: aerosonde.m in byu-magicc/mavsim_public (the repo values, not the
printed book -- the book has errata). Also Beard & McLain Appendix E.2.

Transcription only. No physics here.
"""

import numpy as np


def aerosonde_params():
    """Returns: dict of every constant the model needs, Gammas included."""
    prm = {
        # --- physical ---
        "mass": 11.0,
        "Jx":   0.824,
        "Jy":   1.135,
        "Jz":   1.759,
        "Jxz":  0.120,
        "gravity": 9.81,

        # --- geometry ---
        "S_wing": 0.55,        # wing area            [m^2]
        "b":      2.90,        # wingspan             [m]
        "c":      0.19,        # mean chord           [m]
        "e":      0.9,         # Oswald efficiency    [-]
        "rho":    1.2682,      # air density          [kg/m^3]

        # --- longitudinal aero ---
        "C_L_0":       0.23,
        "C_L_alpha":   5.61,
        "C_L_q":       7.95,
        "C_L_delta_e": 0.13,
        "C_D_0":       0.043,
        "C_D_alpha":   0.030,
        "C_D_q":       0.0,
        "C_D_delta_e": 0.0135,
        "C_D_p":       0.0,
        "C_m_0":       0.0135,
        "C_m_alpha":   -2.74,
        "C_m_q":       -38.21,
        "C_m_delta_e": -0.99,

        # --- stall blending (Ch. 4 lift model) ---
        "M":      50,        # transition rate
        "alpha0": 0.47,        # stall angle          [rad]

        # --- lateral aero ---
        "C_Y_0":         0.0,
        "C_Y_beta":      -0.83,
        "C_Y_p":         0.0,
        "C_Y_r":         0.0,
        "C_Y_delta_a":   0.075,
        "C_Y_delta_r":   0.19,
        "C_ell_0":       0.0,
        "C_ell_beta":    -0.13,
        "C_ell_p":       -0.51,
        "C_ell_r":       0.045,
        "C_ell_delta_a": 0.17,
        "C_ell_delta_r": 0.0024,
        "C_n_0":         0.0,
        "C_n_beta":      0.073,
        "C_n_p":         -0.069,
        "C_n_r":         -0.095,
        "C_n_delta_a":   -0.011,
        "C_n_delta_r":   -0.069,

        # --- propulsion ---
        "D_prop":        20*(0.0254),               # prop diameter [m]
        "K_V":           145,                       # [RPM/V]
        "R_motor":       0.042,                     # [ohms]
        "i0":            1.5,                       # no-load (zero-torque) current [A]
        "ncells":        12,
        "C_Q0":          0.005230,
        "C_Q1":          0.004970,
        "C_Q2":          -0.01664,
        "C_T0":          0.09357,
        "C_T1":          -0.06044,
        "C_T2":          -0.1079
    }

    prm["KQ"] = 1/prm["K_V"]*60/(2*np.pi) # [N-m/A]
    prm["V_max"] = 3.7*prm["ncells"]         # [V]
    prm["AR"]    = prm["b"]**2 / prm["S_wing"]
    
    return add_gammas(prm)

def add_gammas(prm):
    Jx = prm["Jx"]
    Jy = prm["Jy"]
    Jz = prm["Jz"]
    Jxz = prm["Jxz"]

    Gamma = Jx * Jz - Jxz**2

    prm["Gamma"] = Gamma
    prm["Gamma1"] = Jxz * (Jx - Jy + Jz) / Gamma
    prm["Gamma2"] = (Jz * (Jz - Jy) + Jxz**2) / Gamma
    prm["Gamma3"] = Jz / Gamma
    prm["Gamma4"] = Jxz / Gamma
    prm["Gamma5"] = (Jz - Jx) / Jy
    prm["Gamma6"] = Jxz / Jy
    prm["Gamma7"] = (Jx * (Jx - Jy) + Jxz**2) / Gamma
    prm["Gamma8"] = Jx / Gamma
    return prm

def check_complete(prm):
    """Raises if any constant is still None. Catches a missed transcription."""
    missing = sorted(k for k, val in prm.items() if val is None)
    if missing:
        raise ValueError(f"{len(missing)} parameters still unset: {missing}")
    return True


if __name__ == "__main__":
    prm = aerosonde_params()
    check_complete(prm)
    print(f"{len(prm)} parameters, all set")
