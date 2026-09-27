"""
aerodynamics.py -- forces and moments acting on the aircraft.

Reference: Beard & McLain Ch. 4.

Three sources summed into one body-frame force vector and one body-frame
moment vector: gravity, aerodynamics, propulsion. The output plugs straight
into derivatives() in dynamics.py -- nothing in that file changes.

Control input order, fixed everywhere in this file:
    delta_e, delta_a, delta_r, delta_t     elevator, aileron, rudder, throttle

NAMING NOTE: prm["M"] is the stall blending rate, NOT a moment. The pitching
moment is called m in the book and collides with mass. Pick names here that
you can still read in three weeks.
"""

import numpy as np
from frames import vehicle_to_body, airdata_from_body_velocity
from params import aerosonde_params


# ----------------------------------------------------------------------
# Gravity
# ----------------------------------------------------------------------

def gravity_body(x, prm):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    # return prm["mass"]*prm["gravity"]*np.array([-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)])
    return vehicle_to_body(phi, theta, psi) @ np.array([0, 0, prm["mass"]*prm["gravity"]])

# ----------------------------------------------------------------------
# Aerodynamic coefficients
# ----------------------------------------------------------------------

def sigma_blend(alpha, prm):
    neg_exp = np.exp(-prm["M"]*(alpha - prm["alpha0"]))
    pos_exp = np.exp(prm["M"]*(alpha + prm["alpha0"]))
    return (1+ neg_exp + pos_exp)/( (1 + neg_exp)*(1 + pos_exp) )

def lift_coefficient(alpha, prm):
    return (1-sigma_blend(alpha, prm))*(prm["C_L_0"] + prm["C_L_alpha"]*alpha)+sigma_blend(alpha, prm)*(2*np.sign(alpha)*(np.sin(alpha))**2*np.cos(alpha))

def drag_coefficient(alpha, prm):
    return prm["C_D_p"] + (prm["C_L_0"] + prm["C_L_alpha"]*alpha)**2/(np.pi*prm["e"]*prm["AR"])
    
# ----------------------------------------------------------------------
# Aerodynamic forces and moments
# ----------------------------------------------------------------------

def aerodynamic_forces_moments(Va, alpha, beta, p, q, r, delta_e, delta_a, delta_r, prm):
    
    rho, S, b, c = prm["rho"], prm["S_wing"], prm["b"], prm["c"]
    qbar = 0.5 * rho * Va**2 * S
    c2V  = c / (2 * Va)      # nondimensionalizes pitch rate
    b2V  = b / (2 * Va)      # nondimensionalizes roll and yaw rates

    # --- longitudinal ---
    C_L = lift_coefficient(alpha, prm) + prm["C_L_q"]*c2V*q + prm["C_L_delta_e"]*delta_e
    C_D = drag_coefficient(alpha, prm) + prm["C_D_q"]*c2V*q + prm["C_D_delta_e"]*delta_e

    ca, sa = np.cos(alpha), np.sin(alpha)
    
    fx = qbar * (-C_D*ca + C_L*sa)
    fz = qbar * (-C_D*sa - C_L*ca)
    fy = qbar * (prm["C_Y_0"] + prm["C_Y_beta"]*beta + prm["C_Y_p"]*b2V*p + prm["C_Y_r"]*b2V*r + prm["C_Y_delta_a"]*delta_a + prm["C_Y_delta_r"]*delta_r)
    Mx = qbar * b * (prm["C_ell_0"] + prm["C_ell_beta"]*beta + prm["C_ell_p"]*b2V*p + prm["C_ell_r"]*b2V*r + prm["C_ell_delta_a"]*delta_a + prm["C_ell_delta_r"]*delta_r)
    Mz = qbar * b * (prm["C_n_0"] + prm["C_n_beta"]*beta + prm["C_n_p"]*b2V*p + prm["C_n_r"]*b2V*r + prm["C_n_delta_a"]*delta_a + prm["C_n_delta_r"]*delta_r)
    My = qbar * c * (prm["C_m_0"] + prm["C_m_alpha"]*alpha + prm["C_m_q"]*c2V*q + prm["C_m_delta_e"]*delta_e)
    
    return np.array([fx, fy, fz]), np.array([Mx, My, Mz])

# ----------------------------------------------------------------------
# Propulsion
# ----------------------------------------------------------------------

def propeller_thrust_torque(Va, delta_t, prm):
    V_in = prm["V_max"]*delta_t
    a = prm["rho"]*prm["D_prop"]**5/((2*np.pi)**2)*prm["C_Q0"]
    b = prm["rho"]*prm["D_prop"]**4/(2*np.pi)*prm["C_Q1"]*Va + prm["KQ"]**2/prm["R_motor"]
    c = prm["rho"]*prm["D_prop"]**3*prm["C_Q2"]*Va**2 - prm["KQ"]/prm["R_motor"]*V_in + prm["KQ"]*prm["i0"]
    Omega_p = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    J = 2*np.pi*Va/(Omega_p * prm["D_prop"])
    T_P = prm["rho"]*prm["D_prop"]**4/(4*np.pi**2)*Omega_p**2*(prm["C_T2"]*J**2 + prm["C_T1"]*J + prm["C_T0"])
    Q_P = prm["rho"]*prm["D_prop"]**5/(4*np.pi**2)*Omega_p**2*(prm["C_Q2"]*J**2 + prm["C_Q1"]*J + prm["C_Q0"])
    return T_P, Q_P
    
# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def forces_moments(x, deltas, prm):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    delta_e, delta_a, delta_r, delta_t = deltas
    Va, alpha, beta = airdata_from_body_velocity(u, v, w)

    f_g = gravity_body(x, prm)
    f_a, m_a = aerodynamic_forces_moments(Va, alpha, beta, p, q, r, delta_e, delta_a, delta_r, prm)
    T_P, Q_P = propeller_thrust_torque(Va, delta_t, prm)

    return f_g + f_a + np.array([T_P, 0.0, 0.0]), m_a + np.array([-Q_P, 0.0, 0.0])

# ======================================================================
# Tests
# ======================================================================

def _state(pn=0, pe=0, pd=0, u=0, v=0, w=0, phi=0, theta=0, psi=0, p=0, q=0, r=0):
    return np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r], dtype=float)


def test_gravity_orientations():
    # Level: all weight on body z. Pitched 90 deg nose up: all on -body x.
    # Rolled 90 deg right: all on +body y.
    prm = aerosonde_params()
    W = prm["mass"] * prm["gravity"]
    assert np.allclose(gravity_body(_state(), prm), [0, 0, W])
    assert np.allclose(gravity_body(_state(theta=np.pi/2), prm), [-W, 0, 0], atol=1e-9)
    assert np.allclose(gravity_body(_state(phi=np.pi/2), prm), [0, W, 0], atol=1e-9)
    print("gravity rotates correctly                 OK")


def test_lift_below_stall():
    # Well below stall the blend is off, so C_L(0) is just C_L_0,
    # and lift still increases with alpha.
    prm = aerosonde_params()
    assert np.isclose(lift_coefficient(0.0, prm), prm["C_L_0"])
    assert lift_coefficient(0.05, prm) > lift_coefficient(0.0, prm)
    print("lift linear below stall                   OK")


def test_drag_positive():
    # Drag is never negative, and induced drag makes it rise with |alpha|.
    prm = aerosonde_params()
    assert drag_coefficient(0.0, prm) > 0
    assert drag_coefficient(0.2, prm) > drag_coefficient(0.0, prm)
    print("drag positive and rising with alpha       OK")


def test_lateral_symmetry():
    # Symmetric flight, no deflections, no rates: side force, roll and yaw
    # moments must all be exactly zero. Catches a stray sign or term.
    prm = aerosonde_params()
    f, m = aerodynamic_forces_moments(35.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, prm)
    assert np.isclose(f[1], 0.0)
    assert np.isclose(m[0], 0.0)
    assert np.isclose(m[2], 0.0)
    print("lateral terms vanish in symmetric flight  OK")


def test_alpha_zero_is_pure_drag():
    # At alpha = 0 the stability frame IS the body frame, so body-x force
    # must be exactly -drag. Catches a bad lift/drag rotation.
    prm = aerosonde_params()
    Va = 35.0
    f, _ = aerodynamic_forces_moments(Va, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, prm)
    qbar = 0.5 * prm["rho"] * Va**2 * prm["S_wing"]
    assert np.isclose(f[0], -qbar * drag_coefficient(0.0, prm))
    print("alpha=0 gives fx = -drag                  OK")


def test_thrust_falls_with_airspeed():
    # Full throttle: static thrust is largest, and thrust drops as the
    # aircraft speeds up. Torque is positive.
    prm = aerosonde_params()
    T_static, Q_static = propeller_thrust_torque(0.0, 1.0, prm)
    T_fast, _ = propeller_thrust_torque(35.0, 1.0, prm)
    assert T_static > T_fast > 0
    assert Q_static > 0
    print("thrust decreases with airspeed            OK")


if __name__ == "__main__":
    for test in (test_gravity_orientations,
                 test_lift_below_stall,
                 test_drag_positive,
                 test_lateral_symmetry,
                 test_alpha_zero_is_pure_drag,
                 test_thrust_falls_with_airspeed):
        try:
            test()
        except NotImplementedError as e:
            print(f"SKIP  {e}")
