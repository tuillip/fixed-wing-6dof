"""
trim.py -- find the state and control inputs for steady flight.

Reference: Beard & McLain Ch. 5 and Appendix F.2.

Trim is the equilibrium of the whole model. Given a target airspeed and
flight path angle, solve for the inputs that hold the aircraft there:
every velocity and rate derivative zero, only the position rates nonzero.

This is also the real test of aerodynamics.py. An aero model with a sign
error will not trim -- the solver either fails to converge or lands on
something absurd (negative throttle, 40 degrees of elevator).

Unknowns, fixed order everywhere in this file:
    alpha, beta, delta_e, delta_a, delta_r, delta_t

Six unknowns, six equations: u_dot, v_dot, w_dot, p_dot, q_dot, r_dot.
(delta_a and delta_r are not redundant -- they counter propeller torque.)
"""

import numpy as np
from scipy.optimize import fsolve

from params import aerosonde_params
from aerodynamics import forces_moments
from dynamics import derivatives
from frames import body_velocity_from_airdata

# ----------------------------------------------------------------------

def trim_state(Va, alpha, beta, gamma, altitude=100.0):
    """
    Build the 12-element state vector for steady straight flight.

    Args:
        Va        airspeed                           [m/s]
        alpha     angle of attack                    [rad]
        beta      sideslip angle                     [rad]
        gamma     flight path angle, + is climbing   [rad]
        altitude  height above the reference         [m]

    Returns:
        (12,) array in the order
            pn, pe, pd, u, v, w, phi, theta, psi, p, q, r
        Callers name this x_trim.

    Wings level (phi = 0), heading north (psi = 0), not rotating
    (p = q = r = 0). Pitch attitude is alpha + gamma.

    altitude is cosmetic here: rho is a fixed constant in params.py, so
    nothing in the dynamics depends on pd. It would only matter with an
    atmosphere model. 100 m is just a plausible number.
    """ 
    pn = 0
    pe = 0
    pd = -altitude
    u, v, w = body_velocity_from_airdata(Va, alpha, beta)
    phi = 0
    theta = alpha + gamma
    psi = 0
    p = q = r = 0.0
    return np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r])


def trim_residual(z, Va, gamma, prm):
    alpha, beta, delta_e, delta_a, delta_r, delta_t = z
    x = trim_state(Va, alpha, beta, gamma)

    f, m = forces_moments(x, (delta_e, delta_a, delta_r, delta_t), prm)
    xdot = derivatives(x, f, m, prm)

    return np.concatenate([xdot[3:6], xdot[9:12]])
    

def trim(Va, gamma, prm, z0=None):
    if z0 is None:
        z0 = [0.05, 0.0, -0.1, 0.0, 0.0, 0.5]
    z, infodict, ier, mesg = fsolve(trim_residual, z0, args=(Va, gamma, prm), full_output=True)

    if (ier != 1):
        raise RuntimeError(f"trim did not converge at Va={Va}, gamma={gamma}: {mesg}")
    
    alpha, beta, delta_e, delta_a, delta_r, delta_t = z
    
    x_trim = trim_state(Va, alpha, beta, gamma)
    deltas = (delta_e, delta_a, delta_r, delta_t)
    return x_trim, deltas
    

# ======================================================================
# Tests
# ======================================================================

def test_trim_residual_is_zero():
    # The definition of trim. Everything but the position rates must vanish.
    prm = aerosonde_params()
    x, d = trim(35.0, 0.0, prm)
    f, m = forces_moments(x, d, prm)
    xdot = derivatives(x, f, m, prm)
    assert np.allclose(xdot[3:12], 0.0, atol=1e-8), f"not trimmed: {xdot[3:12]}"
    print("trimmed state has zero derivatives        OK")


def test_climb_rate_matches_gamma():
    # Climbing at gamma means descending at -Va*sin(gamma) in NED.
    prm = aerosonde_params()
    Va = 35.0
    for gamma_deg in (0.0, 5.0, -3.0):
        gamma = np.deg2rad(gamma_deg)
        x, d = trim(Va, gamma, prm)
        f, m = forces_moments(x, d, prm)
        xdot = derivatives(x, f, m, prm)
        assert np.isclose(xdot[2], -Va*np.sin(gamma), atol=1e-6)
    print("climb rate matches flight path angle      OK")


def test_slower_flight_needs_more_alpha():
    # Lift must equal weight. Less speed means less dynamic pressure,
    # so the wing has to work at a higher angle of attack.
    prm = aerosonde_params()
    x_fast, _ = trim(35.0, 0.0, prm)
    x_slow, _ = trim(25.0, 0.0, prm)
    alpha_fast = np.arctan2(x_fast[5], x_fast[3])
    alpha_slow = np.arctan2(x_slow[5], x_slow[3])
    assert alpha_slow > alpha_fast, f"{alpha_slow} !> {alpha_fast}"
    print("slower flight needs more alpha            OK")


def test_climbing_needs_more_throttle():
    # Climbing costs power. Same speed, steeper climb, more throttle.
    prm = aerosonde_params()
    _, d_level = trim(35.0, 0.0, prm)
    _, d_climb = trim(35.0, np.deg2rad(5.0), prm)
    assert d_climb[3] > d_level[3], f"{d_climb[3]} !> {d_level[3]}"
    print("climbing needs more throttle              OK")


def test_throttle_and_elevator_are_sane():
    # A physically absurd trim means a sign error upstream in the aero model.
    prm = aerosonde_params()
    x, d = trim(35.0, 0.0, prm)
    delta_e, delta_a, delta_r, delta_t = d
    assert 0.0 < delta_t < 1.0, f"throttle {delta_t} out of range"
    assert abs(np.rad2deg(delta_e)) < 25.0, f"elevator {np.rad2deg(delta_e)} deg"
    print("trim inputs are physically reasonable     OK")


if __name__ == "__main__":
    for test in (test_trim_residual_is_zero,
                 test_climb_rate_matches_gamma,
                 test_slower_flight_needs_more_alpha,
                 test_climbing_needs_more_throttle,
                 test_throttle_and_elevator_are_sane):
        try:
            test()
        except NotImplementedError as e:
            print(f"SKIP  {e}")
