"""
linearize.py -- linear models extracted from the nonlinear plant.

Reference: Beard & McLain Ch. 5; Appendix F.4.

Near a trim point the nonlinear system is well approximated by its tangent:

    delta_xdot = A delta_x + B delta_u

A and B are Jacobians evaluated at trim. They are computed numerically by
perturbing one state (or input) at a time and watching all twelve
derivatives respond -- no symbolic differentiation anywhere.

The eigenvalues of A are the aircraft's modes. That is the payoff of the
whole project: if they land where flight dynamics says they should, every
layer underneath is correct.

State order (from dynamics.py):
    0  1  2   3  4  5   6    7      8    9  10 11
    pn pe pd  u  v  w   phi  theta  psi  p  q  r

Input order (from aerodynamics.py):
    0        1        2        3
    delta_e  delta_a  delta_r  delta_t
"""

import numpy as np

from params import aerosonde_params
from aerodynamics import forces_moments
from dynamics import derivatives
from trim import trim


# ----------------------------------------------------------------------
# Which states and inputs belong to which decoupled set.
# Pure bookkeeping -- read off the state order above.
# ----------------------------------------------------------------------

LON_STATES = [3, 5, 10, 7, 2]    # u, w, q, theta, pd
LAT_STATES = [4, 9, 11, 6, 8]    # v, p, r, phi, psi
LON_INPUTS = [0, 3]              # delta_e, delta_t
LAT_INPUTS = [1, 2]              # delta_a, delta_r


# ----------------------------------------------------------------------

def state_derivative(x, deltas, prm):
    f, m = forces_moments(x, deltas, prm)
    return derivatives(x, f, m, prm)


def linearize(x_trim, deltas_trim, prm, eps=1e-6):
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))
    for i in range(12):
        dx = np.zeros(12)
        dx[i] = eps
        A[:, i] = (state_derivative(x_trim + dx, deltas_trim, prm) - state_derivative(x_trim - dx, deltas_trim, prm)) / (2 * eps)

    for j in range(4):
        dd = np.zeros(4)
        dd[j] = eps
        B[:, j] = (state_derivative(x_trim, deltas_trim + dd, prm) - state_derivative(x_trim, deltas_trim - dd, prm)) / (2 * eps)

    return A, B


def longitudinal(A, B):
    A_lon = A[np.ix_(LON_STATES, LON_STATES)]
    B_lon = B[np.ix_(LON_STATES, LON_INPUTS)]

    return (A_lon, B_lon)


def lateral(A, B):
    A_lat = A[np.ix_(LAT_STATES, LAT_STATES)]
    B_lat = B[np.ix_(LAT_STATES, LAT_INPUTS)]

    return (A_lat, B_lat)


def mode_properties(eigenvalue):
    wn = abs(eigenvalue)
    if abs(eigenvalue) < 1e-9:
        eigen_props = {
            "kind":         "integrator"
        }    
    elif abs(eigenvalue.imag) < 1e-9:
        eigen_props = {
            "kind":         "real",
            "tau":          -1/eigenvalue.real
        }
    else:
        eigen_props = {
            "kind":         "oscillatory",
            "wn":           wn,
            "zeta":         -eigenvalue.real/wn,
            "period":       2*np.pi/abs(eigenvalue.imag)
    }

    return eigen_props


def print_modes(A, label = ""):
    print(label)
    eigenvalues = np.linalg.eigvals(A)
    sorted_eigs = sorted(eigenvalues, key=abs, reverse=True)
    for e in sorted_eigs:
        dict_e = mode_properties(e)
        if dict_e["kind"] == "oscillatory":
            print(f"{e:.4f} is an oscillatory eigenvalue with natural frequency {dict_e['wn']:.4f} [rad/s], damping ratio {dict_e['zeta']:.4f}, and period {dict_e['period']:.4f} [s].")
        elif dict_e["kind"] == "real":
            print(f"{e.real:.4f} is a real eigenvalue with time constant "f"{dict_e['tau']:.4f} [s].")
        else:
            print(f"{e.real:.4f} is a pure integrator eigenvalue.")
            

# ======================================================================
# Tests
# ======================================================================

def test_jacobian_shapes():
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    A, B = linearize(x0, d0, prm)
    assert A.shape == (12, 12), A.shape
    assert B.shape == (12, 4), B.shape
    print("A is 12x12 and B is 12x4                  OK")


def test_trim_point_is_equilibrium():
    # Linearizing somewhere that is not an equilibrium is meaningless.
    # Everything but the position rates must already be zero.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    assert np.allclose(state_derivative(x0, d0, prm)[3:12], 0.0, atol=1e-8)
    print("linearizing about a true equilibrium      OK")


def test_four_pure_integrators():
    # pn, pe, pd and psi accumulate but never feed back -- nothing in the
    # dynamics depends on where you are or which way you point. Four zeros.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    A, _ = linearize(x0, d0, prm)
    n_zero = np.sum(np.abs(np.linalg.eigvals(A)) < 1e-9)
    assert n_zero == 4, f"expected 4 zero eigenvalues, got {n_zero}"
    print("four zero eigenvalues (integrators)       OK")


def test_longitudinal_has_two_oscillatory_modes():
    # Short period and phugoid. The short period must be far faster --
    # an order of magnitude at least, or they are not distinct modes.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    A, B = linearize(x0, d0, prm)
    A_lon, _ = longitudinal(A, B)
    pairs = sorted([e for e in np.linalg.eigvals(A_lon) if e.imag > 1e-8],
                   key=lambda e: -abs(e))
    assert len(pairs) == 2, f"expected 2 complex pairs, got {len(pairs)}"
    assert abs(pairs[0]) > 10 * abs(pairs[1])
    print("short period and phugoid both present     OK")


def test_phugoid_matches_classical_approximation():
    # The independent check. Classical theory says the phugoid frequency is
    # roughly sqrt(2)*g/Va, derived from energy exchange alone with no
    # reference to any coefficient in params.py. Agreement to ~15% means
    # the whole model is right, not just self-consistent.
    prm = aerosonde_params()
    Va = 35.0
    x0, d0 = trim(Va, 0.0, prm)
    A, B = linearize(x0, d0, prm)
    A_lon, _ = longitudinal(A, B)
    pairs = sorted([e for e in np.linalg.eigvals(A_lon) if e.imag > 1e-8],
                   key=lambda e: -abs(e))
    wn_phugoid = abs(pairs[1])
    wn_classical = np.sqrt(2) * prm["gravity"] / Va
    assert 0.85 < wn_phugoid / wn_classical < 1.15, \
        f"{wn_phugoid:.4f} vs {wn_classical:.4f}"
    print("phugoid matches sqrt(2)g/Va               OK")


def test_lateral_modes():
    # One oscillatory pair (Dutch roll), plus two real poles that differ
    # enormously in speed: roll subsidence is fast, spiral is slow.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    A, B = linearize(x0, d0, prm)
    A_lat, _ = lateral(A, B)
    ev = np.linalg.eigvals(A_lat)
    pairs = [e for e in ev if e.imag > 1e-8]
    reals = sorted(e.real for e in ev if abs(e.imag) < 1e-8 and abs(e.real) > 1e-9)
    assert len(pairs) == 1, f"expected 1 Dutch roll pair, got {len(pairs)}"
    assert len(reals) == 2, f"expected roll and spiral, got {reals}"
    assert reals[0] < -5.0, f"roll pole too slow: {reals[0]}"
    assert abs(reals[1]) < 1.0, f"spiral pole too fast: {reals[1]}"
    print("Dutch roll, roll and spiral all present   OK")


def test_short_period_rises_with_airspeed():
    # Physics, not numerology. Dynamic pressure goes as Va^2, so the
    # restoring pitch moment stiffens and the short period speeds up.
    prm = aerosonde_params()
    wns = []
    for Va in (25.0, 35.0):
        x0, d0 = trim(Va, 0.0, prm)
        A, B = linearize(x0, d0, prm)
        A_lon, _ = longitudinal(A, B)
        pairs = sorted([e for e in np.linalg.eigvals(A_lon) if e.imag > 1e-8],
                       key=lambda e: -abs(e))
        wns.append(abs(pairs[0]))
    assert wns[1] > wns[0], f"{wns}"
    print("short period speeds up with airspeed      OK")


if __name__ == "__main__":
    for test in (test_jacobian_shapes,
                 test_trim_point_is_equilibrium,
                 test_four_pure_integrators,
                 test_longitudinal_has_two_oscillatory_modes,
                 test_phugoid_matches_classical_approximation,
                 test_lateral_modes,
                 test_short_period_rises_with_airspeed):
        try:
            test()
        except NotImplementedError as e:
            print(f"SKIP  {e}")

    # Once everything above passes, this is the result worth showing.
    try:
        prm = aerosonde_params()
        x0, d0 = trim(35.0, 0.0, prm)
        A, B = linearize(x0, d0, prm)
        print()
        print_modes(longitudinal(A, B)[0], "LONGITUDINAL, Va = 35 m/s")
        print_modes(lateral(A, B)[0], "LATERAL, Va = 35 m/s")
    except NotImplementedError:
        pass
