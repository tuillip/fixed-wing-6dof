"""
dynamics.py -- twelve-state rigid-body equations of motion.

Reference: Beard & McLain Ch. 3; eom-derivation.md in this repo.

This file is the plant. One pure function, derivatives(), takes the current
state plus the net forces and moments acting right now, and returns how fast
each of the twelve states is changing. No integration, no memory between
calls. Forces and moments are inputs here -- Ch. 4 builds them later.

State order, fixed everywhere in this file:
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r

    pn, pe, pd      inertial position, NED     [m]
    u, v, w         body-frame velocity        [m/s]
    phi, theta, psi roll, pitch, yaw           [rad]
    p, q, r         body-frame angular rates   [rad/s]
"""

import numpy as np
from frames import vehicle_to_body


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

def aerosonde_params():
    """Returns: dict of mass, inertia, gravity, plus the eight Gammas."""
    prm = {
        "mass": 11.0,
        "Jx":   0.824,
        "Jy":   1.135,
        "Jz":   1.759,
        "Jxz":  0.120,
        "gravity": 9.81,
    }
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

def inertia_matrix(prm):
    return np.array([[ prm["Jx"], 0, -prm["Jxz"]],
                     [0, prm["Jy"], 0],
                     [-prm["Jxz"], 0, prm["Jz"] ]])


# ----------------------------------------------------------------------
# The four blocks
# ----------------------------------------------------------------------

def translational_kinematics(x):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    return vehicle_to_body(phi, theta, psi).T @ np.array([u, v, w])


def translational_dynamics(x, forces, prm):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    fx, fy, fz = forces

    return np.array([r*v-q*w, p*w-r*u, q*u-p*v]) + 1/prm["mass"]*np.array([fx, fy, fz])

def rotational_kinematics(x):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    return np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]]) @ np.array([p, q, r])

def rotational_dynamics_direct(x, moments, prm):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    Mx, My, Mz = moments
    J = inertia_matrix(prm)
    omega = np.array([p, q, r])
    return np.linalg.solve(J, moments - np.cross(omega, J @ omega))

def rotational_dynamics_gamma(x, moments, prm):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
    Mx, My, Mz = moments
    return np.array([prm["Gamma1"]*p*q-prm["Gamma2"]*q*r+prm["Gamma3"]*Mx+prm["Gamma4"]*Mz,
                    prm["Gamma5"]*p*r-prm["Gamma6"]*(p**2-r**2)+My/prm["Jy"],
                    prm["Gamma7"]*p*q-prm["Gamma1"]*q*r+prm["Gamma4"]*Mx+prm["Gamma8"]*Mz])

# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def derivatives(x, forces, moments, prm):
    """
    Takes:   x (12,), forces (3,), moments (3,), prm
    Returns: (12,) state derivative, in the state order above
    """
    return np.concatenate([
        translational_kinematics(x),
        translational_dynamics(x, forces, prm),
        rotational_kinematics(x),
        rotational_dynamics_direct(x, moments, prm),
    ])


def simulate(x0, forces, moments, prm, t_end, dt=0.01):
    """
    Takes:   x0 (12,), constant forces (3,) and moments (3,), prm, t_end
    Returns: (t, xs) -- xs is (12, n_times), so it unpacks the same way x does
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda t, y: derivatives(y, forces, moments, prm),
        t_span=(0.0, t_end),
        y0=np.asarray(x0, dtype=float),
        t_eval=np.arange(0.0, t_end + dt, dt),
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, sol.y


# ======================================================================
# Tests
# ======================================================================

def _state(pn=0, pe=0, pd=0, u=0, v=0, w=0, phi=0, theta=0, psi=0, p=0, q=0, r=0):
    """Build a state from named parts, e.g. _state(phi=np.pi/4, p=0.5)."""
    return np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r], dtype=float)


def test_rotational_kinematics_identity():
    # Wings level, nose level: the Euler rate map reduces to the identity.
    x = _state(p=0.1, q=0.2, r=0.3)
    assert np.allclose(rotational_kinematics(x), [0.1, 0.2, 0.3])
    print("rotational kinematics identity at level   OK")


def test_pure_roll_banked():
    # Banked 45 deg, rolling only. Heading must not drift.
    x = _state(phi=np.pi / 4, p=0.5)
    phi_dot, theta_dot, psi_dot = rotational_kinematics(x)
    assert np.isclose(phi_dot, 0.5)
    assert np.isclose(theta_dot, 0.0)
    assert np.isclose(psi_dot, 0.0)
    print("pure roll while banked                    OK")

def test_theta_dot_sign():
    # Banked 45 deg with q = r = 1: cos and sin terms must cancel.
    x = _state(phi=np.pi/4, q=1.0, r=1.0)
    assert np.isclose(rotational_kinematics(x)[1], 0.0)
    print("theta_dot sign                            OK")


def test_two_forms_agree():
    # Same equations written two ways. Catches a Gamma typo.
    prm = aerosonde_params()
    x = _state(p=0.3, q=-0.2, r=0.15)
    moments = np.array([1.2, -0.7, 0.4])
    assert np.allclose(rotational_dynamics_direct(x, moments, prm),
                       rotational_dynamics_gamma(x, moments, prm))
    print("Gamma form matches direct inverse         OK")


def test_angular_momentum_conserved():
    # No moments: omega redistributes among axes, ||J omega|| must not change.
    prm = aerosonde_params()
    J = inertia_matrix(prm)
    _, xs = simulate(_state(p=0.5, q=0.3, r=-0.4),
                     np.zeros(3), np.zeros(3), prm, t_end=20.0)
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = xs

    h_0 = np.linalg.norm(J @ np.array([p[0], q[0], r[0]]))
    h_T = np.linalg.norm(J @ np.array([p[-1], q[-1], r[-1]]))
    assert np.isclose(h_0, h_T, rtol=1e-6), f"drifted: {h_0} -> {h_T}"
    print("angular momentum conserved over 20 s      OK")


def test_free_fall():
    # Gravity only, level, no rotation. NED: z positive down, so pd increases.
    prm = aerosonde_params()
    g, m = prm["gravity"], prm["mass"]
    t_end = 3.0
    forces = np.array([0.0, 0.0, m * g])

    _, xs = simulate(_state(u=30.0), forces, np.zeros(3), prm, t_end=t_end)
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = xs

    assert np.isclose(pd[-1], 0.5 * g * t_end**2, rtol=1e-6)
    assert np.isclose(pn[-1], 30.0 * t_end, rtol=1e-6)
    assert np.allclose([phi[-1], theta[-1], psi[-1]], 0.0, atol=1e-9)
    print("free fall matches (1/2) g t^2             OK")


if __name__ == "__main__":
    for test in (test_rotational_kinematics_identity,
                 test_pure_roll_banked,
                 test_two_forms_agree,
                 test_angular_momentum_conserved,
                 test_free_fall,
                 test_theta_dot_sign):
        try:
            test()
        except NotImplementedError as e:
            print(f"SKIP  {e}")
