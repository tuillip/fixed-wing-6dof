"""
autopilot.py -- successive loop closure autopilot.

Reference: Beard & McLain Ch. 6.

Chapters 3-5 built an aircraft and told you how it behaves on its own.
This file tells it where to go.

The method is successive loop closure. Rather than designing one big
controller for a twelve-state system, you close a fast inner loop around
an attitude, then wrap a slower outer loop around that. The inner loop is
so much faster than the outer one that the outer loop can pretend the
inner loop is instantaneous -- it commands a bank angle and simply assumes
it gets one. Each design is then a single-input single-output problem you
can do on a root locus, which is why this method survives in production
flight software decades after state-space showed up.

Seven loops, in two independent stacks:

    LATERAL                              LONGITUDINAL
    course  chi_c -> phi_c   (PI)        altitude h_c -> theta_c   (PI)
      roll  phi_c -> delta_a (PD)        airspeed Va_c -> theta_c  (PI)
  sideslip  beta  -> delta_r (PI)          pitch theta_c -> delta_e (PD)
                                        airspeed Va_c -> delta_t  (PI)

The two attitude loops (roll, pitch) are PD and use the rate gyro signal
directly -- p and q are measured, so there is no need to differentiate
anything. The outer loops are PI; the integrator is what removes steady
state error against a constant wind or a trim offset.

Airspeed appears twice on purpose. In cruise you hold speed with throttle
and altitude with pitch. In a climb the throttle is already at the stop,
so you hold speed with pitch instead and let altitude take care of itself.
Which of the two is active is the job of the altitude state machine.

State order (from dynamics.py):
    0  1  2   3  4  5   6    7      8    9  10 11
    pn pe pd  u  v  w   phi  theta  psi  p  q  r

Output order (to aerodynamics.py):
    0        1        2        3
    delta_e  delta_a  delta_r  delta_t
"""

import numpy as np
from scipy.integrate import solve_ivp

from params import aerosonde_params
from frames import airdata_from_body_velocity
from linearize import state_derivative
from trim import trim


# ----------------------------------------------------------------------
# Actuator and command limits.
#
# Every loop saturates. Surfaces have mechanical stops, throttle runs 0 to
# 1, and the outer loops must not command an attitude the aircraft cannot
# hold. Check these against Beard's values and against what you think the
# Aerosonde can actually do.
# ----------------------------------------------------------------------

LIMITS = {
    "delta_e": np.radians(45.0),    # elevator, symmetric about zero  [rad]
    "delta_a": np.radians(45.0),    # aileron                          [rad]
    "delta_r": np.radians(45.0),    # rudder                           [rad]
    "delta_t": (0.0, 1.0),          # throttle, one-sided              [-]
    "phi_c":   np.radians(45.0),    # bank angle the course loop may ask for
    "theta_c": np.radians(30.0),    # pitch angle the altitude loop may ask for
}


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------

def saturate(value, low, high):
    """
    Takes:   a number and its two bounds
    Returns: the number clipped into [low, high]

    Used everywhere. Write it once so every loop limits the same way.
    """
    if value > high:
        value = high

    if value < low:
        value = low

    return value


class PIDControl:
    """
    One controller. All seven loops are instances of this with different
    gains and limits -- a PD loop is this with ki = 0, a PI loop is this
    with kd = 0.

    __init__ is written for you; it is bookkeeping. update() is yours.

    The derivative term uses a measured rate rather than the difference of
    successive errors. On the attitude loops that rate is a gyro channel
    (p for roll, q for pitch), which is clean. Differentiating the error
    signal instead would amplify noise and would kick every time the pilot
    moved the command.
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, limit=1.0, dt=0.01,
                 lower=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.high = limit
        self.low = -limit if lower is None else lower
        self.integrator = 0.0       # accumulated error     [error units * s]
        self.error_prev = 0.0       # last error, for trapezoidal integration

    def reset(self):
        """Clear the integrator. Call this when a loop is switched out."""
        self.integrator = 0.0
        self.error_prev = 0.0

    def update(self, error, rate=0.0):
        """
        Takes:   error  = commanded minus measured   [same units as the signal]
                 rate   = measured rate of the signal, for the D term
        Returns: the control output, saturated into [self.low, self.high]

        Four steps:
          1. Advance the integrator by the trapezoidal area over one dt:
             0.5 * dt * (error + error_prev). Save error as error_prev.
          2. Form the unsaturated output: kp*error + ki*integrator - kd*rate.
             Note the minus on the derivative term -- the rate opposes the
             motion, it is damping, not a second proportional term.
          3. Saturate it.
          4. Anti-windup: if the output was clipped, undo the integration
             you just did, so the integrator cannot accumulate authority
             the actuator is not delivering. See the explanation in
             test_anti_windup below for why this matters.
        """
        temp_integrator = self.integrator + 0.5 * self.dt * (error + self.error_prev)
        self.error_prev = error
        output = self.kp * error + self.ki * temp_integrator - self.kd * rate
        saturated_output = saturate(output, self.low, self.high)
        if output == saturated_output:
            self.integrator = temp_integrator

        return saturated_output


# ----------------------------------------------------------------------
# Gain design
#
# These are the numbers you compute once, on the ground, from the linear
# models linearize.py produced. Beard Ch. 6 gives the transfer function
# coefficients (a_phi1, a_phi2, a_theta1..3, a_beta1..2, a_V1..3) and then
# closed-form gains that place the closed-loop poles at a chosen natural
# frequency and damping ratio. Transcribe them from your edition.
#
# The design parameters you choose rather than compute:
#   zeta       damping ratio for each loop, usually around 0.707
#   W          bandwidth separation between an inner and its outer loop.
#              The outer loop is made W times slower. W = 5 to 10. If W is
#              too small the two loops interact and the nice decoupled
#              story stops being true.
# ----------------------------------------------------------------------

def design_gains(prm, Va, zeta_phi=0.707, zeta_theta=0.707, zeta_chi=1.0,
                 e_phi_max=np.radians(20.0), e_theta_max=np.radians(10.0),
                 W_chi=10.0):
    """
    Takes:   prm, trim airspeed Va [m/s], damping ratios, max tracking errors,
             inner/outer bandwidth separation W_chi
    Returns: dict of gains and the transfer-function coefficients they came from

    All equation numbers refer to Beard & McLain, 2nd ed (Oct 2025 printing).

    NOTE ON NAMING: the book writes the rolling-moment coefficient as C_Mx and
    the yawing-moment coefficient as C_Mz. params.py calls the same numbers
    C_ell_* and C_n_*. They are identical.
    """
    rho, S, b, c = prm["rho"], prm["S_wing"], prm["b"], prm["c"]
    Jy, g = prm["Jy"], prm["gravity"]
    G3, G4 = prm["Gamma3"], prm["Gamma4"]

    # --- Inertia-coupled moment coefficients.  Book p.69, after eq (5.12).
    # Jxz couples roll and yaw, so a "rolling moment" coefficient alone does
    # not give roll acceleration.  Gamma3 and Gamma4 are the rows of the
    # inverted inertia matrix that mix them.
    C_p_p       = G3 * prm["C_ell_p"]       + G4 * prm["C_n_p"]
    C_p_delta_a = G3 * prm["C_ell_delta_a"] + G4 * prm["C_n_delta_a"]

    # --- Roll plant:  phi/delta_a = a_phi2 / (s(s + a_phi1)).  Eq (5.24).
    a_phi1 = -0.5 * rho * Va**2 * S * b * C_p_p * (b / (2 * Va))   # eq (5.22)
    a_phi2 =  0.5 * rho * Va**2 * S * b * C_p_delta_a              # eq (5.23)

    # --- Pitch plant: theta/delta_e = a_theta3 / (s^2 + a_theta1 s + a_theta2)
    # Eq (5.27).  Coefficients defined on p.79.
    k_lon = rho * Va**2 * c * S / (2 * Jy)
    a_theta1 = -k_lon * prm["C_m_q"] * (c / (2 * Va))
    a_theta2 = -k_lon * prm["C_m_alpha"]
    a_theta3 =  k_lon * prm["C_m_delta_e"]        # NEGATIVE: C_m_delta_e < 0

    # --- Roll attitude hold.  Eqs (6.1), (6.2).
    # The book leaves wn_phi as a free design parameter.  We pick it from
    # control authority instead: full aileron at e_phi_max of roll error.
    # Then wn_phi follows from eq (6.1) run backwards.
    kp_phi = (LIMITS["delta_a"] / e_phi_max) * np.sign(a_phi2)
    wn_phi = np.sqrt(abs(kp_phi * a_phi2))                         # eq (6.1)
    kd_phi = (2.0 * zeta_phi * wn_phi - a_phi1) / a_phi2           # eq (6.2)

    # --- Pitch attitude hold.  Eqs (6.9), (6.10), (6.11).
    # Same authority argument.  a_theta3 is negative, so kp_theta and
    # kd_theta come out NEGATIVE.  That is correct, not a sign error.
    kp_theta = (LIMITS["delta_e"] / e_theta_max) * np.sign(a_theta3)
    wn_theta = np.sqrt(a_theta2 + kp_theta * a_theta3)             # eq (6.9)
    kd_theta = (2.0 * zeta_theta * wn_theta - a_theta1) / a_theta3 # eq (6.10)

    # The pitch plant has no free integrator, so a PD loop settles short of
    # the command.  K_theta_DC is how much you actually get.  Eq (6.11).
    K_theta_DC = kp_theta * a_theta3 / wn_theta**2

    # --- Course hold.  Eqs (6.5), (6.6).  Outer loop, so slow it down by
    # W_chi relative to the roll loop (p.105).  No wind, so Vg = Va.
    Vg = Va
    wn_chi = wn_phi / W_chi
    kp_chi = 2.0 * zeta_chi * wn_chi * Vg / g                      # eq (6.5)
    ki_chi = wn_chi**2 * Vg / g                                    # eq (6.6)

    return {
        # plant coefficients, kept for the README and for sanity checks
        "a_phi1": a_phi1, "a_phi2": a_phi2,
        "a_theta1": a_theta1, "a_theta2": a_theta2, "a_theta3": a_theta3,
        "wn_phi": wn_phi, "wn_theta": wn_theta, "wn_chi": wn_chi,
        # gains in use
        "kp_phi": kp_phi, "kd_phi": kd_phi,
        "kp_theta": kp_theta, "kd_theta": kd_theta, "K_theta_DC": K_theta_DC,
        "kp_chi": kp_chi, "ki_chi": ki_chi,
        # not yet derived -- altitude, airspeed, sideslip
        "kp_h": 0.0, "ki_h": 0.0,
        "kp_V": 0.0, "ki_V": 0.0,
        "kp_V2": 0.0, "ki_V2": 0.0,
        "kp_beta": 0.0, "ki_beta": 0.0,
    }


def make_controllers(gains, dt=0.01):
    """
    Takes:   the gain dict from design_gains, the control period dt [s]
    Returns: dict of PIDControl objects, keyed "roll", "course", "sideslip",
             "pitch", "altitude", "airspeed_pitch", "airspeed_throttle"

    One place where every loop's gains and limits get bolted together. Pull
    the limits from LIMITS above. The throttle controller is the odd one --
    it runs 0 to 1 rather than symmetric, so pass `lower=0.0`.
    """

    ctrl = {}

    ctrl["roll"] = PIDControl(
        kp=gains["kp_phi"],
        ki=0.0,
        kd=gains["kd_phi"],
        limit=LIMITS["delta_a"],
        dt=dt,
    )

    ctrl["course"] = PIDControl(
        kp=gains["kp_chi"],
        ki=gains["ki_chi"],
        kd=0,
        limit=LIMITS["phi_c"],
        dt=dt,
    )

    ctrl["sideslip"] = PIDControl(
        kp=gains["kp_beta"],
        ki=gains["ki_beta"],
        kd=0,
        limit=LIMITS["delta_r"],
        dt=dt,
    )

    ctrl["pitch"] = PIDControl(
        kp=gains["kp_theta"],
        ki=0.0,
        kd=gains["kd_theta"],
        limit=LIMITS["delta_e"],
        dt=dt,
    )

    ctrl["altitude"] = PIDControl(
        kp=gains["kp_h"],
        ki=gains["ki_h"],
        kd=0,
        limit=LIMITS["theta_c"],
        dt=dt,
    )

    ctrl["airspeed_pitch"] = PIDControl(
        kp=gains["kp_V2"],
        ki=gains["ki_V2"],
        kd=0,
        limit=LIMITS["theta_c"],
        dt=dt,
    )

    ctrl["airspeed_throttle"] = PIDControl(
        kp=gains["kp_V"],
        ki=gains["ki_V"],
        kd=0,
        limit=1.0,
        lower=-1.0,
        dt=dt,
    )
    
    return ctrl


# ----------------------------------------------------------------------
# The lateral stack
# ----------------------------------------------------------------------

def roll_hold(phi_c, phi, p, ctrl):
    """
    Takes:   commanded roll [rad], measured roll [rad], roll rate p [rad/s],
             the controller dict
    Returns: delta_a [rad]

    The innermost lateral loop and the fastest thing in the aircraft -- your
    roll subsidence pole is at -32.9, time constant 30 ms. PD: proportional
    on the roll error, derivative from the p gyro.
    """

    return ctrl["roll"].update(phi_c - phi, p)


def course_hold(chi_c, chi, ctrl):
    """
    Takes:   commanded course [rad], measured course [rad], controller dict
    Returns: commanded roll angle phi_c [rad], limited to LIMITS["phi_c"]

    Wraps the roll loop. A banked aircraft turns, so bank angle is the
    natural actuator for heading. PI -- the integrator is what holds a
    crosswind.

    Wrap the course error into [-pi, pi] before using it. Without that, a
    command of +179 deg against a measurement of -179 deg reads as a 358 deg
    error and the aircraft turns the long way around.
    """
    error = (chi_c - chi + np.pi) % (2 * np.pi) - np.pi
    return ctrl["course"].update(error)


def sideslip_hold(beta, ctrl):
    """
    Takes:   measured sideslip beta [rad], controller dict
    Returns: delta_r [rad]

    Commanded sideslip is always zero -- you want the nose pointed into the
    relative wind, not skidding. PI on the rudder. This is also what damps
    the Dutch roll, which at zeta = 0.183 is the least damped mode you have.
    """
    return ctrl["sideslip"].update(-beta)


# ----------------------------------------------------------------------
# The longitudinal stack
# ----------------------------------------------------------------------

def pitch_hold(theta_c, theta, q, ctrl):
    """
    Takes:   commanded pitch [rad], measured pitch [rad], pitch rate q [rad/s],
             controller dict
    Returns: delta_e [rad]

    The longitudinal mirror of roll_hold. PD, with q from the gyro.

    One wrinkle worth knowing about: a pure PD pitch loop has steady state
    error, because holding a nonzero pitch attitude requires a nonzero
    elevator and proportional control needs an error to produce one. Beard
    handles this with a DC gain correction rather than an integrator, to
    keep the inner loop fast. Check how your edition does it.
    """
    return ctrl["pitch"].update(theta_c - theta, q)


def altitude_hold(h_c, h, ctrl):
    """
    Takes:   commanded altitude [m], measured altitude [m], controller dict
    Returns: commanded pitch theta_c [rad], limited to LIMITS["theta_c"]

    Note h is altitude, positive up. Your state carries pd, positive DOWN.
    h = -pd. Getting this backwards makes the aircraft fly into the ground
    very confidently, so convert at the boundary and only once.
    """
    return ctrl["altitude"].update(h_c - h)


def airspeed_from_pitch(Va_c, Va, ctrl):
    """
    Takes:   commanded airspeed [m/s], measured airspeed [m/s], controller dict
    Returns: commanded pitch theta_c [rad], limited to LIMITS["theta_c"]

    Used in the climb and descend zones, where the throttle is pinned and
    pitch is the only speed control left. Lower the nose to speed up.
    Mind the sign: positive airspeed error wants LESS pitch, not more.
    """
    return ctrl["airspeed_pitch"].update(Va - Va_c)


def airspeed_from_throttle(Va_c, Va, delta_t_trim, ctrl):
    """
    Takes:   commanded airspeed [m/s], measured airspeed [m/s],
             trim throttle [-], controller dict
    Returns: delta_t [-], saturated into [0, 1]

    Used in the altitude-hold zone. The loop controls the deviation from
    trim throttle, not throttle itself -- add delta_t_trim to the PI output.
    Your trim throttle at 35 m/s is 0.938, so there is very little headroom
    above it. Expect this loop to saturate often, which is exactly why the
    anti-windup has to work.
    """
    deviation = ctrl["airspeed_throttle"].update(Va_c - Va)
    return saturate(deviation + delta_t_trim, 0.0, 1.0)


# ----------------------------------------------------------------------
# Mode logic and dispatch
# ----------------------------------------------------------------------

def altitude_zone(h, h_c, prm, takeoff_alt=10.0, hold_band=20.0):
    """
    Takes:   current altitude [m], commanded altitude [m], prm, two thresholds
    Returns: one of "takeoff", "climb", "altitude_hold", "descend"

    The state machine. Four zones:
      takeoff        below takeoff_alt -- full throttle, pitch held at a
                     fixed climb attitude, no altitude feedback at all
      climb          more than hold_band below the command -- full throttle,
                     airspeed held with pitch
      altitude_hold  within hold_band of the command -- throttle holds
                     airspeed, pitch holds altitude
      descend        more than hold_band above the command -- throttle at
                     idle, airspeed held with pitch

    Hysteresis is worth thinking about: an aircraft sitting exactly on a
    zone boundary will chatter between two sets of loops. Not required for
    the tests, but it is the kind of thing an interviewer asks about.
    """

    if (h <= takeoff_alt):
        return "takeoff"
    elif (h_c > h + hold_band):
        return "climb"
    elif (h_c + hold_band < h):
        return "descend"
    else:
        return "altitude_hold"
    

def autopilot(x, commands, ctrl, prm, delta_t_trim):
    """
    Takes:   x (12,), commands dict with keys "Va_c" [m/s], "h_c" [m],
             "chi_c" [rad]; the controller dict; prm; trim throttle
    Returns: (4,) array -- delta_e, delta_a, delta_r, delta_t

    The whole autopilot in one call. Unpack the state, compute airspeed and
    sideslip from the body velocities, work out the course angle, pick the
    altitude zone, run the lateral stack and whichever longitudinal stack
    the zone calls for, and return the four surface commands.

    Course angle chi is the direction the aircraft is actually travelling
    over the ground, which is not the same as heading psi when there is
    wind. With no wind model they are equal, so chi = psi is fine for now --
    but say so in a comment, because the distinction is the entire reason
    the course loop exists.
    """
    raise NotImplementedError("autopilot")


# ----------------------------------------------------------------------
# Closed-loop simulation
#
# Written for you. Note the structure: the autopilot runs once per dt and
# its output is held constant while the plant integrates forward over that
# interval. That is not a simplification -- it is what a real autopilot
# does. Flight software runs at a fixed rate, computes a command, and the
# surfaces hold that command until the next frame. The plant is continuous,
# the controller is discrete, and dt is the control period.
# ----------------------------------------------------------------------

def simulate_closed_loop(x0, commands, ctrl, prm, delta_t_trim,
                         t_end, dt=0.01):
    """
    Takes:   x0 (12,), commands dict, controller dict, prm, trim throttle,
             duration [s], control period [s]
    Returns: (t, xs, us) -- t is (n,), xs is (12, n), us is (4, n)

    xs unpacks exactly the way your state does:
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = xs
    """
    n_steps = int(round(t_end / dt))
    t = np.linspace(0.0, t_end, n_steps + 1)
    xs = np.zeros((12, n_steps + 1))
    us = np.zeros((4, n_steps + 1))

    x = np.asarray(x0, dtype=float)
    xs[:, 0] = x

    for k in range(n_steps):
        deltas = np.asarray(autopilot(x, commands, ctrl, prm, delta_t_trim),
                            dtype=float)
        us[:, k] = deltas
        sol = solve_ivp(lambda t_, y: state_derivative(y, deltas, prm),
                        t_span=(0.0, dt), y0=x, rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        xs[:, k + 1] = x

    us[:, -1] = us[:, -2]
    return t, xs, us


def step_response(loop_fn, x0, d0, prm, command, index, t_end=5.0, dt=0.01):
    """
    Takes:   a single-loop closure function, trim state and inputs, prm,
             the command value, which state index to watch, duration
    Returns: (t, signal) -- the tracked state over time

    Closes ONE loop around the full nonlinear aircraft, leaving every other
    surface at its trim value. This is how you test an inner loop in
    isolation, and it is what produces the step response plot for your
    README. loop_fn(x) must return the single surface deflection.
    """
    n_steps = int(round(t_end / dt))
    t = np.linspace(0.0, t_end, n_steps + 1)
    sig = np.zeros(n_steps + 1)

    x = np.asarray(x0, dtype=float)
    sig[0] = x[index]

    for k in range(n_steps):
        deltas = loop_fn(x)
        sol = solve_ivp(lambda t_, y: state_derivative(y, deltas, prm),
                        t_span=(0.0, dt), y0=x, rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        sig[k + 1] = x[index]

    return t, sig


# ======================================================================
# Tests
# ======================================================================

def test_saturate():
    assert saturate(5.0, -1.0, 1.0) == 1.0
    assert saturate(-5.0, -1.0, 1.0) == -1.0
    assert saturate(0.3, -1.0, 1.0) == 0.3
    assert saturate(0.5, 0.0, 1.0) == 0.5
    print("saturate clips on both sides              OK")


def test_controller_respects_limits():
    # No input, however absurd, may produce an output past the stop. This is
    # the cheapest bug to prevent and the most embarrassing to ship.
    pid = PIDControl(kp=100.0, ki=10.0, kd=1.0, limit=0.5, dt=0.01)
    for err in (1e6, -1e6, 0.0, 3.7, -0.02):
        out = pid.update(err, rate=0.0)
        assert abs(out) <= 0.5 + 1e-12, f"breached the stop: {out}"
    print("controller output stays inside its limit  OK")


def test_zero_error_zero_output():
    # A controller with no error and no accumulated history does nothing.
    pid = PIDControl(kp=2.0, ki=1.0, kd=0.5, limit=1.0, dt=0.01)
    assert np.isclose(pid.update(0.0, rate=0.0), 0.0)
    print("zero error gives zero output              OK")


def test_anti_windup():
    # Pin the controller against its upper stop for five seconds with a
    # constant error it cannot remove. Then reverse the error hard.
    #
    # WITHOUT anti-windup the integrator has been accumulating the whole
    # time -- 500 steps of error 10 at ki = 50 is an enormous stored value --
    # and the output stays pinned at +1 long after the error went negative,
    # because kp*error has to overcome ki*integrator before anything moves.
    # That lag is what makes a real aircraft overshoot and wallow.
    #
    # WITH anti-windup the integrator never grew past what the actuator
    # could deliver, so the output responds on the next frame.
    pid = PIDControl(kp=1.0, ki=50.0, kd=0.0, limit=1.0, dt=0.01)
    for _ in range(500):
        pid.update(10.0)
    assert np.isclose(pid.update(10.0), 1.0), "should be pinned at the stop"

    out = pid.update(-10.0)
    assert out < 0.0, f"still {out:.4f} after the error reversed -- wound up"
    print("integrator does not wind up               OK")


def test_gains_are_finite_and_positive():
    prm = aerosonde_params()
    gains = design_gains(prm, 35.0)
    for key in ("kp_phi", "kd_phi", "kp_theta", "kd_theta"):
        assert key in gains, f"missing gain: {key}"
        assert np.isfinite(gains[key]), f"{key} is not finite: {gains[key]}"
    print("roll and pitch gains computed             OK")


def test_gains_change_with_airspeed():
    # Every aerodynamic coefficient is scaled by dynamic pressure, which
    # goes as Va^2. More dynamic pressure means more control power per
    # degree of surface, so the gains needed to hit the same closed-loop
    # bandwidth must come DOWN as speed goes up. If a gain is unchanged
    # between 25 and 35 m/s, it was not computed from the aerodynamics.
    prm = aerosonde_params()
    slow = design_gains(prm, 25.0)
    fast = design_gains(prm, 35.0)
    assert fast["wn_phi"] > slow["wn_phi"]
    assert abs(fast["kd_phi"]) < abs(slow["kd_phi"])
    print("gains fall as dynamic pressure rises      OK")


def test_roll_hold_tracks_a_bank_command():
    # Close ONLY the roll loop around the full nonlinear aircraft and ask
    # for 20 degrees of bank. Everything else stays at its trim value. If
    # this converges, the inner loop works on the real plant and not just
    # on the linear model it was designed against.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))

    phi_c = np.radians(20.0)

    def loop(x):
        delta_a = roll_hold(phi_c, x[6], x[9], ctrl)
        return (d0[0], delta_a, d0[2], d0[3])

    t, phi = step_response(loop, x0, d0, prm, phi_c, index=6, t_end=5.0)
    err = np.degrees(phi[-1]) - 20.0
    assert abs(err) < 2.0, f"settled at {np.degrees(phi[-1]):.2f} deg"
    print("roll hold tracks a 20 deg command         OK")


def test_roll_hold_is_fast():
    # The roll subsidence pole sits at -32.9, so the open-loop aircraft
    # rolls with a 30 ms time constant. A closed loop slower than about a
    # second is not using the authority available to it.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))

    phi_c = np.radians(20.0)

    def loop(x):
        return (d0[0], roll_hold(phi_c, x[6], x[9], ctrl), d0[2], d0[3])

    t, phi = step_response(loop, x0, d0, prm, phi_c, index=6, t_end=5.0)
    reached = np.where(np.degrees(phi) > 0.9 * 20.0)[0]
    assert len(reached) > 0, "never reached 90% of the command"
    assert t[reached[0]] < 1.0, f"took {t[reached[0]]:.2f} s to reach 90%"
    print("roll hold reaches 90% within a second     OK")


def test_pitch_hold_tracks_a_pitch_command():
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))

    theta_c = np.radians(5.0)

    def loop(x):
        delta_e = pitch_hold(theta_c, x[7], x[10], ctrl)
        return (delta_e, d0[1], d0[2], d0[3])

    t, theta = step_response(loop, x0, d0, prm, theta_c, index=7, t_end=5.0)
    err = np.degrees(theta[-1]) - 5.0
    assert abs(err) < 2.0, f"settled at {np.degrees(theta[-1]):.2f} deg"
    print("pitch hold tracks a 5 deg command         OK")


# ---- Everything below is the stretch goal: the full seven-loop autopilot --

def test_altitude_zones():
    prm = aerosonde_params()
    assert altitude_zone(5.0, 100.0, prm) == "takeoff"
    assert altitude_zone(50.0, 100.0, prm) == "climb"
    assert altitude_zone(100.0, 100.0, prm) == "altitude_hold"
    assert altitude_zone(200.0, 100.0, prm) == "descend"
    print("altitude state machine picks the zone     OK")


def test_autopilot_output_shape_and_limits():
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))
    cmd = {"Va_c": 35.0, "h_c": 100.0, "chi_c": 0.0}

    u = np.asarray(autopilot(x0, cmd, ctrl, prm, d0[3]), dtype=float)
    assert u.shape == (4,), u.shape
    assert abs(u[0]) <= LIMITS["delta_e"] + 1e-12
    assert abs(u[1]) <= LIMITS["delta_a"] + 1e-12
    assert abs(u[2]) <= LIMITS["delta_r"] + 1e-12
    assert 0.0 <= u[3] <= 1.0, f"throttle out of range: {u[3]}"
    print("autopilot returns four legal commands     OK")


def test_holds_trim_when_commanded_trim():
    # Commanded exactly what it is already doing, the autopilot should not
    # wander off. Fly for thirty seconds and check nothing drifts.
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))
    h0 = -x0[2]
    cmd = {"Va_c": 35.0, "h_c": h0, "chi_c": x0[8]}

    t, xs, us = simulate_closed_loop(x0, cmd, ctrl, prm, d0[3], t_end=30.0)
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = xs
    Va = np.array([airdata_from_body_velocity(u[k], v[k], w[k])[0]
                   for k in range(len(t))])

    assert np.all(np.isfinite(xs)), "state went non-finite"
    assert abs(-pd[-1] - h0) < 5.0, f"altitude drifted {-pd[-1] - h0:.2f} m"
    assert abs(Va[-1] - 35.0) < 2.0, f"airspeed drifted to {Va[-1]:.2f} m/s"
    assert abs(np.degrees(phi[-1])) < 5.0, "rolled away from level"
    print("holds trim for 30 s when told to          OK")


def test_climbs_to_a_new_altitude():
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))
    h0 = -x0[2]
    cmd = {"Va_c": 35.0, "h_c": h0 + 50.0, "chi_c": x0[8]}

    t, xs, us = simulate_closed_loop(x0, cmd, ctrl, prm, d0[3], t_end=60.0)
    h = -xs[2]
    assert h[-1] > h0 + 40.0, f"only climbed to {h[-1] - h0:.1f} m"
    print("climbs 50 m and levels off                OK")


def test_turns_to_a_new_course():
    prm = aerosonde_params()
    x0, d0 = trim(35.0, 0.0, prm)
    ctrl = make_controllers(design_gains(prm, 35.0))
    cmd = {"Va_c": 35.0, "h_c": -x0[2], "chi_c": np.radians(45.0)}

    t, xs, us = simulate_closed_loop(x0, cmd, ctrl, prm, d0[3], t_end=30.0)
    chi_final = np.degrees(xs[8][-1])
    assert abs(chi_final - 45.0) < 5.0, f"settled on {chi_final:.1f} deg"
    print("turns onto a 45 deg course                OK")


if __name__ == "__main__":
    # The Oct 15 scope: everything down to pitch hold.
    core = (test_saturate,
            test_controller_respects_limits,
            test_zero_error_zero_output,
            test_anti_windup,
            test_gains_are_finite_and_positive,
            test_gains_change_with_airspeed,
            test_roll_hold_tracks_a_bank_command,
            test_roll_hold_is_fast,
            test_pitch_hold_tracks_a_pitch_command)

    # The stretch goal: the other five loops and the state machine.
    full = (test_altitude_zones,
            test_autopilot_output_shape_and_limits,
            test_holds_trim_when_commanded_trim,
            test_climbs_to_a_new_altitude,
            test_turns_to_a_new_course)

    for test in core + full:
        try:
            test()
        except NotImplementedError as e:
            print(f"SKIP  {e}")

    print()     
    prm = aerosonde_params()
    for Va in (25.0, 35.0):
        g = design_gains(prm, Va)
        print(f"--- gains at Va = {Va} m/s ---")
        for key in ("a_phi1", "a_phi2", "kp_phi", "kd_phi", "wn_phi",
                    "kp_theta", "kd_theta", "wn_theta", "K_theta_DC"):
            print(f"  {key:12s} {g[key]:10.4f}")
