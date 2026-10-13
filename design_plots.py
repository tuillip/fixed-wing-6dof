"""
design_plots.py -- the figures that go in the README.

Four things a flight controls engineer expects to see for a loop you
designed: where you put the closed-loop poles, how much margin you left,
what the linear response looks like, and whether the real nonlinear
aircraft agrees.

WHAT THE LOOP TRANSFER FUNCTION ACTUALLY IS -- read this before writing
roll_loop_tf, because getting it wrong gives plots that look plausible
and are wrong.

The roll loop is not a simple PD controller in series with the plant.
Look at Beard figure 6.6: the kd feedback wraps around the ROLL RATE,
inside the kp loop. It is an inner rate loop, not a derivative term in
series.

Fold the rate loop in first. The plant from aileron to roll rate is

    p / delta_a  =  a_phi2 / (s + a_phi1)

Close kd around that and the rate loop becomes

    a_phi2 / (s + a_phi1 + a_phi2 * kd_phi)

Then roll angle is the integral of roll rate, which adds a 1/s. So the
open-loop transfer function seen by the kp loop is

    L(s)  =  kp_phi * a_phi2 / ( s * (s + a_phi1 + a_phi2 * kd_phi) )

Check it: closing the unity feedback loop on that gives a denominator of
s^2 + (a_phi1 + a_phi2*kd_phi)*s + kp_phi*a_phi2, which is exactly the
expression on book page 103. If your L does not reproduce that, it is
wrong.

The pitch loop is the same idea. kd_theta feeds back q inside the kp
loop, so it adds to the s coefficient of the plant denominator:

    L(s)  =  kp_theta * a_theta3
             / ( s^2 + (a_theta1 + kd_theta*a_theta3)*s + a_theta2 )

Closing that gives the denominator in equation (6.8).
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")           # write files without opening windows
import matplotlib.pyplot as plt
import control

from params import aerosonde_params
from trim import trim
from autopilot import design_gains, make_controllers, roll_hold, pitch_hold
from autopilot import step_response

VA = 35.0
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ----------------------------------------------------------------------
# The two loop transfer functions -- yours to write
# ----------------------------------------------------------------------

def roll_loop_tf(gains):
    """
    Takes:   the gain dict from design_gains
    Returns: a control.tf object, the open-loop transfer function L(s)
             for the roll attitude loop

    control.tf(numerator, denominator) takes two lists of polynomial
    coefficients, highest power first. So s^2 + 3s + 7 is [1, 3, 7], and
    a bare constant k is [k].

    Build the L(s) derived in the module docstring above.
    """
    
    num = [gains["kp_phi"] * gains["a_phi2"]]
    den = [1, gains["a_phi1"] + gains["a_phi2"] * gains["kd_phi"], 0]

    return control.tf(num, den)


def pitch_loop_tf(gains):
    """
    Takes:   the gain dict from design_gains
    Returns: control.tf object, open-loop L(s) for the pitch attitude loop

    Note both kp_theta and a_theta3 are negative, so their product is
    positive. That is what makes this loop stable.
    """

    num = [gains["kp_theta"] * gains["a_theta3"]]
    den = [1, gains["a_theta1"] + gains["a_theta3"] * gains["kd_theta"], gains["a_theta2"]]

    return control.tf(num, den)


# ----------------------------------------------------------------------
# Plotting -- written for you. These are library calls, not design.
# ----------------------------------------------------------------------

def plot_root_locus(L, title, filename):
    """Where the closed-loop poles go as the loop gain sweeps 0 -> inf."""
    fig, ax = plt.subplots(figsize=(6, 5))
    control.root_locus_plot(L, ax=ax)
    ax.grid(True, alpha=0.3)
    cl = control.poles(control.feedback(L, 1))
    ol = control.poles(L)
    open_h, = ax.plot(ol.real, ol.imag, "bx", markersize=9,
                      label="open-loop poles")
    closed_h, = ax.plot(cl.real, cl.imag, "r*", markersize=16,
                        label="closed-loop poles at design gain")
    ax.set_title(title)
    ax.set_xlabel("Real [1/s]")
    ax.set_ylabel("Imaginary [1/s]")
    ax.legend(handles=[open_h, closed_h], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150)
    plt.close(fig)

def plot_bode(L, title, filename):
    """Magnitude and phase vs frequency, phase margin marked by hand."""
    plt.close("all")
    gm, pm, wcg, wcp = control.margin(L)
    fig = plt.figure(figsize=(7, 6))
    control.bode_plot(L, dB=True)
    mag_ax, phase_ax = fig.axes[0], fig.axes[1]
    mag_ax.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
    mag_ax.axvline(wcp, color="r", linestyle="--", linewidth=1.0)
    phase_ax.axhline(-180.0, color="k", linestyle=":", linewidth=0.8)
    phase_ax.axvline(wcp, color="r", linestyle="--", linewidth=1.0)
    phase_ax.annotate(f"phase margin {pm:.1f} deg",
                      xy=(wcp, -180.0 + pm), xytext=(8, 8),
                      textcoords="offset points", color="r", fontsize=10)
    fig.suptitle(f"{title}\nphase margin {pm:.1f} deg at {wcp:.2f} rad/s",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150)
    plt.close(fig)


def plot_linear_step(L, title, filename, t_end=1.0):
    """Closed-loop unit step response of the LINEAR design model."""
    T = control.feedback(L, 1)
    t, y = control.step_response(T, np.linspace(0.0, t_end, 1000))
    dc = float(control.dcgain(T))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y, linewidth=2, label="response")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="command")
    if abs(dc - 1.0) > 0.01:
        ax.axhline(dc, color="r", linestyle=":", linewidth=1.2,
                   label=f"DC gain = {dc:.3f}  (K_theta_DC)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Response / command")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150)
    plt.close(fig)


def plot_nonlinear_step(prm, gains, which, command_deg, filename):
    """
    The one that matters. Closes the loop around the FULL NONLINEAR
    aircraft and plots the result against the command. If this tracks,
    the gains designed on the linear model survive contact with the real
    plant -- which is the entire point of designing on a linear model.
    """
    x0, d0 = trim(VA, 0.0, prm)
    ctrl = make_controllers(gains)
    cmd = np.radians(command_deg)

    if which == "roll":
        index, label = 6, "Roll angle"
        def loop(x):
            return (d0[0], roll_hold(cmd, x[6], x[9], ctrl), d0[2], d0[3])
    else:
        index, label = 7, "Pitch angle"
        def loop(x):
            return (pitch_hold(cmd, x[7], x[10], ctrl), d0[1], d0[2], d0[3])

    t, sig = step_response(loop, x0, d0, prm, cmd, index=index, t_end=3.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, np.degrees(sig), linewidth=2, label="nonlinear response")
    ax.axhline(command_deg, color="k", linestyle="--", linewidth=0.8,
               label="command")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{label} [deg]")
    ax.set_title(f"{label} hold, {command_deg:g} deg step, nonlinear plant")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150)
    plt.close(fig)


def print_margins(L, label):
    """
    Gain margin, phase margin, and the frequencies they occur at.
    These are the numbers that go in the README text.
    """
    gm, pm, wcg, wcp = control.margin(L)
    print(label)
    if np.isfinite(gm):
        print(f"  gain margin   {20*np.log10(gm):8.2f} dB   at {wcg:.3f} rad/s")
    else:
        print("  gain margin   unbounded -- phase never reaches -180 deg")
    print(f"  phase margin  {pm:8.2f} deg  at {wcp:.3f} rad/s")
    cl = control.poles(control.feedback(L, 1))
    print(f"  closed-loop poles: {np.round(cl, 4)}")
    print(f"  wn = {abs(cl[0]):.4f} rad/s, zeta = {-cl[0].real/abs(cl[0]):.4f}")
    print()


# ======================================================================
# Checks
# ======================================================================

def test_closed_loop_matches_design():
    """
    The design asked for a particular wn and zeta. Closing the loop on
    the transfer function should give back exactly those. If it does
    not, either the transfer function is built wrong or the gain
    formulas are.
    """
    prm = aerosonde_params()
    g = design_gains(prm, VA)

    for name, L, wn_key, zeta in (("roll", roll_loop_tf(g), "wn_phi", 0.707),
                                  ("pitch", pitch_loop_tf(g), "wn_theta", 0.707)):
        cl = control.poles(control.feedback(L, 1))
        wn = abs(cl[0])
        z = -cl[0].real / wn
        assert np.isclose(wn, g[wn_key], rtol=1e-3), \
            f"{name}: wn {wn:.4f} vs designed {g[wn_key]:.4f}"
        assert np.isclose(z, zeta, rtol=1e-2), f"{name}: zeta {z:.4f}"
    print("closed-loop poles match the design       OK")


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    prm = aerosonde_params()
    gains = design_gains(prm, VA)

    try:
        test_closed_loop_matches_design()
        print()

        L_roll = roll_loop_tf(gains)
        L_pitch = pitch_loop_tf(gains)

        print_margins(L_roll, f"ROLL LOOP, Va = {VA} m/s")
        print_margins(L_pitch, f"PITCH LOOP, Va = {VA} m/s")

        plot_root_locus(L_roll, "Roll loop root locus", "roll_rlocus.png")
        plot_bode(L_roll, "Roll loop open-loop response", "roll_bode.png")
        plot_linear_step(L_roll, "Roll loop step, linear model",
                         "roll_step_linear.png")
        plot_nonlinear_step(prm, gains, "roll", 20.0, "roll_step_nonlinear.png")

        plot_root_locus(L_pitch, "Pitch loop root locus", "pitch_rlocus.png")
        plot_bode(L_pitch, "Pitch loop open-loop response", "pitch_bode.png")
        plot_linear_step(L_pitch, "Pitch loop step, linear model",
                         "pitch_step_linear.png")
        plot_nonlinear_step(prm, gains, "pitch", 5.0, "pitch_step_nonlinear.png")

        print(f"eight figures written to {FIGDIR}")
    except NotImplementedError as e:
        print(f"SKIP  {e}")
