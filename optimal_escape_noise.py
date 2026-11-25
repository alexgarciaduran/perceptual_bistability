r"""
optimal_escape_noise.py
=======================

Optimal-path analysis for noise-driven and stimulus-driven escape over a
double well, from the Onsager--Machlup (OM) functional.

--------------------------------------------------------------------------
1. MODEL  (single tilted quartic used everywhere)
--------------------------------------------------------------------------
Potential
        V(x) = a ( -x^2/2 + x^4/4 ) - b x ,
drift  f(x) = -V'(x) = a (x - x^3) + b .

Overdamped Langevin dynamics with additive white noise and additive stimulus

        x_dot = a (x - x^3) + b(t) + eta(t) ,   <eta(t) eta(t')> = 2 D delta.

* a  scales the barrier (height Delta V = a/4 at b = 0; wells at x = +/- 1).
* b  is the STIMULUS: a tilt that lowers one well relative to the other.  It
  enters the drift ADDITIVELY, exactly like the noise -- this is the key
  simplification.  Because noise and stimulus act through the same channel,
  the joint optimum has them PROPORTIONAL (b = kappa * eta), so the optimal
  escape trajectory stays smooth: there is no "kick" when stimulus is added
  (unlike a model where the bias sits inside a nonlinearity).

--------------------------------------------------------------------------
2. ONSAGER--MACHLUP FUNCTIONAL  (noise-driven, b = 0)
--------------------------------------------------------------------------
Path probability  P[x] ~ exp(-S[x]),

        S[x] = INT dt [ 1/(4D) (x_dot - f)^2  +  1/2 f'(x) ] .

(a) REDUCED functional  L = (x_dot - f)^2  (Freidlin--Wentzell, D -> 0)
    Euler--Lagrange :  x_ddot = f f' = 1/2 d/dx f^2 .
    First integral  :  x_dot^2 = f^2 + C .
      * zero-energy instanton C = 0 -> x_dot = -f (uphill), giving

                        eta*(t) = -2 f(x)                      [RESULT 1]

      * finite duration C > 0 -> x_dot = sqrt(f^2 + C), eta* = x_dot - f,
        with C fixed by the transition time T.

(b) FULL functional  L = (x_dot - f)^2 + 2 D f'  (Jacobian kept)
    Euler--Lagrange :  x_ddot = f f' + D f'' .
    First integral  :  x_dot^2 = f^2 + 2 D f' + C , so

                        x_dot* = +/- sqrt(f^2 + 2 D f' + C)    [RESULT 2]
                        eta*   = x_dot* - f .

Along the full -1 -> +1 transition eta* is a positive PULSE on the uphill leg
(-1 -> 0), equal to -2f there, and ~0 on the free downhill leg (0 -> +1).

--------------------------------------------------------------------------
3. NOISE-DRIVEN vs STIMULUS-DRIVEN  (joint optimum, Fig 3)
--------------------------------------------------------------------------
Minimise the control cost   J = INT [ eta^2/(4D) + b^2/(4 alpha) ] dt
subject to  x_dot = a(x - x^3) + b + eta ,  x(0) = -1, x(T) = +1.

Pontryagin (costate p):
        eta* = -2 D p ,      b* = -2 alpha p = (alpha/D) eta* ,
        x_dot = a(x - x^3) - 2(D + alpha) p ,
        p_dot = -p * a (1 - 3 x^2) .

alpha is the STIMULUS DIFFUSIVITY (how cheap the stimulus is vs noise):
        alpha = 0    -> pure noise-driven (b = 0, eta* = -2f recovered),
        alpha -> inf -> pure stimulus-driven (eta -> 0),
        0 < alpha    -> joint; stimulus supplies a fraction alpha/(D+alpha)
                        of the total drive U = -2(D+alpha)p, noise the rest,
                        SAME trajectory -> no kick.

Run  ``python optimal_escape_noise.py``  to regenerate every figure.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, quad, cumulative_trapezoid
from scipy.optimize import root_scalar
from matplotlib.lines import Line2D
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["axes.grid"] = False


FIGDIR = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/optimal_escape_eta/figures/'  # Alex
os.makedirs(FIGDIR, exist_ok=True)
DATADIR = os.path.join(FIGDIR, 'sim_data')          # saved simulation kernels
os.makedirs(DATADIR, exist_ok=True)

# =====================================================================
#  Tilted quartic:  V(x) = a(-x^2/2 + x^4/4) - b x ,  f = -V'
# =====================================================================
def V(x, a=1.0, b=0.0):
    return a * (-0.5 * x ** 2 + 0.25 * x ** 4) - b * x


def f(x, a=1.0, b=0.0):
    """Drift  f(x) = a (x - x^3) + b ."""
    return a * (x - x ** 3) + b


def fp(x, a=1.0, b=0.0):
    """f'(x) = a (1 - 3 x^2)  (independent of the tilt b)."""
    return a * (1.0 - 3.0 * x ** 2)


def fpp(x, a=1.0, b=0.0):
    """f''(x) = -6 a x ."""
    return -6.0 * a * x


def wells_and_barrier(a=1.0, b=0.0):
    """Return (x_left, x_barrier, x_right) real roots of f(x)=0, sorted.

    For |b| below the spinodal  b_c = 2a/(3 sqrt 3)  there are three roots
    (two minima + one barrier); above it only one root remains.
    """
    roots = np.roots([-a, 0.0, a, b])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    return real


def spinodal(a=1.0):
    """Critical tilt at which the left well disappears:  b_c = 2a/(3 sqrt 3)."""
    return 2.0 * a / (3.0 * np.sqrt(3.0))


# =====================================================================
#  SECTION A -- noise-driven: reduced vs full functional (b = 0)
# =====================================================================
def reduced_escape(a=1.0, b=0.0, T=12.0, npts=2000, full=True):
    """Reduced-functional escape,  x_dot = sqrt(f^2 + C).

    From the left well to the right well (``full``) or to the barrier.
    C is fixed by the total time T; C -> 0 (large T) is the zero-energy
    instanton where eta* = -2f on the ascent.  Smooth and monotone (no kick).
    Time aligned so the barrier crossing (x = 0 for b = 0) sits at t = 0.
    """
    r = wells_and_barrier(a, b)
    xL, xbar, xR = r[0], r[1], r[-1]
    x0 = xL + 1e-3
    xT = (xR - 1e-3) if full else (xbar - 1e-3)
    xg = np.linspace(x0, xT, npts)

    def travel_time(C):
        val, _ = quad(lambda x: 1.0 / np.sqrt(f(x, a, b) ** 2 + C),
                      x0, xT, limit=400)
        return val - T

    C = root_scalar(travel_time, bracket=[1e-9, 80.0], method="brentq").root
    fx = f(xg, a, b)
    dxdt = np.sqrt(fx ** 2 + C)
    t = cumulative_trapezoid(1.0 / dxdt, xg, initial=0.0)
    t -= np.interp(xbar, xg, t)          # crossing at t = 0
    eta = dxdt - fx
    return t, xg, eta, C


def full_functional(a=1.0, b=0.0, D=0.05, T=12.0, npts=400, full=True):
    """Full OM functional via the BVP  x_ddot = f f' + D f''.

    Left well -> right well (``full``) or -> barrier.  Returns the path,
    eta* = x_dot - f, and the solver status.
    """
    r = wells_and_barrier(a, b)
    xL, xbar, xR = r[0], r[1], r[-1]
    x0 = xL + 1e-3
    xT = (xR - 1e-3) if full else (xbar - 1e-3)

    def rhs(t, y):
        x, dx = y
        return np.vstack((dx, f(x, a, b) * fp(x, a) + D * fpp(x, a)))

    def bc(y0, yT):
        return np.array([y0[0] - x0, yT[0] - xT])

    t = np.linspace(0, T, npts)
    y0 = np.zeros((2, t.size))
    y0[0] = np.linspace(x0, xT, t.size)
    y0[1] = (xT - x0) / T
    sol = solve_bvp(rhs, bc, t, y0, max_nodes=20000, tol=1e-6)
    x, dx = sol.y
    eta = dx - f(x, a, b)
    tt = sol.x - np.interp(xbar, x, sol.x)
    return tt, x, eta, sol.success


def full_om_instanton(a=1.0, b=0.0, D=0.1, npts=2000):
    """Full-OM ascent instanton from Result 2, by quadrature (no BVP).

        x_dot = sqrt(f^2 + 2 D f' + C) ,   eta* = x_dot - f ,

    with C fixed by the start-at-rest condition at the left well
    (f(xL)=0):  C = -2 D f'(xL).  This keeps the radicand >= 0 on the whole
    ascent (=0 only at xL), so it is robust at large D where the BVP fails.
    Returns tau (<=0, crossing at 0), x, and eta*.
    """
    r = wells_and_barrier(a, b)
    xL, xbar = r[0], r[1]
    C = -2.0 * D * fp(xL, a)
    xg = np.linspace(xL + 1e-4, xbar - 1e-4, npts)
    rad = np.maximum(f(xg, a, b) ** 2 + 2 * D * fp(xg, a, b) + C, 1e-12)
    dxdt = np.sqrt(rad)
    eta = dxdt - f(xg, a, b)
    tau = cumulative_trapezoid(1.0 / dxdt, xg, initial=0.0)
    tau -= tau[-1]                       # crossing (xbar) at tau = 0
    return tau, xg, eta


def fullom_well_to_well(a=1.0, b=0.0, D=0.1, npts=4000):
    """Full-OM well->well path by quadrature with a SINGLE C (b = 0).

    For b = 0 both wells share f'(xL)=f'(xR)=-2a, so the rest condition
    C = -2D f'(xL) = 4aD holds at BOTH wells and the radicand

        x_dot^2 = f^2 + 2D f' + C = f^2 + 6 a D (1 - x^2)

    is >= 0 on [xL, xR] (zero only at the wells).  The path rests at both
    wells, crosses the barrier at speed sqrt(6aD), and eta* = x_dot - f rises
    on the ascent and falls back to 0 on the descent (no plateau).  Robust at
    any D.  Crossing (x = xbar) aligned at t = 0.
    """
    r = wells_and_barrier(a, b)
    xL, xbar, xR = r[0], r[1], r[-1]
    C = -2.0 * D * fp(xL, a)
    xg = np.linspace(xL + 1e-5, xR - 1e-5, npts)
    rad = np.maximum(f(xg, a, b) ** 2 + 2 * D * fp(xg, a, b) + C, 1e-12)
    dxdt = np.sqrt(rad)
    eta = dxdt - f(xg, a, b)
    t = cumulative_trapezoid(1.0 / dxdt, xg, initial=0.0)
    t -= np.interp(xbar, xg, t)          # crossing at t = 0
    return t, xg, eta


def committor_eta(a=1.0, b=0.0, D=0.1, x=None, nquad=800):
    """EXACT finite-D switch-conditioned mean noise  <chi|x> = 2 D q'/q.

    Conditioning the diffusion on a left->right switch is a Doob h-transform by
    the committor q(x) = P(reach right well before left).  The reactive process
    gains drift  2 D q'(x)/q(x), so the mean injected noise at position x is

        <chi|x> = 2 D q'/q = 2 D / INT_{xL}^{x} exp([V(y)-V(x)]/D) dy .

    Weak-noise Laplace limit -> 2 V'(x) = -2 f(x) (the instanton) in the bulk,
    but it stays elevated near the wells/barrier (the spikes are real).
    """
    r = wells_and_barrier(a, b)
    xL = r[0]
    if x is None:
        x = np.linspace(xL + 1e-3, r[1] - 1e-3, 200)
    x = np.atleast_1d(x)
    out = np.empty_like(x, dtype=float)
    for i, xx in enumerate(x):
        ys = np.linspace(xL, xx, nquad)
        integ = np.trapz(np.exp((V(ys, a, b) - V(xx, a, b)) / D), ys)
        out[i] = 2.0 * D / integ
    return out


def ascent_instanton(a=1.0, b=0.0, T=15.0, npts=2000):
    """Analytic ascent pulse (left well -> barrier), zero-energy, eta* = -2f.

    Robust closed form used by the parameter sweep (Section B).
    """
    r = wells_and_barrier(a, b)
    xL, xbar = r[0], r[1]
    # integrate the time-reversed relaxation x_dot = -f from just inside the well
    t = np.linspace(0, T, npts)
    dt = t[1] - t[0]
    x = np.zeros(npts)
    x[0] = xL + 1e-3
    for k in range(1, npts):
        x[k] = x[k - 1] - f(x[k - 1], a, b) * dt
    eta = -2.0 * f(x, a, b)
    t = t - t[np.argmax(eta)]            # peak at t = 0 reference
    return t, x, eta


def fig1_escape(a=1.0, b=0.0, D=0.1, T=14.0):
    """Simple optimal escape: potential, x(t), eta(t)  (3 panels).

    Reduced zero-energy instanton (eta* = -2f) overlaid with the full-OM
    well->well path, shown BOTH ways (they agree at small D):
      * single-C quadrature  (fullom_well_to_well, robust at any D)
      * fixed-T BVP          (full_functional)
    """
    tr, xr, etar, Cr = reduced_escape(a, b, T=T, full=True)
    tq, xq, etaq = fullom_well_to_well(a, b, D=D)
    tb, xb, etab, ok = full_functional(a, b, D=D, T=T, full=True)

    # peak-align the noise pulses (the reduced C~0 instanton is slow near the
    # wells; aligning the eta peaks at t=0 compares pulse SHAPES fairly).
    tr = tr - tr[np.argmax(etar)]
    tq = tq - tq[np.argmax(etaq)]
    if ok:
        tb = tb - tb[np.argmax(etab)]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    xv = np.linspace(-1.5, 1.5, 300)
    ax[0].plot(xv, V(xv, a, b), "k", lw=3)
    for xm in wells_and_barrier(a, b):
        ax[0].plot(xm, V(xm, a, b), "o", color="firebrick", ms=9)
    ax[0].set(xlabel="x", ylabel="V(x)", title=f"Potential  (a={a}, b={b})")

    ax[1].plot(tr, xr, "k", lw=3, label="reduced instanton")
    ax[1].plot(tq, xq, color="firebrick", lw=2.5, label=f"full-OM quadrature (D={D})")
    if ok:
        ax[1].plot(tb, xb, "--", color="steelblue", lw=2, label="full-OM BVP")
    ax[1].axhline(0.0, color="gray", ls=":", lw=1)
    ax[1].set(xlabel="time (η* peak at 0)", ylabel="x(t)", xlim=(-7, 7),
              title="Optimal escape path  (−1 → +1)")
    ax[1].legend(frameon=False, fontsize=10)

    ax[2].plot(tr, etar, "k", lw=3, label=r"reduced  $\eta=-2f$")
    ax[2].plot(tq, etaq, color="firebrick", lw=2.5, label="full-OM quadrature")
    if ok:
        ax[2].plot(tb, etab, "--", color="steelblue", lw=2, label="full-OM BVP")
    ax[2].axhline(0.0, color="gray", ls=":", lw=1)
    ax[2].set(xlabel="time (η* peak at 0)", ylabel="η*(t)", xlim=(-7, 7),
              title="Optimal escape noise")
    ax[2].legend(frameon=False, fontsize=10)

    for a_ in ax:
        a_.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


# =====================================================================
#  SECTION B -- dependence on the potential parameters a (barrier) and b (tilt)
# =====================================================================
def fig2_params(a_vals=(0.5, 1.0, 1.5, 2.0), b_fracs=(0.0, 0.3, 0.6, 0.85),
                a_list=None, b_list=None):
    """Explore the potential parameters  (2 rows x 3 cols).

    Top row: vary the barrier a (b = 0).  Bottom row: vary the tilt b (a = 1).
    Cols: potential V(x) | noise pulse eta*(t) | peak eta* vs the parameter,
    with the analytic laws  peak = 4a/(3 sqrt3)  (top)  and  2(b_c - b) (bottom).
    """
    if a_list is None:
        a_list = np.arange(0.1, 2.01, 0.02)
    if b_list is None:
        b_list = np.linspace(0.0, 0.99 * spinodal(1.0), 60)
    bc = spinodal(1.0)
    xv = np.linspace(-1.6, 1.6, 300)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

    # --- top row: vary a (b = 0) -------------------------------------
    for a in a_vals:
        col = plt.cm.viridis(a / 2.2)
        ax[0, 0].plot(xv, V(xv, a, 0.0), lw=2.5, color=col, label=f"a={a}")
        t, x, eta = ascent_instanton(a, 0.0, T=max(6.0, 8.0 / a), npts=4000)
        ax[0, 1].plot(t, eta, lw=2.5, color=col, label=f"a={a}")
    peak_a = [ascent_instanton(a, 0.0, T=max(6.0, 8.0 / a), npts=4000)[2].max()
              for a in a_list]
    ax[0, 2].plot(a_list, peak_a, "k", lw=3, label="numeric")
    ax[0, 2].plot(a_list, 4.0 * a_list / (3.0 * np.sqrt(3.0)), "--",
                  color="firebrick", lw=2, label=r"$4a/3\sqrt{3}$")
    ax[0, 0].set(xlabel="x", ylabel="V(x)", title="Potential vs a  (b=0)")
    ax[0, 1].set(xlabel="time from peak", ylabel="η*(t)", xlim=(-8, 8),
                 title="Noise pulse vs a")
    ax[0, 2].set(xlabel="a", ylabel="peak η*", title="Peak noise vs a")

    # --- bottom row: vary b (a = 1) ----------------------------------
    for frac in b_fracs:
        bb = frac * bc
        col = plt.cm.copper(0.1 + 0.8 * frac)
        ax[1, 0].plot(xv, V(xv, 1.0, bb), lw=2.5, color=col, label=f"b={bb:.2f}")
        t, x, eta = ascent_instanton(1.0, bb, T=20.0, npts=6000)
        ax[1, 1].plot(t, eta, lw=2.5, color=col, label=f"b={bb:.2f}")
    peak_b = [ascent_instanton(1.0, bb, T=20.0, npts=6000)[2].max()
              for bb in b_list]
    ax[1, 2].plot(b_list, peak_b, "k", lw=3, label="numeric")
    ax[1, 2].plot(b_list, 2.0 * (bc - b_list), "--", color="firebrick", lw=2,
                  label=r"$2(b_c-b)$")
    ax[1, 2].axvline(bc, color="gray", ls=":", lw=1.5)
    ax[1, 0].set(xlabel="x", ylabel="V(x)", title="Potential vs b  (a=1)")
    ax[1, 1].set(xlabel="time from peak", ylabel="η*(t)", xlim=(-8, 8),
                 title="Noise pulse vs b")
    ax[1, 2].set(xlabel="b (tilt)", ylabel="peak η*", title="Peak noise vs b")

    for r in (0, 1):
        for c in (0, 1, 2):
            ax[r, c].legend(frameon=False, fontsize=9)
            ax[r, c].spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


# =====================================================================
#  SECTION C -- noise vs stimulus vs joint (additive, no kick)
# =====================================================================
def joint_control_alpha(a=1.0, D=0.1, alpha=0.5, T=12.0, npts=2000):
    """Additive noise+stimulus optimum  (alpha = stimulus diffusivity).

    Cost  J = INT [ eta^2/(4D) + b^2/(4 alpha) ] dt ,  x_dot = f + b + eta.
    Pontryagin gives  eta* = -2D p ,  b* = -2 alpha p ,  p_dot = -p f'(x).
    The TOTAL drive  U = eta + b = -2(D+alpha) p  equals the reduced instanton
    drive (U = x_dot - f), INDEPENDENT of the split, so the trajectory x(t) is
    the same for every alpha; only the allocation changes:

        eta* = D U/(D+alpha) ,  b* = alpha U/(D+alpha) ,  p = -U/(2(D+alpha)) .

    Returns t (crossing at 0), x, eta, p, b.
    """
    t, x, U, C = reduced_escape(a, 0.0, T=T, npts=npts, full=True)
    p = -U / (2.0 * (D + alpha))
    eta = D * U / (D + alpha)
    b = alpha * U / (D + alpha)
    return t, x, eta, p, b


def fig3_control(a=1.0, D=0.1, alphas=(0.3, 2.0), T=12.0):
    """Noise+stimulus optimal control, two alphas (rows).  Cols: x, eta, p, b.

    Same trajectory x(t) for both alphas (additive -> no kick); the stimulus
    takes a fraction alpha/(D+alpha) of the total drive, the noise the rest.
    """
    nrow = len(alphas)
    fig, ax = plt.subplots(nrow, 4, figsize=(14, 3.3 * nrow), squeeze=False)
    cols = ["x(t)", "η(t)", "p(t)", "b(t)"]
    colors = ["k", "firebrick", "steelblue", "seagreen"]
    for i, alpha in enumerate(alphas):
        t, x, eta, p, b = joint_control_alpha(a, D=D, alpha=alpha, T=T)
        frac = alpha / (D + alpha)
        for c, (y, col) in enumerate(zip((x, eta, p, b), colors)):
            ax[i, c].plot(t, y, col, lw=3)
            ax[i, c].axhline(0.0, color="gray", ls=":", lw=1)
            ax[i, c].set(xlim=(-6, 6), xlabel="time from crossing")
            if i == 0:
                ax[i, c].set_title(cols[c])
            ax[i, c].spines[["right", "top"]].set_visible(False)
        ax[i, 0].set_ylabel(f"α={alpha}\n(stim {frac*100:.0f}%)")
    fig.tight_layout()
    return fig


# =====================================================================
#  SECTION D -- stochastic validation: switch-triggered noise kernel
# =====================================================================
#  Let noise alone drive spontaneous switches of
#        x_dot = f(x) + eta ,   f = a(x - x^3) + b ,   <eta eta'> = 2 D delta.
#  Since the stimulus control is off (b(t) = 0), the injected drive is the pure
#  noise, chi(t) = eta(t).  Weak-noise theory predicts the noise conditioned on
#  a switch converges to the optimal escape drive:  <chi | switch> -> -2 f.
#  (b here is a STATIC bias baked into the potential, not a control.)
# =====================================================================
def _instanton_time_aligned(a=1.0, b=0.0, dt=0.005, tmax=60.0):
    """Instanton ascent x_dot = -f; returns tau (<=0, crossing at 0) and -2f."""
    r = wells_and_barrier(a, b)
    xL, xbar = r[0], r[1]
    x, xs, n = xL + 1e-3, [], 0
    xs.append(x)
    while x < xbar - 1e-3 and n < tmax / dt:
        x = x - f(x, a, b) * dt
        xs.append(x)
        n += 1
    xs = np.asarray(xs)
    tau = np.arange(xs.size) * dt
    tau -= tau[-1]
    return tau, -2.0 * f(xs, a, b)


def simulate_switch_kernel(a=1.0, b=0.0, D=None, dt=0.01, steps=120_000,
                           nwalk=200, wt=8.0, seed=0):
    """Euler--Maruyama ensemble; switch-triggered average of the noise.

    Returns dict with the time kernel <chi>(tau) aligned at the barrier
    crossing, the conditional field <chi | x> on the ascent, and counts.
    D defaults to 0.08*a (fixes the barrier/noise ratio across a).
    """
    if D is None:
        D = 0.04 * a
    r = wells_and_barrier(a, b)
    xL, xbar, xR = r[0], r[1], r[-1]
    thrL = xbar - 0.5 * (xbar - xL)
    thrR = xbar + 0.5 * (xR - xbar)
    W = int(round(wt / dt))

    rng = np.random.default_rng(seed)
    sq = np.sqrt(2.0 * D * dt)
    X = np.empty((nwalk, steps), np.float32)
    E = np.empty((nwalk, steps), np.float32)
    x = np.full(nwalk, xL, np.float64)
    for i in range(steps):
        xi = rng.standard_normal(nwalk)
        X[:, i] = x
        E[:, i] = sq * xi / dt                  # chi = eta = sqrt(2D/dt) xi
        x = x + f(x, a, b) * dt + sq * xi
        np.clip(x, -3.0, 3.0, out=x)

    committed = np.full(nwalk, -1, np.int8)      # start in left basin
    recs = []
    for i in range(steps):
        xi = X[:, i]
        committed[xi < thrL] = -1
        newsw = (xi > thrR) & (committed == -1)
        for w in np.nonzero(newsw)[0]:
            j, lo = i, max(0, i - W)
            while j > lo and X[w, j] > xbar:
                j -= 1
            recs.append((w, j))
        committed[xi > thrR] = 1

    # committor drift map  2D q'/q(x)  precomputed on the ascent, for the
    # time-kernel overlay  <2D q'/q>(tau)
    xg_c = np.linspace(xL + 1e-3, xbar - 1e-3, 300)
    cg = committor_eta(a, b, D, xg_c)

    def cmap(xx):
        return np.interp(xx, xg_c, cg, left=cg[0], right=0.0)

    ker_sum = np.zeros(2 * W)
    ker_sq = np.zeros(2 * W)
    fker_sum = np.zeros(2 * W)
    cker_sum = np.zeros(2 * W)
    cnt = 0
    xacc, eacc = [], []
    x_react = xL + 0.1 * (xbar - xL)         # near-well threshold for the climb
    for (w, j) in recs:
        if j - W < 0 or j + W >= steps:
            continue
        segE = E[w, j - W:j + W].astype(np.float64)
        segX = X[w, j - W:j + W].astype(np.float64)
        ker_sum += segE
        ker_sq += segE ** 2
        fker_sum += f(segX, a, b)
        cker_sum += cmap(segX)
        cnt += 1
        # <chi|x>: use only the REACTIVE climb (from the last exit near the
        # well up to the crossing), else equilibrium rattling dilutes it.
        # Drop the first few (launch kick) and last (crossing kick) steps.
        s = j
        while s > j - W and X[w, s] >= x_react:
            s -= 1
        lo, hi = s + 3, j - 2
        if hi > lo:
            xacc.append(X[w, lo:hi].astype(np.float64))
            eacc.append(E[w, lo:hi].astype(np.float64))

    tau = (np.arange(2 * W) - W) * dt
    ker_mean = ker_sum / max(cnt, 1)
    fker_mean = fker_sum / max(cnt, 1)        # <f(x)>(tau) along reactive paths
    cker_mean = cker_sum / max(cnt, 1)        # <2D q'/q>(tau) committor map
    ker_sem = np.sqrt(np.maximum(ker_sq / max(cnt, 1) - ker_mean ** 2, 0)
                      / max(cnt, 1))
    xall = np.concatenate(xacc) if xacc else np.array([0.0])
    eall = np.concatenate(eacc) if eacc else np.array([0.0])
    xbins = np.linspace(xL, xbar, 41)
    xc = 0.5 * (xbins[1:] + xbins[:-1])
    s_e, _ = np.histogram(xall, xbins, weights=eall)
    s_n, _ = np.histogram(xall, xbins)
    chi_of_x = np.divide(s_e, s_n, out=np.full(xc.size, np.nan), where=s_n > 0)
    # per-bin MEDIAN (robust to the heavy upward tail / kicks)
    from scipy.stats import binned_statistic
    chi_med_of_x = binned_statistic(xall, eall, statistic="median",
                                    bins=xbins)[0]

    return dict(a=a, b=b, D=D, xL=xL, xbar=xbar, xR=xR, n_switch=cnt,
                tau=tau, ker_mean=ker_mean, ker_sem=ker_sem, fker_mean=fker_mean,
                cker_mean=cker_mean, xc=xc, chi_of_x=chi_of_x,
                chi_med_of_x=chi_med_of_x)


def _smooth(y, k=9):
    return y if k < 2 else np.convolve(y, np.ones(k) / k, mode="same")


def _kernel_cache_file():
    return os.path.join(DATADIR, "switch_kernels.npz")


def _run_cells(a_list, frac_list, kd, dt, steps, nwalk, seed):
    cells = {}
    for a in tqdm(a_list):
        for frac in frac_list:
            b = round(frac * spinodal(a), 4)
            # D = kd * a
            c = simulate_switch_kernel(a, b, D=0.03, dt=dt, steps=steps,
                                       nwalk=nwalk,
                                       seed=seed + int(1000 * a + 137 * frac))
            cells[(a, frac)] = c
            print(f"  a={a} b={b} ({frac:.0%} b_c) D={c['D']:.3f}"
                  f" -> {c['n_switch']} switches")
    return cells


def _save_cells(cells):
    blob = {}
    for (a, frac), c in cells.items():
        k = f"a{a}_f{frac}"
        for fld in ("tau", "ker_mean", "ker_sem", "fker_mean", "cker_mean",
                    "xc", "chi_of_x", "chi_med_of_x"):
            blob[f"{k}__{fld}"] = c[fld]
        blob[f"{k}__meta"] = np.array([a, frac, c["b"], c["D"], c["n_switch"],
                                       c["xL"], c["xbar"], c["xR"]])
    np.savez_compressed(_kernel_cache_file(), **blob)
    print("saved", _kernel_cache_file())


def _load_cells():
    blob = np.load(_kernel_cache_file())
    prefixes = {key.rsplit("__", 1)[0] for key in blob.files}
    cells = {}
    for p in prefixes:
        a, frac, b, D, n, xL, xbar, xR = blob[f"{p}__meta"]
        cells[(float(a), float(frac))] = dict(
            a=float(a), b=float(b), D=float(D), n_switch=int(n),
            xL=float(xL), xbar=float(xbar), xR=float(xR),
            tau=blob[f"{p}__tau"], ker_mean=blob[f"{p}__ker_mean"],
            ker_sem=blob[f"{p}__ker_sem"], fker_mean=blob[f"{p}__fker_mean"],
            cker_mean=blob[f"{p}__cker_mean"], xc=blob[f"{p}__xc"],
            chi_of_x=blob[f"{p}__chi_of_x"],
            chi_med_of_x=blob[f"{p}__chi_med_of_x"])
    return cells


def fig4_sims(a_list=(0.5, 1.0, 1.5), frac_list=(0.0,), kd=0.15,
              dt=0.05, steps=10_000_000, nwalk=100, seed=0,
              recompute=False):
    """Validate chi(t) = eta(t) against the instanton on an (a,b) grid.

    Compares the switch-triggered noise to BOTH analytic instantons:
      * reduced  eta*(t) = -2 f(x(t))           (zero-energy, Result 1)
      * full OM  eta*(t) = sqrt(f^2+2Df'+C) - f (Result 2, C = -2D f'(xL))
    Rows = a.  Col 0: time kernel <chi>(tau).  Col 1: conditional <chi|x>.
    Col 2 (only if several tilts requested): <chi|x> vs -2f for tilted wells.

    D = kd * a (raise kd to test the full-OM correction).  Set recompute=False
    to reuse the saved simulations in sim_data/switch_kernels.npz.
    """
    cache = _kernel_cache_file()
    need = [(a, frac) for a in a_list for frac in frac_list]
    if (not recompute) and os.path.exists(cache):
        cells = _load_cells()
        if any(k not in cells for k in need):
            print("cache incomplete -> recomputing")
            cells = _run_cells(a_list, frac_list, kd, dt, steps, nwalk, seed)
            _save_cells(cells)
        else:
            print("loaded", cache)
    else:
        cells = _run_cells(a_list, frac_list, kd, dt, steps, nwalk, seed)
        _save_cells(cells)

    show_tilt = len(frac_list) > 1
    ncols = 3 if show_tilt else 2
    na = len(a_list)
    fig, ax = plt.subplots(na, ncols, figsize=(5 * ncols, 4 * na),
                           squeeze=False)
    for ia, a in enumerate(a_list):
        c0 = cells[(a, 0.0)]
        D = c0["D"]
        tr, er = _instanton_time_aligned(a, 0.0)                # reduced -2f
        tfl, xfl, efl = full_om_instanton(a, 0.0, D=D)          # full OM

        # col 0: time kernel + both analytic instantons (crossing at tau=0)
        axx = ax[ia, 0]
        km = _smooth(c0["ker_mean"])
        axx.axhline(0, color="gray", ls=":", lw=1)
        axx.fill_between(c0["tau"], km - 2 * c0["ker_sem"],
                         km + 2 * c0["ker_sem"], color="0.85")
        axx.plot(c0["tau"], km, "k", lw=2.5, label=r"sim $\langle\chi\rangle$")
        # The absolute launch time of an instanton is arbitrary (set by the
        # escape rate), so shift each analytic pulse to the sim's pulse peak
        # (ignoring the tau=0 triggering spike) and compare SHAPES.
        pre = c0["tau"] < -0.3
        t_peak = c0["tau"][pre][np.argmax(km[pre])]
        tr_s = tr - tr[np.argmax(er)] + t_peak
        tfl_s = tfl - tfl[np.argmax(efl)] + t_peak
        axx.plot(tr_s, er, "firebrick", lw=2.5, ls="--",
                 label=r"reduced $-2f$ (peak-aligned)")
        axx.plot(tfl_s, efl, "steelblue", lw=2.5, ls="--",
                 label="full OM (peak-aligned)")
        # reduced law mapped through the actual reactive ensemble (bounded).
        # (The committor 2Dq'/q is shown in x-space, col 1: mapped to time it
        #  is dominated by its x->xL divergence and the pre-commit equilibrium,
        #  so it does not correspond to <chi>(tau).)
        axx.plot(c0["tau"], -2 * c0["fker_mean"], "darkorange", lw=2,
                 label=r"$-2\langle f\rangle(\tau)$")
        ytop = 1.35 * max(er.max(), efl.max())
        axx.set(xlim=(-8, 3), ylim=(-0.25 * ytop, ytop))
        axx.set_title(f"a={a}, b=0, D={D:.2f}  (n={c0['n_switch']})")
        axx.set_xlabel("time from crossing τ")
        axx.set_ylabel(r"$\langle\chi\rangle(\tau)$")
        if ia == 0:
            axx.legend(frameon=False, fontsize=10)

        # col 1: conditional field <chi|x>: sim vs -2f vs EXACT committor
        axx = ax[ia, 1]
        xx = np.linspace(c0["xL"] + 1e-3, c0["xbar"] - 1e-3, 300)
        axx.plot(xx, -2 * f(xx, a, 0.0), "firebrick", lw=2.5,
                 label=r"reduced $-2f(x)$")
        axx.plot(xx, committor_eta(a, 0.0, D, xx), "seagreen", lw=2.5,
                 label=r"committor $2Dq'/q$")
        axx.plot(c0["xc"], c0["chi_of_x"], "ko", ms=5,
                 label=r"sim mean $\langle\chi|x\rangle$")
        axx.plot(c0["xc"], c0["chi_med_of_x"], "s", ms=4, mfc="none",
                 mec="darkorange", label=r"sim median")
        axx.set_title(f"a={a}, b=0  conditional field")
        axx.set_xlabel("x"); axx.set_ylabel(r"$\langle\chi|x\rangle$")
        axx.set_ylim(-0.3 * efl.max(), 1.7 * efl.max())    # clip near-well edge
        if ia == 0:
            axx.legend(frameon=False, fontsize=10)

        # col 2: tilt effect (only if several tilts requested)
        if show_tilt:
            axx = ax[ia, 2]
            for frac in frac_list:
                if frac == 0.0:
                    continue
                c = cells[(a, frac)]
                col = plt.cm.copper(0.15 + 0.7 * frac / max(frac_list))
                xx = np.linspace(c["xL"], c["xbar"], 200)
                axx.plot(xx, -2 * f(xx, a, c["b"]), lw=2.5, color=col)
                axx.plot(c["xc"], c["chi_of_x"], "o", ms=5, color=col,
                         label=f"b={c['b']:.2f} ({frac:.0%} b_c)")
            axx.set_title(f"a={a}  tilt lowers the kernel")
            axx.set_xlabel("x"); axx.set_ylabel(r"$\langle\chi|x\rangle$")
            axx.legend(frameon=False, title="lines: −2f", fontsize=9)

    for a_ in ax.ravel():
        a_.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


# =====================================================================
#  SECTION E -- Hamiltonian phase portrait  (x, p)
# =====================================================================
def fig5_phase(a=1.0, b=0.0, D=0.1):
    """Phase portrait of the escape Hamiltonian  H = p f - D p^2.

    Canonical (x, p) plane for  x_dot = f + eta ,  eta = -2D p  (sigma^2=2D):
        x_dot =  dH/dp = f - 2D p ,     p_dot = -dH/dx = -p f'(x) .
    The two zero-energy (H=0) manifolds are the noiseless relaxation line
    p = 0 (eta = 0, downhill) and the escape branch p = f/D (eta = -2f,
    uphill).  The optimal switch is the heteroclinic orbit: ascent
    xL -> xbar on p = f/D, then free descent xbar -> xR on p = 0.
    """
    r = wells_and_barrier(a, b)
    xL, xbar, xR = r[0], r[1], r[-1]

    xv = np.linspace(xL - 0.4, xR + 0.4, 400)
    pmax = 1.4 * np.max(np.abs(f(np.linspace(xL, xR, 400), a, b) / D))
    pv = np.linspace(-pmax, pmax, 400)
    X, P = np.meshgrid(xv, pv)
    H = P * f(X, a, b) - D * P ** 2
    dX = f(X, a, b) - 2 * D * P                 # x_dot = dH/dp
    dP = -P * fp(X, a)                          # p_dot = -dH/dx

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.streamplot(xv, pv, dX, dP, color="0.82", density=1.1, linewidth=0.7,
                  arrowsize=0.8)
    ax.contour(X, P, H, levels=np.linspace(-2, 2, 13), colors="0.88",
               linewidths=0.6)

    ax.axhline(0.0, color="steelblue", lw=3)
    ax.plot(xv, f(xv, a, b) / D, color="firebrick", lw=3)

    # optimal switch: ascent xL->xbar on p=f/D, descent xbar->xR on p=0
    xa = np.linspace(xL, xbar, 200)
    ax.plot(xa, f(xa, a, b) / D, color="firebrick", lw=6, alpha=0.5, zorder=1)
    xd = np.linspace(xbar, xR, 120)
    ax.plot(xd, np.zeros_like(xd), color="steelblue", lw=7, alpha=0.5, zorder=1)

    # finite-T orbits H=const (C=-4DH): H<0 cross the barrier (jump wells),
    # H>0 are closed flank loops that turn back without crossing.
    f2max = np.max(f(np.linspace(xL, xR, 400), a, b) ** 2)
    H_esc = sorted([-1.2 * f2max / (4 * D), -0.3 * f2max / (4 * D)])
    ax.contour(X, P, H, levels=H_esc, colors=["seagreen", "darkorange"],
               linewidths=2.2, linestyles="--", zorder=4)
    ax.contour(X, P, H, levels=[0.4 * f2max / (4 * D)], colors=["purple"],
               linewidths=2.0, zorder=4)

    for xs in (xL, xbar, xR):                   # wells + barrier on p=0
        ax.plot(xs, 0.0, "ko", ms=10, zorder=5)

    handles = [
        Line2D([0], [0], color="firebrick", lw=3, label=r"instanton $H=0$ ($\eta=-2f$)"),
        Line2D([0], [0], color="steelblue", lw=3, label=r"relaxation $p=0$ ($\eta=0$)"),
        Line2D([0], [0], color="darkorange", lw=2.2, ls="--",
               label=r"finite-$T$ escape $H<0$ (crosses)"),
        Line2D([0], [0], color="purple", lw=2.0,
               label=r"$H>0$: turns back (no crossing)"),
    ]
    ax.set(xlabel="position x", ylabel="momentum p",
           xlim=(xv[0], xv[-1]), ylim=(-pmax, pmax),
           title=f"Escape Hamiltonian H=pf−Dp²  (a={a}, b={b}, D={D})")
    ax.legend(handles=handles, frameon=False, fontsize=14, loc="upper right")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


# =====================================================================
#  main
# =====================================================================
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    figs = {
        "1_escape.png": fig1_escape(a=1.0, b=0.0, D=0.1),
        "2_params.png": fig2_params(),
        "3_control.png": fig3_control(a=1.0, D=0.1, alphas=(0.3, 2.0)),
        "4_sims.png": fig4_sims(recompute=False),
        "5_phase.png": fig5_phase(a=1.0, b=0.0, D=0.1),
    }
    for name, fig in figs.items():
        path = os.path.join(FIGDIR, name)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("saved", path)
    # plt.close("all")


if __name__ == "__main__":
    main()
