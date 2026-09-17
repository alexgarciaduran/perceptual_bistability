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
3. NOISE-DRIVEN vs STIMULUS-DRIVEN  (joint optimum, Section C)
--------------------------------------------------------------------------
Minimise the control cost   J = 1/2 INT ( eta^2 + b^2 / kappa ) dt
subject to  x_dot = a(x - x^3) + b + eta ,  x(0) = -1, x(T) = +1.

Pontryagin (costate p):
        eta* = -p ,      b* = -kappa p = kappa * eta* ,
        x_dot = a(x - x^3) - (1 + kappa) p ,
        p_dot = -p * a (1 - 3 x^2) .

kappa is the STIMULUS GAIN (how cheap the stimulus is relative to noise):
        kappa = 0    -> pure noise-driven (b = 0, eta* = -2f recovered),
        kappa -> inf -> pure stimulus-driven (eta -> 0),
        0 < kappa    -> joint; stimulus supplies a fraction kappa/(1+kappa)
                        of the drive, noise the rest, SAME shape -> no kick.

Run  ``python optimal_escape_noise.py``  to regenerate every figure.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, quad, cumulative_trapezoid
from scipy.optimize import root_scalar

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


def figA_compare(a=1.0, b=0.0, D=0.05, T_long=14.0, T_short=4.0):
    """Reduced (C~0), reduced (finite-T) and full-OM on the -1 -> +1 escape."""
    tr, xr, etar, Cr = reduced_escape(a, b, T=T_long)
    tf, xf, etaf, Cf = reduced_escape(a, b, T=T_short)
    tb, xb, etab, ok = full_functional(a, b, D=D, T=T_long)

    xv = np.linspace(-1.6, 1.6, 300)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].plot(xv, V(xv, a, b), "k", lw=2.5)
    for xm in wells_and_barrier(a, b):
        ax[0, 0].plot(xm, V(xm, a, b), "ro")
    ax[0, 0].set(xlabel="x", ylabel="V(x)",
                 title=f"Tilted quartic  (a={a}, b={b}, ΔV={a/4:.3f})")

    ax[0, 1].plot(tr, xr, "k", lw=3, label=f"reduced, C≈0 (T={T_long:g})")
    ax[0, 1].plot(tf, xf, "--", color="peru", lw=2.5,
                  label=f"reduced, T={T_short:g} (C={Cf:.3f})")
    if ok:
        ax[0, 1].plot(tb, xb, ":", color="firebrick", lw=2.5,
                      label=f"full OM (D={D})")
    ax[0, 1].axhline(0.0, color="gray", ls=":", lw=1)
    ax[0, 1].set(xlabel="time from crossing", ylabel="x(t)", xlim=(-7, 7),
                 title="Optimal escape path  (-1 → +1)")
    ax[0, 1].legend(frameon=False, fontsize=8)

    ax[1, 0].plot(tr, etar, "k", lw=3, label="reduced C≈0")
    ax[1, 0].plot(tf, etaf, "--", color="peru", lw=2.5, label=f"reduced T={T_short:g}")
    if ok:
        ax[1, 0].plot(tb, etab, ":", color="firebrick", lw=2.5, label=f"full OM (D={D})")
    ax[1, 0].axhline(0.0, color="gray", ls=":", lw=1)
    ax[1, 0].set(xlabel="time from crossing", ylabel="η*(t)", xlim=(-7, 7),
                 title="Optimal noise η*(t):  pulse up, free fall down")
    ax[1, 0].legend(frameon=False, fontsize=8)

    ax[1, 1].plot(xr, etar, "k", lw=3, label="reduced C≈0")
    ax[1, 1].plot(xf, etaf, "--", color="peru", lw=2.5, label=f"reduced T={T_short:g}")
    if ok:
        ax[1, 1].plot(xb, etab, ":", color="firebrick", lw=2.5, label=f"full OM (D={D})")
    xa = np.linspace(wells_and_barrier(a, b)[0], 0, 100)
    ax[1, 1].plot(xa, -2 * f(xa, a, b), color="steelblue", lw=1.5, alpha=0.7,
                  label="-2f(x) (ascent)")
    ax[1, 1].axhline(0.0, color="gray", ls=":", lw=1)
    ax[1, 1].axvline(0.0, color="gray", ls=":", lw=1)
    ax[1, 1].set(xlabel="x", ylabel="η*(x)", title="Optimal noise vs position")
    ax[1, 1].legend(frameon=False, fontsize=8)

    for a_ in ax.ravel():
        a_.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


# =====================================================================
#  SECTION B -- dependence on the potential parameters a (barrier) and b (tilt)
# =====================================================================
def figB_param_dependence(a_list=np.arange(0.1, 2.01, 0.02),
                           b_list=None):
    """Left column: peak / latency / width of eta* vs barrier a (at b = 0).
    Right column: peak eta* and escape cost vs tilt b (at a = 1), showing the
    stimulus lowering the noise needed until the well vanishes at b_c.
    """
    if b_list is None:
        b_list = np.linspace(0.0, 0.99 * spinodal(1.0), 60)

    # --- vs a (b = 0) -------------------------------------------------
    peak, peak_an, auc = [], [], []
    for a in a_list:
        Ta = max(6.0, 8.0 / a)
        t, x, eta = ascent_instanton(a, 0.0, T=Ta, npts=4000)
        peak.append(eta.max())
        peak_an.append(4.0 * a / (3.0 * np.sqrt(3.0)))     # 2 f(-1/sqrt3)=4a/3√3
        auc.append(np.trapz(eta, t))                    # unnormalized width~1/a

    # --- vs b (a = 1) -------------------------------------------------
    # analytic: peak eta = -2 f(-1/sqrt3) = 2 (b_c - b);  action = 2 (V_bar - V_L)
    bc = spinodal(1.0)
    peak_b, peak_b_an, action_b = [], [], []
    for b in b_list:
        _, _, eta = ascent_instanton(1.0, b, T=max(8.0, 20.0), npts=6000)
        peak_b.append(eta.max())
        peak_b_an.append(2.0 * (bc - b))
        r = wells_and_barrier(1.0, b)
        action_b.append(2.0 * (V(r[1], 1.0, b) - V(r[0], 1.0, b)))  # 2 ΔV_left

    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))

    ax[0, 0].plot(a_list, peak, "k", lw=3, label="reduced C=0")
    ax[0, 0].plot(a_list, peak_an, "--", color="gray", lw=2,
                  label=r"$4a/(3\sqrt{3})$")
    ax[0, 0].set(xlabel="a", ylabel="peak η*", title="Peak drive vs a  (∝ a)")
    ax[0, 0].legend(frameon=False, fontsize=8)

    # pulse shape for several a: amplitude grows (∝a), width shrinks (∝1/a)
    for a in (0.3, 0.6, 1.0, 1.5):
        t, x, eta = ascent_instanton(a, 0.0, T=max(6.0, 8.0 / a), npts=4000)
        ax[0, 1].plot(t, eta, lw=2.5, color=plt.cm.viridis(a / 1.7),
                      label=f"a={a}")
    ax[0, 1].set(xlabel="time from peak", ylabel="η*(t)", xlim=(-8, 8),
                 title="Pulse shape vs a")
    ax[0, 1].legend(frameon=False, fontsize=8)

    ax[0, 2].plot(a_list, auc, "k", lw=3, label="numeric")
    ax[0, 2].axhline(2.0, color="gray", ls="--", lw=2,
                     label=r"$2\,\Delta x=2$")
    ax[0, 2].set(xlabel="a", ylabel=r"$\int \eta^*\,dt$", ylim=(1.9, 2.1),
                 title="Pulse area = impulse invariant")
    ax[0, 2].legend(frameon=False, fontsize=8)

    ax[1, 0].plot(b_list, peak_b, "firebrick", lw=3, label="numeric")
    ax[1, 0].plot(b_list, peak_b_an, "--", color="gray", lw=2,
                  label=r"$2(b_c-b)$")
    ax[1, 0].axvline(bc, color="gray", ls=":", lw=1.5)
    ax[1, 0].set(xlabel="b (stimulus)", ylabel="peak η*",
                 title="Noise needed shrinks with tilt")
    ax[1, 0].legend(frameon=False, fontsize=8)
    ax[1, 1].plot(b_list, action_b, "firebrick", lw=3)
    ax[1, 1].axvline(bc, color="gray", ls=":", lw=1.5)
    ax[1, 1].set(xlabel="b (stimulus)", ylabel=r"$2\,\Delta V_{\rm left}$",
                 title="Escape action vs tilt")

    # potential tilting with b
    xv = np.linspace(-1.6, 1.6, 300)
    for bb in np.linspace(0, bc, 5):
        ax[1, 2].plot(xv, V(xv, 1.0, bb), lw=2,
                      color=plt.cm.copper(bb / bc), label=f"b={bb:.2f}")
    ax[1, 2].set(xlabel="x", ylabel="V(x)", title="Potential vs tilt b")
    ax[1, 2].legend(frameon=False, fontsize=7)

    for a_ in ax.ravel():
        a_.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


# =====================================================================
#  SECTION C -- noise vs stimulus vs joint (additive, no kick)
# =====================================================================
def joint_control(a=1.0, kappa=1.0, T=12.0, npts=2000):
    """Joint noise+stimulus optimum for the additive control.

    Substituting the total drive  U = eta + b = -(1+kappa) p  in the Pontryagin
    equations gives a system INDEPENDENT of kappa:

        x_dot = a(x - x^3) + U ,   U_dot = -U f'(x) ,

    which is exactly the reduced instanton (U = sqrt(f^2 + C) - f).  Hence the
    optimal TRAJECTORY and total drive do not depend on the noise/stimulus
    split; only the allocation does:

        eta* = U / (1 + kappa) ,   b* = kappa U / (1 + kappa) = kappa * eta* .

    Both channels have the SAME shape, so the escape path is smooth for every
    kappa (no kick).  Cost  J = 1/2 INT U^2 / (1 + kappa)  falls as stimulus
    gets cheaper.  Returns t (aligned at the crossing), x, eta, b, cost.
    """
    t, x, U, C = reduced_escape(a, 0.0, T=T, npts=npts, full=True)
    eta = U / (1.0 + kappa)
    b = kappa * U / (1.0 + kappa)
    cost = np.trapz(U ** 2 / (1.0 + kappa), t)
    return t, x, eta, b, cost, True


def figC_noise_vs_stimulus(a=1.0, kappas=(0.0, 0.5, 2.0, 20.0), T=12.0):
    """Sweep the stimulus gain kappa from pure-noise to stimulus-dominated.

    Trajectories stay smooth throughout (no kick): eta and b are proportional
    because both act additively on the drift.
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(kappas)))

    # trajectory + total drive are the same for every kappa -> draw once
    t0, x0, eta0, b0, _, _ = joint_control(a, kappa=kappas[0], T=T)
    U = eta0 + b0
    ax[0].plot(t0, x0, "k", lw=3)
    ax[0].axhline(0.0, color="gray", ls=":", lw=1)
    ax[0].set(xlabel="time from crossing", ylabel="x(t)", xlim=(-6, 6),
              title="Escape trajectory\n(same for every κ → smooth, no kick)")

    for chan in (1, 2):
        ax[chan].plot(t0, U, "--", color="gray", lw=1.5,
                      label=r"total drive $\eta+b$")

    for kap, col in zip(kappas, colors):
        t, x, eta, b, cost, ok = joint_control(a, kappa=kap, T=T)
        frac = kap / (1 + kap)
        lab = ("pure noise (κ=0)" if kap == 0
               else f"κ={kap:g} (stim {frac*100:.0f}%)")
        ax[1].plot(t, eta, lw=2.5, color=col, label=lab)
        ax[2].plot(t, b, lw=2.5, color=col, label=lab)

    ax[1].set(xlabel="time from crossing", ylabel="η(t)", xlim=(-6, 6),
              title="Noise channel  η* = U/(1+κ)")
    ax[1].legend(frameon=False, fontsize=7)
    ax[2].set(xlabel="time from crossing", ylabel="b(t)", xlim=(-6, 6),
              title="Stimulus channel  b* = κU/(1+κ)")
    ax[2].legend(frameon=False, fontsize=7)
    for a_ in ax:
        a_.spines[["right", "top"]].set_visible(False)
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

    ker_sum = np.zeros(2 * W)
    ker_sq = np.zeros(2 * W)
    fker_sum = np.zeros(2 * W)
    cnt = 0
    xacc, eacc = [], []
    x_react = xL + 0.1 * (xbar - xL)         # near-well threshold for the climb
    for (w, j) in recs:
        if j - W < 0 or j + W >= steps:
            continue
        seg = E[w, j - W:j + W].astype(np.float64)
        ker_sum += seg
        ker_sq += seg ** 2
        fker_sum += f(X[w, j - W:j + W].astype(np.float64), a, b)
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
    ker_sem = np.sqrt(np.maximum(ker_sq / max(cnt, 1) - ker_mean ** 2, 0)
                      / max(cnt, 1))
    xall = np.concatenate(xacc) if xacc else np.array([0.0])
    eall = np.concatenate(eacc) if eacc else np.array([0.0])
    xbins = np.linspace(xL, xbar, 41)
    xc = 0.5 * (xbins[1:] + xbins[:-1])
    s_e, _ = np.histogram(xall, xbins, weights=eall)
    s_n, _ = np.histogram(xall, xbins)
    chi_of_x = np.divide(s_e, s_n, out=np.full(xc.size, np.nan), where=s_n > 0)

    return dict(a=a, b=b, D=D, xL=xL, xbar=xbar, xR=xR, n_switch=cnt,
                tau=tau, ker_mean=ker_mean, ker_sem=ker_sem, fker_mean=fker_mean,
                xc=xc, chi_of_x=chi_of_x)


def _smooth(y, k=9):
    return y if k < 2 else np.convolve(y, np.ones(k) / k, mode="same")


def _kernel_cache_file():
    return os.path.join(DATADIR, "switch_kernels.npz")


def _run_cells(a_list, frac_list, kd, dt, steps, nwalk, seed):
    cells = {}
    for a in a_list:
        for frac in frac_list:
            b = round(frac * spinodal(a), 4)
            c = simulate_switch_kernel(a, b, D=kd * a, dt=dt, steps=steps,
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
        for fld in ("tau", "ker_mean", "ker_sem", "xc", "chi_of_x"):
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
            ker_sem=blob[f"{p}__ker_sem"], xc=blob[f"{p}__xc"],
            chi_of_x=blob[f"{p}__chi_of_x"])
    return cells


def figD_sim_vs_theory(a_list=(0.5, 1.0, 1.5), frac_list=(0.0,), kd=0.15,
                       dt=0.01, steps=200_000, nwalk=1000, seed=0,
                       recompute=True):
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
        axx.plot(tr, er, "firebrick", lw=2.5, ls="--", label=r"reduced $-2f(x(t))$")
        axx.plot(tfl, efl, "steelblue", lw=2.5, ls="--", label="full OM")
        ytop = 1.35 * max(er.max(), efl.max())
        axx.set(xlim=(-8, 3), ylim=(-0.25 * ytop, ytop))
        axx.set_title(f"a={a}, b=0, D={D:.2f}  (n={c0['n_switch']})")
        axx.set_xlabel("time from crossing τ")
        axx.set_ylabel(r"$\langle\chi\rangle(\tau)$")
        if ia == 0:
            axx.legend(frameon=False, fontsize=10)

        # col 1: conditional field <chi|x> + both analytic instantons
        axx = ax[ia, 1]
        xx = np.linspace(c0["xL"], c0["xbar"], 200)
        axx.plot(xx, -2 * f(xx, a, 0.0), "firebrick", lw=2.5, label=r"reduced $-2f(x)$")
        axx.plot(xfl, efl, "steelblue", lw=2.5, label="full OM")
        axx.plot(c0["xc"], c0["chi_of_x"], "ko", ms=5, label=r"sim $\langle\chi|x\rangle$")
        axx.set_title(f"a={a}, b=0  conditional field")
        axx.set_xlabel("x"); axx.set_ylabel(r"$\langle\chi|x\rangle$")
        axx.set_ylim(-0.3 * efl.max(), 1.4 * efl.max())    # clip residual edge kicks
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
#  main
# =====================================================================
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    figs = {
        "A_quartic_compare.png": figA_compare(a=1.0, b=0.0, D=0.05),
        "B_param_dependence.png": figB_param_dependence(),
        "C_noise_vs_stimulus.png": figC_noise_vs_stimulus(a=1.0),
        "D_sim_vs_theory.png": figD_sim_vs_theory(),
    }
    for name, fig in figs.items():
        path = os.path.join(FIGDIR, name)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("saved", path)
    # plt.close("all")


if __name__ == "__main__":
    main()
