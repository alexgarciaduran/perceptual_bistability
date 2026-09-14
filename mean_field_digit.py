# -*- coding: utf-8 -*-
r"""
Two-unit mean-field system with cross coupling J (J<0), logistic gain sigma:

    x = sigma( J (1 - y) + B_x )
    y = sigma( J (1 - x) + B_y ),      sigma(z) = 1/(1+e^{-z}).

Read as the continuous mean-field relaxation
    xdot = -x + sigma(J(1-y)+B_x),
    ydot = -y + sigma(J(1-x)+B_y),
whose fixed points are the self-consistent solutions above. This script computes
the fixed points (numerically by multi-start, and the symmetric branch
analytically), the Jacobian eigenvalues (analytic and numeric), sweeps J to draw
the bifurcation diagram, and estimates the stationary density from long Langevin
simulations. Default B_x=B_y=0.

Note on the coupling: J(1-y) = |J| y + J for J<0, i.e. an EFFECTIVE positive
(cooperative) coupling |J| between the units plus a negative self-bias J. The
linearised interaction is d(xdot)/dy = -J sigma' > 0, so the two units are
positively coupled; the symmetric mode (1,1) is the one that can lose stability.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

OUT = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/mean_field_digits/'  # Alex
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigma(z):
    s = sigma(z)
    return s * (1.0 - s)


def rhs(v, J, Bx=0.0, By=0.0):
    """Continuous mean-field vector field (xdot, ydot)."""
    x, y = v
    return np.array([-x + sigma(J * (1.0 - y) + Bx),
                     -y + sigma(J * (1.0 - x) + By)])


def fp_residual(v, J, Bx=0.0, By=0.0):
    """Zero at a fixed point (same as rhs, phrased for a root finder)."""
    x, y = v
    return [sigma(J * (1.0 - y) + Bx) - x,
            sigma(J * (1.0 - x) + By) - y]


def jacobian(x, y, J, Bx=0.0, By=0.0):
    """Analytic Jacobian of the continuous dynamics at (x,y)."""
    ux = J * (1.0 - y) + Bx
    uy = J * (1.0 - x) + By
    return np.array([[-1.0,            -J * dsigma(ux)],
                     [-J * dsigma(uy), -1.0]])


# ----------------------------------------------------------------------------
# Fixed points
# ----------------------------------------------------------------------------
def fixed_points(J, Bx=0.0, By=0.0, grid=13, tol=1e-4):
    """All fixed points in [0,1]^2 by multi-start root finding (deduplicated)."""
    sols = []
    for x0 in np.linspace(0.0, 1.0, grid):
        for y0 in np.linspace(0.0, 1.0, grid):
            s, _, flag, _ = fsolve(fp_residual, [x0, y0], args=(J, Bx, By), full_output=True)
            if flag == 1 and -1e-6 <= s[0] <= 1 + 1e-6 and -1e-6 <= s[1] <= 1 + 1e-6:
                if not any(np.allclose(s, t, atol=tol) for t in sols):
                    sols.append(np.clip(s, 0, 1))
    return np.array(sols) if sols else np.empty((0, 2))


def symmetric_fp_analytic(J, B=0.0):
    """Symmetric fixed point(s) m = sigma(J(1-m)+B) on the diagonal x=y=m, found as
    roots of h(m)=sigma(J(1-m)+B)-m on [0,1] via sign-change bracketing + brentq."""
    h = lambda m: sigma(J * (1.0 - m) + B) - m
    ms = np.linspace(0.0, 1.0, 2001)
    hv = h(ms)
    roots = []
    for i in np.flatnonzero(np.sign(hv[:-1]) != np.sign(hv[1:])):
        roots.append(brentq(h, ms[i], ms[i + 1], xtol=1e-12))
    return np.array(roots)


def eig_analytic_symmetric(m, J):
    """Analytic Jacobian eigenvalues at a symmetric fixed point x=y=m (B arbitrary,
    enters only through m). p = sigma' = m(1-m):
        lambda_sym  (mode (1,1))  = -1 - J p
        lambda_anti (mode (1,-1)) = -1 + J p .
    For J<0 (-J=|J|): lambda_sym = -1 + |J|p (destabilises first), lambda_anti<0."""
    p = m * (1.0 - m)
    return np.array([-1.0 - J * p, -1.0 + J * p])   # (sym (1,1), anti (1,-1))


def classify(x, y, J, Bx=0.0, By=0.0):
    ev = np.linalg.eigvals(jacobian(x, y, J, Bx, By))
    return ev, bool(np.all(ev.real < 0))


# ----------------------------------------------------------------------------
# Langevin simulation -> stationary density
# ----------------------------------------------------------------------------
def simulate(J, Bx=0.0, By=0.0, D=0.01, dt=0.02, T=20000.0, seed=0, v0=None):
    """Euler-Maruyama on vdot = rhs(v) + sqrt(2D) xi, reflecting at [0,1]^2.
    Returns the (x,y) trajectory (post short burn-in)."""
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    s = np.sqrt(2.0 * D * dt)
    x, y = (rng.random(2) if v0 is None else np.asarray(v0, float))
    xs = np.empty(n); ys = np.empty(n)
    for t in range(n):
        dx, dy = rhs((x, y), J, Bx, By)
        x += dx * dt + s * rng.standard_normal()
        y += dy * dt + s * rng.standard_normal()
        # reflect into the unit square
        if x < 0: x = -x
        elif x > 1: x = 2 - x
        if y < 0: y = -y
        elif y > 1: y = 2 - y
        xs[t] = x; ys[t] = y
    b = n // 20
    return xs[b:], ys[b:]


# ----------------------------------------------------------------------------
# Analyses / figures
# ----------------------------------------------------------------------------
def report_pointwise(J_list=(-8.0, -4.0, -2.0, 2.0, 4.0), B=0.0):
    """Print analytic-vs-numeric fixed points and eigenvalues."""
    print(f"\n=== fixed points & eigenvalues (B_x=B_y={B}) ===")
    for J in J_list:
        print(f"\nJ = {J:+.2f}")
        ms = symmetric_fp_analytic(J, B)
        for m in ms:
            ev_a = eig_analytic_symmetric(m, J)
            ev_n, stab = classify(m, m, J, B, B)
            print(f"  symmetric m={m:.5f} | eig analytic (sym,anti)="
                  f"({ev_a[0]:+.4f}, {ev_a[1]:+.4f})  numeric={np.sort(ev_n.real)}"
                  f"  {'stable' if stab else 'UNSTABLE'}")
        fps = fixed_points(J, B, B)
        asym = [s for s in fps if abs(s[0] - s[1]) > 1e-3]
        for s in asym:
            ev_n, stab = classify(s[0], s[1], J, B, B)
            print(f"  asymmetric (x,y)=({s[0]:.4f},{s[1]:.4f}) eig={np.round(ev_n.real,4)}"
                  f"  {'stable' if stab else 'UNSTABLE'}")
        print(f"  total fixed points found: {len(fps)}")


def plot_bifurcation(J_sweep=np.linspace(-10, 6, 321), B=0.0, save=True):
    """Bifurcation diagram: fixed-point coordinates vs J, coloured by stability;
    plus the analytic symmetric branch and the |J|p=1 / Jp=1 onset lines."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for J in J_sweep:
        fps = fixed_points(J, B, B)
        for s in fps:
            _, stab = classify(s[0], s[1], J, B, B)
            c = 'k' if stab else '0.7'
            m = 'o' if stab else 'x'
            ax[0].plot(J, s[0], m, color=c, ms=2.5, mew=0.5)          # x-coordinate
            ax[0].plot(J, s[1], m, color=c, ms=2.5, mew=0.5)          # y-coordinate
            ax[1].plot(J, abs(s[0] - s[1]), m, color=c, ms=2.5, mew=0.5)  # asymmetry
    ax[0].set(xlabel='coupling J', ylabel='fixed-point coords x*, y*',
              title=f'Bifurcation diagram (B={B})')
    ax[1].set(xlabel='coupling J', ylabel='|x* - y*|  (asymmetry)',
              title='competition order parameter')
    # analytic symmetric branch overlay
    Js = np.linspace(J_sweep[0], J_sweep[-1], 400)
    for J in Js:
        for m in symmetric_fp_analytic(J, B):
            ax[0].plot(J, m, '.', color='tab:red', ms=1.5, zorder=0)
    for a in ax:
        a.axvline(0, color='0.85', lw=0.8)
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    ax[0].plot([], [], 'ko', ms=4, label='stable'); ax[0].plot([], [], 'x', color='0.7', label='unstable')
    ax[0].plot([], [], '.', color='tab:red', label='analytic symmetric')
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(OUT, 'bifurcation.png'), dpi=160, bbox_inches='tight')
    return fig


def plot_eigs_vs_J(J_sweep=np.linspace(-10, 6, 400), B=0.0, save=True):
    """Analytic eigenvalues of the symmetric branch vs J (sym & anti modes)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    Js, lam_s, lam_a, mm = [], [], [], []
    for J in J_sweep:
        for m in symmetric_fp_analytic(J, B):
            ev = eig_analytic_symmetric(m, J)
            Js.append(J); lam_s.append(ev[0]); lam_a.append(ev[1]); mm.append(m)
    Js = np.array(Js)
    ax.plot(Js, lam_s, '.', ms=2, color='tab:blue', label=r'$\lambda_{sym}$ (mode $(1,1)$)')
    ax.plot(Js, lam_a, '.', ms=2, color='tab:orange', label=r'$\lambda_{anti}$ (mode $(1,-1)$)')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='0.85', lw=0.8)
    ax.set(xlabel='coupling J', ylabel='eigenvalue', title=f'Symmetric-branch eigenvalues (B={B})')
    ax.legend(frameon=False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(OUT, 'eigenvalues.png'), dpi=160, bbox_inches='tight')
    return fig


def plot_density(J_list=(-6.0, -2.0, 3.0), B=0.0, D=0.05, T=50000.0, save=True):
    """Stationary density (2D histogram + x-marginal) from long Langevin runs,
    with fixed points overlaid."""
    fig, axes = plt.subplots(2, len(J_list), figsize=(4.0 * len(J_list), 7),
                             squeeze=False)
    for c, J in enumerate(J_list):
        xs, ys = simulate(J, B, B, D=D, T=T, seed=0)
        H, xe, ye = np.histogram2d(xs, ys, bins=80, range=[[0, 1], [0, 1]], density=True)
        axes[0][c].imshow(H.T, origin='lower', extent=[0, 1, 0, 1], aspect='auto',
                          cmap='magma')
        for s in fixed_points(J, B, B):
            _, stab = classify(s[0], s[1], J, B, B)
            axes[0][c].plot(s[0], s[1], 'o', mfc=('c' if stab else 'none'),
                            mec='c', ms=8, mew=1.5)
        axes[0][c].set(title=f'J={J}  density p(x,y)', xlabel='x', ylabel='y')
        axes[1][c].hist(xs, bins=100, range=(0, 1), density=True, color='0.3')
        axes[1][c].set(xlabel='x', ylabel='p(x)', title='x-marginal')
        for a in (axes[0][c], axes[1][c]):
            a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.suptitle(f'Long-simulation stationary densities (B={B}, D={D}, T={T:g})')
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(OUT, 'densities.png'), dpi=160, bbox_inches='tight')
    return fig


def plot_phase_portrait(J, B=0.0, save=True):
    """Nullclines + vector field + fixed point(s) with eigenvector directions, for
    the continuous system xdot=-x+sigma(J(1-y)+B), ydot=-y+sigma(J(1-x)+B)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    g = np.linspace(0, 1, 400)
    ax.plot(sigma(J * (1 - g) + B), g, color='tab:blue', lw=2, label='x-nullcline  $\\dot x=0$')
    ax.plot(g, sigma(J * (1 - g) + B), color='tab:orange', lw=2, label='y-nullcline  $\\dot y=0$')
    X, Y = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    U = -X + sigma(J * (1 - Y) + B); V = -Y + sigma(J * (1 - X) + B)
    ax.quiver(X, Y, U, V, color='0.7', width=0.003, scale=8)
    for s in fixed_points(J, B, B):
        ev, stab = classify(s[0], s[1], J, B, B)
        ax.plot(*s, 'o', mfc=('k' if stab else 'none'), mec='k', ms=10, mew=1.5, zorder=5)
        w, V2 = np.linalg.eig(jacobian(s[0], s[1], J, B, B))       # eigenvector directions
        for k in range(2):
            d = 0.13 * V2[:, k].real / (np.linalg.norm(V2[:, k].real) + 1e-12)
            col = 'tab:green' if w[k].real < 0 else 'tab:red'
            ax.plot([s[0] - d[0], s[0] + d[0]], [s[1] - d[1], s[1] + d[1]], color=col, lw=2, zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y',
           title=f'Phase portrait  J={J}, B={B}\n(green=stable eigvec, red=unstable)')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(OUT, f'phase_J{J:g}.png'), dpi=160, bbox_inches='tight')
    return fig


def scan_multiplicity(J_sweep=np.linspace(-10, 6, 161), B=0.0):
    """Print where the number of fixed points changes (bifurcation locations)."""
    print(f"\n=== fixed-point count vs J (B={B}) ===")
    prev = None
    for J in J_sweep:
        n = len(fixed_points(J, B, B))
        if n != prev:
            print(f"  J={J:+.3f}: {n} fixed point(s)")
            prev = n


def pitchfork_J(B=0.0, J_hi=12.0):
    """Onset J* of the (1,-1) pitchfork: smallest J with Jp=1 at the symmetric FP,
    i.e. sign change of lambda_anti = -1 + J*m*(1-m) with m=sigma(J(1-m)+B)."""
    def lam_anti(J):
        ms = symmetric_fp_analytic(J, B)
        return min(-1.0 + J * m * (1 - m) for m in ms)   # most-unstable symmetric root
    Js = np.linspace(0.01, J_hi, 4000)
    vals = np.array([lam_anti(J) for J in Js])
    i = np.flatnonzero((vals[:-1] < 0) & (vals[1:] >= 0))
    return float(brentq(lam_anti, Js[i[0]], Js[i[0] + 1])) if len(i) else np.nan


if __name__ == '__main__':
    B = -2
    print(f"\npitchfork onset J* = {pitchfork_J(B):.4f}  (B={B})")
    report_pointwise(J_list=(0., 0.5, 1, 3, 5, 6), B=B)
    scan_multiplicity(J_sweep=np.linspace(0, 12, 121), B=B)
    plot_bifurcation(J_sweep=np.linspace(0, 12, 241), B=B)
    plot_eigs_vs_J(J_sweep=np.linspace(0, 12, 400), B=B)
    plot_phase_portrait(4.0, B=B)                            # below onset: single node
    plot_phase_portrait(8.0, B=B)                            # above onset: two winners + saddle
    plot_density(J_list=(0.5, 3, 6), B=B)              # unimodal -> bimodal
    print('\nfigures saved to', OUT)
