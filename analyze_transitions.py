# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 2025

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import os
import gibbs_necker as gn
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from numba import njit

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["axes.grid"] = False


# ---GLOBAL VARIABLES
THETA = gn.THETA * 1.
DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/transition_analysis/simulations/'
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/transition_analysis/'


def face_eigvects(theta):
    """
    Orthonormal eigenvectors of the connectivity split by eigenvalue.

    Returns the uniform mode (lambda=3) and the three face/domain modes
    (lambda=1). The lambda=1 vectors are the +-1 face splits, found by
    brute force so they are interpretable regardless of node labelling.

    Parameters
    ----------
    theta : array (n, n)
        Connectivity (cube adjacency).

    Returns
    -------
    v_uni : array (n,)
        Uniform mode, normalised.
    v_face : array (3, n)
        Face modes, orthonormal.
    """
    n = theta.shape[0]
    v_uni = np.ones(n) / np.sqrt(n)
    faces = []
    for ups in itertools.combinations(range(n), n // 2):
        s = -np.ones(n)
        s[list(ups)] = 1
        if np.allclose(theta @ s, 1 * s):          # eigenvector with eigenvalue 1
            faces.append(s)
    faces = np.array(faces)
    faces = faces * faces[:, 0][:, None]           # canonical sign (v and -v merge)
    v_face = np.unique(faces, axis=0) / np.sqrt(n)  # 3 distinct face axes
    return v_uni, v_face


def frustration(m, theta):
    """
    Mean number of anti-aligned (frustrated) edges.

    F = sum_<ij> (1 - m_i m_j) / 2, with m the mean spin in [-1, 1].
    Equals 0 when aligned, n_edges/2 at the undecided centre, and the
    domain-wall length for a sharp single domain.
    """
    n_edges = theta.sum() / 2
    return n_edges / 2 - 0.25 * np.einsum('...i,ij,...j->...', m, theta, m)


def projections(m, v_face):
    """Projection of state(s) m onto the three face modes, and its norm."""
    proj = m @ v_face.T                 # (..., 3)
    return proj, np.linalg.norm(proj, axis=-1)


def moving_average(x, w):
    """O(n) centred moving average via cumulative sum."""
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0))
    ma = (c[w:] - c[:-w]) / w
    pad = (len(x) - len(ma)) // 2
    return np.concatenate([np.full(pad, ma[0]), ma,
                           np.full(len(x) - len(ma) - pad, ma[-1])])


def switch_windows(m_states, mean_spin, steps_back, steps_front, w_smooth):
    """
    Locate up-switches and cut aligned windows of the spin trajectory.

    Parameters
    ----------
    m_states : array (T, n)
        Spin trajectory in [-1, 1].
    mean_spin : array (T,)
        Population mean spin.
    steps_back, steps_front : int
        Window half-widths (in samples).
    w_smooth : int
        Smoothing window for switch detection.

    Returns
    -------
    windows : array (n_sw, steps_back + steps_front, n)
        Spin trajectories aligned to each up-switch.
    """
    p = moving_average(np.clip((mean_spin + 1) / 2, 0, 1), w_smooth)
    p[p > 0.5] = 1
    p[p < 0.5] = 0
    _, starts, vals = gn.rle(p)
    idx = starts[vals == 1]
    idx = idx[(idx > steps_back) & (idx < len(p) - steps_front)]
    windows = np.array([m_states[i - steps_back:i + steps_front] * 1.
                        for i in idx])
    return windows


def neighbor_list(theta):
    """Adjacency as an (n, deg) array of neighbour indices."""
    return np.array([np.where(theta[i])[0] for i in range(theta.shape[0])])


@njit(cache=True)
def gibbs_chain(j, b, n_iter, neigh, seed):
    """
    Single-site Gibbs sampler on the cube, returned as a +-1 chain.

    Flip of node i has energy change dK = -2 x_i (J h_i + b), with h_i the
    sum of neighbour spins, accepted with probability sigmoid(dK).
    """
    np.random.seed(seed)
    n = neigh.shape[0]
    x = np.where(np.random.rand(n) > 0.5, 1, -1).astype(np.int8)
    out = np.empty((n_iter, n), dtype=np.int8)
    for t in range(n_iter):
        i = np.random.randint(n)
        h = 0
        for k in range(neigh.shape[1]):
            h += x[neigh[i, k]]
        dk = -2 * x[i] * (j * h + b)
        if np.random.rand() < 1 / (1 + np.exp(-dk)):
            x[i] = -x[i]
        out[t] = x
    return out


def subsample(w, n_max):
    """Keep at most n_max random windows."""
    if len(w) > n_max:
        w = w[np.random.choice(len(w), n_max, replace=False)]
    return w


def gibbs_switches(j, b=0, n_iter=20000000, n_chains=8, max_switches=400,
                   steps_back=400, steps_front=400, w_smooth=400, theta=THETA):
    """Collect spin windows around spontaneous up-switches of the Gibbs chain."""
    neigh = neighbor_list(theta)
    cap = max_switches // n_chains + 1
    out = []
    for seed in range(n_chains):
        states = gibbs_chain(float(j), float(b), int(n_iter), neigh, seed)
        mean_spin = states.sum(axis=1) / theta.shape[0]      # int8-safe mean
        w = switch_windows(states, mean_spin, steps_back, steps_front, w_smooth)
        out.append(subsample(w, cap))
    out = [w for w in out if len(w)]
    return np.concatenate(out, axis=0) if out else np.empty((0, steps_back + steps_front, theta.shape[0]))


@njit(cache=True)
def mf_chain(j, b, theta, noise, tau, n_steps, dt, seed):
    """JIT'd Euler-Maruyama integration of the mean-field SDE (beliefs q)."""
    np.random.seed(seed)
    n = theta.shape[0]
    x = np.random.rand(n)
    out = np.empty((n_steps, n), dtype=np.float32)       # float32 to bound memory
    c = noise * np.sqrt(dt / tau)
    for t in range(n_steps):
        field = theta @ (2 * x - 1)
        for i in range(n):
            drift = (1.0 / (1.0 + np.exp(-(2 * j * field[i] + 2 * b))) - x[i]) / tau
            x[i] = x[i] + dt * drift + c * np.random.randn()
        out[t] = x
    return out


def mf_switches(j, b=0, noise=0.2, tau=0.1, dt=1e-2, t_end=200000, n_chains=10,
                max_switches=400, steps_back=250, steps_front=250,
                w_smooth=1000, theta=THETA):
    """Collect mean-spin windows around spontaneous up-switches of the MF SDE."""
    n_steps = int(t_end / dt)
    cap = max_switches // n_chains + 1
    def one_chain(seed):
        q = mf_chain(float(j), float(b), theta, float(noise), float(tau),
                     n_steps, float(dt), seed)
        m = 2 * np.clip(q, 0, 1) - 1
        return subsample(switch_windows(m, np.mean(m, axis=1),
                                        steps_back, steps_front, w_smooth), cap)
    out = Parallel(n_jobs=3)(delayed(one_chain)(s) for s in range(n_chains))
    out = [w for w in out if len(w)]
    return np.concatenate(out, axis=0) if out else np.empty((0, steps_back + steps_front, theta.shape[0]))


def cached(name, fn, recompute=False):
    """Load a saved simulation from DATA_FOLDER, or compute it and save."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    path = DATA_FOLDER + name + '.npy'
    if os.path.exists(path) and not recompute:
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def mf_flow(q, j, b, theta):
    """Deterministic mean-field velocity."""
    return gn.sigmoid(2 * j * (theta @ (2 * q - 1)) + 2 * b) - q


def mf_jacobian(q, j, b, theta):
    """Jacobian of the mean-field flow at q."""
    s = gn.sigmoid(2 * j * (theta @ (2 * q - 1)) + 2 * b)
    return 4 * j * (s * (1 - s))[:, None] * theta - np.eye(theta.shape[0])


def mf_saddle(j, b=0, theta=THETA, n_seeds=400):
    """
    Lowest-frustration index-1 saddle (transition state) of the MF flow.

    For J<1 this is the symmetric point q=0.5; for J>1 it is an off-diagonal
    face-domain saddle. Found by a random search, keeping the index-1 saddle
    with smallest frustration (the barrier the switch actually crosses).
    """
    n = theta.shape[0]
    best, best_f = np.full(n, 0.5), np.inf
    for _ in range(n_seeds):
        q, _, ok, _ = fsolve(mf_flow, np.random.rand(n), args=(j, b, theta),
                             full_output=True)
        if ok == 1 and np.max(np.abs(mf_flow(q, j, b, theta))) < 1e-8:
            if (np.linalg.eigvals(mf_jacobian(q, j, b, theta)).real > 0).sum() == 1:
                f = frustration(2 * q - 1, theta)
                if f < best_f:
                    best, best_f = q, f
    return best


def mf_map(j, b=0, theta=THETA, steps=400000, tol_stop=1e-30, numiters=200,
           dt=5e-3, noise=0.1):
    """
    Minimum-action path (Freidlin-Wentzell), as mean spin (numiters, n).

    Uses the action optimiser (mean_field_necker.minimum_action_path); with
    enough gradient-descent steps it converges off the diagonal for J>1. This
    is the correct object where the flow is non-gradient (off-diagonal), unlike
    the MEP (mf_mep), which equals the MAP only on the gradient subspace.
    """
    import mean_field_necker as mf
    _, x, _, _, _ = mf.minimum_action_path(j, b, noise=noise, theta=theta.astype(int),
                                           numiters=numiters, dt=dt, steps=steps,
                                           tol_stop=tol_stop)
    return 2 * x.detach().numpy() - 1


def mf_map_seeded(j, b=0, theta=THETA, eps=0.3, numiters=200, steps=200000,
                  step_size=1e-5, dt=5e-3):
    """
    Action MAP with a lambda=1 (face) seed and no early stop, as q-path.

    The straight diagonal guess sits in the symmetric subspace, which the
    action descent escapes only slowly for J just above 1. Seeding the
    midpoints with a face perturbation (zero at the endpoints) and disabling
    tol_stop lets it converge to the true off-diagonal MAP.
    """
    import mean_field_necker as mf
    import torch
    n = theta.shape[0]
    _, v_face = face_eigvects(theta)
    x1, x2, _ = mf.get_unst_and_stab_fp(j, b)
    k = np.arange(numiters)
    slope = (x2 - x1) / numiters
    x0 = (x1 - x2) + np.outer(k, np.repeat(slope, n))
    bump = np.sin(np.pi * k / (numiters - 1))
    x0 = x0 + eps * np.outer(bump, v_face[0] * np.sqrt(n))
    x0[0], x0[-1] = np.repeat(x1, n), np.repeat(x2, n)
    x0 = torch.tensor(x0, dtype=torch.float)
    th = torch.tensor(theta * 1., dtype=torch.float)
    _, x, _, _ = mf.get_path_between(x0.clone(), j, b, 0.1, th, steps=steps,
                                     step_size=step_size, dt=dt, tol_stop=-1.)
    return x.detach().numpy()


def plot_map_detail(j_list=(0.6, 1.2, 2.0), b=0, theta=THETA):
    """Per-node action-MAP trajectories and its projection onto all 8 eigenvectors."""
    os.makedirs(SV_FOLDER, exist_ok=True)
    evals, evecs = np.linalg.eigh(theta * 1.0)
    srt = np.argsort(-evals)
    evals, evecs = evals[srt], evecs[:, srt]
    fig, ax = plt.subplots(len(j_list), 2, figsize=(11, 4.0 * len(j_list)))
    ax = np.atleast_2d(ax)
    for a in ax.flatten():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    for r, j in enumerate(j_list):
        m = mf_map(j, b=b, theta=theta)                   # action MAP, mean spin
        q = (m + 1) / 2
        t = np.arange(len(m))
        proj = m @ evecs                                  # (numiters, 8)
        for i in range(theta.shape[0]):
            ax[r, 0].plot(t, q[:, i], lw=1.2)
        ax[r, 0].set_title('MAP node trajectories, J = {}'.format(j))
        ax[r, 0].set_xlabel('Path index')
        ax[r, 0].set_ylabel(r'$q_i$')
        for k in range(evecs.shape[1]):
            ax[r, 1].plot(t, proj[:, k], lw=1.8,
                          label=str(int(round(evals[k]))))
        ax[r, 1].set_title(r'MAP projection onto eigenvectors, J = {}'.format(j))
        ax[r, 1].set_xlabel('Path index')
        ax[r, 1].set_ylabel('projection')
        ax[r, 1].legend(title='Eigenvalue', ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_map_detail.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'fig_map_detail.pdf', dpi=400, bbox_inches='tight')
    return fig


def mf_mep(j, b=0, theta=THETA, dt=5e-3, nsteps=12000, delta=1e-2, uniform=False):
    """
    Minimum-energy path, as mean spin (2*nsteps+1, n).

    Descends the deterministic flow from a saddle along its unstable manifold
    into each well. With uniform=False uses the lowest-F index-1 saddle (which
    at high J connects a face domain to a well). With uniform=True descends from
    the symmetric point q=0.5 along the uniform (lambda=3) direction, giving the
    all-0 -> all-1 path.
    """
    if uniform:
        q = np.full(theta.shape[0], 0.5)
        w, v = np.linalg.eigh(theta * 1.0)
        u = v[:, np.argmax(w)]                           # lambda=3 uniform mode
    else:
        q = mf_saddle(j, b, theta)
        w, v = np.linalg.eig(mf_jacobian(q, j, b, theta))
        u = v[:, np.argmax(w.real)].real
    u = u / np.linalg.norm(u)
    halves = []
    for sgn in (1, -1):
        p = q + sgn * delta * u
        traj = [p.copy()]
        for _ in range(nsteps):
            p = p + dt * mf_flow(p, j, b, theta)
            traj.append(p.copy())
        halves.append(np.array(traj))
    full = np.vstack([halves[0][::-1], halves[1]])
    return 2 * full - 1


def count_fixed_points(j, b=0, theta=THETA, n_seeds=2000, atol=1e-2):
    """Distinct fixed points of the mean-field flow, counted by stability."""
    n = theta.shape[0]
    fps, idxs = [], []
    for _ in range(n_seeds):
        q, _, ok, _ = fsolve(mf_flow, np.random.rand(n), args=(j, b, theta),
                             full_output=True)
        if ok == 1 and np.max(np.abs(mf_flow(q, j, b, theta))) < 1e-8:
            if not any(np.allclose(q, f, atol=atol) for f in fps):
                fps.append(q)
                idxs.append(int((np.linalg.eigvals(
                    mf_jacobian(q, j, b, theta)).real > 0).sum()))
    idxs = np.array(idxs)
    return len(fps), int((idxs == 0).sum()), int((idxs == 1).sum())


def plot_n_fixed_points(j_list=np.arange(0., 2.01, 0.05), b=0, theta=THETA,
                        n_seeds=2000):
    """Number of mean-field fixed points as a function of coupling J."""
    os.makedirs(SV_FOLDER, exist_ok=True)
    total, stable, saddle = np.array([count_fixed_points(j, b, theta, n_seeds)
                                      for j in j_list]).T
    fig, ax = plt.subplots(1, figsize=(6, 4.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(j_list, total, color='k', marker='o', ms=3, label='all')
    ax.plot(j_list, stable, color='seagreen', marker='o', ms=3, label='stable')
    ax.plot(j_list, saddle, color='firebrick', marker='o', ms=3, label='index-1 saddles')
    ax.axvline(1 / 3, ls=':', color='gray', alpha=0.7)
    ax.axvline(1, ls='--', color='gray', alpha=0.7)
    ax.set_xlabel('Coupling J')
    ax.set_ylabel('Number of fixed points')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_nfp_vs_j.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'fig_nfp_vs_j.pdf', dpi=400, bbox_inches='tight')
    return j_list, total, stable, saddle


def crossing_metric(windows, theta, v_face, metric='F'):
    """Value of F or ||nu|| at the mu=0 crossing of each window."""
    vals = []
    for w in windows:
        mu = np.mean(w, axis=1)
        k = np.argmin(np.abs(mu))                     # mu=0 crossing
        m = w[k]
        vals.append(frustration(m, theta) if metric == 'F'
                    else projections(m, v_face)[1])
    return np.array(vals)


def bin_metric(windows, theta, v_face, metric='F', edges=np.linspace(-1, 1, 21)):
    """Mean metric as a function of binned magnetization mu."""
    mu = np.concatenate([np.mean(w, axis=1) for w in windows])
    m = np.concatenate(list(windows))
    val = frustration(m, theta) if metric == 'F' else projections(m, v_face)[1]
    centers = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(mu, edges) - 1
    mean = np.array([np.nanmean(val[which == i]) if np.any(which == i) else np.nan
                     for i in range(len(centers))])
    return centers, mean


def smooth_time(windows, k=51):
    """Moving average along time (axis 1) of a window stack, per node."""
    if k <= 1 or not len(windows):
        return windows
    ker = np.ones(k) / k
    out = np.empty_like(windows)
    for i in range(windows.shape[0]):
        for c in range(windows.shape[2]):
            out[i, :, c] = np.convolve(windows[i, :, c], ker, mode='same')
    return out


def bin_mean(mu, val, edges):
    """Mean of val in bins of mu (nan where empty)."""
    centers = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(mu, edges) - 1
    mean = np.array([np.nanmean(val[which == i]) if np.any(which == i) else np.nan
                     for i in range(len(centers))])
    return centers, mean


def plot_norm_panels(j_list=(0.6, 1.3), b=0, theta=THETA, smooth_mf=51,
                     n_scatter=30000):
    """
    Per J, three panels: Gibbs and MF ||nu|| vs mu (cloud + binned mean),
    then the two averages compared with the MAP. MF windows are temporally
    smoothed so the mean ||nu|| is not inflated by per-node noise.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    _, v_face = face_eigvects(theta)
    edges = np.linspace(-1, 1, 25)
    nrow = len(j_list)
    fig, ax = plt.subplots(nrow, 3, figsize=(15, 4.5 * nrow))
    ax = np.atleast_2d(ax)
    for a in ax.flatten():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    for r, j in enumerate(j_list):
        gw = cached('gibbs_j{}'.format(j), lambda: gibbs_switches(j, b=b, theta=theta))
        mw = cached('mf_j{}'.format(j), lambda: mf_switches(j, b=b, theta=theta))
        mp = cached('map_j{}'.format(j), lambda: mf_map(j, b=b, theta=theta))
        mw = smooth_time(mw, smooth_mf)

        means = {}
        for col, (w, key, color) in enumerate([(gw, 'Gibbs', 'k'),
                                               (mw, 'MF', 'firebrick')]):
            m = np.concatenate(list(w))
            mu = np.mean(m, axis=1)
            nu = projections(m, v_face)[1]
            sel = np.random.choice(len(mu), min(n_scatter, len(mu)), replace=False)
            ax[r, col].scatter(mu[sel], nu[sel], s=2, alpha=0.05, color=color)
            c, mean = bin_mean(mu, nu, edges)
            ok = ~np.isnan(mean)                       # Gibbs mu is discrete
            ax[r, col].plot(c[ok], mean[ok], color=color, lw=2.5, marker='o', ms=4)
            means[key] = (c[ok], mean[ok])
            ax[r, col].set_title('{}, J = {}'.format(key, j))
            ax[r, col].set_xlabel(r'Magnetization $\mu$')
            ax[r, col].set_ylabel(r'$\|\nu\|$')

        ax[r, 2].plot(*means['Gibbs'], color='k', lw=2.5, label='Gibbs')
        ax[r, 2].plot(*means['MF'], color='firebrick', lw=2.5, label='MF')
        ax[r, 2].plot(np.mean(mp, axis=1), projections(mp, v_face)[1],
                      color='royalblue', lw=2.5, label='MF MAP')
        ax[r, 2].set_title('Averages, J = {}'.format(j))
        ax[r, 2].set_xlabel(r'Magnetization $\mu$')
        ax[r, 2].set_ylabel(r'mean $\|\nu\|$')
        ax[r, 2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_norm_panels.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'fig_norm_panels.pdf', dpi=400, bbox_inches='tight')
    return fig


def plot_nu_and_directions(j_list=(0.6, 1.2), theta=THETA, smooth_mf=21):
    """
    ||nu|| vs mu (MF + Gibbs), and the three lambda=1 projections aligned to
    each switch's leading face. MF comes from the saved noise=0.3 sims.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    _, v_face = face_eigvects(theta)
    edges = np.linspace(-1, 1, 25)
    fig, ax = plt.subplots(len(j_list), 2, figsize=(11, 4.5 * len(j_list)))
    ax = np.atleast_2d(ax)
    for a in ax.flatten():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    dcol = ['darkorange', 'seagreen', 'purple']

    for r, j in enumerate(j_list):
        mw = smooth_time(np.load(DATA_FOLDER + 'mf_j{}_n0.3_tau0.05.npy'.format(j)), smooth_mf)
        gw = cached('gibbs_j{}'.format(j), lambda: gibbs_switches(j, theta=theta))

        # --- left: ||nu|| vs mu, MF and Gibbs
        for w, key, color in [(mw, 'MF', 'firebrick'), (gw, 'Gibbs', 'k')]:
            m = np.concatenate(list(w))
            c, mean = bin_mean(np.mean(m, axis=1), projections(m, v_face)[1], edges)
            ok = ~np.isnan(mean)
            ax[r, 0].plot(c[ok], mean[ok], color=color, marker='o', ms=3, label=key)
        ax[r, 0].set_title(r'$\|\nu\|$ vs $\mu$, J = {}'.format(j))
        ax[r, 0].set_xlabel(r'$\mu$')
        ax[r, 0].set_ylabel(r'$\|\nu\|$')
        ax[r, 0].legend(frameon=False)

        # --- right: three lambda=1 projections, aligned to the leading face
        for w, ls in [(mw, '-'), (gw, '--')]:
            mus, prs = [], []
            for win in w:
                mu = np.mean(win, axis=1)
                p = win @ v_face.T                       # (T, 3)
                k = np.argmin(np.abs(mu))                # mu=0 crossing
                order = np.argsort(-np.abs(p[k]))        # leading face first
                prs.append(p[:, order] * np.sign(p[k, order[0]]))
                mus.append(mu)
            mus, prs = np.concatenate(mus), np.concatenate(prs)
            for d in range(3):
                c, mean = bin_mean(mus, prs[:, d], edges)
                ok = ~np.isnan(mean)
                ax[r, 1].plot(c[ok], mean[ok], ls, color=dcol[d], lw=2,
                              label=(None if ls == '--' else 'dir {}'.format(d + 1)))
        ax[r, 1].set_title(r'aligned $\lambda=1$ projections (-- Gibbs), J = {}'.format(j))
        ax[r, 1].set_xlabel(r'$\mu$')
        ax[r, 1].set_ylabel('projection')
        ax[r, 1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_nu_directions.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'fig_nu_directions.pdf', dpi=400, bbox_inches='tight')
    return fig


def corrected_map(j, b=0, theta=THETA, numiters=500, steps=300000, dt=5e-3,
                  step_size=1e-5, noise=0.1, seed=0, seed_dir=None, seed_eps=0.3):
    """
    Minimum-action path with the CORRECTED FW residual xdot - (sigmoid - q),
    i.e. including the -q decay term the original lagrangian omits. Path runs
    from the all-0 well to the all-1 well. Returns mean spin (numiters, n).

    seed_dir: optional 8-vector (an eigenvector) to seed the midpoints with a
    sin-bump of amplitude seed_eps, to test whether the path leaves that mode.
    """
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    n = theta.shape[0]
    th = torch.tensor(theta * 1., dtype=torch.float)
    k = np.arange(numiters)
    x0 = np.outer(k / (numiters - 1), np.ones(n)) + np.random.randn(numiters, n) * 0.05
    if seed_dir is not None:
        bump = np.sin(np.pi * k / (numiters - 1))
        x0 = x0 + seed_eps * np.outer(bump, seed_dir)
    x0[0], x0[-1] = np.zeros(n), np.ones(n)
    x = torch.tensor(x0, dtype=torch.float)
    for i in range(steps):
        x.requires_grad_(True)
        xdot = (x[1:] - x[:-1]) / dt
        xdot = torch.cat([xdot, xdot[-1:]], 0)
        fx = (1 / (1 + torch.exp(-2 * j * (2 * torch.matmul(th, x.T) - 3) - 2 * b))).T
        flow = fx - x
        S = torch.linalg.norm(xdot - flow) / (2 * noise ** 2)
        g = torch.autograd.grad(S, x)[0]
        g[[0, -1]] *= 0
        x = (x - g * step_size).detach()
    return 2 * x.numpy() - 1


def action_density(m, j, b=0, theta=THETA, dt=5e-3):
    """Per-point action density |xdot - fx|^2 (same fx as minimum_action_path)."""
    q = (m + 1) / 2
    qdot = np.vstack([np.diff(q, axis=0) / dt, (q[-1] - q[-2])[None] / dt])
    fx = gn.sigmoid(2 * j * (2 * (q @ theta) - 3) + 2 * b)
    return ((qdot - fx) ** 2).sum(1)


def plot_path_analysis(path_m, j, name='path', b=0, theta=THETA):
    """
    Four-panel analysis of a transition path (MAP or MEP), given as mean spin:
    node trajectories, all-8 eigenvector projections, action density vs mu,
    and the (lambda=3, lambda=-3) phase portrait coloured by density.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    evals, evecs = np.linalg.eigh(theta * 1.0)
    srt = np.argsort(-evals)
    evals, evecs = evals[srt], evecs[:, srt]
    V3 = evecs[:, 0]
    m = path_m
    q = (m + 1) / 2
    t = np.arange(len(m))
    proj = m @ evecs
    dens = action_density(m, j, b, theta)
    mu = m.mean(1)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for a in ax.flatten():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    for i in range(theta.shape[0]):
        ax[0, 0].plot(t, q[:, i], lw=1.2)
    ax[0, 0].set_xlabel('path index'); ax[0, 0].set_ylabel(r'$q_i$')
    ax[0, 0].set_title('{} node trajectories, J={}'.format(name, j))
    for k in range(evecs.shape[1]):
        ax[0, 1].plot(t, proj[:, k], lw=1.6, label=str(int(round(evals[k]))))
    ax[0, 1].set_xlabel('path index'); ax[0, 1].set_ylabel('projection')
    ax[0, 1].set_title('eigenvector projections')
    ax[0, 1].legend(title='Eigenvalue', ncol=2, frameon=False, fontsize=9)
    ax[0, 2].plot(mu, dens, color='firebrick', lw=2)
    ax[0, 2].set_xlabel(r'$\mu$'); ax[0, 2].set_ylabel(r'action density $|\dot x-f|^2$')
    ax[0, 2].set_title('where the cost sits')
    # two phase portraits: (lambda=3, lambda=-3) and (lambda=3, lambda=1)
    for col, (vy, laby) in enumerate([(evecs[:, -1], '-3'), (evecs[:, 1], '1')]):
        axc = ax[1, col]
        uu, ss = np.meshgrid(np.linspace(-2.9, 2.9, 19), np.linspace(-1.6, 1.6, 13))
        U, S = np.zeros_like(uu), np.zeros_like(ss)
        for p in range(uu.shape[0]):
            for c in range(uu.shape[1]):
                mm = uu[p, c] * V3 + ss[p, c] * vy
                f = mf_flow(np.clip((mm + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
                U[p, c] = (2 * f) @ V3; S[p, c] = (2 * f) @ vy
        axc.quiver(uu, ss, U, S, color='lightgray', alpha=0.8)
        sc = axc.scatter(m @ V3, m @ vy, c=dens, cmap='viridis', s=16)
        plt.colorbar(sc, ax=axc, label='density')
        axc.set_xlabel(r'$\lambda=3$'); axc.set_ylabel(r'$\lambda={}$'.format(laby))
        axc.set_title(r'$(\lambda=3,\ \lambda={})$ plane'.format(laby))
    ax[1, 2].axis('off')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_pathanalysis_{}_j{}.png'.format(name, j), dpi=350, bbox_inches='tight')
    return fig


def plot_overlay_density(paths, labels, j, b=0, theta=THETA, name='MAP_MEP'):
    """
    Overlay several paths (list of mean-spin arrays) in the (lambda=3, lambda=-3)
    plane, coloured by action density, over the projected flow field.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0)
    o = np.argsort(-w); V3, Vm3 = V[:, o][:, 0], V[:, o][:, -1]
    vmax = max(action_density(p, j, b, theta).max() for p in paths)
    fig, ax = plt.subplots(figsize=(9, 7))
    uu, ss = np.meshgrid(np.linspace(-2.9, 2.9, 21), np.linspace(-1.6, 1.6, 15))
    U, S = np.zeros_like(uu), np.zeros_like(ss)
    for p in range(uu.shape[0]):
        for c in range(uu.shape[1]):
            m = uu[p, c] * V3 + ss[p, c] * Vm3
            f = mf_flow(np.clip((m + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
            U[p, c] = (2 * f) @ V3; S[p, c] = (2 * f) @ Vm3
    ax.quiver(uu, ss, U, S, color='lightgray', alpha=0.8)
    markers = ['o', 's', '^', 'D']
    for i, (pth, lab) in enumerate(zip(paths, labels)):
        d = action_density(pth, j, b, theta)
        sc = ax.scatter(pth @ V3, pth @ Vm3, c=d, cmap='viridis', vmin=0, vmax=vmax,
                        s=16, marker=markers[i % 4], label=lab)
    plt.colorbar(sc, ax=ax, label=r'action density $|\dot x-f|^2$')
    ax.set_xlabel(r'$\lambda=3$ (magnetization)'); ax.set_ylabel(r'$\lambda=-3$ (bipartite)')
    ax.set_title('Paths in (uniform, bipartite) plane, J={}'.format(j))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_overlay_density_{}_j{}.png'.format(name, j), dpi=400, bbox_inches='tight')
    return fig


def plot_phase_grid(j=2.0, b=0, theta=THETA, path=None, name='MAP'):
    """
    Phase portraits of a path for every pair of eigenvalue directions, in a
    grid, over the projected flow field. Diagonal cells show each mode vs path
    index. `path` is mean spin (numiters, n); defaults to the action MAP.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    evals, evecs = np.linalg.eigh(theta * 1.0)
    srt = np.argsort(-evals)
    evals, evecs = evals[srt], evecs[:, srt]
    reps = [0, 1, 4, 7]                                   # lambda = 3, 1, -1, -3
    labs = [int(round(evals[r])) for r in reps]
    rng = {3: 2.9, 1: 1.6, -1: 1.6, -3: 1.6}
    mp = mf_map(j, b=b, theta=theta) if path is None else path
    proj = mp @ evecs                                    # (numiters, 8)
    n = len(reps)
    fig, ax = plt.subplots(n, n, figsize=(13, 13))
    for a in range(n):
        for c in range(n):
            axc = ax[a, c]
            axc.spines['right'].set_visible(False)
            axc.spines['top'].set_visible(False)
            vi, vj = evecs[:, reps[c]], evecs[:, reps[a]]   # x = col mode, y = row mode
            if a == c:
                axc.plot(np.arange(len(mp)), proj[:, reps[a]], color='firebrick', lw=2)
                axc.set_ylabel(r'$\lambda={}$'.format(labs[a]))
                axc.set_xlabel('path index')
                continue
            xs = np.linspace(-rng[labs[c]], rng[labs[c]], 15)
            ys = np.linspace(-rng[labs[a]], rng[labs[a]], 13)
            XX, YY = np.meshgrid(xs, ys)
            U, W = np.zeros_like(XX), np.zeros_like(YY)
            for p in range(XX.shape[0]):
                for qd in range(XX.shape[1]):
                    m = XX[p, qd] * vi + YY[p, qd] * vj
                    f = mf_flow(np.clip((m + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
                    U[p, qd] = (2 * f) @ vi
                    W[p, qd] = (2 * f) @ vj
            axc.quiver(XX, YY, U, W, color='gray', alpha=0.5)
            axc.plot(proj[:, reps[c]], proj[:, reps[a]], color='firebrick', lw=2.5)
            axc.set_xlabel(r'$\lambda={}$'.format(labs[c]))
            axc.set_ylabel(r'$\lambda={}$'.format(labs[a]))
    fig.suptitle('{} phase portraits, J = {}'.format(name, j), y=1.0)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_phase_grid_{}_j{}.png'.format(name, j), dpi=350, bbox_inches='tight')
    return fig


def plot_eigen_traces(j=1.2, theta=THETA, n_show=80, smooth_mf=21):
    """
    Per-eigendirection switch traces, MF vs Gibbs. Rows = eigenvalue groups
    (3, 1, -1, -3); faint lines are single switches, bold is the mean of the
    projection magnitude (norm for the 3-fold groups, |.| for lambda=+-3).
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    evals, evecs = np.linalg.eigh(theta * 1.0)
    srt = np.argsort(-evals)
    evals, evecs = evals[srt], evecs[:, srt]
    groups = [('$\\lambda=3$', [0], 'signed'), ('$\\lambda=1$', [1, 2, 3], 'norm'),
              ('$\\lambda=-1$', [4, 5, 6], 'norm'), ('$\\lambda=-3$', [7], 'abs')]
    mw = smooth_time(np.load(DATA_FOLDER + 'mf_j{}_n0.3_tau0.05.npy'.format(j)), smooth_mf)
    gw = cached('gibbs_j{}'.format(j), lambda: gibbs_switches(j, theta=theta))
    fig, ax = plt.subplots(4, 2, figsize=(11, 13), sharex=True)
    for col, (w, name, mc) in enumerate([(mw, 'MF', 'firebrick'), (gw, 'Gibbs', 'k')]):
        sel = np.random.choice(len(w), min(n_show, len(w)), replace=False)
        T = w.shape[1]
        t = np.arange(T) - T // 2
        for row, (lab, idx, mode) in enumerate(groups):
            tr_all = []
            for s in sel:
                p = w[s] @ evecs[:, idx]
                tr = (np.linalg.norm(p, axis=1) if mode == 'norm'
                      else (np.abs(p[:, 0]) if mode == 'abs' else p[:, 0]))
                ax[row, col].plot(t, tr, color='gray', alpha=0.15, lw=0.6)
                tr_all.append(tr)
            ax[row, col].plot(t, np.mean(tr_all, axis=0), color=mc, lw=2.2)
            ax[row, col].spines['right'].set_visible(False)
            ax[row, col].spines['top'].set_visible(False)
            ax[row, col].set_ylabel(lab)
            if row == 0:
                ax[row, col].set_title('{}, J = {}'.format(name, j))
        ax[3, col].set_xlabel('Samples from switch')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_eigen_traces_j{}.png'.format(j), dpi=400, bbox_inches='tight')
    return fig


def plot_transitions(j_list=(0.6, 1.3), b=0, theta=THETA, recompute=False):
    """Main driver: figures A (||nu|| vs mu), B (F vs mu), C (hist), D (modes)."""
    os.makedirs(SV_FOLDER, exist_ok=True)
    v_uni, v_face = face_eigvects(theta)
    colors = {'gibbs': 'k', 'mf': 'firebrick', 'map': 'royalblue'}

    figA, axA = plt.subplots(ncols=len(j_list), figsize=(5 * len(j_list), 4.5))
    figB, axB = plt.subplots(ncols=len(j_list), figsize=(5 * len(j_list), 4.5))
    figC, axC = plt.subplots(ncols=len(j_list), figsize=(5 * len(j_list), 4.5))
    figD, axD = plt.subplots(ncols=len(j_list), figsize=(5 * len(j_list), 4.5))
    for a in list(axA) + list(axB) + list(axC) + list(axD):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    for col, j in enumerate(j_list):
        gw = cached('gibbs_j{}'.format(j), lambda: gibbs_switches(j, b=b, theta=theta), recompute)
        mw = cached('mf_j{}'.format(j), lambda: mf_switches(j, b=b, theta=theta), recompute)
        mp = cached('map_j{}'.format(j), lambda: mf_map(j, b=b, theta=theta), recompute)
        print('J={}: gibbs switches={}, mf switches={}'.format(j, len(gw), len(mw)))

        # --- Fig A: ||nu|| vs mu
        for w, key in [(gw, 'gibbs'), (mw, 'mf')]:
            if len(w):
                m = np.concatenate(list(w))
                mu = np.mean(m, axis=1)
                axA[col].scatter(mu, projections(m, v_face)[1], s=2, alpha=0.05,
                                 color=colors[key], label=key)
        mu_map = np.mean(mp, axis=1)
        axA[col].plot(mu_map, projections(mp, v_face)[1], color=colors['map'],
                      lw=2.5, label='MF MAP')
        axA[col].set_title('J = ' + str(j))
        axA[col].set_xlabel(r'Magnetization $\mu$')
        axA[col].set_ylabel(r'$\|\nu\|$ ($\lambda=1$ projection)')

        # --- Fig B: F vs mu
        for w, key in [(gw, 'gibbs'), (mw, 'mf')]:
            if len(w):
                c, f = bin_metric(w, theta, v_face, metric='F')
                axB[col].plot(c, f, color=colors[key], marker='o', ms=4, label=key)
        axB[col].plot(mu_map, frustration(mp, theta), color=colors['map'],
                      lw=2.5, label='MF MAP')
        axB[col].axhline(6, color='gray', ls=':', alpha=0.6)
        axB[col].axhline(4, color='gray', ls='--', alpha=0.6)
        axB[col].set_title('J = ' + str(j))
        axB[col].set_xlabel(r'Magnetization $\mu$')
        axB[col].set_ylabel(r'Frustration $F$')

        # --- Fig C: histogram of F at mu=0 crossing
        for w, key in [(gw, 'gibbs'), (mw, 'mf')]:
            if len(w):
                axC[col].hist(crossing_metric(w, theta, v_face, metric='F'),
                              bins=np.linspace(2, 8, 25), density=True, alpha=0.5,
                              color=colors[key], label=key)
        axC[col].axvline(6, color='gray', ls=':', alpha=0.6)
        axC[col].axvline(4, color='gray', ls='--', alpha=0.6)
        axC[col].set_title('J = ' + str(j))
        axC[col].set_xlabel(r'Frustration $F$ at $\mu=0$')
        axC[col].set_ylabel('Density')

        # --- Fig D: MF per-mode lambda=1 projections vs time from switch
        if len(mw):
            sb = mw.shape[1] // 2
            t = (np.arange(mw.shape[1]) - sb) * 1e-3
            mode_colors = ['darkorange', 'seagreen', 'purple']
            for jump in mw:
                proj, _ = projections(jump, v_face)        # (T, 3)
                sign = np.sign(proj[np.argmax(np.abs(proj).sum(1))])  # align leading sign
                for k in range(3):
                    axD[col].plot(t, proj[:, k] * sign[k], color=mode_colors[k],
                                  alpha=0.15)
            axD[col].set_title('J = ' + str(j))
            axD[col].set_xlabel('Time from switch (s)')
            axD[col].set_ylabel(r'$\lambda=1$ projection')

    axA[0].legend(markerscale=4, framealpha=1)
    axB[0].legend()
    axC[0].legend()
    for fig, name in [(figA, 'figA_norm_vs_mu'), (figB, 'figB_frustration_vs_mu'),
                      (figC, 'figC_crossing_hist'), (figD, 'figD_mf_modes')]:
        fig.tight_layout()
        fig.savefig(SV_FOLDER + name + '.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + name + '.pdf', dpi=400, bbox_inches='tight')
    return figA, figB, figC, figD


if __name__ == '__main__':
    plot_transitions(j_list=(0.6, 1.3))
