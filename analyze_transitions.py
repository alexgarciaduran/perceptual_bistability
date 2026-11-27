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
from tqdm import tqdm

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


def cached_gibbs(j, recompute=False, theta=THETA, **kw):
    """Gibbs switch windows at coupling j, saved to / loaded from DATA_FOLDER."""
    return cached('sw_gibbs_j{}'.format(j),
                  lambda: gibbs_switches(j, theta=theta, **kw), recompute)


def cached_mf(j, noise=0.4, tau=0.05, recompute=False, theta=THETA, **kw):
    """MF-SDE switch windows at (j, noise, tau), saved to / loaded from DATA_FOLDER."""
    return cached('sw_mf_j{}_n{}_tau{}'.format(j, noise, tau),
                  lambda: mf_switches(j, noise=noise, tau=tau, theta=theta, **kw), recompute)


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


def mf_mep(j, b=0, theta=THETA, dt=5e-3, nsteps=12000, delta=1e-2, uniform=False,
           mode_dir=None):
    """
    Minimum-energy path, as mean spin (2*nsteps+1, n).

    Descends the deterministic flow from a seed point along a chosen direction
    into each well:
      mode_dir given -- descend from the symmetric point q=0.5 along that
                        eigenvector of theta (e.g. a lambda=1 face mode). This is
                        the "escape along mode_dir": the two branches roll into
                        the two attractors on either side of that direction.
      uniform=True   -- same but along the lambda=3 (uniform) mode (all-0->all-1).
      else           -- lowest-F index-1 saddle (face-domain <-> well at high J).
    """
    if mode_dir is not None:
        q = np.full(theta.shape[0], 0.5)
        u = np.asarray(mode_dir, dtype=float)
    elif uniform:
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
                  step_size=1e-4, noise=0.1, seed=0, seed_dir=None, seed_eps=0.3):
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
    for i in tqdm(range(steps)):
        x.requires_grad_(True)
        xdot = (x[1:] - x[:-1]) / dt
        xdot = torch.cat([xdot, xdot[-1:]], 0)
        fx = (1 / (1 + torch.exp(-2 * j * (2 * torch.matmul(th, x.T) - 3) - 2 * b))).T
        flow = fx - x
        S = torch.linalg.norm(xdot - flow) / (2 * noise ** 2)
        g = torch.autograd.grad(S, x)[0]
        g[[0, -1]] *= 0
        x = (x - g * step_size).detach()
        if i % 10000 == 0:
            tqdm.write(f" Action {S.detach().float(): .1f}")
    return 2 * x.numpy() - 1


def action_density(m, j, b=0, theta=THETA, dt=5e-3):
    """Per-point action density |qdot - (sigmoid - q)|^2 (corrected MF drift)."""
    q = (m + 1) / 2
    qdot = np.vstack([np.diff(q, axis=0) / dt, (q[-1] - q[-2])[None] / dt])
    flow = gn.sigmoid(2 * j * (2 * (q @ theta) - 3) + 2 * b) - q   # f = sigmoid - q
    return ((qdot - flow) ** 2).sum(1)


def plot_instantaneous_cost(paths, labels, j, b=0, theta=THETA, name='cost', dt=5e-3):
    """Instantaneous action density |qdot-(sigmoid-q)|^2 vs mu and vs time, per path."""
    os.makedirs(SV_FOLDER, exist_ok=True)
    colors = ['firebrick', 'royalblue', 'orange', 'seagreen']
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4.6))
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        d = action_density(p, j, b, theta)
        mu = p.mean(1)
        ax[0].plot(mu, d, color=colors[i % 4], lw=2, label=lab)
        ax[1].plot(np.arange(len(p)) * dt, d, color=colors[i % 4], lw=2, label=lab)
    ax[0].set_xlabel(r'$\mu$'); ax[0].set_ylabel(r'cost $|\dot q-(\sigma-q)|^2$')
    ax[0].set_title('instantaneous cost vs magnetization')
    ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('cost'); ax[1].set_title('vs time')
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_instcost_{}_j{}.png'.format(name, j), dpi=400, bbox_inches='tight')
    return fig


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


def plot_map_vs_mep(mapp, mep, j, theta=THETA):
    """MAP vs MEP: |lambda=-3| and ||lambda=1|| projections vs magnetization."""
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0)
    o = np.argsort(-w); Vm3 = V[:, o][:, -1]
    _, v_face = face_eigvects(theta)
    fig, ax = plt.subplots(ncols=2, figsize=(11, 4.5))
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    for p, lab, c in [(mapp, 'MAP', 'firebrick'), (mep, 'MEP', 'royalblue')]:
        mu = p.mean(1)
        ax[0].plot(mu, np.abs(p @ Vm3), color=c, lw=2, label=lab)
        ax[1].plot(mu, projections(p, v_face)[1], color=c, lw=2, label=lab)
    ax[0].set_xlabel(r'$\mu$'); ax[0].set_ylabel(r'$|\lambda=-3|$'); ax[0].set_title('bipartite')
    ax[1].set_xlabel(r'$\mu$'); ax[1].set_ylabel(r'$\|\lambda=1\|$'); ax[1].set_title('face')
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_map_vs_mep_j{}.png'.format(j), dpi=400, bbox_inches='tight')
    return fig


def plot_variance_envelope(js=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), noise=0.4, theta=THETA):
    """
    Transverse-fluctuation std vs the harmonic OU envelope std = noise*sqrt(2/|4J sigma' lam - 1|).
    Each mode fluctuates around the diagonal as an OU process whose stiffness is the local
    Jacobian eigenvalue; at the crossing the denominator is |J lam_k - 1|.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0); o = np.argsort(-w); V = V[:, o]
    groups = {3: [0], 1: [1, 2, 3], -1: [4, 5, 6], -3: [7]}
    lam = {3: 3., 1: 1., -1: -1., -3: -3.}

    def envelope(mbar, j, k):
        s = 1 / (1 + np.exp(-6 * j * mbar)); sp = s * (1 - s)   # sigma'(uniform path)
        return noise * np.sqrt(2 / np.clip(np.abs(4 * j * sp * lam[k] - 1), 0.02, None))

    def base(curve, t0, w=80):                                 # off-crossing floor (the tails)
        idx = np.abs(np.arange(len(curve)) - t0) > w
        return curve[idx].mean() if idx.any() else curve.min()

    data = {}
    for j in js:
        win = cached_mf(j, noise=noise)
        P = win @ V
        mbar = win.mean((0, 2)); t0 = int(np.argmin(np.abs(mbar)))
        std = {k: np.sqrt(P[:, :, idx].var(0).mean(1)) for k, idx in groups.items()}
        data[j] = dict(std=std, mbar=mbar, t0=t0)
    cmap = mpl.cm.viridis; nrm = mpl.colors.Normalize(min(js), max(js))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    for j in [js[1], js[len(js) // 2], js[-2]]:                 # (A) lam=1 excess std(t), baseline at 0
        d = data[j]; c = cmap(nrm(j)); t0 = d['t0']; t = np.arange(len(d['mbar'])) - t0
        sm = d['std'][1] - base(d['std'][1], t0)
        en = envelope(d['mbar'], j, 1); en = en - base(en, t0)
        ax[0].plot(t, sm, color=c, lw=2.2, label='J={:.1f}'.format(j))
        ax[0].plot(t, en, color=c, lw=1.4, ls='--')
    ax[0].set_xlim(-120, 120); ax[0].set_xlabel('samples from switch')
    ax[0].set_ylabel(r'$\lambda=1$ std $-$ baseline'); ax[0].set_title('measured (-) vs harmonic (- -)')
    ax[0].legend(frameon=False, fontsize=12)
    jm = js[len(js) // 2]; d = data[jm]; t0 = d['t0']; ks = [1, -1, -3]; x = np.arange(3)
    meas = [d['std'][k][t0] - base(d['std'][k], t0) for k in ks]   # (B) ordering at crossing, baselined
    pred = []
    for k in ks:
        e = envelope(d['mbar'], jm, k); pred.append(e[t0] - base(e, t0))
    ax[1].bar(x - 0.18, meas, 0.36, color='#2c7fb8', label='measured')
    ax[1].bar(x + 0.18, pred, 0.36, color='#b0b0b0', label='harmonic')
    ax[1].set_xticks(x); ax[1].set_xticklabels([r'$\lambda=1$', r'$\lambda=-1$', r'$\lambda=-3$'])
    ax[1].set_ylabel('crossing std $-$ baseline'); ax[1].set_title('mode ordering (J={:.1f})'.format(jm))
    ax[1].legend(frameon=False, fontsize=12)
    jar = np.array(js); pm, pp = [], []                          # (C) peak lam=1 excess vs J, baselined
    for j in js:
        d2 = data[j]; tt = d2['t0']
        pm.append(d2['std'][1][tt] - base(d2['std'][1], tt))
        e = envelope(d2['mbar'], j, 1); pp.append(e[tt] - base(e, tt))
    ax[2].plot(jar, pm, 'o-', color='#2c7fb8', lw=2, label='measured')
    ax[2].plot(jar, pp, 's--', color='#777777', lw=1.8, label=r'harmonic $\propto 1/\sqrt{|J-1|}$')
    ax[2].axvline(1.0, color='k', ls=':', lw=1); ax[2].set_xlabel('J')
    ax[2].set_ylabel(r'$\lambda=1$ crossing excess'); ax[2].set_title('soft-mode growth toward J=1')
    ax[2].legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_variance_envelope.png', dpi=400)
    fig.savefig(SV_FOLDER + 'fig_variance_envelope.pdf', dpi=400)
    return fig


def plot_peak_projection_vs_J(js=(0.4, 0.6, 0.8, 1.0, 1.2, 1.4), theta=THETA):
    """
    Peak swing (max - min) of the aligned-mean switch projection per eigenmode vs J, for the
    mean-field SDE and Gibbs. lambda=3 is the full switch (~6); lambda=1 grows near J=1; the
    rest stay near 0.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0); o = np.argsort(-w); V = V[:, o]
    groups = {3: [0], 1: [1, 2, 3], -1: [4, 5, 6], -3: [7]}

    def peaks(win):
        P = mean_aligned_switch(win, theta) @ V
        return {k: max(P[:, i].max() - P[:, i].min() for i in idx) for k, idx in groups.items()}

    colors = {3: 'firebrick', 1: 'seagreen', -1: 'royalblue', -3: 'purple'}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for col, (cache, title) in enumerate([(cached_mf, 'mean field'), (cached_gibbs, 'Gibbs')]):
        pk = {k: [] for k in groups}
        for j in js:
            p = peaks(cache(j))
            for k in groups:
                pk[k].append(p[k])
        for k in groups:
            ax[col].plot(js, pk[k], 'o-', color=colors[k], lw=2, label=r'$\lambda={}$'.format(k))
        ax[col].set_xlabel('J'); ax[col].set_title(title)
        ax[col].spines['right'].set_visible(False); ax[col].spines['top'].set_visible(False)
    ax[0].set_ylabel('peak projection swing (max - min)'); ax[0].legend(frameon=False, title='mode')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_peakproj_vs_J.png', dpi=400)
    fig.savefig(SV_FOLDER + 'fig_peakproj_vs_J.pdf', dpi=400)
    return fig


def nucleation_chain(j, face=(0, 0, 0, 0, 1, 1, 1, 1), theta=THETA, dt=5e-3, nsteps=15000):
    """
    Deterministic [0] -> face-domain -> [1] path (flow-MEP chain) through the two face saddles,
    computed in the symmetric (face-value y, rest-value z) 2-D invariant subspace. Returns the
    mean-spin path, or None if the face domain is not a stable attractor at this J (J < ~1.7).
    """
    face = np.asarray(face, bool)
    lift = lambda y, z: np.where(face, y, z)
    def f2(yz):
        fl = mf_flow(lift(yz[0], yz[1]), j, 0, theta); return np.array([fl[face].mean(), fl[~face].mean()])
    def jac2(s, h=1e-6):
        J = np.zeros((2, 2))
        for i in range(2):
            d = np.zeros(2); d[i] = h; J[:, i] = (f2(s + d) - f2(s - d)) / (2 * h)
        return J
    idx2 = lambda s: int((np.linalg.eigvals(jac2(s)).real > 0).sum())
    fps = []                                            # all 2-D fixed points (grid of seeds)
    for y0 in np.linspace(0.03, 0.97, 9):
        for z0 in np.linspace(0.03, 0.97, 9):
            s, _, ok, _ = fsolve(f2, [y0, z0], full_output=True)
            if ok == 1 and np.max(np.abs(f2(s))) < 1e-9 and (0 <= s).all() and (s <= 1).all():
                if not any(np.linalg.norm(s - u) < 1e-4 for u in fps):
                    fps.append(s)
    attr = [s for s in fps if idx2(s) == 0]; sadd = [s for s in fps if idx2(s) == 1]
    if len(attr) < 4:
        return None
    near = lambda pt: attr[int(np.argmin([np.linalg.norm(pt - a) for a in attr]))]
    ll, hl, hh = near([0.02, 0.02]), near([0.98, 0.02]), near([0.98, 0.98])
    def descend(S):
        wv, vv = np.linalg.eig(jac2(S)); u = np.real(vv[:, np.argmax(wv.real)]); u = u / np.linalg.norm(u)
        out = []
        for sgn in (1, -1):
            p = S + sgn * 1e-3 * u; tr = [p.copy()]
            for _ in range(nsteps):
                p = p + dt * f2(p); tr.append(p.copy())
                if np.max(np.abs(f2(p))) < 1e-8:
                    break
            out.append(np.array(tr))
        return out
    def leg(A, B):                                      # the saddle whose flow connects A and B
        for S in sadd:
            d = descend(S); e = [o[-1] for o in d]
            if (min(np.linalg.norm(e[0] - A), np.linalg.norm(e[1] - A)) < 0.05 and
                    min(np.linalg.norm(e[0] - B), np.linalg.norm(e[1] - B)) < 0.05):
                return d, e
        return None
    c1, c2 = leg(ll, hl), leg(hl, hh)
    if c1 is None or c2 is None:
        return None
    d1, e1 = c1; d2, e2 = c2
    a = int(np.argmin([np.linalg.norm(e1[0] - ll), np.linalg.norm(e1[1] - ll)]))
    s1lo, s1F = d1[a], d1[1 - a]
    b = int(np.argmin([np.linalg.norm(e2[0] - hh), np.linalg.norm(e2[1] - hh)]))
    s2hi, s2F = d2[b], d2[1 - b]
    yz = np.vstack([s1lo[::-1], s1F, s2F[::-1], s2hi])   # [0] -> S1 -> F -> S2 -> [1]
    return 2 * np.array([lift(y, z) for y, z in yz]) - 1


def plot_nucleation_grid(js=(1.7, 2.0, 2.5, 3.0), face=(0, 0, 0, 0, 1, 1, 1, 1), theta=THETA):
    """Eigenmode projections (lam=3,1,-1,-3) of the deterministic nucleation chain, grid over J."""
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0); o = np.argsort(-w); V = V[:, o]
    reps = {3: 0, 1: 1, -1: 4, -3: 7}; modes = [3, 1, -1, -3]
    chains = [(jv, nucleation_chain(jv, face, theta)) for jv in js]
    chains = [(jv, m) for jv, m in chains if m is not None]
    fig, ax = plt.subplots(len(modes), len(chains), figsize=(3.2 * len(chains), 2.5 * len(modes)),
                           sharey='row', squeeze=False)
    for col, (jv, m) in enumerate(chains):
        P = m @ V; s = np.linspace(0, 1, len(m))
        for row, k in enumerate(modes):
            a = ax[row][col]
            a.plot(s, P[:, reps[k]], color='firebrick', lw=2)
            a.axhline(0, color='k', lw=0.6, alpha=0.4)
            a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
            if row == 0:
                a.set_title('J = {:.1f}'.format(jv))
            if col == 0:
                a.set_ylabel(r'$\lambda={}$'.format(k))
            if row == len(modes) - 1:
                a.set_xlabel('path progress')
    fig.suptitle('nucleation chain  [0] -> face -> [1],  eigenmode projections', y=1.0)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_nucleation_grid.png', dpi=350, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'fig_nucleation_grid.pdf', dpi=350, bbox_inches='tight')
    return fig


def plot_noise_sweep(js=(0.5, 0.7, 0.9, 1.1), noises=(0.25, 0.3, 0.4), theta=THETA, n_chains=8):
    """
    lambda=1 crossing-excess vs noise, one line per J. Below J=1 the excursion is a soft-mode
    fluctuation and scales ~ noise (extrapolating to 0); above J=1 it is nucleation and departs
    from the linear law. Lower noise -> rarer switches, so t_end is raised (capped by memory:
    mf_chain stores the whole trajectory, so each chain's (t_end/dt, 8) array must stay < ~1 GB).
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0); o = np.argsort(-w); V = V[:, o]
    def base(curve, t0, wd=80):
        idx = np.abs(np.arange(len(curve)) - t0) > wd
        return curve[idx].mean() if idx.any() else curve.min()
    tend = {0.25: 300000, 0.3: 300000, 0.4: 200000, 0.5: 150000}
    res = {j: [] for j in js}; counts = {j: [] for j in js}
    for j in js:
        for nz in noises:
            win = cached_mf(j, noise=nz, t_end=tend.get(nz, 300000), n_chains=n_chains, theta=theta)
            P = win @ V; mbar = win.mean((0, 2)); t0 = int(np.argmin(np.abs(mbar)))
            std1 = np.sqrt(P[:, :, [1, 2, 3]].var(0).mean(1))
            res[j].append(std1[t0] - base(std1, t0)); counts[j].append(len(win))
            print('J={:.1f} noise={:.2f}: switches={} excess={:.3f}'.format(j, nz, len(win), res[j][-1]))
    cmap = mpl.cm.viridis; nrm = mpl.colors.Normalize(min(js), max(js)); nz = np.array(noises)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for j in js:
        ax.plot(nz, res[j], 'o-', color=cmap(nrm(j)), lw=2, label='J={:.1f}'.format(j))
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('noise'); ax.set_ylabel(r'$\lambda=1$ crossing excess')
    ax.set_title('fluctuation (J<1, $\\propto$ noise) vs nucleation (J>1)')
    ax.set_xlim(0, max(noises) * 1.05); ax.legend(frameon=False, title='coupling')
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_noise_sweep.png', dpi=400)
    fig.savefig(SV_FOLDER + 'fig_noise_sweep.pdf', dpi=400)
    return fig


def plot_everything(j, theta=THETA, recompute=False):
    """
    Full figure suite comparing the deterministic routes with the simulated switch.

    Deterministic routes (all [0]->[1]): diagonal (lambda=3), uniform MEP, and the
    cached gradient-descent MAPs seeded along lambda=-3 (the min-action bipartite
    instanton) and lambda=1. Simulated switch from cached_mf: the raw ensemble mean
    (washes out to the diagonal) and the sign-aligned mean (the real lambda=1 bulge).
    """
    w, v = np.linalg.eigh(theta * 1.0); V3 = v[:, np.argmax(w)]

    def orient(m):                                        # force lambda=3 increasing (lo->hi)
        return m[::-1] if (m @ V3)[0] > (m @ V3)[-1] else m

    mapm3_n = cached_map(j, 'lam-3', recompute, theta)           # native direction (hi->lo)
    map1_n = cached_map(j, 'lam1', recompute, theta)
    diag_n = cached_map(j, 'diagonal', recompute, theta)
    mapm3, map1, diag = orient(mapm3_n), orient(map1_n), orient(diag_n)   # lo->hi for plots
    mep = orient(mf_mep(j, theta=theta, uniform=True))           # standard uniform MEP (flow-timed)
    win = cached_mf(j)
    sim_raw = orient(win.mean(0))                               # ensemble mean (-> diagonal)
    sim_aln = orient(mean_aligned_switch(win, theta))          # sign-aligned mean (lambda=1 bulge)

    # action over time (direct |qdot-f|^2/2sigma^2). Score the deterministic paths in their
    # NATIVE direction: the action is not time-reversal invariant (reversing -> |qdot+f|^2).
    plot_action_vs_time([diag_n, mapm3_n, map1_n, sim_raw, sim_aln],
                        ['diagonal', r'MAP $\lambda$=-3', r'MAP $\lambda$=1', 'sim raw', 'sim aligned'],
                        j, theta=theta, dts=[5e-3, 5e-3, 5e-3, 1e-2, 1e-2])
    # action landscapes with the simulated switch overlaid, both transverse planes
    plot_action_landscape([mapm3, diag, sim_aln], [r'MAP $\lambda$=-3', 'diagonal', 'sim aligned'],
                          j, theta=theta, name='routes', trans='lam-3')
    plot_action_landscape([map1, diag, sim_aln], [r'MAP $\lambda$=1', 'diagonal', 'sim aligned'],
                          j, theta=theta, name='routes', trans='lam1')
    # averaged sims on the action-landscape eigengrid (drift background)
    plot_eigengrid(mapm3, [sim_aln], j, theta=theta, name='MAPsim', background='drift')
    plot_eigengrid(mep, [sim_aln], j, theta=theta, name='MEPsim', background='drift')
    # detailed per-path views and overlays
    plot_path_analysis(mapm3, j, name='MAP', theta=theta)
    plot_path_analysis(mep, j, name='MEP', theta=theta)
    plot_overlay_density([mapm3, diag, sim_aln], ['MAP', 'diagonal', 'sim'], j, theta=theta)
    plot_map_vs_mep(mapm3, mep, j, theta=theta)
    return dict(map_lam3=mapm3, map_lam1=map1, diagonal=diag, mep=mep,
                sim_raw=sim_raw, sim_aligned=sim_aln)


def mode_field(uu, ss, j, b, theta, V3, Vm3, field='drift'):
    """Scalar field over the (lambda=3, lambda=-3) plane for a landscape plot."""
    Z = np.zeros_like(uu)
    for p in range(uu.shape[0]):
        for c in range(uu.shape[1]):
            m = uu[p, c] * V3 + ss[p, c] * Vm3
            if field == 'energy':                  # naive MF energy -1/2 m^T th m
                Z[p, c] = -0.5 * m @ theta @ m
            else:                                  # squared drift |f|^2 (mean spin)
                f = mf_flow(np.clip((m + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
                Z[p, c] = ((2 * f) ** 2).sum()
    return Z


def mean_aligned_switch(windows, theta=THETA):
    """
    Average switch, aligned so they don't cancel. Each window is projected onto
    the eigenbasis; the lambda=3 (order-parameter) sign is aligned by its value
    in the end well, and every transverse mode by its sign at the mu=0 crossing;
    then averaged and mapped back to node space (T, n).
    """
    w, V = np.linalg.eigh(theta * 1.0)
    o = np.argsort(-w); V = V[:, o]
    Ps = []
    for win in windows:
        P = win @ V                                   # (T, n) projections
        k = np.argmin(np.abs(win.mean(1)))            # crossing
        sign = np.sign(P[k])
        sign[0] = np.sign(P[-1, 0])                   # lambda=3: align by end well
        Ps.append(P * sign)
    return np.mean(Ps, axis=0) @ V.T                  # back to node space


def plot_eigengrid(path, sims, j, b=0, theta=THETA, name='MAP', background='flow', ngrid=40):
    """
    Grid of eigenvalue-pair projections (lambda=3,1,-1,-3). Each off-diagonal
    cell shows the deterministic `path` (red) and the simulated switches `sims`
    (gray) over a background that is either the flow field (background='flow')
    or the squared-drift landscape (background='drift'). Diagonal cells show
    each mode vs path index.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    evals, evecs = np.linalg.eigh(theta * 1.0)
    o = np.argsort(-evals); evals, evecs = evals[o], evecs[:, o]
    reps = [0, 1, 4, 7]
    labs = [int(round(evals[r])) for r in reps]
    rng = {3: 2.9, 1: 1.6, -1: 1.6, -3: 1.6}
    proj = path @ evecs
    simproj = [s @ evecs for s in sims]
    scol = 'white' if background == 'drift' else 'gray'
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    # precompute the drift backgrounds so every cell shares one colour scale
    bg, gmax = {}, 0.0
    for a in range(4):
        for c in range(4):
            if a == c:
                continue
            vx, vy = evecs[:, reps[c]], evecs[:, reps[a]]
            XX, YY = np.meshgrid(np.linspace(-rng[labs[c]], rng[labs[c]], ngrid),
                                 np.linspace(-rng[labs[a]], rng[labs[a]], ngrid))
            if background == 'drift':
                Z = np.zeros_like(XX)
                for p in range(ngrid):
                    for q in range(ngrid):
                        m = XX[p, q] * vx + YY[p, q] * vy
                        f = mf_flow(np.clip((m + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
                        Z[p, q] = ((2 * f) ** 2).sum()
                bg[(a, c)] = (XX, YY, Z); gmax = max(gmax, Z.max())
            else:
                bg[(a, c)] = (XX, YY, None)
    im = None
    for a in range(4):
        for c in range(4):
            axc = ax[a, c]
            axc.spines['right'].set_visible(False); axc.spines['top'].set_visible(False)
            vx, vy = evecs[:, reps[c]], evecs[:, reps[a]]
            if a == c:
                axc.plot(np.arange(len(path)), proj[:, reps[a]], color='firebrick', lw=2)
                for sp in simproj:
                    axc.plot(np.linspace(0, len(path), len(sp)), sp[:, reps[a]], color='gray', alpha=0.9, lw=1.8)
                axc.set_xlabel('path'); axc.set_ylabel(r'$\lambda={}$'.format(labs[a]))
                continue
            XX, YY, Z = bg[(a, c)]
            if background == 'drift':
                im = axc.contourf(XX, YY, Z, levels=np.linspace(0, gmax, 21), cmap='viridis')
            else:
                qX, qY = np.meshgrid(np.linspace(-rng[labs[c]], rng[labs[c]], 13),
                                     np.linspace(-rng[labs[a]], rng[labs[a]], 11))
                U, W = np.zeros_like(qX), np.zeros_like(qY)
                for p in range(qX.shape[0]):
                    for q in range(qX.shape[1]):
                        m = qX[p, q] * vx + qY[p, q] * vy
                        f = mf_flow(np.clip((m + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
                        U[p, q] = (2 * f) @ vx; W[p, q] = (2 * f) @ vy
                axc.quiver(qX, qY, U, W, color='lightgray', alpha=0.7)
            for sp in simproj:
                axc.plot(sp[:, reps[c]], sp[:, reps[a]], color=scol, alpha=0.9, lw=1.8)
            axc.plot(proj[:, reps[c]], proj[:, reps[a]], color='firebrick', lw=2)
            axc.set_xlabel(r'$\lambda={}$'.format(labs[c])); axc.set_ylabel(r'$\lambda={}$'.format(labs[a]))
    fig.suptitle('{} on {} grid (red=MAP, gray=avg sim), J={}'.format(name, background, j), y=1.0)
    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label=r'drift $|2f|^2$')
    fig.savefig(SV_FOLDER + 'fig_eigengrid_{}_{}_j{}.png'.format(name, background, j), dpi=300, bbox_inches='tight')
    return fig


def plot_action_landscape(paths, labels, j, b=0, theta=THETA, name='MAP',
                          field='drift', ngrid=60, quiver=True, trans='lam-3'):
    """
    (lambda=3, transverse) plane coloured by a scalar landscape, with path traces
    on top. trans='lam-3' (bipartite) or 'lam1' (face). field='drift' -> squared
    drift |f|^2 (the local cost the FW action integrates); 'energy' -> MF energy.
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    w, V = np.linalg.eigh(theta * 1.0)
    o = np.argsort(-w); V3 = V[:, o][:, 0]
    if trans == 'lam1':
        _, v_face = face_eigvects(theta); Vm3 = v_face[0]; ylab = r'$\lambda=1$ (face)'
    else:
        Vm3 = V[:, o][:, -1]; ylab = r'$\lambda=-3$ (bipartite)'
    ymax = max(1.6, max(np.abs(p @ Vm3).max() for p in paths) * 1.1)
    us = np.linspace(-2.9, 2.9, ngrid)
    ss_ = np.linspace(-ymax, ymax, ngrid)
    uu, ss = np.meshgrid(us, ss_)
    Z = mode_field(uu, ss, j, b, theta, V3, Vm3, field)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.contourf(uu, ss, Z, levels=30, cmap='viridis')
    plt.colorbar(im, ax=ax, label=(r'energy' if field == 'energy' else r'drift $|f|^2$'))
    if quiver:
        qu, qs = np.meshgrid(np.linspace(-2.9, 2.9, 19), np.linspace(-1.6, 1.6, 13))
        U = np.zeros_like(qu); S = np.zeros_like(qs)
        for p in range(qu.shape[0]):
            for c in range(qu.shape[1]):
                m = qu[p, c] * V3 + qs[p, c] * Vm3
                f = mf_flow(np.clip((m + 1) / 2, 1e-4, 1 - 1e-4), j, b, theta)
                U[p, c] = (2 * f) @ V3; S[p, c] = (2 * f) @ Vm3
        ax.quiver(qu, qs, U, S, color='white', alpha=0.5)
    colors = ['firebrick', 'orange', 'deepskyblue', 'navy', 'seagreen', 'purple']
    for i, (pth, lab) in enumerate(zip(paths, labels)):
        ax.plot(pth @ V3, pth @ Vm3, color=colors[i % 6], lw=2.5, label=lab)
    ax.set_xlabel(r'$\lambda=3$ (magnetization)'); ax.set_ylabel(ylab)
    ax.set_title('{} landscape ({}), J={}'.format(name, field, j))
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_landscape_{}_{}_{}_j{}.png'.format(name, trans, field, j), dpi=400, bbox_inches='tight')
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


def cached_map(j, seed='lam-3', recompute=False, theta=THETA, optimizer='adam'):
    """Minimum-action path (mean spin), cached. seed: 'lam-3' | 'lam1'.

    Adam (default) with tol_stop=-1 finds the true global MAP (lambda=-3) fast from
    either seed; its loss is non-monotonic so any finite tol_stop can fire
    mid-descent. From the lambda=1 seed, Adam's momentum carries it down to a
    lower-action, more-lambda=3 trajectory in the same lambda=-3 basin (not a
    failure). Plain GD stays in the higher lambda=1 saddle -- pass optimizer='gd'
    for that distinct route.
    """
    import mean_field_necker as mf
    w, v = np.linalg.eigh(theta)                       # ascending: [:,0]=lam-3, [:,-2]=lam1
    eig = None if seed == 'diagonal' else (v[:, 0] if seed == 'lam-3' else v[:, -2])
    def fn():
        if optimizer == 'adam':
            steps, step_size, tol = 60000, 3e-3, -1
        else:
            steps, step_size, tol = 400000, 1e-5, 1e-30
        _, x, _, _, _ = mf.minimum_action_path(j, 0, 0.1, theta.astype(int), numiters=500,
                                               dt=5e-3, steps=steps, tol_stop=tol,
                                               init='line', eig_dir=eig, init_noise=0,
                                               step_size=step_size, optimizer=optimizer)
        return 2 * x.detach().numpy() - 1
    return cached('map_{}_j{}'.format(seed, j), fn, recompute)


def plot_action_vs_time(paths, labels, j, b=0, theta=THETA, name='actiontime', dts=5e-3, noise=0.1):
    """
    Direct Freidlin-Wentzell action density |qdot - f|^2/(2 sigma^2) and its cumulative.

    No resampling: each path keeps its own points and its own dt (scalar or per-path
    list in dts); the time axis is normalized to each path's own duration (x in [0, 1])
    so the shapes overlay. Each total S is then that path's genuine action -- the
    optimiser MAPs recover ~390 (lambda=-3) / ~412 (diagonal) / ~416 (lambda=1).
    """
    os.makedirs(SV_FOLDER, exist_ok=True)
    colors = ['firebrick', 'royalblue', 'navy', 'orange', 'purple', 'seagreen']
    if np.isscalar(dts):
        dts = [dts] * len(paths)
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
    for i, (p, lab, pdt) in enumerate(zip(paths, labels, dts)):
        d = action_density(p, j, b, theta, pdt) / (2 * noise ** 2)
        tn = np.linspace(0, 1, len(p))
        ax[0].plot(tn, d, color=colors[i % 6], lw=2, label='{} (S={:.1f})'.format(lab, d.sum() * pdt))
        ax[1].plot(tn, np.cumsum(d) * pdt, color=colors[i % 6], lw=2, label=lab)
    ax[0].set_ylabel(r'action density $|\dot q-f|^2/2\sigma^2$')
    ax[0].set_title('instantaneous action, J={}'.format(j))
    ax[1].set_ylabel(r'cumulative action $S(t)$'); ax[1].set_title('cumulative action, J={}'.format(j))
    for a in ax:
        a.set_xlabel('time (normalized to each duration)')
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    ax[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'fig_{}_j{}.png'.format(name, j), dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'fig_{}_j{}.pdf'.format(name, j), dpi=400, bbox_inches='tight')
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
                yv = proj[:, reps[a]]
                axc.plot(np.arange(len(mp)), yv, color='firebrick', lw=2)
                if yv.max() - yv.min() < 2e-3:        # flatten 1e-16 noise
                    c0 = 0.5 * (yv.max() + yv.min())
                    axc.set_ylim(c0 - 1e-3, c0 + 1e-3)
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
    # plot_transitions(j_list=(0.6, 1.3))
    plot_eigen_traces(j=1.2, theta=THETA, n_show=80, smooth_mf=21)
    # plot_eigen_traces(j=0.6, theta=THETA, n_show=80, smooth_mf=21)
    plot_everything(j=1.2)
    # plot_everything(j=0.6)
    # plot_phase_grid(j=2.0, b=0, theta=THETA, path=None, name='MAP')
    plot_variance_envelope()
    plot_peak_projection_vs_J()
