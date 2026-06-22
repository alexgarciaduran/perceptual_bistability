# -*- coding: utf-8 -*-
"""
@author: alexg
"""

import os
import numpy as np
import networkx as nx
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm


mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams["axes.grid"] = False


sig = lambda x: 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
#  CONFIG: where sweep data and figures are saved / loaded from
# --------------------------------------------------------------------------- #
DATA_DIR = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/rnn_aux/undirected/'  # Alex


def get_graph(N=40, d=3, seed=0):
    G = nx.random_regular_graph(d, N, seed=seed)
    A = nx.to_numpy_array(G, dtype=int)
    iu, ju = np.where(np.triu(A, 1) > 0)
    return A, iu.astype(np.int64), ju.astype(np.int64)


@njit(cache=True, fastmath=True)
def _simulate(q0, up0, um0, ei, ej, zedge, K, wqu, wuq,
              tau_q, tau_u, noise, steps, dt, rq, rup, rum, rec_every):
    N = q0.shape[0]; E = ei.shape[0]
    q = q0.copy(); up = up0.copy(); um = um0.copy()
    sdt = np.sqrt(dt)
    nrec = steps // rec_every
    Q = np.empty((nrec, N)); UP = np.empty((nrec, E))
    ridx = 0
    for t in range(steps):
        upn = up.copy(); umn = um.copy()
        for k in range(E):
            i = ei[k]; j = ej[k]
            a = wqu*((2.0*q[i]-1.0) + (2.0*q[j]-1.0))     # symmetric: pair sum
            sp = 1.0/(1.0+np.exp(-(a + zedge[k])))
            sm = 1.0/(1.0+np.exp(-(a - zedge[k])))
            upn[k] = up[k] + dt*(sp - up[k])/tau_u
            umn[k] = um[k] + dt*(sm - um[k])/tau_u
            if noise > 0.0:
                upn[k] += noise*sdt*rup[t, k]
                umn[k] += noise*sdt*rum[t, k]
            if upn[k] < 1e-6: upn[k] = 1e-6
            elif upn[k] > 1.0-1e-6: upn[k] = 1.0-1e-6
            if umn[k] < 1e-6: umn[k] = 1e-6
            elif umn[k] > 1.0-1e-6: umn[k] = 1.0-1e-6
        inp = np.zeros(N)
        for k in range(E):
            i = ei[k]; j = ej[k]
            aux = wuq*((2.0*up[k]-1.0) + (2.0*um[k]-1.0))  # symmetric output
            inp[i] += K*(2.0*q[j]-1.0) + aux
            inp[j] += K*(2.0*q[i]-1.0) + aux
        qnew = q.copy()
        for i in range(N):
            s = 1.0/(1.0+np.exp(-inp[i]))
            qnew[i] = q[i] + dt*(s - q[i])/tau_q
            if noise > 0.0:
                qnew[i] += noise*sdt*rq[t, i]
            if qnew[i] < 1e-6: qnew[i] = 1e-6
            elif qnew[i] > 1.0-1e-6: qnew[i] = 1.0-1e-6
        q = qnew; up = upn; um = umn
        if (t % rec_every) == 0 and ridx < nrec:
            for i in range(N): Q[ridx, i] = q[i]
            for k in range(E): UP[ridx, k] = up[k]
            ridx += 1
    return q, up, um, Q, UP


def _biased_q0(N, seed, tilt_lo=0.05, tilt_hi=0.25, jitter=0.0):
    """Init near a small GLOBAL tilt so the network commits coherently
    (random q0 freezes into domains -> noisy branch scatter)."""
    rng = np.random.default_rng(seed)
    tilt = rng.choice(np.array([-1.0, 1.0])) * rng.uniform(tilt_lo, tilt_hi)
    return np.clip(0.5 + tilt + jitter*rng.standard_normal(N), 0.01, 0.99)


def simulate(ei, ej, N, z, K=0.2, wqu=1.5, wuq=1.0, tau_q=5.0, tau_u=0.5,
             noise=0.0, T=600.0, dt=0.05, seed=0, q0=None, rec_every=2,
             biased_init=True):
    rng = np.random.default_rng(seed)
    E = ei.shape[0]
    steps = int(T/dt)
    if q0 is None:
        q0 = _biased_q0(N, seed) if biased_init else rng.uniform(0.3, 0.7, N)
    zedge = np.full(E, float(z))
    a0 = wqu*((2*q0[ei]-1) + (2*q0[ej]-1))
    up0 = sig(a0 + zedge); um0 = sig(a0 - zedge)
    if noise > 0:
        rq = rng.standard_normal((steps, N))
        rup = rng.standard_normal((steps, E)); rum = rng.standard_normal((steps, E))
    else:
        rq = np.zeros((steps, N)); rup = np.zeros((steps, E)); rum = np.zeros((steps, E))
    return _simulate(q0.astype(np.float64), up0.astype(np.float64),
                     um0.astype(np.float64), ei, ej, zedge.astype(np.float64),
                     float(K), float(wqu), float(wuq), float(tau_q), float(tau_u),
                     float(noise), steps, float(dt), rq, rup, rum, int(rec_every))


def final_mean(ei, ej, N, z, n_inits=40, **kw):
    return np.array([simulate(ei, ej, N, z, seed=s, **kw)[0].mean()
                     for s in range(n_inits)])


def classify(means, lo=0.4, hi=0.6):
    mid = np.sum((means >= lo) & (means <= hi))
    if mid >= 0.8*len(means): return "monostable"
    if np.sum(means > hi) > 0 and np.sum(means < lo) > 0: return "bistable"
    return "committed-one-side"


def keff_matrix(ei, ej, N, z, wqu, wuq, K):
    """Effective q-q connectivity at q=1/2 for a given stimulus z."""
    eps = 1e-4
    M = np.zeros((N, N))
    q0 = np.full(N, 0.5)

    def q_input(q):
        a = wqu * ((2*q[ei]-1) + (2*q[ej]-1))      # symmetric pair-sum drive
        up = 1.0/(1.0+np.exp(-(a + z)))            # u+ : +z additive bias
        um = 1.0/(1.0+np.exp(-(a - z)))            # u- : -z additive bias
        aux = wuq * ((2*up-1) + (2*um-1))          # symmetric output
        inp = np.zeros(N)
        np.add.at(inp, ei, K*(2*q[ej]-1) + aux)    # fixed K + emergent aux term
        np.add.at(inp, ej, K*(2*q[ei]-1) + aux)
        return inp

    for k in range(len(ei)):
        i, j = ei[k], ej[k]
        for a_, b_ in ((i, j), (j, i), (i, i), (j, j)):
            qp = q0.copy(); qp[b_] += eps
            qm = q0.copy(); qm[b_] -= eps
            M[a_, b_] = (q_input(qp)[a_] - q_input(qm)[a_]) / (2*eps)
    return M


# --------------------------------------------------------------------------- #
#  z-sweep with save / load to a chosen folder
# --------------------------------------------------------------------------- #
def run_or_load_sweep(ei, ej, N, zs, pars, n_inits=40, T=600.0,
                      data_dir=DATA_DIR, fname="sweep.npz", force=False):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, fname)
    if (not force) and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"loaded sweep from {path}")
        return np.asarray(d["zs"]), d["means"]
    means = np.zeros((len(zs), n_inits))
    for zi, z in enumerate(tqdm(zs)):
        means[zi] = final_mean(ei, ej, N, z, n_inits=n_inits, T=T, **pars)
        print(f"  z={z:.2f} done", flush=True)
    np.savez(path, zs=np.asarray(zs), means=means,
             pars=np.array([pars], dtype=object))
    print(f"saved sweep to {path}")
    return np.asarray(zs), means


def solve_z(K, d, W):
    # step 1: compute A
    A = (2/d - K) / (8*W)

    # check validity
    disc = 1 - 4*A
    if disc < 0:
        raise ValueError("No real solution: discriminant < 0")

    sqrt_disc = np.sqrt(disc)

    # step 2: two possible sigma solutions
    s1 = (1 + sqrt_disc) / 2
    s2 = (1 - sqrt_disc) / 2

    def logit(s):
        return np.log(s / (1 - s))

    # avoid numerical issues (clip if needed)
    eps = 1e-12
    s1 = np.clip(s1, eps, 1 - eps)
    s2 = np.clip(s2, eps, 1 - eps)

    z1 = logit(s1)
    z2 = logit(s2)

    return z1, z2


def plot_critical_z_vs_W_K(W_list=np.arange(-0.1, 0.3, 1e-3),
                           J_list=np.arange(0., 0.5, 1e-4), d=4, z=0):
    W, J = np.meshgrid(W_list, J_list, indexing='ij')
    denom = 2 - J*d*2
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = 4*d*W/denom
    has_zc    = (denom > 0) & (arg >= 1)   # stim-induced bistability
    kbistable = (denom <= 0)               # K-induced (bistable for all z)
    zc = np.full(arg.shape, np.nan)
    zc[has_zc] = 2*np.arccosh(np.sqrt(arg[has_zc]))

    regime = np.zeros(arg.shape)
    regime[has_zc] = 1
    regime[kbistable] = 2

    extent = [J_list.min(), J_list.max(), W_list.min(), W_list.max()]
    cmap = ListedColormap(['#2c3e50', '#c0392b', '#e6b800'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # effective connectivity at z=0 (peak): J = 2K + 16*W*sigma'(0),  sigma'(0)=1/4
    Jeff = 4*J + 16*W*sig(z)*(1-sig(z))

    fig, ax = plt.subplots(ncols=3, figsize=(15.5, 3.7))
    im0 = ax[0].imshow(regime, aspect='auto', extent=extent, origin='lower',
                       cmap=cmap, norm=norm)
    ax[1].set_title('Regimes', fontsize=13)
    im1 = ax[1].imshow(zc, aspect='auto', extent=extent, origin='lower')
    ax[1].set_title(r'Critical stimulus $z_c$', fontsize=13)
    im2 = ax[2].imshow(Jeff, aspect='auto', extent=extent, origin='lower', cmap='magma')
    
    # overlay panel-1 regime boundaries on panel 3
    ax[2].contour(J, W, regime, levels=[0.5, 1.5], colors='cyan', linewidths=1.2)
    
    for a in ax:
        a.set_xlabel('Fixed coupling, J')
        a.set_ylabel(r'Stim-induced coupling, $W=w_{qu}\cdot w_{uq}$')
    cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, ticks=[0, 1, 2])
    cb0.ax.set_yticklabels(['Monostable', 'Stim-induced\nbistability',
                            'K-induced\nbistability'])
    fig.colorbar(im1, ax=ax[1], fraction=0.046, label=r'$z_c$')
    fig.colorbar(im2, ax=ax[2], fraction=0.046, label=r'$J_{eff}(z{=}$' + str(z) +r'$)=4K+8W$')
    ax[2].set_title(f'Effective connectivity (z={z})', fontsize=13)
    fig.tight_layout()
    for term in ['.png', '.svg']:
        f = os.path.join(DATA_DIR, "regime_vs_params"+term)
        fig.savefig(f, dpi=300, bbox_inches="tight")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    N, d = 50, 4
    A, ei, ej = get_graph(N=N, d=d, seed=0)
    PARS = dict(K=0.3, wqu=0.8, wuq=0.8, tau_q=10.0, tau_u=0.2, noise=0.0)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- symmetry check ----
    print("=== symmetry check: q=1/2 fixed for all z? ===")
    for z in [0.0, 1.0, 2.0, 4.0]:
        qf = simulate(ei, ej, N, z, q0=np.full(N, 0.5), T=600, **PARS)[0]
        print(f"  z={z:+.1f}: mean {qf.mean():.5f}  std {qf.std():.5f}")

    # ---- z sweep (cached in DATA_DIR) ----
    print("=== z sweep (save/load) ===")
    zs = np.linspace(0.0, 7.0, 101)
    zs, all_means = run_or_load_sweep(ei, ej, N, zs, PARS, n_inits=50, T=1000.0,
                                      data_dir=DATA_DIR, fname="sweep.npz",
                                      force=False)
    labels = [classify(m) for m in all_means]
    z_c = next((z for z, l in zip(zs, labels) if l == "monostable"), None)
    print("z_critical ~", z_c)

    # ---- example dynamics: a bistable z and a monostable z ----
    z_bi = 0.0
    z_mono = 6.
    ex = {}
    for tag, z in [(f"z={z_bi:.1f} (bistable)", z_bi),
                   (f"z={z_mono:.1f} (monostable)", z_mono)]:
        ex[tag] = simulate(ei, ej, N, z, seed=1, T=600.0, rec_every=2,
                           biased_init=False, **PARS)[3]

    # ---- effective connectivity matrices ----
    z_mats = [0.0, 1.0, 2.0, 3.0, 4.0]
    mats = [keff_matrix(ei, ej, N, z, PARS["wqu"], PARS["wuq"], PARS["K"])
            for z in z_mats]
    vmax = max(np.abs(m).max() for m in mats)

    # ====================== FIGURE 1: pitchfork + dynamics ==================
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for zi, z in enumerate(zs):
        ax[0].plot(np.full(all_means.shape[1], z), all_means[zi], 'o', ms=1.5,
                   color="#c0392b", alpha=0.55)
    ax[0].axhline(0.5, color="gray", lw=0.8, ls=":")
    zc_real = 2*np.arccosh(np.sqrt(2*d*PARS['wuq']*PARS['wqu'] / (2-PARS['K']*d)))
    
    zc1, zc2 = solve_z(PARS['K'], d, PARS['wuq']*PARS['wqu'])
    if z_c is not None:
        ax[0].axvline(z_c, color="#2471a3", lw=1.2, ls="--",
                      label=f"z_c ~ {z_c:.2f}")
        ax[0].axvline(zc_real, color="#2471a3", lw=1.2, ls="--",
                      label=f"z_c_true ~ {zc_real:.2f}")
        ax[0].legend(frameon=False)
    ax[0].set_xlabel("Stimulus: difference in velocity"); ax[0].set_ylabel("Fixed point")
    # ax[0].set_title("Pitchfork (no gating; K always on)")
    ax[0].set_ylim(-0.04, 1.04); ax[0].spines[["top", "right"]].set_visible(False)

    dt = 0.05; rec_every = 2
    
    for col, (tag, Q) in enumerate(ex.items()):
        a = ax[1 + col]
    
        t = np.arange(Q.shape[0]) * dt * rec_every
    
        for i in range(N):
            a.plot(
                t,
                Q[:, i],
                lw=0.8,
                alpha=0.7,
                color="#c0392b" if Q[-1, i] > 0.5 else "#2471a3"
            )
    
        a.set_title(tag, fontsize=14)
        a.set_xlabel("Time")
        a.set_ylabel("q_i")
        a.set_ylim(-0.04, 1.04)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    f1 = os.path.join(DATA_DIR, "pitchfork_and_dynamics.png")
    fig.savefig(f1, dpi=300, bbox_inches="tight")
    f1 = os.path.join(DATA_DIR, "pitchfork_and_dynamics.svg")
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    print("saved", f1)

    # ====================== FIGURE 2: effective connectivity ================
    fig, ax = plt.subplots(1, len(z_mats), figsize=(4*len(z_mats), 4.2))
    for k, (z, M) in enumerate(zip(z_mats, mats)):
        im = ax[k].imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[k].set_title(f"z={z}\nmean edge K_eff={M[A>0].mean():.3f}",
                        fontsize=13)
        ax[k].set_xlabel("q_j")
        if k == 0: ax[k].set_ylabel("q_i")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.tight_layout()
    f2 = os.path.join(DATA_DIR, "keff_vs_z.png")
    fig.savefig(f2, dpi=300, bbox_inches="tight")
    f2 = os.path.join(DATA_DIR, "keff_vs_z.svg")
    fig.savefig(f2, dpi=300, bbox_inches="tight")
    print("saved", f2)

    # ---- console summary ----
    m0 = all_means[0]; up = m0[m0 > 0.5]; dn = m0[m0 < 0.5]
    print(f"\nz=0 branches: up={up.mean():.3f}+-{up.std():.3f}, "
          f"dn={dn.mean():.3f}+-{dn.std():.3f}")
    for z, M in zip(z_mats, mats):
        print(f"  z={z}: mean edge K_eff = {M[A>0].mean():.4f}")
    
    
    # ---- plot matrices (regime & z_c) ----
    plot_critical_z_vs_W_K()
