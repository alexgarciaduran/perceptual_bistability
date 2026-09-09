# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
import posterior_computation_comparison as pc

THETA = pc.THETA_NECKER
N = THETA.shape[0]
_G = nx.from_numpy_array(THETA)
_DIST = dict(nx.all_pairs_shortest_path_length(_G))
DIST0 = np.array([_DIST[0][j] for j in range(N)])          # distance from vertex 0
B0 = 0.10
_MU, _V = np.linalg.eigh(THETA)                            # adjacency eigenbasis


# ---------------------------------------------------------------- engines ---
def _marginals(kind, J, B, alpha=1.0):
    Jm, Bv = J * THETA, np.full(N, B)
    if kind == 'exact':
        return pc.exact_marginals(Jm, Bv)
    if kind == 'mf':
        return (pc._mf_magnetization(Jm, Bv, np.zeros(N)) + 1) / 2
    return pc.fractional_bp(Jm, Bv, alpha=alpha, max_iter=6000, tol=1e-12)


def _confidence(kind, J, B, alpha=1.0):
    return float(_marginals(kind, J, B, alpha)[0])


def match_J(kind, q_target, alpha=1.0, B=B0, hi=3.0):
    """Find J such that the node marginal equals q_target at uniform field B."""
    f = lambda J: _confidence(kind, J, B, alpha) - q_target
    if f(1e-4) > 0 or f(hi) < 0:
        return np.nan
    return scipy.optimize.brentq(f, 1e-4, hi, xtol=1e-8)


def _lr_kind(kind):
    return {'exact': 'exact', 'mf': 'mf', 'fbp': 'lbp', 'lbp': 'lbp'}[kind]


def response_matrix(kind, J, alpha=1.0, B=B0):
    """chi_ij = d<x_i>/dB_j via the linear-response engine (FDT)."""
    C, _ = pc.linear_response_cov(_lr_kind(kind), J, B, alpha=alpha)
    return C


# ---------------------------------------------------------------- S1 --------
def cue_spread(kind, q_target, alpha=1.0):
    """r_d (response at graph distance d to a single-vertex cue) and rho=r1/r0,
    at matched confidence q_target. Returns (q, rho, r_profile[0..3])."""
    J = match_J(kind, q_target, alpha)
    if not np.isfinite(J):
        return np.nan, np.nan, np.full(4, np.nan)
    chi = response_matrix(kind, J, alpha)
    r = np.array([chi[DIST0 == d, 0].mean() for d in range(4)])
    return q_target, r[1] / r[0], r / r[0]


# ---------------------------------------------------------------- S3 --------
def onsager_R(kind, q_target, alpha=1.0):
    """R = (chi^-1)_ii (1 - m_i^2) at matched confidence."""
    J = match_J(kind, q_target, alpha)
    if not np.isfinite(J):
        return np.nan
    chi = response_matrix(kind, J, alpha)
    m = 2 * _marginals(kind, J, B0, alpha) - 1
    return float(np.linalg.inv(chi)[0, 0] * (1 - m[0] ** 2))


# ---------------------------------------------------------------- S2 --------
def eigenmode_precision(kind, q_target, alpha=1.0):
    """1/chi_k vs adjacency eigenvalue mu_k, and the affine-fit residual."""
    J = match_J(kind, q_target, alpha)
    if not np.isfinite(J):
        return _MU, np.full(N, np.nan), np.nan
    chi = response_matrix(kind, J, alpha)
    chik = np.einsum('ik,ij,jk->k', _V, chi, _V)            # v_k^T chi v_k
    inv = 1.0 / chik
    coef = np.polyfit(_MU, inv, 1)
    resid = float(np.sqrt(np.mean((inv - np.polyval(coef, _MU)) ** 2)))
    return _MU, inv, resid


# --------------------------------------------------- S(neg): hysteresis -----
def _roots_1d(kind, J, B, alpha=1.0, grid=np.linspace(-4, 4, 4001)):
    """Fixed points of the 1D uniform-mode map (variational schemes)."""
    if kind == 'mf':
        g = np.tanh(N * J * grid + B) - grid          # grid = m
    else:
        f = (1 / alpha) * np.arctanh(np.tanh(J * alpha) * np.tanh(grid * (N - alpha) + B))
        g = f - grid                                   # grid = M
    idx = np.flatnonzero(np.signbit(g[:-1]) != np.signbit(g[1:]))
    return len(idx)


def hysteresis_Bc(kind, q_star, alpha=1.0, dB=0.002, Bmax=1.5):
    """Evidence B_c at which the non-dominant branch folds, at equidominance
    confidence q_star (which sets J via the B=0 stable fixed point)."""
    # J from q_star at B=0
    def qstar_of_J(J):
        if kind == 'mf':
            m = pc._mf_magnetization(N * 0 + J * THETA, np.full(N, 0.0), 0.5 * np.ones(N))
            return (m[0] + 1) / 2
        q = pc.fractional_bp(J * THETA, np.zeros(N), alpha=alpha, max_iter=6000,
                             tol=1e-12, M_init=0.3 * (THETA != 0))
        return float(q[0])
    fJ = lambda J: qstar_of_J(J) - q_star
    if fJ(3.0) < 0:
        return np.nan
    J = scipy.optimize.brentq(fJ, 1e-3, 3.0, xtol=1e-6)
    for B in np.arange(0.0, Bmax, dB):
        if _roots_1d(kind, J, B, alpha) < 3:
            return float(B)
    return np.nan


# --------------------------------------------------- S4: frustration --------
def _signed_theta(flip_edges):
    T = THETA.copy()
    for (i, j) in flip_edges:
        T[i, j] = -1; T[j, i] = -1
    return T


def bp_history(Tsigned, J, B=0.0, alpha=1.0, damping=0.5, iters=200, seed=1):
    """Loopy BP (log-message) on a signed graph, recording the readout m_0(t).
    Records the whole trajectory so a limit cycle is visible."""
    rng = np.random.default_rng(seed)
    Jm = J * Tsigned
    M = (Tsigned != 0) * 0.01 * rng.standard_normal((N, N))
    hist = np.zeros(iters)
    for t in range(iters):
        Q = B + (M * (Tsigned != 0)).sum(0)
        newM = M.copy()
        for i in range(N):
            for j in range(N):
                if Tsigned[i, j] == 0:
                    continue
                h = Q[i] - alpha * M[j, i]
                v = (1 / alpha) * np.arctanh(np.tanh(Jm[i, j] * alpha) * np.tanh(h))
                newM[i, j] = damping * v + (1 - damping) * M[i, j]
        M = newM
        hist[t] = np.tanh(B + M[:, 0].sum())
    return hist


def gibbs_history(J, B=0.0, iters=200, seed=1):
    """Gibbs magnetisation m_0-ish trace (order parameter) for the noise line."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], N)
    Jm = J * THETA
    hist = np.zeros(iters)
    for t in range(iters):
        for i in range(N):
            h = B + Jm[i] @ s
            s[i] = 1.0 if rng.random() < 1 / (1 + np.exp(-2 * h)) else -1.0
        hist[t] = s.mean() * 2   # scale to overlay
    return hist


# -------------------------------------------------- coupling perturbation ---
# distance of each node from edge (0,1): min distance to either endpoint
_DIST_EDGE = np.array([min(_DIST[0][k], _DIST[1][k]) for k in range(N)])


def coupling_perturbation(kind, q_target, alpha=1.0, edge=(0, 1)):
    """d<x_k>/dJ_edge at matched confidence, grouped by graph distance of k from
    the perturbed edge. Returns (J, response_by_distance)."""
    J = match_J(kind, q_target, alpha)
    if not np.isfinite(J):
        return np.nan, np.full(4, np.nan)
    dJ = pc.coupling_response(_lr_kind(kind), J * THETA, np.full(N, B0), edge, alpha=alpha)
    prof = np.array([dJ[_DIST_EDGE == d].mean() if np.any(_DIST_EDGE == d) else np.nan
                     for d in range(4)])
    return J, prof


def plot_coupling_perturbation(q_target=0.80, save=True):
    """Spatial profile of the coupling response dm_k/dJ_{01} vs graph distance
    from the perturbed edge, at matched confidence -- the J-perturbation analog
    of S1's field-cue spread."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    dists = np.arange(4)
    for kind, a, c, mk, lab in [('exact', 1.0, 'k', 'o', 'exact / sampling'),
                                ('fbp', 1.0, 'C0', 's', 'loopy BP'),
                                ('mf', 1.0, 'r', '^', 'mean field')]:
        J, prof = coupling_perturbation(kind, q_target, a)
        ax[0].plot(dists, prof, mk + '-', color=c, label=f'{lab} (J={J:.2f})')
        ax[1].plot(dists, prof / prof[0], mk + '-', color=c, label=lab)
    ax[0].set(xlabel='graph distance from perturbed edge', ylabel=r'$dm_k/dJ_{01}$',
              xticks=dists, title=f'coupling response (matched q={q_target})')
    ax[1].set(xlabel='graph distance from perturbed edge',
              ylabel=r'$dm_k/dJ_{01}$ (normalised)', xticks=dists,
              title='normalised profile')
    for a_ in ax:
        a_.legend(frameon=False, fontsize=9)
        a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(pc.DATA_FOLDER + 'coupling_perturbation.png', dpi=200,
                    bbox_inches='tight')
    return fig


# ---------------------------------------------------------------- figure ----
def plot_signatures(save=True):
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    q_grid = np.round(np.arange(0.56, 0.87, 0.01), 3)
    schemes = [('exact', 1.0, 'k', '-', 'exact / sampling'),
               ('fbp', 2.0, 'orange', '--', r'FBP $\alpha$=2.0'),
               ('fbp', 1.5, 'purple', '--', r'FBP $\alpha$=1.5'),
               ('fbp', 1.0, 'C0', '-', 'loopy BP'),
               ('fbp', 0.5, 'green', '--', r'FBP $\alpha$=0.5'),
               ('mf', 1.0, 'r', '-', 'mean field')]

    # (a) rho vs q
    for kind, a, c, ls, lab in schemes:
        qs, rhos = [], []
        for q in q_grid:
            qq, rho, _ = cue_spread(kind, q, a)
            if np.isfinite(rho):
                qs.append(q); rhos.append(rho)
        ax[0, 0].plot(qs, rhos, ls, color=c, label=lab)
    ax[0, 0].set(xlabel='perceived confidence q', ylabel=r'spread index $\rho=r_1/r_0$',
                 title='(a) spread of a local cue vs confidence\nJ only runs along each curve')
    ax[0, 0].legend(fontsize=9, frameon=False)

    # (b) normalized response vs graph distance, matched at q=0.80
    for kind, c, lab in [('exact', 'k', 'exact / sampling'),
                         ('fbp', 'C0', 'loopy BP'), ('mf', 'r', 'mean field')]:
        J = match_J(kind, 0.80, 1.0)
        _, _, prof = cue_spread(kind, 0.80, 1.0)
        off = {'exact': -0.27, 'fbp': 0.0, 'mf': 0.27}[kind]
        ax[0, 1].bar(np.arange(4) + off, prof, width=0.27, color=c,
                     label=f'{lab}  (J={J:.2f})')
    ax[0, 1].set(xlabel='graph distance from the cued vertex',
                 ylabel='normalized response $r_d/r_0$', xticks=range(4),
                 title='(b) same data, matched at q = 0.80')
    ax[0, 1].legend(fontsize=9, frameon=False)

    # (c) 1/chi_k vs mu_k at q=0.80
    for kind, a, c, mk, lab in [('exact', 1.0, 'k', 'o', 'exact / sampling'),
                                ('fbp', 1.0, 'C0', 's', 'loopy BP'),
                                ('mf', 1.0, 'r', '^', 'mean field')]:
        mu, inv, resid = eigenmode_precision(kind, 0.80, a)
        inv = inv / np.abs(inv).max()
        ax[0, 2].scatter(mu, inv, color=c, marker=mk, s=60,
                         label=f'{lab} (resid {resid:.1e})')
        coef = np.polyfit(mu, inv, 1)
        xs = np.linspace(mu.min(), mu.max(), 10)
        ax[0, 2].plot(xs, np.polyval(coef, xs), color=c, lw=1, alpha=0.6)
    ax[0, 2].set(xlabel=r'adjacency eigenvalue $\mu_k$', ylabel=r'$1/\chi_k$ (normalised)',
                 title='(c) MF and BP are exactly affine in $\\mu_k$\nexact inference is not')
    ax[0, 2].legend(fontsize=9, frameon=False)

    # (d) Onsager ratio R vs q
    for kind, a, c, ls, lab in [('exact', 1.0, 'k', '-', 'exact / sampling'),
                                ('fbp', 1.0, 'C0', '-', 'loopy BP'),
                                ('fbp', 0.5, 'green', '--', r'FBP $\alpha$=0.5'),
                                ('fbp', 1.4, 'orange', '--', r'FBP $\alpha$=1.4'),
                                ('mf', 1.0, 'r', '-', 'mean field')]:
        qs, Rs = [], []
        for q in q_grid:
            R = onsager_R(kind, q, a)
            if np.isfinite(R):
                qs.append(q); Rs.append(R)
        ax[1, 0].plot(qs, Rs, ls, color=c, label=lab)
    ax[1, 0].set(xlabel='perceived confidence q', ylabel=r'$R=(\chi^{-1})_{ii}(1-m_i^2)$',
                 yscale='log', title='(d) Onsager ratio: flat / unimodal / divergent')
    ax[1, 0].legend(fontsize=9, frameon=False)

    # (e) hysteresis collapse (negative)
    qstar = np.round(np.arange(0.62, 0.995, 0.01), 3)
    for kind, a, c, lab in [('mf', 1.0, 'r', 'mean field'),
                            ('fbp', 0.5, 'green', r'FBP $\alpha$=0.5'),
                            ('fbp', 1.0, 'C0', 'loopy BP'),
                            ('fbp', 1.4, 'orange', r'FBP $\alpha$=1.4')]:
        Bc = np.array([hysteresis_Bc(kind, q, a) for q in qstar])
        ref = Bc[np.argmin(np.abs(qstar - 0.70))]
        ax[1, 1].plot(qstar, Bc / ref, color=c, label=lab)
    ax[1, 1].set(xlabel='confidence at equidominance $q^*$',
                 ylabel=r'$B_c/B_c(q^*=0.7)$', yscale='log',
                 title='(e) NEGATIVE: confidence vs hysteresis\ncollapses (<6% spread)')
    ax[1, 1].legend(fontsize=9, frameon=False)

    # (f) frustration: BP limit cycle vs noise-driven switching
    ax[1, 2].plot(gibbs_history(0.7, iters=200), color='gray', lw=0.8,
                  label='noise-driven switching')
    ax[1, 2].plot(bp_history(_signed_theta([(0, 1)]), J=1.1, iters=200),
                  color='purple', lw=1.5, label='BP on a frustrated graph')
    ax[1, 2].set(xlabel='iteration', ylabel='$m_0$',
                 title='(f) frustration: BP has a limit cycle\nMF and sampling never do')
    ax[1, 2].legend(fontsize=9, frameon=False)

    for a in ax.flat:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(pc.DATA_FOLDER + 'signatures.png', dpi=180, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    plot_signatures()
    plt.show()
