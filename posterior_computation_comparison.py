# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:30:09 2026

@author: alexg
"""

import numpy as np
import os
import itertools
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy.optimize
from numba import njit
from tqdm import tqdm


mpl.rcParams['font.size'] = 18
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 16
plt.rcParams['ytick.labelsize']= 16



DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/comparison_algorithms/'  # Alex

np.random.seed(0)

# ----------------------------
# Graph + Ising model creation
# ----------------------------
def generate_ising_graph(n, p):
    A = (np.random.rand(n, n) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T  # symmetric adjacency

    J = np.random.randn(n, n) * A
    J = (J + J.T) / 2  # ensure symmetry
    np.fill_diagonal(J, 0)

    B = np.random.randn(n)

    return A, J, B

# ----------------------------
# Exact inference (enumeration)
# ----------------------------
def exact_marginals(J, B):
    n = len(B)
    states = list(itertools.product([-1, 1], repeat=n))

    probs = []
    for s in states:
        s = np.array(s)
        energy = 0.5 * s @ J @ s + B @ s
        probs.append(np.exp(energy))
    probs = np.array(probs)
    probs /= probs.sum()

    marginals = np.zeros(n)
    for i in range(n):
        marginals[i] = sum(p for s, p in zip(states, probs) if s[i] == 1)

    return marginals

# ----------------------------
# Gibbs sampling (numba-accelerated)
# ----------------------------
@njit(cache=True)
def _gibbs_kernel(J, B, steps, burn_in):
    n = B.shape[0]
    s = np.where(np.random.random(n) < 0.5, -1.0, 1.0)
    counts = np.zeros(n)
    nsamp = 0
    for t in range(steps):
        for i in range(n):
            h = B[i]
            for k in range(n):
                h += J[i, k] * s[k]
            prob = 1.0 / (1.0 + np.exp(-2.0 * h))
            s[i] = 1.0 if np.random.random() < prob else -1.0
        if t >= burn_in:
            for i in range(n):
                if s[i] == 1.0:
                    counts[i] += 1.0
            nsamp += 1
    return counts / nsamp


def gibbs_sampling(J, B, steps=10000, burn_in=1000):
    return _gibbs_kernel(np.asarray(J, float), np.asarray(B, float),
                         int(steps), int(burn_in))

# ----------------------------
# Mean-field inference
# ----------------------------
def mean_field(J, B, max_iter=100, tol=1e-6):
    n = len(B)
    m = np.zeros(n)+np.random.randn(n)*0.01

    for _ in range(max_iter):
        m_new = np.tanh(B + J @ m)
        if np.max(np.abs(m_new - m)) < tol:
            break
        m = m_new

    return (m + 1) / 2  # convert to P(s=1)

# ----------------------------
# Loopy Belief Propagation
# ----------------------------
def loopy_bp(J, B, max_iter=100, tol=1e-6, alpha=1.0, damping=0.5):
    n = len(B)

    messages = np.zeros((n, n))+np.random.randn(n, n)*0.01

    for _ in range(max_iter):
        new_messages = np.zeros_like(messages)

        for i in range(n):
            for j in range(n):
                if J[i, j] == 0:
                    continue

                incoming = sum(alpha * messages[k, i] for k in range(n) if k != j)

                h = B[i] + incoming
                new_messages[i, j] = np.arctanh(np.tanh(J[i, j]) * np.tanh(h))/alpha
        # damping
        new_messages = damping*new_messages + (1-damping)*messages
        if np.max(np.abs(new_messages - messages)) < tol:
            break

        messages = new_messages

    marginals = np.zeros(n)
    for i in range(n):
        h = B[i] + messages[:, i].sum()
        marginals[i] = (1 + np.tanh(h)) / 2

    return marginals

# ----------------------------------------------------------------------------
# Fractional Belief Propagation (numba), paper log-ratio convention
# ----------------------------------------------------------------------------
@njit(cache=True)
def _fbp_kernel(J, B, alpha, M, max_iter, tol, damping):
    n = B.shape[0]
    for _ in range(max_iter):
        Q = np.empty(n)
        for i in range(n):
            acc = B[i]
            for k in range(n):
                if J[k, i] != 0.0:
                    acc += M[k, i]
            Q[i] = acc
        newM = M.copy()
        maxchange = 0.0
        for i in range(n):
            for j in range(n):
                if J[i, j] == 0.0:
                    continue
                h = Q[i] - alpha * M[j, i]          # cavity field for i->j
                val = (1.0 / alpha) * np.arctanh(np.tanh(J[i, j] * alpha) * np.tanh(h))
                val = damping * val + (1.0 - damping) * M[i, j]
                c = abs(val - M[i, j])
                if c > maxchange:
                    maxchange = c
                newM[i, j] = val
        M[:, :] = newM
        if maxchange < tol:
            break
    q = np.zeros(n)
    for i in range(n):
        acc = B[i]
        for k in range(n):
            if J[k, i] != 0.0:
                acc += M[k, i]
        q[i] = 1.0 / (1.0 + np.exp(-2.0 * acc))
    return q, M


def fractional_bp(J, B, alpha=1.0, max_iter=300, tol=1e-8, damping=0.5, seed=0,
                  M_init=None, return_messages=False):
    """Fractional BP marginals P(x_i=1) in the paper's log-ratio convention.

    alpha=1 recovers loopy BP; alpha->0 tends to mean field. Uses a numba
    kernel; the reverse message M_{j->i} enters the cavity field with weight
    alpha (the m_{j->i}^{1-alpha} term of the message update). Pass M_init (a
    converged message matrix) to warm-start -- needed for stable linear-response
    finite differences that must stay on one fixed-point branch."""
    n = len(B)
    if M_init is None:
        rng = np.random.default_rng(seed)
        M = (np.asarray(J) != 0.0).astype(np.float64) * 0.01 * rng.standard_normal((n, n))
    else:
        M = np.array(M_init, dtype=float)
    q, M = _fbp_kernel(np.asarray(J, float), np.asarray(B, float),
                       float(alpha), M, int(max_iter), float(tol), float(damping))
    return (q, M) if return_messages else q


def r_stim(x, j_e, b_e, n_neigh=3, alpha=1):
    return b_e*x**(n_neigh) - (j_e**alpha) * b_e * x**(n_neigh-alpha) + (j_e**alpha) * x**alpha - 1


def find_solution_bp(j, b, min_r=0., max_r=30, w_size=0.1,
                     tol=1e-2, n_neigh=3, alpha=1, max_sols=3):
    """Roots of the FBP fixed-point polynomial r_stim, via grid sign-change
    detection + bracketed bisection. Copied from loop_belief_prop_necker.py."""
    j_e = np.exp(2.0 * j)
    b_e = np.exp(2.0 * b)

    grid = np.arange(min_r, max_r + w_size, w_size)
    vals = r_stim(grid, j_e, b_e, n_neigh=n_neigh, alpha=alpha)

    idx = np.flatnonzero(np.signbit(vals[:-1]) != np.signbit(vals[1:]))

    sols = []
    for k in idx:
        root = scipy.optimize.bisect(r_stim, grid[k], grid[k + 1],
                                     args=(j_e, b_e, n_neigh, alpha), xtol=1e-12)
        if not sols or min(abs(root - s) for s in sols) > tol:
            sols.append(root)
            if len(sols) == max_sols:
                break
    return sols


def bistability_onset_J(alpha, b, n, J_scan, min_r=1e-9, max_r=30, w_size=0.05):
    """Smallest J in J_scan at which FBP becomes bistable at field b and degree
    n (i.e. g(r)=r_stim has >1 positive root -- the saddle-node/pitchfork onset).

    Valid for any b: at b=0 it reproduces the closed form J*(alpha)=
    1/(2a)log(n/(n-2a)); for b!=0 the transition is a saddle-node found here
    numerically. Returns np.nan if no fold occurs on J_scan."""
    for J in J_scan:
        sols = find_solution_bp(J, b, min_r=min_r, max_r=max_r, w_size=w_size,
                                tol=1e-2, n_neigh=n, alpha=alpha)
        if len(sols) > 1:
            return float(J)
    return np.nan


def jstar_curve(d, b, alpha_grid, J_max=1.0, dJ=0.01):
    """J*(alpha) at fixed field b for a degree-d node: onset J per alpha."""
    J_scan = np.arange(0.0, J_max + 1e-9, dJ)
    return np.array([bistability_onset_J(a, b, d, J_scan) for a in alpha_grid])


def d_kl_d_q(q, p):
    # derivative of D_KL(q || p) w.r.t. q  [minimised objective: D_KL(q||p)]
    return np.log(q/p) - np.log((1-q)/(1-p))


def _q_of_r(r, b, n=3):
    # FBP single-node marginal from the message ratio r, for a degree-n node
    return np.exp(b)*r**n / (np.exp(-b) + np.exp(b)*r**n)


def optimal_alpha_grad_descent(j, b, p, n=3, lr=1e-2, n_iter=5000, a0=2,
                               tol=1e-12, eps=1e-3, epsgrad=1e-10):
    """alpha minimising D_KL(q(alpha) || p) by AdaGrad + heavy-ball descent.

    Degree-aware: q(alpha) is the FBP marginal at the upper fixed point r(alpha)
    for a node with n neighbours, and p is the TRUE marginal of that same n-node
    system (passed in). dKL/dalpha = dKL/dq * dq/dalpha, with dq/dalpha a central
    finite difference (spacing 2*eps). Generalises the inference-mode
    optimal_alpha_grad_descent from loop_belief_prop_necker.py from n=3 to any n."""
    a = float(a0)
    alpha_vals = [a]
    der_memory = [0.0]
    grad_memory = 0.0
    for i in range(n_iter):
        sols_m = find_solution_bp(j, b, min_r=0., max_r=30, w_size=0.1,
                                  tol=1e-2, n_neigh=n, alpha=a-eps)
        sols_p = find_solution_bp(j, b, min_r=0., max_r=30, w_size=0.1,
                                  tol=1e-2, n_neigh=n, alpha=a+eps)
        if not sols_m or not sols_p:
            break  # no fixed point bracketed -> stop at current alpha
        q_a = _q_of_r(np.max(sols_m), b, n)        # q(alpha - eps)
        q_a_eps = _q_of_r(np.max(sols_p), b, n)    # q(alpha + eps)
        d_q_d_a = (q_a_eps - q_a)/eps/2             # central difference
        derivative = d_kl_d_q(q_a, p)*d_q_d_a
        grad_memory += derivative**2
        learning_rate = lr / (np.sqrt(grad_memory) + epsgrad)
        change_a = derivative*learning_rate + der_memory[i]*0.9
        a = a - change_a
        alpha_vals.append(a)
        der_memory.append(change_a)
        if np.abs(alpha_vals[i] - alpha_vals[i-1]) <= tol:
            break
    return a


# Memoised wrapper: alpha-hat now depends on (j, b, p, n) -- so degree d enters
# through n, and the true marginal p enters directly.
_OPT_ALPHA_CACHE = {}


def optimal_alpha(j, b, p, n):
    key = (round(float(j), 6), round(float(b), 6),
           round(float(p), 6), int(round(n)))
    if key not in _OPT_ALPHA_CACHE:
        _OPT_ALPHA_CACHE[key] = optimal_alpha_grad_descent(
            j, b, p, n=int(round(n)), lr=1e-2, tol=1e-7, epsgrad=1e-3)
    return _OPT_ALPHA_CACHE[key]


# ----------------------------
# Main experiments
# ----------------------------
def run_experiment_multi_p(p_list, N=30, n=8):
    all_results = {}

    for p in p_list:
        results = []

        for graph_id in tqdm(range(N), desc=f"multi_p p={p}"):
            A, J, B = generate_ising_graph(n, p)

            res = {
                "A": A,
                "J": J,
                "B": B,
            }

            # Exact
            res["exact"] = exact_marginals(J, B)

            # Gibbs
            res["gibbs"] = gibbs_sampling(J, B)

            # Mean-field
            res["mean_field"] = mean_field(J, B, max_iter=200)

            # LBP
            res["lbp"] = loopy_bp(J, B, alpha=1.0, max_iter=200)

            # Fractional BP (paper log-ratio convention)
            for alpha in [0.5, 0.75, 1.25, 1.5, 2, 2.5, 3]:
                res[f"fbp_{alpha}"] = fractional_bp(J, B, alpha=alpha, max_iter=200)

            # NB: no optimal-alpha row here. The homogeneous n-regular theory
            # does not transfer to heterogeneous Erdos-Renyi graphs: feeding it a
            # mean-field surrogate (mean j/b/degree) gives an alpha that is worse
            # than plain LBP (empirically 10-30x higher L2 at low p).

            results.append(res)

        all_results[p] = results

    return all_results


def plot_grid_by_method_and_p(all_results, methods=None):
    """
    Grid plot:
    - rows = methods
    - columns = p values
    - each cell = scatter (approx vs true)
    """

    if methods is None:
        methods = ["gibbs", "mean_field", "lbp",
                   "fbp_0.5", "fbp_1.5", "fbp_3"]

    method_names = ['Gibbs\nsampling', 'Mean-Field',
                    'LBP', r'FBP ($\alpha=0.5$)',
                    r'FBP ($\alpha=1.5$)', r'FBP ($\alpha=3$)']
    p_list = sorted(all_results.keys())
    p_list = [0.2, 0.4, 0.6, 0.8, 1]

    n_rows = len(methods)
    n_cols = len(p_list)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5*n_cols, 2.5*n_rows),
                             sharex=True, sharey=True)
    for a in axes.flatten():
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)

    # Handle edge case (1 row or 1 col)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, method in enumerate(methods):
        for j, p in enumerate(p_list):

            ax = axes[i, j]

            x_all, y_all = [], []

            for res in all_results[p]:
                x_all.extend(res["exact"])
                y_all.extend(res[method])

            x_all = np.array(x_all)
            y_all = np.array(y_all)

            ax.plot([-0.1, 1.1], [-0.1, 1.1], linestyle="--",
                    color='k', zorder=1)
            ax.scatter(x_all, y_all, alpha=0.4, s=10, zorder=20)

            # Titles (top row)
            if i == 0:
                ax.set_title(f"p = {p}")

            # Row labels (left column)
            if j == 0:
                ax.set_ylabel(method_names[i])

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            if i == (len(methods)-1):
                ax.set_xlabel('Exact')

    fig.tight_layout()
    plt.show()
    fig.savefig(DATA_FOLDER + 'multi_p_erdos_renyi.png')
    fig.savefig(DATA_FOLDER + 'multi_p_erdos_renyi.svg')


# ----------------------------
# Graph generator
# ----------------------------
def get_regular_graph(d=4, n=10, seed=None):
    G = nx.random_regular_graph(d, n, seed=seed)
    A = nx.to_numpy_array(G, dtype=int)
    return A

# ----------------------------
# Create Ising parameters with variable B
# ----------------------------
def create_ising_params_with_B(A, J_val, B_val):
    """
    A: adjacency matrix
    J_val: coupling strength
    B_val: scalar external field
    """
    n = A.shape[0]
    J = A * J_val
    B = np.ones(n) * B_val  # variable external field
    return J, B


def run_regular_graph_experiment(d_list, J_list, n=10, N=30, fbp_alphas=[0.5,0.75,1.25,1.5]):
    """
    Runs all methods for all d-regular graphs and J values
    """
    results = {}
    B_list = np.linspace(-0.5, 0.5, N)

    for d in d_list:
        results[d] = {}
        A = get_regular_graph(d, n, seed=0)
        for J_val in tqdm(J_list, desc=f"regular d={d}"):
            results[d][J_val] = []

            for graph_id in range(N):
                B_val = B_list[graph_id]
                J_mat, B = create_ising_params_with_B(A, J_val, B_val)

                res = {"A": A, "J": J_mat, "B": B}

                # Exact
                res["exact"] = exact_marginals(J_mat, B)

                # Gibbs
                res["gibbs"] = gibbs_sampling(J_mat, B, steps=10000)

                # Mean-field
                res["mean_field"] = mean_field(J_mat, B, max_iter=200)

                # LBP
                res["lbp"] = loopy_bp(J_mat, B, max_iter=200)

                # Fractional BP (paper log-ratio convention)
                for alpha in [0.5, 0.75, 1.25, 1.5, 2, 2.5, 3]:
                    res[f"fbp_{alpha}"] = fractional_bp(J_mat, B, alpha=alpha, max_iter=200)

                # Fractional BP at the degree-aware optimal (inference) alpha.
                # (J_val, B_val) scalar, every node has degree d -> n = d,
                # true marginal p = mean(exact). Same convention as g(r), so
                # fbp_opt sits on the diagonal (up to loop effects).
                res["alpha_opt"] = optimal_alpha(J_val, B_val,
                                                 float(np.mean(res["exact"])), d)
                res["fbp_opt"] = fractional_bp(J_mat, B, alpha=res["alpha_opt"], max_iter=200)

                results[d][J_val].append(res)

    return results


def plot_regular_results_dcolor(all_results, d_list, J_list=None, methods=None, N=30):
    """
    Scatter plot: rows = methods, columns = J values, color = d

    Parameters
    ----------
    all_results : dict
        results_regular[d][J_val][graph_id][method]
    d_list : list
        degrees to include (used for color)
    J_list : list
        J values to include (columns)
    methods : list
        inference methods (rows)
    N : int
        number of graphs to include per J (here 1)
    """
    if methods is None:
        methods = ["gibbs", "mean_field", "lbp",
                   "fbp_2", "fbp_3", "fbp_opt"]
    method_names = ['Gibbs\nsampling', 'Mean-Field',
                    'LBP', r'FBP ($\alpha=2$)',
                    r'FBP ($\alpha=3$)', r'FBP ($\hat{\alpha}$)']

    if J_list is None:
        # assume all d have same J values
        J_list = sorted(all_results[d_list[0]].keys())
    J_list = [0., 0.2, 0.4, 0.6, 0.8, 1]

    n_rows = len(methods)
    n_cols = len(J_list)

    # colormap for d
    cmap = cm.get_cmap("viridis", len(d_list))
    d_to_color = {d: cmap(i) for i, d in enumerate(d_list)}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5*n_cols, 2.5*n_rows),
                             sharex=True, sharey=True)

    for a in axes.flatten():
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, method in enumerate(methods):
        for j, J_val in enumerate(J_list):
            ax = axes[i, j]
            # diagonal line
            ax.plot([0, 1], [0, 1], linestyle="--", color='k', lw=0.8)
            for d in d_list:
                for n in range(N):
                    res = all_results[d][J_val][n]  # N=1 graph
                    x = res["exact"]
                    y = res[method]
                    ax.scatter(x, y, alpha=0.8, s=20, color=d_to_color[d], label=f"d={d}")

            
            # column title
            if i == 0:
                ax.set_title(f"J = {J_val:.2f}")

            # row label
            if j == 0:
                ax.set_ylabel(method_names[i])

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)

            # remove inner ticks
            if i < n_rows - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == (len(methods)-1):
                ax.set_xlabel('Exact')

    # create one legend for all
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=d_to_color[d], markersize=8, label=f"d={d}")
               for d in d_list]
    fig.legend(handles=handles, loc='upper right', title="d")
    plt.tight_layout()
    plt.show()
    fig.savefig(DATA_FOLDER + 'multi_dJ_regular.png')
    fig.savefig(DATA_FOLDER + 'multi_dJ_regular.svg')


def compute_error_vs_alpha(d_list=(2, 3, 4, 5, 6), B_values=(0.0, 0.1, 0.3, 0.5),
                           J_grid=np.round(np.arange(0.0, 1.01, 0.04), 3),
                           alpha_grid=np.linspace(0.1, 3.5, 40),
                           n=8, max_iter=300):
    """Compute FBP error maps E[(d,B)] = mean|q_FBP(alpha,J) - exact| (nodes),
    and analytic alpha_hat per (d,B,J). Returned as a dict ready to pickle."""
    def _kl(q, p, eps=1e-9):
        q = np.clip(q, eps, 1 - eps)
        p = np.clip(p, eps, 1 - eps)
        return np.mean(q*np.log(q/p) + (1-q)*np.log((1-q)/(1-p)))

    mae, mse, kl = {}, {}, {}
    ah = {}
    jstar = {}
    Jmax = float(np.asarray(J_grid)[-1])
    for d in d_list:
        A = get_regular_graph(d, n, seed=0)
        for B0 in B_values:
            Emae = np.zeros((len(alpha_grid), len(J_grid)))
            Emse = np.zeros((len(alpha_grid), len(J_grid)))
            Ekl = np.zeros((len(alpha_grid), len(J_grid)))
            ahl = []
            for jj, J in enumerate(tqdm(J_grid, desc=f"err(alpha) d={d} B={B0}")):
                Jm, Bv = create_ising_params_with_B(A, J, B0)
                ex = exact_marginals(Jm, Bv)
                for ai, a in enumerate(alpha_grid):
                    q = fractional_bp(Jm, Bv, alpha=a, max_iter=max_iter)
                    Emae[ai, jj] = np.mean(np.abs(q - ex))
                    Emse[ai, jj] = np.mean((q - ex)**2)
                    Ekl[ai, jj] = _kl(q, ex)
                ahl.append(optimal_alpha(J, B0, float(np.mean(ex)), d))
            key = (d, round(float(B0), 4))
            mae[key], mse[key], kl[key] = Emae, Emse, Ekl
            ah[key] = np.array(ahl)
            # numeric bistability onset J*(alpha) at this field B (saddle-node
            # for B!=0, pitchfork for B=0) -- the per-B red curve
            jstar[key] = jstar_curve(d, B0, alpha_grid, J_max=Jmax)
    return {"mae": mae, "mse": mse, "kl": kl, "ah": ah, "jstar": jstar,
            "J_grid": np.asarray(J_grid), "alpha_grid": np.asarray(alpha_grid),
            "d_list": list(d_list), "B_values": [round(float(b), 4) for b in B_values]}


def plot_error_vs_alpha_regular(d_list=(2, 3, 4, 5, 6),
                                B_values=(0.0, 0.1, 0.3, 0.5),
                                J_grid=np.round(np.arange(0.0, 1.01, 0.04), 3),
                                alpha_grid=np.linspace(0.1, 3.5, 40),
                                n=8, max_iter=300, metric='kl',
                                load_data=True, data_path=None, save=True):
    """
    Grid of FBP error heatmaps: rows = field B, columns = degree d.

    metric : 'kl' (mean D_KL(q||exact), the minimised objective -- default),
             'mse' (mean squared error) or 'mae' (mean abs error). The white
             ridge is the argmin of the displayed metric; for 'kl' it coincides
             with the analytic alpha_hat (cyan) by construction.
    """
    if metric not in ('kl', 'mse', 'mae'):
        raise ValueError("metric must be 'kl', 'mse' or 'mae'")
    if data_path is None:
        data_path = DATA_FOLDER + 'error_vs_alpha_data.pkl'
    cache = None
    if load_data and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            cache = pickle.load(f)
        if metric not in cache:   # old-schema cache -> recompute
            print("cache lacks metric maps; recomputing ...")
            cache = None
        else:
            print(f"loaded error-vs-alpha data from {data_path}")
    if cache is None:
        cache = compute_error_vs_alpha(d_list, B_values, J_grid, alpha_grid,
                                       n=n, max_iter=max_iter)
        with open(data_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"saved error-vs-alpha data to {data_path}")

    E, ah = cache[metric], cache["ah"]
    metric_label = {'kl': r'mean $D_{KL}(q\,\|\,$exact$)$',
                    'mse': 'mean squared error',
                    'mae': 'mean |q - exact|'}[metric]
    J_grid, alpha_grid = cache["J_grid"], cache["alpha_grid"]
    d_list, B_values = cache["d_list"], cache["B_values"]
    # backward-compat: older caches have no numeric onset -> compute + resave
    if "jstar" not in cache:
        Jmax = float(np.asarray(J_grid)[-1])
        cache["jstar"] = {(d, round(float(B0), 4)):
                          jstar_curve(d, B0, alpha_grid, J_max=Jmax)
                          for d in d_list for B0 in B_values}
        with open(data_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"added numeric J*(alpha,B) to {data_path}")
    jstar = cache["jstar"]
    vmax = max(np.nanmax(v) for v in E.values())

    nr, nc = len(B_values), len(d_list)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1*nc, 2.9*nr),
                             sharex=True, sharey=True, squeeze=False)
    for ri, B0 in enumerate(B_values):
        for ci, d in enumerate(d_list):
            ax = axes[ri, ci]
            Emat = E[(d, round(float(B0), 4))]
            im = ax.imshow(Emat, origin='lower', aspect='auto', cmap='viridis',
                           vmin=0, vmax=vmax,
                           extent=[J_grid[0], J_grid[-1], alpha_grid[0], alpha_grid[-1]])
            ridge = alpha_grid[np.argmin(Emat, axis=0)]
            ax.plot(J_grid, ridge, color='w', lw=1.8)
            ax.plot(J_grid, ah[(d, round(float(B0), 4))], color='cyan', lw=1.4, ls=':')
            # numeric bistability onset J*(alpha) at this B (red). Empty where
            # no fold exists (e.g. large B). B=0 matches the closed form.
            Js = jstar[(d, round(float(B0), 4))]
            ax.plot(Js, alpha_grid, color='r', lw=1.2, ls='--')
            ax.set_xlim(J_grid[0], J_grid[-1])
            ax.set_ylim(alpha_grid[0], alpha_grid[-1])
            if ri == 0:
                ax.set_title(f"d = {d}")
            if ci == 0:
                ax.set_ylabel(f"B = {B0}\n" + r"$\alpha$")
            if ri == nr - 1:
                ax.set_xlabel('Coupling J')
    # one legend (proxy handles) + shared colorbar
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='w', lw=2, label=r'empirical $\alpha^\ast$'),
               Line2D([0], [0], color='cyan', lw=2, ls=':', label=r'analytic $\hat\alpha$'),
               Line2D([0], [0], color='r', lw=2, ls='--', label=r'$J^\ast(\alpha,B)$')]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label(metric_label)
    if save:
        fig.savefig(DATA_FOLDER + f'error_vs_alpha_regular_{metric}.png', dpi=300,
                    bbox_inches='tight')
        fig.savefig(DATA_FOLDER + f'error_vs_alpha_regular_{metric}.svg',
                    bbox_inches='tight')
    return fig


# ============================================================================
# Algorithm-discriminating signatures
# ============================================================================
# Necker cube adjacency (3-regular, 8 nodes) -- from gibbs_necker.THETA.
THETA_NECKER = np.array([[0, 1, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0],
                         [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                         [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]],
                        dtype=float)


# ----------------------------------------------------------------------------
# #1  Critical coupling vs connectivity N  (theory / identifiability view)
# ----------------------------------------------------------------------------
def plot_critical_coupling_vs_N(N_list=np.arange(3, 13), alphas=(1.0, 1.5),
                                save=True):
    """
    J*(N) for MF (1/N) and FBP/LBP (numeric onset at B=0), in three views:
      (a) raw J*(N); (b) normalized J*(N)/J*(N0); (c) the product J*(N)*N.
    View (c) makes the identifiability caveat explicit: since only the effective
    coupling ~ J*lambda_max is observable and N (the PGM connectivity) is a
    modelling choice, curves that differ in (a)/(b) can collapse when the
    controllable quantity is the product J*N. MF gives J*N = 1 exactly (a flat
    line); BP-family curves are not flat, i.e. they *would* separate IF N were
    controllable -- which experimentally it is not.
    """
    N_list = np.asarray(N_list, float)
    J_scan = np.arange(0.0, 3.0001, 0.005)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    series = {'MF (1/N)': 1.0 / N_list}
    for a in alphas:
        lab = 'LBP' if a == 1.0 else rf'FBP $\alpha$={a}'
        series[lab] = np.array([bistability_onset_J(a, 0.0, int(N), J_scan,
                                                     w_size=0.02)
                                for N in N_list])
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(series)))
    for c, (lab, Js) in zip(colors, series.items()):
        axes[0].plot(N_list, Js, 'o-', color=c, label=lab)
        axes[1].plot(N_list, Js / Js[0], 'o-', color=c, label=lab)
        axes[2].plot(N_list, Js * N_list, 'o-', color=c, label=lab)
    for ax in axes:
        ax.set_xlabel('Connectivity N')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_ylabel(r'$J^\ast$'); axes[0].set_title('raw critical coupling')
    axes[1].set_ylabel(r'$J^\ast(N)/J^\ast(N_0)$'); axes[1].set_title('normalized')
    axes[2].set_ylabel(r'$J^\ast \cdot N$')
    axes[2].set_title('product view (only this axis is\nexperimentally accessible)')
    axes[0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'critical_coupling_vs_N.png', dpi=200,
                    bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# #3  Sampling vs variational: switching dynamics
# ----------------------------------------------------------------------------
@njit(cache=True)
def _gibbs_traj(J, B, steps, burn_in):
    """Gibbs trajectory: returns the magnetisation m_t = mean(s) per sweep."""
    n = B.shape[0]
    s = np.where(np.random.random(n) < 0.5, -1.0, 1.0)
    out = np.zeros(steps - burn_in)
    for t in range(steps):
        for i in range(n):
            h = B[i]
            for k in range(n):
                h += J[i, k] * s[k]
            s[i] = 1.0 if np.random.random() < 1.0/(1.0+np.exp(-2.0*h)) else -1.0
        if t >= burn_in:
            out[t - burn_in] = s.mean()
    return out


def langevin_1d(kind, J, B, N=3, alpha=1.0, sigma=0.25, tau=1.0, dt=0.01,
                T=4000.0, seed=0):
    """1D reduced Langevin for the variational schemes (returns q(t)).

    kind='mf' : dq = (sigmoid(2NJ(2q-1)+2B) - q) dt/tau + noise
    kind='fbp': dM = (f(M(N-a)+B,a) - M) dt/tau + noise, q = sigmoid(2(NM+B)),
                f(x,a) = (1/a) arctanh(tanh(Ja) tanh(x)).  (a=1 -> LBP)
    """
    rng = np.random.default_rng(seed)
    nstep = int(T / dt)
    s = np.sqrt(dt / tau) * sigma
    q = np.empty(nstep)
    if kind == 'mf':
        x = 0.5
        for t in range(nstep):
            x += (1.0/(1.0+np.exp(-(2*N*J*(2*x-1)+2*B))) - x)*dt/tau + s*rng.standard_normal()
            q[t] = x
    elif kind == 'fbp':
        M = 0.0
        for t in range(nstep):
            f = (1.0/alpha)*np.arctanh(np.tanh(J*alpha)*np.tanh(M*(N-alpha)+B))
            M += (f - M)*dt/tau + s*rng.standard_normal()
            q[t] = 1.0/(1.0+np.exp(-2.0*(N*M + B)))
    else:
        raise ValueError("kind must be 'mf' or 'fbp'")
    return q


def _schmitt_switches(order, hi, lo):
    """Count committed switches of a scalar series using a Schmitt trigger
    with thresholds (lo, hi); returns switch indices. Ignores small jitter
    around the centre so a monostable trace registers no switches."""
    state = 0  # +1 above hi, -1 below lo, 0 undecided
    idx = []
    for t, v in enumerate(order):
        if v > hi:
            if state == -1:
                idx.append(t)
            state = 1
        elif v < lo:
            if state == 1:
                idx.append(t)
            state = -1
    return np.array(idx)


def switching_dynamics(J, B=0.0, sigma=0.25, dt=0.01, T=4000.0,
                       gibbs_steps=120000, gibbs_burn=2000, seed=0):
    """Run Gibbs, MF-Langevin and LBP-Langevin on the Necker cube and return
    per-algorithm order-parameter traces + switch statistics (rate, dominance
    times). Sampling (Gibbs) and variational (MF/LBP) are compared by the SHAPE
    and J-scaling of these statistics, not absolute rates (which depend on the
    arbitrary noise/temperature)."""
    Jm = THETA_NECKER * J
    Bv = np.full(THETA_NECKER.shape[0], B)
    out = {}
    # Gibbs: magnetisation trace (order parameter in [-1,1])
    m = _gibbs_traj(Jm, Bv, int(gibbs_steps), int(gibbs_burn))
    sw = _schmitt_switches(m, 0.5, -0.5)
    out['gibbs'] = {'trace': m, 'switch_idx': sw,
                    'rate': len(sw)/len(m), 'dwell': np.diff(sw) if len(sw) > 1 else np.array([])}
    # Variational: q(t) in [0,1] -> centre at 0.5
    for kind, alpha in [('mf', None), ('fbp', 1.0)]:
        q = langevin_1d(kind, J, B, N=3, alpha=(alpha or 1.0), sigma=sigma,
                        dt=dt, T=T, seed=seed)
        sw = _schmitt_switches(q, 0.75, 0.25)
        out['mf' if kind == 'mf' else 'lbp'] = {
            'trace': q, 'switch_idx': sw,
            'rate': len(sw)/len(q), 'dwell': np.diff(sw)*dt if len(sw) > 1 else np.array([])}
    return out


def plot_sampling_vs_variational(J_traj=0.8, J_list=np.round(np.arange(0.3, 1.21, 0.1), 2),
                                 B=0.0, sigma=0.25, save=True):
    """
    Sampling vs variational discriminating dynamics on the Necker cube:
      (a) example order-parameter traces at J_traj;
      (b) switch rate vs coupling J (scaling / onset);
      (c) dominance-time distributions at J_traj (shape).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    dyn0 = switching_dynamics(J_traj, B=B, sigma=sigma)
    labels = {'gibbs': 'Gibbs (sampling)', 'mf': 'Mean-Field', 'lbp': 'LBP'}
    colors = {'gibbs': 'k', 'mf': 'r', 'lbp': 'C0'}
    # (a) traces (Gibbs magnetisation mapped to [0,1] for overlay)
    for key in ['gibbs', 'mf', 'lbp']:
        tr = dyn0[key]['trace']
        y = (tr + 1)/2 if key == 'gibbs' else tr
        axes[0].plot(np.linspace(0, 1, len(y))[:3000], y[:3000],
                     color=colors[key], lw=0.8, alpha=0.8, label=labels[key])
    axes[0].set_title(f'traces (J={J_traj}, B={B})')
    axes[0].set_xlabel('time (norm.)'); axes[0].set_ylabel('percept / q')
    axes[0].legend(frameon=False, fontsize=10)
    # (b) switch rate vs J
    rates = {k: [] for k in ['gibbs', 'mf', 'lbp']}
    for Jv in tqdm(J_list, desc='switch-rate vs J'):
        d = switching_dynamics(Jv, B=B, sigma=sigma)
        for k in rates:
            rates[k].append(d[k]['rate'])
    for k in rates:
        r = np.array(rates[k]); r = r/ (r.max() + 1e-12)
        axes[1].plot(J_list, r, 'o-', color=colors[k], label=labels[k])
    axes[1].axvline(0.5*np.log(3), color='gray', ls=':', label=r'LBP $J^\ast$')
    axes[1].set_title('switch rate vs J (norm.)')
    axes[1].set_xlabel('coupling J'); axes[1].set_ylabel('rate / max')
    axes[1].legend(frameon=False, fontsize=9)
    # (c) dominance-time distributions at J_traj
    for key in ['gibbs', 'mf', 'lbp']:
        dw = dyn0[key]['dwell']
        if len(dw) > 5:
            dw = dw / dw.mean()
            axes[2].hist(dw, bins=25, density=True, histtype='step',
                         color=colors[key], label=labels[key])
    axes[2].set_title(f'dominance times (J={J_traj}, norm.)')
    axes[2].set_xlabel('dwell / mean'); axes[2].set_ylabel('density')
    axes[2].legend(frameon=False, fontsize=10)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'sampling_vs_variational.png', dpi=200,
                    bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# A  Variability signature: does internal variance track the TRUE posterior?
# ----------------------------------------------------------------------------
# Sampling (Gibbs) *is* the posterior, so the variance of its percept trace
# equals the true posterior variance. A variational point estimate (MF/FBP
# Langevin) has a variance set by the injected noise + well curvature -- a floor
# decoupled from the true uncertainty. Coupling-invariant test (Orban 2016;
# Festa 2021): calibrate the variational noise to match the true variance at ONE
# operating point, then sweep -- sampling stays on the identity, variational
# drifts off it.
def exact_order_param_moments(J, B, theta=THETA_NECKER):
    """Mean and variance of the order parameter om=(mean_i x_i + 1)/2 under the
    exact Boltzmann distribution (uniform coupling J, field B on the cube)."""
    n = theta.shape[0]
    states = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
    k = 0.5 * J * np.einsum('si,ij,sj->s', states, theta, states) + B * states.sum(1)
    w = np.exp(k - k.max())
    w /= w.sum()
    om = (states.mean(1) + 1) / 2
    mean = float((w * om).sum())
    var = float((w * om**2).sum() - mean**2)
    return mean, var


def _percept_variance(kind, J, B, sigma, alpha=1.0, dt=0.01, T=3000.0,
                      gibbs_steps=150000, gibbs_burn=5000, seed=0):
    """Variance of the global percept in [0,1] for one scheme."""
    if kind == 'gibbs':
        m = _gibbs_traj(THETA_NECKER * J,
                        np.full(THETA_NECKER.shape[0], B),
                        int(gibbs_steps), int(gibbs_burn))
        return float(np.var((m + 1) / 2))
    q = langevin_1d('mf' if kind == 'mf' else 'fbp', J, B, N=3, alpha=alpha,
                    sigma=sigma, dt=dt, T=T, seed=seed)
    return float(np.var(q[len(q)//5:]))


def _calibrate_sigma(kind, J_ref, B_ref, target_var, alpha=1.0,
                     sigmas=np.linspace(0.05, 0.7, 14)):
    """Pick the variational noise sigma whose percept variance best matches
    target_var at the reference (J_ref, B_ref)."""
    errs = [abs(_percept_variance(kind, J_ref, B_ref, s, alpha=alpha) - target_var)
            for s in sigmas]
    return float(sigmas[int(np.argmin(errs))])


def plot_variability_signature(J_curve=0.5,
                               B_curve=np.round(np.linspace(-0.5, 0.5, 31), 3),
                               J_grid=(0.1, 0.2, 0.4, 0.5, 0.7, 0.8),
                               B_grid=np.round(np.linspace(-0.4, 0.4, 11), 3),
                               J_ref=0.4, B_ref=0.15, save=True):
    """
    (a) percept variance vs B at fixed coupling J_curve: true posterior (black),
        Gibbs (should overlap true), MF and LBP Langevin (noise floors,
        calibrated to true at the reference point);
    (b) internal variance vs true posterior variance across a (J,B) grid: Gibbs
        on the identity line, variational off it -- the coupling-invariant
        signature that sampling represents uncertainty and variational does not.
    """
    # calibrate variational noise once, at a monostable reference point
    _, var_ref = exact_order_param_moments(J_ref, B_ref)
    sig_mf = _calibrate_sigma('mf', J_ref, B_ref, var_ref)
    sig_lbp = _calibrate_sigma('lbp', J_ref, B_ref, var_ref, alpha=1.0)
    print(f"calibrated sigma: MF={sig_mf:.3f}, LBP={sig_lbp:.3f} (target var={var_ref:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) variance vs B at fixed J
    tv, gv, mv, lv = [], [], [], []
    for B0 in tqdm(B_curve, desc='var vs B'):
        tv.append(exact_order_param_moments(J_curve, B0)[1])
        gv.append(_percept_variance('gibbs', J_curve, B0, 0.0))
        mv.append(_percept_variance('mf', J_curve, B0, sig_mf))
        lv.append(_percept_variance('lbp', J_curve, B0, sig_lbp, alpha=1.0))
    axes[0].plot(B_curve, tv, 'k-', lw=2.5, label='true posterior')
    axes[0].plot(B_curve, gv, 'o-', color='0.4', ms=4, label='Gibbs (sampling)')
    axes[0].plot(B_curve, mv, 's-', color='r', ms=4, label='Mean-Field')
    axes[0].plot(B_curve, lv, '^-', color='C0', ms=4, label='LBP')
    axes[0].set_xlabel('Sensory evidence B'); axes[0].set_ylabel('percept variance')
    axes[0].set_title(f'variance vs evidence (J={J_curve})')
    axes[0].legend(frameon=False, fontsize=10)

    # (b) internal var vs true var over a grid
    pts = {'gibbs': ([], []), 'mf': ([], []), 'lbp': ([], [])}
    for J0 in tqdm(J_grid, desc='var scatter'):
        for B0 in B_grid:
            true_v = exact_order_param_moments(J0, B0)[1]
            pts['gibbs'][0].append(true_v); pts['gibbs'][1].append(_percept_variance('gibbs', J0, B0, 0.0))
            pts['mf'][0].append(true_v);    pts['mf'][1].append(_percept_variance('mf', J0, B0, sig_mf))
            pts['lbp'][0].append(true_v);   pts['lbp'][1].append(_percept_variance('lbp', J0, B0, sig_lbp, alpha=1.0))
    mx = max(max(v[1]) for v in pts.values()) * 1.05
    axes[1].plot([0, mx], [0, mx], 'k--', lw=1, label='identity')
    for key, c, mk, lab in [('gibbs', '0.4', 'o', 'Gibbs (sampling)'),
                            ('mf', 'r', 's', 'Mean-Field'), ('lbp', 'C0', '^', 'LBP')]:
        axes[1].scatter(pts[key][0], pts[key][1], c=c, marker=mk, s=28, alpha=0.75, label=lab)
    axes[1].scatter([var_ref], [var_ref], marker='*', s=200, facecolor='none',
                    edgecolor='green', linewidth=1.6, label='calibration point', zorder=5)
    axes[1].set_xlabel('true posterior variance'); axes[1].set_ylabel('internal variance')
    axes[1].set_title('does variance track uncertainty?')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'variability_signature.png', dpi=200,
                    bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# A (rigorous)  Implied covariance via linear response (FDT), sigma-free
# ----------------------------------------------------------------------------
# Every scheme defines an implied covariance through Cov(x_i,x_j)=d<x_i>/dB_j.
# For the exact model this is an identity; for MF/BP it is the linear-response
# covariance of that algorithm's own marginals -- a parameter-free notion of the
# uncertainty each scheme represents, with no injected noise (no sigma). We
# compare the order-parameter variance Var(om)=(1/4n^2) sum_ij d<x_i>/dB_j
# across exact / Gibbs / MF / LBP. Warm-starting keeps the finite differences on
# one fixed-point branch.
def _mf_magnetization(Jmat, Bvec, m0, max_iter=2000, tol=1e-12):
    """Deterministic MF fixed point (magnetisation <x_i>), warm-started at m0."""
    m = m0.copy()
    for _ in range(max_iter):
        mn = np.tanh(Bvec + Jmat @ m)
        if np.max(np.abs(mn - m)) < tol:
            break
        m = mn
    return m


def linear_response_cov(kind, J, B, theta=THETA_NECKER, delta=1e-3, alpha=1.0,
                        max_iter=4000):
    """Implied covariance matrix C_ij = d<x_i>/dB_j (symmetric finite diff).

    kind in {'exact','mf','lbp'}. Uses spins <x_i> = 2*P(x_i=1) - 1. Warm-starts
    each perturbed solve at the unperturbed fixed point so the response is that
    of a single branch. Returns (C, var_om) with var_om = C.sum()/(4 n^2)."""
    n = theta.shape[0]
    Jmat = J * theta
    Bvec = np.full(n, float(B))

    if kind == 'exact':
        def spins(Bv):
            return 2.0 * exact_marginals(Jmat, Bv) - 1.0
        base = spins(Bvec)
        pert = lambda Bv: spins(Bv)
    elif kind == 'mf':
        base = _mf_magnetization(Jmat, Bvec, np.zeros(n), max_iter=max_iter)
        pert = lambda Bv: _mf_magnetization(Jmat, Bv, base, max_iter=max_iter)
    elif kind == 'lbp':
        q0, M0 = fractional_bp(Jmat, Bvec, alpha=alpha, max_iter=max_iter,
                               return_messages=True)
        base = 2.0 * q0 - 1.0
        pert = lambda Bv: 2.0 * fractional_bp(Jmat, Bv, alpha=alpha,
                                              max_iter=max_iter, M_init=M0) - 1.0
    else:
        raise ValueError("kind must be 'exact', 'mf' or 'lbp'")

    C = np.zeros((n, n))
    for j in range(n):
        Bp = Bvec.copy(); Bp[j] += delta
        Bm = Bvec.copy(); Bm[j] -= delta
        C[:, j] = (pert(Bp) - pert(Bm)) / (2.0 * delta)
    C = 0.5 * (C + C.T)                     # symmetrise (exact for 'exact')
    var_om = float(C.sum() / (4.0 * n**2))
    return C, var_om


def plot_variability_signature_lr(J_list=(0.3, 0.5),
                                  B_curve=np.round(np.linspace(-0.6, 0.6, 25), 3),
                                  gibbs_steps=400000, gibbs_burn=20000, save=True):
    """
    Rigorous, sigma-free variability signature: order-parameter variance Var(om)
    vs sensory evidence B, one panel per coupling J. Curves:
      - exact (enumeration, ground truth),
      - exact via linear response (correctness check; must overlap exact),
      - Gibbs (sample variance of the trace -> the sampling estimate),
      - MF and LBP linear-response covariance (each scheme's *implied* variance).
    No injected noise, no free scale: differences are intrinsic to the algorithm.
    """
    fig, axes = plt.subplots(1, len(J_list), figsize=(5.2*len(J_list), 4.2),
                             squeeze=False)
    for ax, J in zip(axes[0], J_list):
        v_true, v_lr_exact, v_gibbs, v_mf, v_lbp = [], [], [], [], []
        for B0 in tqdm(B_curve, desc=f'Var(om) vs B, J={J}'):
            v_true.append(exact_order_param_moments(J, B0)[1])
            v_lr_exact.append(linear_response_cov('exact', J, B0)[1])
            v_mf.append(linear_response_cov('mf', J, B0)[1])
            v_lbp.append(linear_response_cov('lbp', J, B0)[1])
            m = _gibbs_traj(THETA_NECKER * J, np.full(THETA_NECKER.shape[0], B0),
                            int(gibbs_steps), int(gibbs_burn))
            v_gibbs.append(float(np.var((m + 1) / 2)))
        ax.plot(B_curve, v_true, 'k-', lw=2.5, label='exact (enumeration)')
        ax.plot(B_curve, v_lr_exact, color='0.6', lw=4, alpha=0.4,
                label='exact via linear response')
        ax.plot(B_curve, v_gibbs, 'o', color='C2', ms=4, label='Gibbs (sampling)')
        ax.plot(B_curve, v_mf, 's-', color='r', ms=4, label='MF (linear response)')
        ax.plot(B_curve, v_lbp, '^-', color='C0', ms=4, label='LBP (linear response)')
        ax.set_xlabel('Sensory evidence B')
        ax.set_ylabel('order-parameter variance')
        ax.set_title(f'J = {J}')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'variability_signature_lr.png', dpi=200,
                    bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Task 1  Cycle-structure graph family: approximation error vs loopiness
# ----------------------------------------------------------------------------
def make_cycle_family(n=9, levels=(0, 1, 2, 4, 7, 11), n_graphs=6, seed=0):
    """Graph family indexed by loopiness L = number of edges added to a random
    spanning tree (= independent cycles E-N+1). L=0 is a tree; larger L adds
    short cycles. Returns {L: [adjacency, ...]}."""
    rng = np.random.default_rng(seed)
    try:
        rand_tree = nx.random_labeled_tree
    except AttributeError:
        rand_tree = nx.random_tree
    fam = {}
    for L in levels:
        graphs = []
        for _ in range(n_graphs):
            A = nx.to_numpy_array(rand_tree(n, seed=int(rng.integers(1_000_000_000))))
            non = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
            rng.shuffle(non)
            for (i, j) in non[:L]:
                A[i, j] = A[j, i] = 1.0
            graphs.append(A)
        fam[L] = graphs
    return fam


def run_cycle_family(fam, J_list=(0.3, 0.6, 0.9), B=0.1, alphas=(0.5, 1.0, 1.5)):
    """Mean |q_alg - exact| (per node, averaged over graphs) vs loopiness and J,
    for MF and FBP(alpha). Returns {(L, J): {method: err, 'degree': deg}}."""
    res = {}
    for L, graphs in fam.items():
        for J in tqdm(J_list, desc=f"cycle family L={L}"):
            acc = {'mf': [], **{f'fbp_{a}': [] for a in alphas}}
            degs = []
            for A in graphs:
                n = A.shape[0]
                Jm, Bv = A * J, np.full(n, B)
                ex = exact_marginals(Jm, Bv)
                degs.append(A.sum(1).mean())
                acc['mf'].append(np.mean(np.abs(mean_field(Jm, Bv, max_iter=500) - ex)))
                for a in alphas:
                    acc[f'fbp_{a}'].append(
                        np.mean(np.abs(fractional_bp(Jm, Bv, alpha=a, max_iter=3000) - ex)))
            res[(L, J)] = {k: float(np.mean(v)) for k, v in acc.items()}
            res[(L, J)]['degree'] = float(np.mean(degs))
    return res


def plot_cycle_family(n=9, levels=(0, 1, 2, 4, 7, 11), n_graphs=6,
                      J_list=(0.3, 0.6, 0.9), B=0.1, alphas=(0.5, 1.0, 1.5),
                      load_data=True, data_path=None, save=True):
    """Approximation error (vs exact) as cycle structure grows, per J. On a tree
    (L=0) BP is exact; error grows with the number of short cycles."""
    if data_path is None:
        data_path = DATA_FOLDER + 'cycle_family_data.pkl'
    if load_data and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            res = pickle.load(f)
    else:
        res = run_cycle_family(make_cycle_family(n, levels, n_graphs), J_list, B, alphas)
        with open(data_path, 'wb') as f:
            pickle.dump(res, f)
    methods = ['mf'] + [f'fbp_{a}' for a in alphas]
    labels = {'mf': 'MF', **{f'fbp_{a}': ('LBP' if a == 1.0 else f'FBP {a}') for a in alphas}}
    fig, axes = plt.subplots(1, len(J_list), figsize=(5*len(J_list), 4.2), squeeze=False)
    for ax, J in zip(axes[0], J_list):
        for meth in methods:
            ax.plot(levels, [res[(L, J)][meth] for L in levels], 'o-', label=labels[meth])
        deg = np.mean([res[(L, J)]['degree'] for L in levels])
        ax.set(xlabel='loopiness  L = independent cycles', ylabel='mean |q - exact|',
               title=f'J = {J}  (mean deg {deg:.1f})')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'cycle_family.png', dpi=200, bbox_inches='tight')
    return fig


def run_cycle_family_grid(fam, J_list=(0.2, 0.4, 0.6, 0.8),
                          B_list=np.round(np.linspace(-0.5, 0.5, 7), 3),
                          alphas=(0.5, 1.0, 1.5), gibbs_steps=8000):
    """Per-node marginals of every method vs exact, for each graph type (loopiness
    level), swept over J and B. Returns {L: [ {method: q_array, 'J':J,'exact':..}, ...]}
    with one entry per (graph, J, B) -- the raw data for the method x graph-type grid."""
    out = {}
    for L, graphs in fam.items():
        entries = []
        for A in tqdm(graphs, desc=f"cycle grid L={L}"):
            n = A.shape[0]
            for J in J_list:
                for B in B_list:
                    Jm, Bv = A * J, np.full(n, B)
                    e = {'J': J, 'B': B, 'exact': exact_marginals(Jm, Bv),
                         'gibbs': gibbs_sampling(Jm, Bv, steps=gibbs_steps),
                         'mean_field': mean_field(Jm, Bv, max_iter=500),
                         'lbp': fractional_bp(Jm, Bv, alpha=1.0, max_iter=3000)}
                    for a in alphas:
                        if a != 1.0:
                            e[f'fbp_{a}'] = fractional_bp(Jm, Bv, alpha=a, max_iter=3000)
                    entries.append(e)
        out[L] = entries
    return out


def plot_cycle_by_method_and_type(n=9, levels=(0, 1, 3, 6, 11), n_graphs=5,
                                  J_list=(0.2, 0.4, 0.6, 0.8),
                                  B_list=np.round(np.linspace(-0.5, 0.5, 7), 3),
                                  methods=None, gibbs_steps=8000,
                                  load_data=True, data_path=None, save=True):
    """Grid like plot_grid_by_method_and_p: rows = inference method, columns =
    graph type (loopiness L). Each cell scatters approximate q vs exact q over
    all graphs/nodes, coloured by the graph coupling J, swept over fields B."""
    if methods is None:
        methods = ['gibbs', 'mean_field', 'lbp', 'fbp_0.5', 'fbp_1.5']
    method_names = {'gibbs': 'Gibbs\nsampling', 'mean_field': 'Mean-Field',
                    'lbp': 'LBP', 'fbp_0.5': r'FBP ($\alpha$=0.5)',
                    'fbp_1.5': r'FBP ($\alpha$=1.5)'}
    if data_path is None:
        data_path = DATA_FOLDER + 'cycle_grid_data.pkl'
    if load_data and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            res = pickle.load(f)
        levels = res['levels']; J_list = res['J_list']
        data = res['data']
    else:
        fam = make_cycle_family(n, levels, n_graphs)
        data = run_cycle_family_grid(fam, J_list, B_list, gibbs_steps=gibbs_steps)
        with open(data_path, 'wb') as f:
            pickle.dump({'data': data, 'levels': levels, 'J_list': J_list}, f)

    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=min(J_list), vmax=max(J_list))
    cmap = cm.get_cmap('viridis')
    nr, nc = len(methods), len(levels)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6*nc, 2.6*nr),
                             sharex=True, sharey=True, squeeze=False)
    for i, meth in enumerate(methods):
        for j, L in enumerate(levels):
            ax = axes[i, j]
            ax.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', lw=0.8, zorder=1)
            x = np.concatenate([e['exact'] for e in data[L]])
            y = np.concatenate([e[meth] for e in data[L]])
            cvals = np.concatenate([np.full(len(e['exact']), e['J']) for e in data[L]])
            ax.scatter(x, y, c=cvals, cmap=cmap, norm=norm, s=8, alpha=0.5, zorder=20)
            ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_title(f'L = {L}')
            if j == 0:
                ax.set_ylabel(method_names.get(meth, meth))
            if i == nr - 1:
                ax.set_xlabel('Exact')
    fig.tight_layout()
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, label='coupling J')
    if save:
        fig.savefig(DATA_FOLDER + 'cycle_grid.png', dpi=180, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Task 2  Coupling perturbation: d<x_k>/dJ_ij (warm-started finite difference)
# ----------------------------------------------------------------------------
def coupling_response(kind, Jmat, Bvec, edge, delta=1e-3, alpha=1.0, max_iter=4000):
    """Response vector d<x_k>/dJ_ij (spins <x>=2q-1) to perturbing the single
    coupling on `edge`=(i,j). Warm-started so BP/MF stay on one branch.
    kind in {'exact','mf','lbp'} ('lbp' with alpha!=1 is FBP)."""
    i, j = edge
    n = len(Bvec)

    def spins(Jm, warm):
        if kind == 'exact':
            return 2 * exact_marginals(Jm, Bvec) - 1, None
        if kind == 'mf':
            m = _mf_magnetization(Jm, Bvec,
                                  warm if warm is not None else np.zeros(n),
                                  max_iter=max_iter)
            return m, m
        q, M = fractional_bp(Jm, Bvec, alpha=alpha, max_iter=max_iter,
                             tol=1e-12, M_init=warm, return_messages=True)
        return 2 * q - 1, M

    base, warm = spins(Jmat, None)
    Jp = Jmat.copy(); Jp[i, j] += delta; Jp[j, i] += delta
    Jm_ = Jmat.copy(); Jm_[i, j] -= delta; Jm_[j, i] -= delta
    mp, _ = spins(Jp, warm)
    mm, _ = spins(Jm_, warm)
    return (mp - mm) / (2 * delta)


# ----------------------------------------------------------------------------
# Input susceptibility  chi_ij = d<x_i>/dB_j  vs graph distance d(i,j)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _gibbs_cov_kernel(J, B, steps, burn_in):
    n = B.shape[0]
    s = np.where(np.random.random(n) < 0.5, -1.0, 1.0)
    sum1 = np.zeros(n)
    sum2 = np.zeros((n, n))
    nsamp = 0
    for t in range(steps):
        for i in range(n):
            h = B[i]
            for k in range(n):
                h += J[i, k] * s[k]
            prob = 1.0 / (1.0 + np.exp(-2.0 * h))
            s[i] = 1.0 if np.random.random() < prob else -1.0
        if t >= burn_in:
            for i in range(n):
                sum1[i] += s[i]
                for k in range(n):
                    sum2[i, k] += s[i] * s[k]
            nsamp += 1
    mean = sum1 / nsamp
    cov = sum2 / nsamp
    for i in range(n):
        for k in range(n):
            cov[i, k] -= mean[i] * mean[k]
    return cov


def gibbs_susceptibility(J, B, theta=THETA_NECKER, steps=300000, burn_in=20000):
    """Gibbs estimate of chi_ij = d<x_i>/dB_j = Cov(x_i, x_j) (fluctuation-
    dissipation theorem). This is the sampler's implied input-susceptibility and
    converges to the exact chi as steps -> inf."""
    n = theta.shape[0]
    return _gibbs_cov_kernel(theta * float(J), np.full(n, float(B)),
                             int(steps), int(burn_in))


def plot_input_susceptibility(J_list=(0.15, 0.30), B=0.0, alphas=(0.5, 1.0, 1.5),
                              gibbs_steps=300000, gibbs_burn=20000,
                              theta=THETA_NECKER, save=True):
    """Input susceptibility chi_ij = d<x_i>/dB_j on the Necker cube, grouped by
    graph distance d(i,j). Shows how a perturbation of the evidence at node j
    propagates to node i's marginal as a function of their separation, per
    algorithm. Exact = ground truth; Gibbs -> exact (sampling, via FDT);
    MF/LBP/FBP give each scheme's *implied* response and its (generally wrong)
    decay with distance. One panel per coupling J.

    Note the MF mean-field onset for a 3-regular graph is J*=1/3: as J -> 1/3
    the symmetric-branch MF susceptibility inflates, a visible signature."""
    G = nx.from_numpy_array(theta)
    D = dict(nx.all_pairs_shortest_path_length(G))
    n = theta.shape[0]
    dist = np.array([[D[i][j] for j in range(n)] for i in range(n)])
    dvals = np.arange(0, int(dist.max()) + 1)

    methods = [dict(kind='exact', lab='exact', c='k', ls='-'),
               dict(kind='gibbs', lab='Gibbs', c='0.5', ls=':')]
    ac = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))
    for a, c in zip(alphas, ac):
        lab = 'LBP' if abs(a - 1.0) < 1e-9 else rf'FBP $\alpha$={a}'
        methods.append(dict(kind='fbp', lab=lab, c=c, ls='-', alpha=a))
    methods.append(dict(kind='mf', lab='MF', c='r', ls='--'))

    fig, axes = plt.subplots(1, len(J_list), figsize=(5.8 * len(J_list), 4.6),
                             squeeze=False)
    for ax, J in zip(axes[0], J_list):
        for md in methods:
            if md['kind'] == 'gibbs':
                C = gibbs_susceptibility(J, B, theta, gibbs_steps, gibbs_burn)
            elif md['kind'] == 'fbp':
                C, _ = linear_response_cov('lbp', J, B, theta=theta, alpha=md['alpha'])
            else:
                C, _ = linear_response_cov(md['kind'], J, B, theta=theta)
            mu = np.array([C[dist == d].mean() for d in dvals])
            sd = np.array([C[dist == d].std() for d in dvals])
            ax.errorbar(dvals, mu, yerr=sd, fmt=md['ls'], marker='o', ms=5,
                        color=md['c'], label=md['lab'], capsize=3, lw=2)
        ax.set_xlabel('graph distance  d(i, j)')
        ax.set_ylabel(r'$\partial \langle x_i\rangle / \partial B_j$')
        ax.set_title(f'J = {J},  B = {B}')
        ax.set_xticks(dvals)
        ax.axhline(0, color='0.8', lw=1)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=11)
    fig.suptitle('Input susceptibility vs graph distance (Necker cube)')
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'input_susceptibility.png', dpi=200,
                    bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Susceptibility analyses: matched-q (fit J), ratios, and J-sweeps, per algorithm
# ----------------------------------------------------------------------------
def _susc_methods(alphas):
    """Ordered list of algorithms for the susceptibility plots. 'exact' doubles
    as sampling (they coincide); 'gibbs' is the finite-sample estimate."""
    ms = [dict(kind='exact', lab='exact/sampling', c='k', ls='-', alpha=1.0),
          dict(kind='mf', lab='MF', c='r', ls='--', alpha=1.0)]
    ac = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))
    for a, c in zip(alphas, ac):
        ms.append(dict(kind='fbp', c=c, ls='-', alpha=a,
                       lab=('LBP' if abs(a - 1.0) < 1e-9 else rf'FBP $\alpha$={a}')))
    return ms


def _dist_matrix(theta):
    Gr = nx.from_numpy_array(theta)
    D = dict(nx.all_pairs_shortest_path_length(Gr))
    n = theta.shape[0]
    return np.array([[D[i][j] for j in range(n)] for i in range(n)])


def _marg_q(kind, J, B, alpha, theta):
    """Mean perceived confidence q = P(x_i=1) at uniform (J, B)."""
    n = theta.shape[0]; Jm = J * theta; Bv = np.full(n, float(B))
    if kind in ('exact', 'gibbs'):
        m = 2 * exact_marginals(Jm, Bv) - 1
    elif kind == 'mf':
        m = _mf_magnetization(Jm, Bv, np.zeros(n))
    else:
        m = 2 * fractional_bp(Jm, Bv, alpha=alpha) - 1
    return float((m.mean() + 1) / 2)


def _fit_J_for_q(kind, q_target, B, alpha, theta, J_grid):
    """Coupling J at which algorithm `kind` reaches confidence q_target (field B)."""
    qs = np.array([_marg_q(kind, J, B, alpha, theta) for J in J_grid])
    return float(np.interp(q_target, qs, J_grid))


def _chi(kind, J, B, alpha, theta, gibbs=(300000, 20000)):
    """Susceptibility matrix chi_ij = d<x_i>/dB_j for one algorithm."""
    if kind == 'gibbs':
        return gibbs_susceptibility(J, B, theta, *gibbs)
    if kind == 'fbp':
        return linear_response_cov('lbp', J, B, theta=theta, alpha=alpha)[0]
    return linear_response_cov(kind, J, B, theta=theta)[0]


def _rd(C, dist, dvals):
    return np.array([C[dist == d].mean() for d in dvals])


def plot_susc_vs_q(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                   q_grid=np.round(np.linspace(0.55, 0.9, 8), 3),
                   J_grid=np.round(np.arange(0.0, 2.0, 0.01), 3),
                   theta=THETA_NECKER, save=True):
    """(1) Susceptibility r_d vs perceived confidence q, with J FIT per algorithm
    to reach each q (matched operating point). One panel per graph distance d;
    lines = algorithms. r_0 = self-susceptibility, r_1.. = response at distance d."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = _susc_methods(alphas)
    fig, axes = plt.subplots(1, len(dvals), figsize=(3.6 * len(dvals), 3.6), squeeze=False)
    for md in methods:
        R = np.full((len(q_grid), len(dvals)), np.nan)
        for iq, q in enumerate(q_grid):
            J = _fit_J_for_q(md['kind'], q, B, md['alpha'], theta, J_grid)
            R[iq] = _rd(_chi(md['kind'], J, B, md['alpha'], theta), dist, dvals)
        for d in dvals:
            axes[0][d].plot(q_grid, R[:, d], md['ls'], color=md['c'], marker='o',
                            ms=3, label=md['lab'])
    for d in dvals:
        axes[0][d].set(title=f'distance d={d}', xlabel='perceived confidence q',
                       ylabel=(r'$r_d=\partial\langle x_i\rangle/\partial B_j$' if d == 0 else ''))
        axes[0][d].spines['top'].set_visible(False); axes[0][d].spines['right'].set_visible(False)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Susceptibility vs confidence (J fit per q, B={B})'); fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'susc_vs_q.png', dpi=180, bbox_inches='tight')
    return fig


def plot_susc_ratios(q_star=0.8, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                     J_grid=np.round(np.arange(0.0, 2.0, 0.01), 3), include_gibbs=True,
                     gibbs=(400000, 30000), theta=THETA_NECKER, save=True):
    """(2) Response ratios r_0/r_d vs distance d at matched confidence q_star.
    r_0/r_d = how much stronger the self-response is than the response at distance
    d (a gauge-free number). Steeper => cue stays local; flatter => spreads."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = list(_susc_methods(alphas))
    if include_gibbs:
        methods.append(dict(kind='gibbs', lab='Gibbs', c='0.5', ls=':', alpha=1.0))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for md in methods:
        k_fit = 'exact' if md['kind'] == 'gibbs' else md['kind']   # gibbs shares exact's J(q)
        J = _fit_J_for_q(k_fit, q_star, B, md['alpha'], theta, J_grid)
        r = _rd(_chi(md['kind'], J, B, md['alpha'], theta, gibbs), dist, dvals)
        ax.plot(dvals, r / r[0], md['ls'], color=md['c'], marker='o', ms=6, label=md['lab'])
    ax.set(xlabel='graph distance d', ylabel=r'$r_d / r_0$',
           title=f'Normalised response vs distance at matched q={q_star} (B={B})')
    ax.set_xticks(dvals); ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + f'susc_ratios_q{q_star}.png', dpi=180, bbox_inches='tight')
    return fig


def plot_susc_vs_J(d=1, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                   J_grid=np.round(np.arange(0.05, 1.0, 0.05), 3), include_gibbs=False,
                   gibbs=(150000, 10000), theta=THETA_NECKER, save=True):
    """(3) Average susceptibility at a GIVEN distance d vs coupling J, all
    algorithms on one panel. Shows how the response at separation d grows (and,
    for MF, diverges near its spurious critical point) with coupling."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = list(_susc_methods(alphas))
    if include_gibbs:
        methods.append(dict(kind='gibbs', lab='Gibbs', c='0.5', ls=':', alpha=1.0))
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for md in methods:
        rd = [ _rd(_chi(md['kind'], J, B, md['alpha'], theta, gibbs), dist, dvals)[d]
               for J in J_grid ]
        ax.plot(J_grid, rd, md['ls'], color=md['c'], marker='.', ms=5, label=md['lab'])
    ax.set(xlabel='coupling J', ylabel=rf'$r_{{{d}}}$  (mean $\chi$ at distance {d})',
           title=f'Susceptibility at distance d={d} vs coupling (B={B})')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + f'susc_vs_J_d{d}.png', dpi=180, bbox_inches='tight')
    return fig


def plot_susc_overview(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                       q_grid=np.round(np.linspace(0.55, 0.9, 8), 3),
                       J_grid_q=np.round(np.arange(0.0, 2.0, 0.01), 3),
                       J_grid=np.round(np.arange(0.05, 1.0, 0.05), 3),
                       theta=THETA_NECKER, save=True):
    """(4) Combined susceptibility summary: (a) spread rho=r_1/r_0 vs confidence q
    (J fit per q) -- the S1 signature; (b) self r_0 and neighbour r_1 vs coupling
    J -- the raw scale/decay. Gives the vs-q and vs-J views side by side."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = _susc_methods(alphas)
    fig, (axq, axj) = plt.subplots(1, 2, figsize=(12, 4.8))
    for md in methods:
        rho = []
        for q in q_grid:
            J = _fit_J_for_q(md['kind'], q, B, md['alpha'], theta, J_grid_q)
            r = _rd(_chi(md['kind'], J, B, md['alpha'], theta), dist, dvals)
            rho.append(r[1] / r[0])
        axq.plot(q_grid, rho, md['ls'], color=md['c'], marker='o', ms=3, label=md['lab'])
        r0 = []; r1 = []
        for J in J_grid:
            r = _rd(_chi(md['kind'], J, B, md['alpha'], theta), dist, dvals)
            r0.append(r[0]); r1.append(r[1])
        axj.plot(J_grid, r1, md['ls'], color=md['c'], marker='.', ms=4, label=md['lab'])
    axq.set(xlabel='perceived confidence q', ylabel=r'spread $\rho=r_1/r_0$',
            title='(a) cue spread vs confidence (J fit per q)')
    axj.set(xlabel='coupling J', ylabel=r'neighbour response $r_1$',
            title='(b) neighbour susceptibility vs coupling')
    for ax in (axq, axj):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axq.legend(frameon=False, fontsize=8)
    fig.suptitle(f'Susceptibility overview (Necker, B={B})'); fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + 'susc_overview.png', dpi=180, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    p_list = np.round(np.arange(0.2, 1.01, 0.1), 2)
    d_list = list(range(2, 7))   # degrees 2-6
    J_list = np.round(np.arange(0., 1.01, 0.1), 2)  # J = 0.0, 0.1, ..., 1.0

    # Set REGENERATE = True to recompute the experiments (needed after any change
    # to the inference engines); otherwise cached pkls are loaded if present.
    REGENERATE = False
    # reg_pkl = DATA_FOLDER + "ising_results_multi_J_d.pkl"
    # p_pkl = DATA_FOLDER + "ising_results_multi_p.pkl"

    # # --- regular d-regular graphs: generate (once) or load ------------------
    # if REGENERATE or not os.path.exists(reg_pkl):
    #     print("Generating regular-graph experiment ...")
    #     results_regular = run_regular_graph_experiment(d_list, J_list, n=8, N=30)
    #     with open(reg_pkl, "wb") as f:
    #         pickle.dump(results_regular, f)
    #     print(f"Saved {reg_pkl}")
    # with open(reg_pkl, "rb") as f:
    #     all_results = pickle.load(f)
    # plot_regular_results_dcolor(all_results, d_list, N=30)

    # # --- Erdos-Renyi multi-p graphs: generate (once) or load ----------------
    # if REGENERATE or not os.path.exists(p_pkl):
    #     print("Generating multi-p (Erdos-Renyi) experiment ...")
    #     results = run_experiment_multi_p(p_list, N=30, n=8)
    #     with open(p_pkl, "wb") as f:
    #         pickle.dump(results, f)
    #     print(f"Saved {p_pkl}")
    # with open(p_pkl, "rb") as f:
    #     all_results = pickle.load(f)
    # plot_grid_by_method_and_p(all_results, methods=None)

    # # --- effect of alpha: error vs (alpha, J) grid over B (cached) ----------
    # plot_error_vs_alpha_regular(d_list=(2, 3, 4, 5, 6),
    #                             B_values=(0.0, 0.1, 0.3, 0.5),
    #                             load_data=True, metric='kl')
    
    
    # plot_cycle_by_method_and_type(n=9, levels=(0, 1, 3, 6, 11), n_graphs=5,
    #                               J_list=(0.2, 0.4, 0.6, 0.8),
    #                               B_list=np.repeat(np.round(np.linspace(-0.5, 0.5, 7), 3), 2),
    #                               methods=None, gibbs_steps=8000,
    #                               load_data=True, data_path=None, save=True)
    plot_input_susceptibility(J_list=(0.15, 0.30), B=0.0, alphas=(0.5, 1.0, 1.5),
                              gibbs_steps=300000, gibbs_burn=20000,
                              theta=THETA_NECKER, save=True)
