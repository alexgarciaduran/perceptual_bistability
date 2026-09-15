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


mpl.rcParams['font.size'] = 15
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 15
plt.rcParams['ytick.labelsize']= 15



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
        M = (np.asarray(J) != 0.0).astype(np.float64) * 0.3 * rng.standard_normal((n, n))
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
                                  methods=None, gibbs_steps=8000, symmetrize=True,
                                  inset_graphs=True,
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
            if symmetrize:                       # q <-> 1-q symmetry: mirror the cloud
                x = np.concatenate([x, 1 - x]); y = np.concatenate([y, 1 - y])
                cvals = np.concatenate([cvals, cvals])
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
    if inset_graphs:                             # small black graph cartoon per L (top row)
        fam_g = make_cycle_family(n, levels, n_graphs)   # deterministic (same seed) -> matches data
        for j, L in enumerate(levels):
            A = fam_g[L][0]; G = nx.from_numpy_array(A)
            iax = axes[0, j].inset_axes([0.02, 0.60, 0.38, 0.38])
            pg = nx.spring_layout(G, seed=1)
            nx.draw_networkx_edges(G, pg, ax=iax, edge_color='k', width=0.6)
            nx.draw_networkx_nodes(G, pg, ax=iax, node_color='k', node_size=10)
            iax.set_axis_off()
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        fig.savefig(DATA_FOLDER + 'cycle_grid.png', dpi=180)   # no tight bbox (insets can make it invalid)
    return fig


def _cycle_style(key):
    """Colour/linestyle/label for a cycle-family method key (plot_susc_vs_J scheme)."""
    if key == 'mean_field':
        return dict(c='firebrick', ls='--', lab='MF')
    if key == 'gibbs':
        return dict(c='0.5', ls=':', lab='Gibbs')
    if key == 'lbp':
        return dict(c=plt.cm.Blues(0.60), ls='-', lab='LBP')
    if key.startswith('fbp_'):
        a = float(key.split('_')[1])
        shade = {0.5: 0.42, 1.0: 0.60, 1.5: 0.78, 2.0: 0.97}.get(a, 0.6)
        return dict(c=plt.cm.Blues(shade), ls='-', lab=rf'FBP $\alpha$={a:g}')
    return dict(c='k', ls='-', lab=key)


def _clearest_graph(adjs):
    """Among candidate adjacency matrices, pick the (graph, layout) whose nodes are
    most separated (largest minimum pairwise distance) so edges/loops are countable."""
    best = None
    for A in adjs:
        G = nx.from_numpy_array(A)
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, k=1.5, iterations=300, seed=1)
        P = np.array(list(pos.values()))
        dmin = min(np.linalg.norm(P[i] - P[j])
                   for i in range(len(P)) for j in range(i + 1, len(P)))
        if best is None or dmin > best[2]:
            best = (G, pos, dmin)
    return best[0], best[1]


def plot_error_vs_complexity(n=9, levels=tuple(range(0, 12)), n_graphs=5,
                             cartoon_levels=(0, 1, 3, 5, 11),
                             J_list=(0.2, 0.4, 0.6, 0.8),
                             B_list=np.round(np.linspace(-0.5, 0.5, 7), 3),
                             metric='kl', xaxis='L', methods=None, gibbs_steps=10000,
                             signed=False, load_data=True, data_path=None, save=True):
    """Summary of algorithm behaviour vs graph complexity, with a top strip of graph
    cartoons per level. Panel (a): error to exact (metric='mse' or 'kl') averaged over
    nodes/graphs/J/B, one line per algorithm. Panel (b): signed over-confidence
    mean(|q-0.5| - |p_exact-0.5|) (>0 over-confident e.g. MF; <0 under-confident e.g.
    FBP alpha>1). xaxis='L' (independent cycles) or 'lambda' (mean largest adjacency
    eigenvalue). Reuses the cycle_grid_data.pkl cache."""
    if data_path is None:
        data_path = DATA_FOLDER + 'cycle_grid_data.pkl'
    if load_data and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            res = pickle.load(f)
        levels = res['levels']; data = res['data']
    else:
        data = run_cycle_family_grid(make_cycle_family(n, levels, n_graphs),
                                     J_list, B_list, gibbs_steps=gibbs_steps)
        with open(data_path, 'wb') as f:
            pickle.dump({'data': data, 'levels': levels, 'J_list': J_list}, f)

    if methods is None:                              # all inference keys present in the data
        skip = ('exact', 'J', 'B')
        keys = [k for k in data[levels[0]][0] if k not in skip]
        methods = [k for k in ('gibbs', 'mean_field', 'lbp') if k in keys] + \
                  sorted([k for k in keys if k.startswith('fbp_')], key=lambda s: float(s.split('_')[1]))

    fam = make_cycle_family(n, levels, n_graphs)      # deterministic -> matches data; for x + cartoons
    if xaxis == 'lambda':
        lams = [[float(np.max(np.linalg.eigvalsh(A))) for A in fam[L]] for L in levels]
        xvals = [float(np.mean(v)) for v in lams]
        xerr = [float(np.std(v) / np.sqrt(len(v))) for v in lams]   # SEM of lambda_max across graphs
        xlabel = r'$\lambda_{\max}$'
    else:
        xvals = [float(L) for L in levels]; xerr = None
        xlabel = 'Loopiness  L (independent cycles)'

    def err(p, q):
        if metric == 'mse':
            return (q - p) ** 2
        e = 1e-9; p = np.clip(p, e, 1 - e); q = np.clip(q, e, 1 - e)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    ylab_err = 'MSE to exact' if metric == 'mse' else r'KL(exact$\parallel$approx)'

    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 4], hspace=0.4, wspace=0.25)
    cl = [L for L in cartoon_levels if L in levels]   # cartoons only for a subset of levels
    gtop = gs[0, :].subgridspec(1, len(cl), wspace=0.25)
    for j, L in enumerate(cl):                        # top strip: clearest graph cartoon per shown level
        cax = fig.add_subplot(gtop[0, j])
        G, pg = _clearest_graph(fam[L])               # most-separated example, so loops are countable
        nx.draw_networkx_edges(G, pg, ax=cax, edge_color='k', width=0.8)
        nx.draw_networkx_nodes(G, pg, ax=cax, node_color='k', node_size=14)
        cax.set_axis_off(); cax.margins(0.18)
        lam = float(np.mean([np.max(np.linalg.eigvalsh(A)) for A in fam[L]]))
        cax.set_title(rf'$\lambda$={lam:.1f}' if xaxis == 'lambda' else f'L={L}',
                      fontsize=mpl.rcParams['font.size'] * 0.8)

    axE = fig.add_subplot(gs[1, 0]); axO = fig.add_subplot(gs[1, 1])
    for meth in methods:
        st = _cycle_style(meth)
        E, Ee, OC, OCe = [], [], [], []
        for L in levels:
            ents = data[L]
            # group entries by graph (graph-major order) for a per-graph SEM; else pool
            bs = len(ents) // n_graphs if n_graphs and len(ents) % n_graphs == 0 else 0
            blocks = [ents[g * bs:(g + 1) * bs] for g in range(n_graphs)] if bs else [[e] for e in ents]
            ev, ov = [], []
            for blk in blocks:
                p = np.concatenate([e['exact'] for e in blk])
                q = np.concatenate([e[meth] for e in blk])
                ev.append(float(err(p, q).mean()))
                # over-confidence = area between psychometric q(B) and exact, integrated over the
                # true posterior (as in plot_gibbs_jstar_overconfidence), per J, averaged over J
                byJ = {}
                for e in blk:
                    byJ.setdefault(e['J'], []).append(e)
                aJ = []
                for es in byJ.values():
                    es = sorted(es, key=lambda e: e['B'])
                    qexB = np.array([np.mean(e['exact']) for e in es])
                    qapB = np.array([np.mean(e[meth]) for e in es])
                    dd = (qapB - qexB) if signed else np.abs(qapB - qexB)
                    aJ.append(float(np.trapz(dd, qexB)))
                ov.append(float(np.mean(aJ)))
            ev, ov = np.array(ev), np.array(ov); s = np.sqrt(len(ev))
            E.append(ev.mean());  Ee.append(ev.std() / s)      # SEM across graphs
            OC.append(ov.mean()); OCe.append(ov.std() / s)
        axE.errorbar(xvals, E, yerr=Ee, xerr=xerr, fmt='o', ls=st['ls'], color=st['c'], ms=4,
                     capsize=2, label=st['lab'])
        axO.errorbar(xvals, OC, yerr=OCe, xerr=xerr, fmt='o', ls=st['ls'], color=st['c'], ms=4,
                     capsize=2, label=st['lab'])
    axE.set_xlabel(xlabel); axE.set_ylabel(ylab_err); # axE.set_title('(a) Error vs complexity')
    if signed:
        axO.axhline(0, color='k', lw=0.8)
    axO.set_xlabel(xlabel); axO.set_ylabel('Over-confidence (area vs exact)')  # int|q-q_exact| dq_exact
    for a_ in (axE, axO):
        a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    axO.legend(frameon=False)
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        fig.savefig(DATA_FOLDER + f'error_vs_complexity_{metric}_{xaxis}.png', dpi=400)
        fig.savefig(DATA_FOLDER + f'error_vs_complexity_{metric}_{xaxis}.svg', dpi=180)
    return fig


def plot_gibbs_jstar_overconfidence(
        T_grid=np.round(np.logspace(2, 6, 9)).astype(int),
        J_list=(0.5, 0.7, 0.9, 1.1),
        B_grid=np.round(np.linspace(0.0, 0.5, 11), 3),   # favored side (B>=0); B<0 is the mirror
        Bstar_list=(0.0, 0.1, 0.2, 0.3),
        n_seeds=10, c=10.0, tilt=6.0, burn=1000, node_mean=True, signed=False,
        theta=THETA_NECKER, recompute=False, save=True, fname='gibbs_jstar_overconfidence'):
    """Two panels for Gibbs sampling vs chain length T.
    LEFT: critical coupling J*_Gibbs(T,B)=(ln T + tilt*|B|)/c, one analytic line per B.
    RIGHT: over-confidence vs T, conditioned on coupling J. Over-confidence at (J,T) is
    the SIGNED area between the Gibbs and exact psychometric curves q(B), integrated
    against the true posterior, int (q_gibbs - q_exact) dq_exact (as in the MF/BP
    over-confidence), averaged over n_seeds chains (+/- SEM). It decays toward 0 as
    T->inf; at fixed T it grows with J (mixing slows ~ e^{cJ})."""
    import hashlib
    n = theta.shape[0]
    qfun = (lambda a: float(np.mean(a))) if node_mean else (lambda a: float(a[0]))

    # exact psychometric per J (deterministic, cheap)
    qexact = {J: np.array([qfun(exact_marginals(J * theta, np.full(n, B))) for B in B_grid])
              for J in J_list}

    # cache the (expensive, stochastic) Gibbs marginals QG[(J,T)] = array[n_seeds, len(B_grid)];
    # keyed by the sweep params only, so replots / signed toggles reload instead of resampling.
    key = repr((tuple(float(j) for j in J_list), tuple(int(t) for t in T_grid),
                tuple(float(b) for b in B_grid), int(n_seeds), int(burn), bool(node_mean),
                np.asarray(theta, float).tobytes()))
    cache_fn = os.path.join(DATA_FOLDER, 'gibbs_oc_qg_%s.pkl' % hashlib.md5(key.encode()).hexdigest()[:12])
    if not recompute and os.path.exists(cache_fn):
        with open(cache_fn, 'rb') as f:
            QG = pickle.load(f)
    else:
        QG = {}
        for J in J_list:
            Jm = J * theta
            for T in tqdm(T_grid, desc=f'Gibbs J={J}', leave=False):
                arr = np.empty((n_seeds, len(B_grid)))
                for s in range(n_seeds):
                    arr[s] = [qfun(gibbs_sampling(Jm, np.full(n, B), int(burn + T), int(burn)))
                              for B in B_grid]
                QG[(float(J), int(T))] = arr
        os.makedirs(DATA_FOLDER, exist_ok=True)
        with open(cache_fn, 'wb') as f:
            pickle.dump(QG, f)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 3.4))

    # LEFT: J*(T) per B
    colsB = plt.cm.Greens(np.linspace(0.2, 0.85, len(Bstar_list)))
    for B, col in zip(Bstar_list, colsB):
        axL.plot(T_grid, (np.log(T_grid) + tilt * abs(B)) / c, '-o', color=col, ms=3, label=B)
    axL.set_xscale('log'); axL.set_xlabel('Chain length T'); axL.set_ylabel(r'$J^\ast_{\mathrm{Gibbs}}(T)$')
    axL.legend(frameon=False, title='Evidence, B')
    axL.spines['top'].set_visible(False); axL.spines['right'].set_visible(False)

    # RIGHT: over-confidence (area between psychometrics) vs T, per J
    colsJ = plt.cm.plasma(np.linspace(0.1, 0.82, len(J_list)))
    for J, col in zip(J_list, colsJ):
        qex = qexact[J]
        means, sems = [], []
        for T in T_grid:
            qg = QG[(float(J), int(T))]                          # [n_seeds, len(B_grid)]
            d = (qg - qex) if signed else np.abs(qg - qex)       # signed (MF/BP-style) or absolute
            areas = np.trapz(d, qex, axis=1)                     # per-seed area over the true posterior
            means.append(areas.mean()); sems.append(areas.std() / np.sqrt(n_seeds))
        axR.errorbar(T_grid, means, yerr=sems, fmt='o-', color=col, ms=3, capsize=2, label=J)
    axR.set_xscale('log'); axR.set_xlabel('Chain length, T')
    axR.set_ylabel('Over-confidence')
    # axR.set_title('(b) Over-confidence vs T');
    axR.legend(frameon=False, title='Coupling, J')
    axR.spines['top'].set_visible(False); axR.spines['right'].set_visible(False)

    fig.tight_layout()
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=300)
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
        fig.savefig(DATA_FOLDER + 'input_susceptibility.svg', bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Susceptibility analyses: matched-q (fit J), ratios, and J-sweeps, per algorithm
# ----------------------------------------------------------------------------
def _susc_methods(alphas):
    """Ordered list of algorithms for the susceptibility plots. 'exact' doubles
    as sampling (they coincide); 'gibbs' is the finite-sample estimate."""
    ms = [dict(kind='exact', lab='exact/sampling', c='k', ls='-', alpha=1.0),
          dict(kind='mf', lab='MF', c='r', ls='--', alpha=1.0)]
    ac = plt.cm.Blues(np.linspace(0.15, 0.85, len(alphas)))
    for a, c in zip(alphas, ac):
        ms.append(dict(kind='fbp', c=c, ls='-', alpha=a,
                       lab=('LBP' if abs(a - 1.0) < 1e-9 else rf'FBP $\alpha$={a}')))
    return ms


# optimal-alpha (KL-optimal FBP) line, shared across the susceptibility plots.
# alpha is a placeholder (unused: alpha_hat is computed per (J,B) inside _chi/_marg_q).
_OPT_FIT_J = np.round(np.arange(0.0, 3.0, 0.05), 3)      # coarse grid for the alpha-fit


def _opt_method():
    return dict(kind='fbp_opt', lab=r'FBP $\hat\alpha$ (optimal)', c='#2ca02c', ls='-.', alpha=1.0)


def _jstar(kind, alpha, theta):
    """Closed-form critical coupling J* (bistability onset at B=0), generalised to
    any graph via the largest adjacency eigenvalue lambda_max: MF (alpha->0) gives
    1/lambda_max, FBP/LBP give (1/2a)log(l/(l-2a)). Returns nan when no bifurcation
    exists (alpha>=lambda_max/2) or for schemes without one (exact/gibbs/fbp_opt)."""
    lmax = float(np.max(np.linalg.eigvalsh(theta)))
    if kind == 'mf':
        return 1.0 / lmax
    if kind == 'fbp':
        return (1.0 / (2 * alpha)) * np.log(lmax / (lmax - 2 * alpha)) if lmax - 2 * alpha > 1e-6 else np.nan
    return np.nan            # exact, gibbs, fbp_opt: no closed-form onset


def _dist_matrix(theta):
    Gr = nx.from_numpy_array(theta)
    D = dict(nx.all_pairs_shortest_path_length(Gr))
    n = theta.shape[0]
    return np.array([[D[i][j] for j in range(n)] for i in range(n)])


def _alpha_hat(J, B, theta):
    """KL-optimal FBP alpha at (J, B) for this graph's degree, using the exact
    Necker marginal as the target p. Floored to a small positive value."""
    n_deg = int(round(theta.sum(1).mean()))
    p_true = _marg_q('exact', J, B, 1.0, theta)
    return max(float(optimal_alpha(J, B, p_true, n_deg)), 1e-2)


def _marg_q(kind, J, B, alpha, theta):
    """Mean posterior q = P(x_i=1) at uniform (J, B)."""
    n = theta.shape[0]; Jm = J * theta; Bv = np.full(n, float(B))
    if kind in ('exact', 'gibbs'):
        m = 2 * exact_marginals(Jm, Bv) - 1
    elif kind == 'mf':
        m = _mf_magnetization(Jm, Bv, np.zeros(n))
    elif kind == 'fbp_opt':
        m = 2 * fractional_bp(Jm, Bv, alpha=_alpha_hat(J, B, theta)) - 1
    else:
        m = 2 * fractional_bp(Jm, Bv, alpha=alpha) - 1
    return float((m.mean() + 1) / 2)


def _fit_J_for_q(kind, q_target, B, alpha, theta, J_grid):
    qs = np.array([_marg_q(kind, J, B, alpha, theta) for J in J_grid])
    if q_target > qs.max() + 1e-6 or q_target < qs.min() - 1e-6:
        return np.nan                      # unreachable q -> don't clamp
    return float(np.interp(q_target, qs, J_grid))


def _chi(kind, J, B, alpha, theta, gibbs=(300000, 20000)):
    """Susceptibility matrix chi_ij = d<x_i>/dB_j for one algorithm."""
    if kind == 'gibbs':
        return gibbs_susceptibility(J, B, theta, *gibbs)
    if kind == 'fbp_opt':
        return linear_response_cov('lbp', J, B, theta=theta, alpha=_alpha_hat(J, B, theta))[0]
    if kind == 'fbp':
        return linear_response_cov('lbp', J, B, theta=theta, alpha=alpha)[0]
    return linear_response_cov(kind, J, B, theta=theta)[0]


def _rd(C, dist, dvals):
    return np.array([C[dist == d].mean() for d in dvals])


def plot_susc_vs_q(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                   q_grid=np.round(np.linspace(0.55, 0.99, 12), 3),
                   J_grid=np.round(np.arange(0.0, 6.0, 0.02), 3),
                   theta=THETA_NECKER, save=True,
                   normalize_y=False):
    """(1) Susceptibility r_d vs perceived confidence q, with J FIT per algorithm
    to reach each q (matched operating point). One panel per graph distance d;
    lines = algorithms. r_0 = self-susceptibility, r_1.. = response at distance d.
    q values an algorithm cannot reach at this B are left blank (not clamped)."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = list(_susc_methods(alphas)) + [_opt_method()]
    fig, axes = plt.subplots(1, len(dvals), figsize=(3.3 * len(dvals), 3.2), squeeze=False)
    for md in methods:
        # optimal-alpha J(q) is smooth and expensive (alpha fit per J): use a
        # coarser grid for it; fixed-alpha lines keep the fine grid.
        Jg = np.round(np.arange(0.0, 3.0, 0.05), 3) if md['kind'] == 'fbp_opt' else J_grid
        R = np.full((len(q_grid), len(dvals)), np.nan)
        for iq, q in enumerate(q_grid):
            J = _fit_J_for_q(md['kind'], q, B, md['alpha'], theta, Jg)
            if np.isnan(J):
                continue                                   # q unreachable -> blank
            R[iq] = _rd(_chi(md['kind'], J, B, md['alpha'], theta), dist, dvals)
        for d in dvals:
            axes[0][d].plot(q_grid, R[:, d], md['ls'], color=md['c'], marker='o',
                            ms=3, label=md['lab'])
    for d in dvals:
        axes[0][d].set(title=f'distance d={d}', xlabel=r'Posterior probability',
                       ylabel=(r'$r_d=\partial\langle x_i\rangle/\partial B_j$' if d == 0 else ''))
        axes[0][d].spines['top'].set_visible(False); axes[0][d].spines['right'].set_visible(False)
        if normalize_y:
            axes[0][d].set_ylim(-0.05, 1.25)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        label_y = 'norm' if normalize_y else ''
        fig.savefig(DATA_FOLDER + f'susc_vs_q_{label_y}_{B}.png', dpi=180)
        fig.savefig(DATA_FOLDER + f'susc_vs_q_{label_y}_{B}.svg', bbox_inches='tight')
    return fig


def plot_susc_Bq_grid(B_grid=np.round(np.linspace(0, 1, 15), 3),
                      q_grid=np.round(np.linspace(0.55, 0.95, 15), 3),
                      alphas=(0.5, 1.0, 1.5, 2.0),
                      J_grid=np.round(np.arange(0, 6, 0.02), 3),
                      theta=THETA_NECKER, save=True, recompute=False):

    dist = _dist_matrix(theta)
    dvals = np.arange(int(dist.max()) + 1)
    methods = list(_susc_methods(alphas)) + [_opt_method()]
    B_grid, q_grid = np.asarray(B_grid), np.asarray(q_grid)

    fname = DATA_FOLDER + f'susc_Bq_all_{len(B_grid)}x{len(q_grid)}.npz'

    if not recompute and os.path.exists(fname):
        data = np.load(fname, allow_pickle=True)
        matrices, mse = data['matrices'].item(), data['mse'].item()
    else:
        def get_R(md, d):
            R = np.full((len(B_grid), len(q_grid)), np.nan)
            Jg = np.round(np.arange(0, 3, 0.05), 3) \
                if md['kind'] == 'fbp_opt' else J_grid

            for i, B in enumerate(B_grid):
                for j, q in enumerate(q_grid):
                    J = _fit_J_for_q(md['kind'], q, B, md['alpha'],
                                     theta, Jg)
                    if np.isnan(J):
                        continue
                    chi = _chi(md['kind'], J, B, md['alpha'], theta)
                    R[i, j] = _rd(chi, dist, np.array([d]))[0]
            return R

        matrices = {md['lab']: {} for md in methods}
        matrices['Exact'] = {}

        for d in tqdm(dvals):
            for md in methods:
                print(f"{md['lab']}, d={d}")
                matrices[md['lab']][d] = get_R(md, d)

            print(f"Exact, d={d}")
            matrices['Exact'][d] = get_R(
                {'kind': 'exact', 'alpha': None}, d
            )

        mse = {md['lab']: {} for md in methods}

        for md in methods:
            label = md['lab']
            for d in dvals:
                R, E = matrices[label][d], matrices['Exact'][d]
                valid = np.isfinite(R) & np.isfinite(E)
                mse[label][d] = np.mean((R[valid] - E[valid]) ** 2)

        np.savez(fname, matrices=matrices, mse=mse)

    vals = np.concatenate([
        R[np.isfinite(R)].ravel()
        for Rdict in matrices.values()
        for R in Rdict.values()
    ])
    vmin, vmax = vals.min(), 1.5

    nrows = len(methods) + 2
    fig, axes = plt.subplots(
        nrows, len(dvals),
        figsize=(3.2 * len(dvals), 3.0 * nrows),
        squeeze=False
    )

    # Susceptibility heatmaps
    for i, md in enumerate(methods):
        label = md['lab']
        for j, d in enumerate(dvals):
            ax = axes[i, j]
            im = ax.imshow(
                matrices[label][d],
                origin='lower', aspect='auto',
                extent=[q_grid.min(), q_grid.max(),
                        B_grid.min(), B_grid.max()],
                cmap='viridis', vmin=vmin, vmax=vmax
            )
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(f'd={d}')
            if i == len(methods) - 1:
                ax.set_xlabel('q')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Exact heatmap row
    i = len(methods)
    for j, d in enumerate(dvals):
        ax = axes[i, j]
        ax.imshow(
            matrices['Exact'][d],
            origin='lower', aspect='auto',
            extent=[q_grid.min(), q_grid.max(),
                    B_grid.min(), B_grid.max()],
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        if j == 0:
            ax.set_ylabel('Exact')
        ax.set_xlabel('q')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # MSE row
    i = len(methods) + 1
    for j, d in enumerate(dvals):
        ax = axes[i, j]
        x = np.arange(len(methods))
        y = [mse[md['lab']][d] for md in methods]

        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [md['lab'] for md in methods],
            rotation=45, ha='right'
        )
        ax.set_title(f'd={d}')
        ax.set_ylabel('MSE' if j == 0 else '')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.colorbar(im, ax=axes[:-1], shrink=.8, label=r'$r_d$')
    fig.tight_layout()

    if save:
        fig.savefig(DATA_FOLDER + 'susc_Bq_grid.png',
                    dpi=180, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'susc_Bq_grid.svg',
                    bbox_inches='tight')

    return matrices, mse, fig


def plot_optimal_alpha_vs_J(B_list=(0.05, 0.1, 0.2, 0.3),
                            J_grid=np.round(np.arange(0.05, 1.5, 0.02), 3),
                            theta=THETA_NECKER, save=True):
    """Companion to plot_susc_vs_q: the KL-optimal FBP exponent alpha-hat as a
    function of coupling J, one curve per sensory evidence B. Reference lines mark
    LBP (alpha=1) and the bistability-suppression boundary (alpha=N/2). alpha-hat
    is degenerate at B=0 (the symmetric marginal is 0.5 for every alpha), so only
    B>0 is shown. This explains where the green 'FBP alpha-hat' susceptibility line
    sits relative to LBP/FBP: alpha-hat grows with J (loops discounted more as
    coupling strengthens)."""
    n_deg = int(round(theta.sum(1).mean()))
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(B_list)))
    for B, c in zip(B_list, cols):
        ah = [_alpha_hat(J, B, theta) for J in J_grid]
        ax.plot(J_grid, ah, '-', color=c, lw=2, label=f'B={B}')
    ax.axhline(1.0, color='0.5', ls='--', lw=1, label='LBP ($\\alpha=1$)')
    ax.axhline(n_deg / 2, color='0.5', ls=':', lw=1,
               label=rf'bistability off ($\alpha=N/2={n_deg/2:g}$)')
    ax.set(xlabel='coupling J', ylabel=r'optimal $\hat\alpha$',
           title=r'KL-optimal $\hat\alpha$ vs coupling (per sensory evidence $B$)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        for ext in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'optimal_alpha_vs_J.{ext}', dpi=180, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Over-confidence and posterior-matrix comparisons (ported from
# loop_belief_prop_necker.plot_over_conf_mf_bp_gibbs / all_comparison_together),
# self-contained and extended with the FBP-alpha family + optimal-alpha line.
# ----------------------------------------------------------------------------
def _cmp_methods(alphas, gibbs_T=(), burn=1000, include_opt=False):
    """Algorithm set + colours as plot_susc_vs_J: exact/sampling (black), MF (red),
    FBP family (Blues by alpha), optional FBP-optimal (green). gibbs_T is a tuple of
    Gibbs chain lengths (number of post-burn samples); each becomes its own grey
    line/panel labelled by T, with (steps, burn_in)=(burn+T, burn)."""
    ms = list(_susc_methods(alphas)) + ([_opt_method()] if include_opt else [])
    gts = list(gibbs_T or [])
    greys = plt.cm.Greys(np.linspace(0.45, 0.88, max(len(gts), 1)))
    for T, c in zip(gts, greys):
        ms.append(dict(kind='gibbs', lab=f'Gibbs T={T:g}', c=c, ls=':', alpha=1.0,
                       gibbs=(int(burn + T), int(burn))))
    return ms


def _q_node(kind, J, B, alpha, theta, node=0, gibbs=(20000, 2000), steps=100, init='uniform'):
    """P(x_node = 1) for one algorithm at uniform (J, B). MF and FBP start from a
    random state and iterate `steps`. init='uniform' -> q0~U(0,1) / messages~U(0,1)
    (full random: each cell picks a well -> wide speckle above J*, for the matrices).
    init='small' -> tiny random near symmetric (follows the evidence-consistent well
    -> smooth over-confidence)."""
    n = theta.shape[0]; Jm = J * theta; Bv = np.full(n, float(B))
    if kind == 'exact':
        return float(exact_marginals(Jm, Bv)[node])
    if kind == 'gibbs':
        return float(gibbs_sampling(Jm, Bv, gibbs[0], gibbs[1])[node])
    if kind == 'mf':
        if init == 'uniform':
            m = np.random.uniform(-1.0, 1.0, n)            # q0 ~ U(0,1): full random -> speckle
        elif init == 'det':
            m = np.full(n, np.sign(B) * 0.9)               # evidence-aligned -> evidence well
        else:
            m = np.random.randn(n) * 0.01                  # tiny random near symmetric
        for _ in range(int(steps)):
            m = np.tanh(Bv + Jm @ m)
        return float((m[node] + 1) / 2)
    a = _alpha_hat(J, B, theta) if kind == 'fbp_opt' else alpha
    mask = (Jm != 0.0)
    if init == 'uniform':
        u = np.random.uniform(1e-6, 1.0 - 1e-6, (n, n))    # message beliefs ~ U(0,1)
        M0 = mask * 0.5 * np.log(u / (1.0 - u))            # log-ratio M = 0.5*logit(u)
    elif init == 'det':
        M0 = mask * (np.sign(B) * 0.5)                     # evidence-aligned messages
    else:
        M0 = mask * np.random.randn(n, n) * 0.01           # tiny random messages
    q = fractional_bp(Jm, Bv, alpha=a, M_init=M0, max_iter=max(int(steps), 100))
    return float(q[node])


def _q_matrix(md, j_list, b_list, node=0, theta=THETA_NECKER, steps=100, init='uniform',
              recompute=False):
    """Posterior grid Q[j, b] = P(x_node=1) for one algorithm, cached to disk under
    DATA_FOLDER/matrix_cache so repeated plots don't recompute. The sampler's chain
    length is in md['gibbs']=(steps, burn); MF/FBP use `steps` random-init iterations.
    Key = hash of (kind, alpha, node, steps, init, j_list, b_list, theta[, gibbs])."""
    import hashlib
    g = md.get('gibbs', (20000, 2000))
    cache_dir = os.path.join(DATA_FOLDER, 'matrix_cache'); os.makedirs(cache_dir, exist_ok=True)
    a = round(float(md['alpha']), 6)
    ini = init if md['kind'] not in ('exact', 'gibbs') else '-'   # init irrelevant for exact/gibbs
    key = repr((md['kind'], a, int(node), int(steps), ini, np.asarray(j_list, float).tobytes(),
                np.asarray(b_list, float).tobytes(), np.asarray(theta, float).tobytes(),
                tuple(g) if md['kind'] == 'gibbs' else None))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    tag = f"T{g[0]-g[1]}" if md['kind'] == 'gibbs' else f"a{round(a, 3)}_{ini}"
    fn = os.path.join(cache_dir, f"qmat_{md['kind']}_{tag}_{h}.npy")
    if not recompute and os.path.exists(fn):
        return np.load(fn)
    M = np.empty((len(j_list), len(b_list)))
    for ij, j in enumerate(j_list):
        for ib, b in enumerate(b_list):
            M[ij, ib] = _q_node(md['kind'], j, b, md['alpha'], theta, node, g, steps, init)
    np.save(fn, M)
    return M


def _mf_nfp(J, b, lmax, ngrid=2001):
    """Number of fixed points of the 1D MF map q=sigmoid(2*lmax*J*(2q-1)+2b)."""
    qs = np.linspace(0.0, 1.0, ngrid)
    f = 1.0 / (1.0 + np.exp(-(2 * lmax * J * (2 * qs - 1) + 2 * b))) - qs
    return int(np.count_nonzero(np.diff(np.sign(f)) != 0))


def _onset_bisect(is_bi, J_hi=2.0, coarse=0.03, tol=1e-5, iters=40):
    """Precise onset J*: coarse-bracket the mono->bistable transition of the
    predicate is_bi(J), then bisect to tolerance tol. nan if never bistable."""
    lo, hi, J = 0.0, None, coarse
    while J <= J_hi + 1e-9:
        if is_bi(J):
            hi = J; break
        lo = J; J += coarse
    if hi is None:
        return np.nan
    for _ in range(iters):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        (hi, lo) = (mid, lo) if is_bi(mid) else (hi, mid)
    return 0.5 * (lo + hi)


def _jstar_curve(md, b_list, theta, gibbs_c=10.0):
    """Precise onset J*(B) over b_list for one scheme (the black curve), by
    bisection: MF via fixed-point counting, FBP/LBP via find_solution_bp root count,
    Gibbs via (ln T + 8|B|)/c. nan array for exact / optimal-alpha."""
    b_arr = np.asarray(b_list, float)
    lmax = float(np.max(np.linalg.eigvalsh(theta))); n = int(round(lmax))
    if md['kind'] == 'mf':
        return np.array([_onset_bisect(lambda J, b=b: _mf_nfp(J, b, lmax) >= 3) for b in b_arr])
    if md['kind'] == 'fbp':
        a = md['alpha']
        return np.array([_onset_bisect(
            lambda J, b=b: len(find_solution_bp(J, abs(b), n_neigh=n, alpha=a, w_size=0.02)) > 1)
            for b in b_arr])
    if md['kind'] == 'gibbs':
        T = md['gibbs'][0] - md['gibbs'][1]
        return (np.log(T) + 8 * np.abs(b_arr)) / gibbs_c
    return np.full(len(b_arr), np.nan)


def plot_overconfidence_vs_J(j_list=np.round(np.arange(0.0, 1.0001, 0.02), 3),
                             b_list=np.round(np.arange(-0.5, 0.5001, 0.02), 3),
                             alphas=(0.5, 1.0, 1.5, 2.0), gibbs=(100, 1000, 10000),
                             include_opt=False, node=0, steps=100, init='det',
                             theta=THETA_NECKER, recompute=False, save=True):
    """Over-confidence vs coupling J for every scheme (colours as plot_susc_vs_J).
    Over-confidence at fixed J is the L1 gap to the exact marginal integrated over
    the sensory sweep, OC(J) = int |q_algo(B) - q_true(B)| dq_true(B). Exact is 0
    by construction; MF is largest, LBP/FBP ordered by alpha. `gibbs` is a tuple of
    Gibbs chain lengths T (each a grey line). include_opt adds the (slow) optimal-
    alpha line. Posterior grids are cached to disk (shared with plot_posterior_matrices
    when the grids match)."""
    methods = _cmp_methods(alphas, gibbs_T=gibbs, include_opt=include_opt)
    Qtrue = _q_matrix(dict(kind='exact', alpha=1.0), j_list, b_list, node, theta, steps, init, recompute)
    fig, ax = plt.subplots(figsize=(5.3, 3.6))
    for md in tqdm(methods):
        Q = _q_matrix(md, j_list, b_list, node, theta, steps, init, recompute)
        oc = [float(np.trapz(np.abs(Q[ij] - Qtrue[ij]), Qtrue[ij])) for ij in range(len(j_list))]
        ax.plot(j_list, oc, md['ls'], color=md['c'], lw=3.5, label=md['lab'])
    ax.set(xlabel='Coupling J', ylabel='Over-confidence')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        for ext in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'overconfidence_vs_J.{ext}', dpi=180, bbox_inches='tight')
    return fig


def plot_posterior_matrices(j_list=np.round(np.arange(0.0, 1.0001, 0.01), 4),
                            b_list=np.round(np.arange(-0.5, 0.5001, 0.01), 4),
                            alphas=(0.5, 1.0, 1.5, 2.0),
                            gibbs=(100, 1000, 10000), include_opt=False, node=0,
                            steps=100, init='uniform', show_jstar=True,
                            gibbs_c=10.0, theta=THETA_NECKER, recompute=False, save=True,
                            fname='posterior_matrices'):
    """Posterior q(x=1) over the (J, B) plane, one coolwarm heatmap per algorithm
    (exact, MF, FBP family, optional FBP-optimal, and one Gibbs panel per chain length
    in `gibbs`; each Gibbs cell is a single random-start chain). MSE vs the exact
    posterior is annotated per panel. Overlays the onset curve J*(B) for all B: numeric
    saddle-node for MF/FBP, and J*_Gibbs(T)=(ln T + 8|B|)/c for the sampler
    (c=gibbs_c, the Necker barrier slope). Grids are cached to disk."""
    methods = _cmp_methods(alphas, gibbs_T=gibbs, include_opt=include_opt)
    looper = tqdm(methods)
    mats = [(md, _q_matrix(md, j_list, b_list, node, theta, steps, init, recompute)) for md in looper]
    Mtrue = next(M for md, M in mats if md['kind'] == 'exact')

    ncols = min(4, len(mats)); nrows = int(np.ceil(len(mats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axes = axes.flatten()
    ext = [b_list[0], b_list[-1], j_list[0], j_list[-1]]
    im = None
    for a, (md, M) in zip(axes, mats):
        im = a.imshow(np.flipud(M), aspect='auto', extent=ext, cmap='coolwarm_r',
                      vmin=0, vmax=1, interpolation='none')
        mse = float(np.mean((M - Mtrue) ** 2))
        ttl = md['lab'] if md['kind'] == 'exact' else f"{md['lab']}\nMSE={mse:.3f}"
        a.set_title(ttl, fontsize=10)
        if show_jstar:
            jc = _jstar_curve(md, b_list, theta, gibbs_c=gibbs_c)
            if np.isfinite(jc).any():
                a.plot(b_list, jc, color='k', lw=1.4)
                a.set_ylim(j_list[0], j_list[-1]); a.set_xlim(b_list[0], b_list[-1])
        a.set_xticks([-0.5, 0, 0.5])
    for a in axes[len(mats):]:
        a.set_visible(False)
    for i, a in enumerate(axes[:len(mats)]):
        if i % ncols == 0:
            a.set_ylabel('Coupling J')
        if i // ncols == nrows - 1:
            a.set_xlabel('Sensory evidence B')
    fig.subplots_adjust(hspace=0.5, wspace=0.28)
    if im is not None:
        fig.colorbar(im, ax=axes[:len(mats)], fraction=0.02, label='Posterior q(x=1)')
    if save:
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=180, bbox_inches='tight')
    return fig


def plot_mse_vs_alpha(j_list=np.round(np.arange(0.0, 1.0001, 0.02), 3),
                      b_list=np.round(np.arange(-0.5, 0.5001, 0.02), 3),
                      alpha_grid=np.round(np.arange(0.1, 2.501, 0.1), 2),
                      J_show=(0.3, 0.6, 0.9, 1.2), init='det', steps=100, node=0,
                      theta=THETA_NECKER, recompute=False, save=True, fname='mse_vs_alpha'):
    """Supplementary: MSE of the FBP posterior vs exact over the (J,B) plane as a
    function of alpha. Left: grid-mean MSE(alpha) with the minimiser alpha-hat and
    the LBP (alpha=1) / MF (alpha->0) references. Right: MSE(alpha) at fixed coupling
    J (mean over B), showing the optimum shifting right with J. Deterministic init
    for a smooth curve; matrices are cached to disk."""
    Qexact = _q_matrix(dict(kind='exact', alpha=1.0), j_list, b_list, node, theta, steps, init, recompute)
    jidx = {J: int(np.argmin(np.abs(np.asarray(j_list) - J))) for J in J_show}
    mse_all, mse_J = [], {J: [] for J in J_show}
    for a in tqdm(alpha_grid, desc='MSE vs alpha'):
        Qa = _q_matrix(dict(kind='fbp', alpha=float(a)), j_list, b_list, node, theta, steps, init, recompute)
        se = (Qa - Qexact) ** 2
        mse_all.append(float(se.mean()))
        for J in J_show:
            mse_J[J].append(float(se[jidx[J]].mean()))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    ax[0].plot(alpha_grid, mse_all, color='k', ms=3, linewidth=3)
    a_min = float(alpha_grid[int(np.argmin(mse_all))])
    ax[0].axvline(a_min, color='#2ca02c', ls='--', label=rf'$\hat\alpha={a_min:g}$')
    ax[0].axvline(1.0, color='0.6', ls=':', label='LBP')
    ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel('MSE (grid mean)'); ax[0].set_title(r'MSE vs $\alpha$')
    ax[0].legend(frameon=False)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(J_show)))
    for J, c in zip(J_show, cols):
        ax[1].plot(alpha_grid, mse_J[J], ms=3, color=c, label=f'J={J:g}',
                   linewidth=3)
        am = alpha_grid[int(np.argmin(mse_J[J]))]
        ax[1].plot(am, min(mse_J[J]), '*', color=c, ms=13, mec='k', mew=0.5,
                   linewidth=3)
    ax[1].set_xlabel(r'$\alpha$'); ax[1].set_ylabel('MSE (mean over B)')
    ax[1].set_title(r'MSE vs $\alpha$ per coupling J'); ax[1].legend(frameon=False)
    for a_ in ax:
        a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
        a_.set_yscale('log')
    fig.tight_layout()
    if save:
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=300, bbox_inches='tight')
    return fig


def plot_paper_figure(j_list=np.round(np.arange(0.0, 1.0001, 0.005), 4),
                      b_list=np.round(np.arange(-0.5, 0.5001, 0.005), 4),
                      gibbs=(1000, 10000, 100000),
                      susc_alphas=(0.5, 1.0, 1.5, 2.0), susc_B=0.1,
                      susc_q_grid=np.round(np.linspace(0.55, 0.99, 12), 3),
                      susc_J_grid=np.round(np.arange(0.0, 6.0, 0.02), 3),
                      rdJ_d=1, rdJ_J_grid=np.round(np.arange(0.05, 1.0, 0.05), 3),
                      oc_j_list=np.round(np.arange(0.0, 1.0001, 0.02), 3),
                      oc_b_list=np.round(np.arange(-0.5, 0.5001, 0.02), 3),
                      oc_include_opt=True,
                      steps=100, node=0, gibbs_c=10.0,
                      theta=THETA_NECKER, recompute=False, save=True, fname='paper_figure'):
    """Composite main-paper figure (4x4 GridSpec), all text at the current rcParams
    font size (nothing is hard-coded here):
      rows 0-1 : the 7 posterior matrices (exact, MF, LBP, FBP-alpha_hat, Gibbs T)
                 with J*(B) overlay and MSE, plus a colour bar;
      row 2    : susceptibility r_d vs confidence q, one panel per graph distance d;
      row 3    : over-confidence vs J (left) and r_d vs J at d=rdJ_d (right).
    Matrices/over-confidence reuse the disk cache (uniform init for the matrices,
    deterministic evidence-following init for over-confidence)."""
    import matplotlib.gridspec as gridspec
    ts = mpl.rcParams['font.size'] * 0.9            # titles a bit smaller than body
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    fig = plt.figure(figsize=(14, 13), constrained_layout=True)
    gs = gridspec.GridSpec(4, 4, figure=fig)

    # (1) posterior matrices: exact, MF, LBP, FBP-alpha_hat, Gibbs T=... (7 panels)
    mmeth = _cmp_methods(alphas=(1.0,), gibbs_T=gibbs, include_opt=True)
    mats = [(md, _q_matrix(md, j_list, b_list, node, theta, steps, 'uniform', recompute)) for md in mmeth]
    Mtrue = next(M for md, M in mats if md['kind'] == 'exact')
    ext = [b_list[0], b_list[-1], j_list[0], j_list[-1]]
    pos = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    bycol = {}
    for (r, c) in pos:
        bycol[c] = max(bycol.get(c, -1), r)
    bottom = {(bycol[c], c) for c in bycol}         # lowest panel in each column
    ax0 = None; im = None
    for (r, c), (md, M) in zip(pos, mats):
        ax = fig.add_subplot(gs[r, c], sharex=ax0, sharey=ax0)   # shared x/y
        if ax0 is None:
            ax0 = ax
        im = ax.imshow(np.flipud(M), aspect='auto', extent=ext, cmap='coolwarm_r',
                       vmin=0, vmax=1, interpolation='none')
        ax.set_title('True Posterior' if md['kind'] == 'exact'
                     else f"{md['lab']}\nMSE={np.mean((M - Mtrue) ** 2):.3f}", fontsize=ts)
        jc = _jstar_curve(md, b_list, theta, gibbs_c=gibbs_c)
        if np.isfinite(jc).any():
            ax.plot(b_list, jc, 'k', lw=1)
        ax.set_xlim(b_list[0], b_list[-1]); ax.set_ylim(j_list[0], j_list[-1]); ax.set_xticks([-0.5, 0, 0.5])
        if (r, c) in bottom:
            ax.set_xlabel('Sensory evidence B')
        else:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel('Coupling J')
        else:
            ax.tick_params(labelleft=False)
    # thin colour bar + shared legend share the free top-right cell
    legcell = fig.add_subplot(gs[1, 3]); legcell.axis('off')
    cbax = legcell.inset_axes([0.02, 0.08, 0.10, 0.84])
    fig.colorbar(im, cax=cbax, label='Posterior q(x=1)')

    # (2) susceptibility r_d vs q, one panel per distance d
    smeth = list(_susc_methods(susc_alphas)) + [_opt_method()]
    sax = [fig.add_subplot(gs[2, di]) for di in range(len(dvals))]
    for md in smeth:
        R = np.full((len(susc_q_grid), len(dvals)), np.nan)
        Jg = _OPT_FIT_J if md['kind'] == 'fbp_opt' else susc_J_grid
        for iq, q in enumerate(susc_q_grid):
            Jf = _fit_J_for_q(md['kind'], q, susc_B, md['alpha'], theta, Jg)
            if not np.isnan(Jf):
                R[iq] = _rd(_chi(md['kind'], Jf, susc_B, md['alpha'], theta), dist, dvals)
        for di in range(len(dvals)):
            sax[di].plot(susc_q_grid, R[:, di], md['ls'], color=md['c'], marker='o', ms=3, label=md['lab'])
    for di, ax in enumerate(sax):
        ax.set_title(f'Distance d={dvals[di]}', fontsize=ts); ax.set_xlabel('Posterior q')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    sax[0].set_ylabel(r'$r_d=\partial\langle x_i\rangle/\partial B_j$')

    # (3a) over-confidence vs J (deterministic init; OWN grid so it reuses the
    # plot_overconfidence_vs_J cache -- match oc_j_list/oc_b_list/oc_include_opt/gibbs
    # to how you ran that function)
    axoc = fig.add_subplot(gs[3, 0:2])
    ocm = _cmp_methods(susc_alphas, gibbs_T=gibbs, include_opt=oc_include_opt)
    Qtrue = _q_matrix(dict(kind='exact', alpha=1.0), oc_j_list, oc_b_list, node, theta, steps, 'det', recompute)
    for md in tqdm(ocm):
        Q = _q_matrix(md, oc_j_list, oc_b_list, node, theta, steps, 'det', recompute)
        oc = [float(np.trapz(np.abs(Q[ij] - Qtrue[ij]), Qtrue[ij])) for ij in range(len(oc_j_list))]
        axoc.plot(oc_j_list, oc, md['ls'], color=md['c'], label=md['lab'])
    axoc.set_xlabel('Coupling J'); axoc.set_ylabel('Over-confidence'); axoc.set_title('Over-confidence', fontsize=ts)
    axoc.spines['top'].set_visible(False); axoc.spines['right'].set_visible(False)

    # (3b) r_d vs J at fixed distance, with J* stars
    axr = fig.add_subplot(gs[3, 2:4])
    for md in smeth:
        rd = [_rd(_chi(md['kind'], J, susc_B, md['alpha'], theta), dist, dvals)[rdJ_d] for J in rdJ_J_grid]
        axr.plot(rdJ_J_grid, rd, md['ls'], color=md['c'], marker='.', label=md['lab'])
        Js = _jstar(md['kind'], md['alpha'], theta)
        if np.isfinite(Js) and rdJ_J_grid.min() <= Js <= rdJ_J_grid.max():
            rstar = _rd(_chi(md['kind'], Js, susc_B, md['alpha'], theta), dist, dvals)[rdJ_d]
            axr.plot(Js, rstar, marker='*', color=md['c'], ms=12, mec='k', mew=0.5, ls='none')
    axr.set_xlabel('Coupling J'); axr.set_ylabel(rf'$r_{{{rdJ_d}}}$')
    axr.set_title(f'Susceptibility at d={rdJ_d} vs J', fontsize=ts)
    axr.spines['top'].set_visible(False); axr.spines['right'].set_visible(False)

    # single shared legend for all line plots, in the free top-right cell
    handles, labels = axoc.get_legend_handles_labels()
    legcell.legend(handles, labels, loc='center left', bbox_to_anchor=(0.42, 0.5), frameon=False)

    if save:
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=300, bbox_inches='tight')
    return fig


def plot_susc_ratios(q_star=0.8, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                     J_grid=np.round(np.arange(0.0, 6.0, 0.02), 3), include_gibbs=True,
                     gibbs=(400000, 30000), theta=THETA_NECKER, save=True):
    """(2) Response ratios r_0/r_d vs distance d at matched confidence q_star.
    r_0/r_d = how much stronger the self-response is than the response at distance
    d (a gauge-free number). Steeper => cue stays local; flatter => spreads."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = list(_susc_methods(alphas)) + [_opt_method()]
    if include_gibbs:
        methods.append(dict(kind='gibbs', lab='Gibbs', c='0.5', ls=':', alpha=1.0))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for md in methods:
        k_fit = 'exact' if md['kind'] == 'gibbs' else md['kind']   # gibbs shares exact's J(q)
        Jg = _OPT_FIT_J if md['kind'] == 'fbp_opt' else J_grid
        J = _fit_J_for_q(k_fit, q_star, B, md['alpha'], theta, Jg)
        if np.isnan(J):
            print(f"  {md['lab']}: q={q_star} unreachable (q ceiling < target) -- skipped")
            continue
        r = _rd(_chi(md['kind'], J, B, md['alpha'], theta, gibbs), dist, dvals)
        ax.plot(dvals, r / r[0], md['ls'], color=md['c'], marker='o', ms=6, label=md['lab'])
    ax.set(xlabel='graph distance d', ylabel=r'$r_d / r_0$',
           title=f'Normalised response vs distance at matched q={q_star} (B={B})')
    ax.set_xticks(dvals); ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + f'susc_ratios_q{q_star}.png', dpi=180, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + f'susc_ratios_q{q_star}.svg', bbox_inches='tight')
    return fig


def plot_susc_vs_J(d=1, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                   J_grid=np.round(np.arange(0.05, 1.0, 0.05), 3), include_gibbs=False,
                   gibbs=(150000, 10000), theta=THETA_NECKER, save=True):
    """(3) Average susceptibility at a GIVEN distance d vs coupling J, all
    algorithms on one panel. Shows how the response at separation d grows (and,
    for MF, diverges near its spurious critical point) with coupling."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = list(_susc_methods(alphas)) + [_opt_method()]
    if include_gibbs:
        methods.append(dict(kind='gibbs', lab='Gibbs', c='0.5', ls=':', alpha=1.0))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for md in methods:
        rd = [ _rd(_chi(md['kind'], J, B, md['alpha'], theta, gibbs), dist, dvals)[d]
               for J in J_grid ]
        ax.plot(J_grid, rd, md['ls'], color=md['c'], marker='.', ms=5, label=md['lab'])
        Js = _jstar(md['kind'], md['alpha'], theta)          # star at critical coupling
        if np.isfinite(Js) and J_grid.min() <= Js <= J_grid.max():
            rstar = _rd(_chi(md['kind'], Js, B, md['alpha'], theta, gibbs), dist, dvals)[d]
            ax.plot(Js, rstar, marker='*', color=md['c'], ms=16, mec='k', mew=0.6,
                    ls='none', zorder=6)
    ax.plot([], [], marker='*', color='0.4', mec='k', mew=0.6, ls='none', ms=13,
            label=r'$J^*$ (onset, $B=0$)')
    ax.set(xlabel='Coupling J', ylabel=rf'$r_{{{d}}}$')
    # title=f'Susceptibility at distance d={d} vs coupling (B={B})')
    ax.legend(frameon=False, fontsize=12, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(DATA_FOLDER + f'susc_vs_J_d{d}.png', dpi=180, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + f'susc_vs_J_d{d}.svg', bbox_inches='tight')
    return fig


def plot_susc_overview(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                       q_grid=np.round(np.linspace(0.55, 0.99, 12), 3),
                       J_grid_q=np.round(np.arange(0.0, 6.0, 0.02), 3),
                       J_grid=np.round(np.arange(0.05, 1.5, 0.05), 3),
                       theta=THETA_NECKER, save=True):
    """(4) Combined susceptibility summary: (a) spread rho=r_1/r_0 vs confidence q
    (J fit per q) -- the S1 signature; (b) self r_0 and neighbour r_1 vs coupling
    J -- the raw scale/decay. Gives the vs-q and vs-J views side by side."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    methods = list(_susc_methods(alphas)) + [_opt_method()]
    fig, (axq, axj) = plt.subplots(1, 2, figsize=(12, 4.8))
    for md in methods:
        rho = []
        Jgq = _OPT_FIT_J if md['kind'] == 'fbp_opt' else J_grid_q
        for q in q_grid:
            J = _fit_J_for_q(md['kind'], q, B, md['alpha'], theta, Jgq)
            if np.isnan(J):
                rho.append(np.nan); continue                # q unreachable -> blank
            r = _rd(_chi(md['kind'], J, B, md['alpha'], theta), dist, dvals)
            rho.append(r[1] / r[0])
        axq.plot(q_grid, rho, md['ls'], color=md['c'], marker='o', ms=6, label=md['lab'])
        r0 = []; r1 = []
        for J in J_grid:
            r = _rd(_chi(md['kind'], J, B, md['alpha'], theta), dist, dvals)
            r0.append(r[0]); r1.append(r[1])
        axj.plot(J_grid, r1, md['ls'], color=md['c'], marker='.', ms=6, label=md['lab'])
    axq.set(xlabel='Posterior q', ylabel=r'spread $\rho=r_1/r_0$',
            title='(a) cue spread vs confidence (J fit per q)')
    axj.set(xlabel='coupling J', ylabel=r'neighbour response $r_1$',
            title='(b) neighbour susceptibility vs coupling')
    for ax in (axq, axj):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axq.legend(frameon=False, fontsize=8)
    if save:
        fig.savefig(DATA_FOLDER + 'susc_overview.png', dpi=180, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'susc_overview.svg', bbox_inches='tight')
    return fig


def plot_evidence_interaction_vs_J(J_grid=np.round(np.arange(0.1, 0.85, 0.05), 3),
                                   h=0.4, B0=0.0, t=0, alphas=(0.5, 1.0, 1.5),
                                   normalize=True, theta=THETA_NECKER, save=True):
    """Evidence-source OVER-COUNTING vs coupling J (parameter-reduced).

    Target node t gets a SENSORY field h on itself; the REST of the figure
    (context) gets the same field h (reaching t only through the graph/loops).
    Read the target's perceived log-odds L_t = logit(q_t) and form the two-source
    interaction
        I = L_t(h,h) - L_t(h,0) - L_t(0,h) + L_t(0,0).
    I ~ 0  additive (Bayes-optimal, exact);  I < 0 sub-additive (MF under-combines,
    ignores loop correlations);  I > 0 super-additive (circular-inference OVER-
    counting, alpha>1). Swept over J. normalize=True divides by the single-source
    responses (|R1|+|R2|) -> dimensionless, removes the evidence-magnitude scale
    and MF's near-onset blow-up, so the sub/additive/super ORDERING is what shows."""
    n = theta.shape[0]; ctx = [i for i in range(n) if i != t]

    def Lt(hs, hc, J, kind, alpha):
        Bv = np.full(n, float(B0)); Bv[t] += hs
        for i in ctx:
            Bv[i] += hc
        Jm = J * theta
        if kind == 'exact':
            q = exact_marginals(Jm, Bv)[t]
        elif kind == 'mf':
            q = (_mf_magnetization(Jm, Bv, np.zeros(n))[t] + 1) / 2
        else:
            q = fractional_bp(Jm, Bv, alpha=alpha)[t]
        q = min(max(float(q), 1e-9), 1 - 1e-9)
        return np.log(q / (1 - q))

    methods = _susc_methods(alphas)                     # exact, MF, FBP(alpha)...
    fig, ax = plt.subplots(figsize=(7, 5))
    for md in methods:
        k, a = md['kind'], md['alpha']; ys = []
        for J in J_grid:
            L00 = Lt(0, 0, J, k, a)
            R1 = Lt(h, 0, J, k, a) - L00
            R2 = Lt(0, h, J, k, a) - L00
            I = Lt(h, h, J, k, a) - Lt(h, 0, J, k, a) - Lt(0, h, J, k, a) + L00
            ys.append(I / (abs(R1) + abs(R2) + 1e-9) if normalize else I)
        ax.plot(J_grid, ys, md['ls'], color=md['c'], marker='o', ms=3, label=md['lab'])
    ax.axhline(0, color='0.7', lw=1)
    ax.set_xlabel('coupling J')
    ax.set_ylabel('interaction / single-source' if normalize else r'interaction $I$')
    ax.set_title('Evidence over-counting vs J  (>0 super-additive, <0 sub-additive)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        for ext in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'evidence_interaction_vs_J.{ext}', dpi=180, bbox_inches='tight')
    return fig


def plot_evidence_interaction_grid(J_grid=np.round(np.arange(0.1, 0.85, 0.05), 3),
                                   h_grid=np.round(np.linspace(0.0, 0.6, 13), 3),
                                   B0=0.0, t=0, alphas=(0.5, 1.0, 1.5),
                                   clip=0.1, white_eps=1e-2, resp_min=0.5,
                                   theta=THETA_NECKER, save=True):
    """RAW evidence-source interaction I over a (J x evidence h) grid, one heatmap
    per algorithm, SHARED diverging colormap (bwr). Source A = field h on the
    readout node t; source B = field h on the other nodes (reaches t only through
    the graph/loops). I = L_t(h,h)-L_t(h,0)-L_t(0,h)+L_t(0,0), L_t=logit(q_t).
    Colour is CLIPPED at +-clip (|I|>clip saturates to full blue/red) and a tiny
    white deadzone |I|<white_eps marks additive. blue<0 = sub-additive (under-count,
    MF); white = additive (Bayes/exact); red>0 = super-additive (over-count)."""
    n = theta.shape[0]; ctx = [i for i in range(n) if i != t]

    def Lt(hs, hc, J, kind, alpha):
        Bv = np.full(n, float(B0)); Bv[t] += hs
        for i in ctx:
            Bv[i] += hc
        Jm = J * theta
        if kind == 'exact':
            q = exact_marginals(Jm, Bv)[t]
        elif kind == 'mf':
            q = (_mf_magnetization(Jm, Bv, np.zeros(n))[t] + 1) / 2
        else:
            q = fractional_bp(Jm, Bv, alpha=alpha)[t]
        q = min(max(float(q), 1e-9), 1 - 1e-9)
        return np.log(q / (1 - q))

    methods = _susc_methods(alphas)                     # exact, MF, FBP(alpha)...
    mats = {}
    for md in methods:
        k, a = md['kind'], md['alpha']
        M = np.zeros((len(J_grid), len(h_grid))); RC = np.zeros_like(M)
        for ij, J in enumerate(J_grid):
            for ih, h in enumerate(h_grid):
                L00 = Lt(0, 0, J, k, a)
                R1 = Lt(h, 0, J, k, a) - L00
                R2 = Lt(0, h, J, k, a) - L00                 # CONTEXT response
                I = Lt(h, h, J, k, a) - Lt(h, 0, J, k, a) - Lt(0, h, J, k, a) + L00
                M[ij, ih] = I; RC[ij, ih] = abs(R2)          # I and context-response mag
        mats[md['lab']] = (M, RC)
    ext = [h_grid[0], h_grid[-1], J_grid[0], J_grid[-1]]
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(-clip, clip); cmap = plt.cm.bwr
    fig, axes = plt.subplots(1, len(methods), figsize=(2.8 * len(methods), 3.4),
                             squeeze=False, sharey=True)
    for ax, md in zip(axes[0], methods):
        M, RC = mats[md['lab']]
        rgba = cmap(norm(np.clip(M, -clip, clip)))           # bwr by sign/magnitude
        rgba[np.abs(M) < white_eps] = [1, 1, 1, 1]           # additive -> white
        rgba[RC < resp_min] = [0.8, 0.8, 0.8, 1]             # UNRESPONSIVE -> gray (overrides)
        ax.imshow(rgba, origin='lower', aspect='auto', extent=ext)
        ax.set_title(md['lab'], fontsize=10); ax.set_xlabel('evidence h', fontsize=9)
    axes[0][0].set_ylabel('coupling J', fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes[0], fraction=0.02, extend='both',
                 label=f'interaction I (clip +-{clip}; red=over-count)')
    fig.suptitle(f'Evidence interaction over (J x evidence)  [gray = context response |R2|<{resp_min}: unresponsive, not additive]', fontsize=9)
    if save:
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'evidence_interaction_grid.{ext_}', dpi=170, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Testable prediction #1 : confidence calibration (over-confidence below J*)
# ----------------------------------------------------------------------------
def _global_percept(cfg):
    """Global interpretation of a spin configuration: sign of the net magnetisation."""
    s = float(np.sum(cfg))
    return 1 if s >= 0 else -1


def _calibration_trials(J, n_trials, b_lo=0.001, b_hi=0.49,
                        alphas=(0.5, 1.0, 1.5, 2.0),
                        theta=THETA_NECKER, seed=0):
    """Simulated-observer trials on the cube at coupling J, scored per node.

    The same sampled trials are used for every inference scheme and every FBP alpha,
    so differences in calibration are due to the inference method rather than trial
    sampling. Returns keys ``exact``, ``mf`` and ``fbp_<alpha>``.
    """
    n = theta.shape[0]
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    rng = np.random.default_rng(seed)
    schemes = ['exact', 'mf'] + [f'fbp_{a:g}' for a in alphas]
    conf = {k: [] for k in schemes}; corr = {k: [] for k in schemes}
    for tr in range(n_trials):
        u = 1.0 if rng.random() < 0.5 else -1.0
        b = rng.uniform(b_lo, b_hi)
        Bv = np.full(n, u * b); Jm = J * theta
        E = 0.5 * np.einsum('si,ij,sj->s', states, Jm, states) + states @ Bv
        w = np.exp(E - E.max()); w /= w.sum()
        marg_ex = (states == 1).T @ w
        truth = states[rng.choice(len(states), p=w)]
        qs = {'exact': marg_ex, 'mf': mean_field(Jm, Bv)}
        qs.update({f'fbp_{a:g}': fractional_bp(Jm, Bv, alpha=a)
                   for a in alphas})
        for k, q in qs.items():
            dec = np.where(q >= 0.5, 1.0, -1.0)
            conf[k].append(np.maximum(q, 1 - q))
            corr[k].append((dec == truth).astype(float))
    return ({k: np.concatenate(conf[k]) for k in schemes},
            {k: np.concatenate(corr[k]) for k in schemes})


def _ece(conf, correct, n_bins=10):
    """Expected calibration error and signed over-confidence (mean conf - mean acc)."""
    edges = np.linspace(0.5, 1.0, n_bins + 1); ece = 0.0
    for k in range(n_bins):
        m = (conf >= edges[k]) & (conf <= edges[k + 1] if k == n_bins - 1 else conf < edges[k + 1])
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return ece, float(conf.mean() - correct.mean())


def plot_confidence_calibration(J_list=(0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8),
                                J_show=0.1, n_trials=1500,
                                alphas=(0.5, 1.0, 1.5, 2.0),
                                theta=THETA_NECKER, seed=0, save=True,
                                fname='confidence_calibration'):
    """Testable prediction #1: confidence calibration across inference schemes.

    Uses the common plotting/computation convention of the susceptibility analyses:
    exact = black solid, MF = red dashed, and the FBP family = blue shades with
    alpha increasing from light to dark. Alpha=1 is labelled LBP.
    """
    methods = _susc_methods(alphas)
    method_keys = [('exact', md) if md['kind'] == 'exact' else
                   ('mf', md) if md['kind'] == 'mf' else
                   (f"fbp_{md['alpha']:g}", md) for md in methods]
    JstarMF = 1.0 / int(round(theta.sum(1)[0]))
    deg = int(round(theta.sum(1)[0]))
    JstarLBP = 0.5 * np.log(deg / (deg - 2))

    ece = {k: [] for k, _ in method_keys}
    over = {k: [] for k, _ in method_keys}
    over_se = {k: [] for k, _ in method_keys}
    calib_show = None
    for J in tqdm(J_list, desc='calibration vs J'):
        conf, corr = _calibration_trials(J, n_trials, alphas=alphas,
                                         theta=theta, seed=seed)
        for k, _ in method_keys:
            e, o = _ece(conf[k], corr[k]); ece[k].append(e); over[k].append(o)
            rng = np.random.default_rng(seed + 1); m = len(conf[k])
            bs = [np.mean(conf[k][ix]) - np.mean(corr[k][ix])
                  for ix in (rng.integers(0, m, m) for _ in range(200))]
            over_se[k].append(np.std(bs))
        if abs(J - J_show) < 1e-9:
            calib_show = (conf, corr)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    if calib_show is None:
        conf, corr = _calibration_trials(J_show, n_trials, alphas=alphas,
                                         theta=theta, seed=seed)
    else:
        conf, corr = calib_show

    edges = np.linspace(0.5, 1.0, 9); ctr = 0.5 * (edges[:-1] + edges[1:])
    for k, md in method_keys:
        acc = [corr[k][(conf[k] >= edges[i]) & (conf[k] <= edges[i + 1])].mean()
               if ((conf[k] >= edges[i]) & (conf[k] <= edges[i + 1])).any() else np.nan
               for i in range(len(edges) - 1)]
        ax[0].plot(ctr, acc, md['ls'], color=md['c'], marker='o', ms=6,
                   lw=2.2, label=md['lab'])
    ax[0].plot([0.5, 1], [0.5, 1], 'k:', lw=1)
    ax[0].set(xlabel='confidence', ylabel='empirical accuracy',
              title=f'(a) calibration at J={J_show} (< $J^*_{{MF}}$)')
    ax[0].legend(frameon=False)

    for k, md in method_keys:
        ax[1].plot(J_list, ece[k], md['ls'], color=md['c'], marker='o', ms=6,
                   lw=2.2, label=md['lab'])
    ax[1].axvline(JstarMF, color='r', ls=':', lw=1)
    ax[1].axvline(JstarLBP, color=plt.cm.Blues(0.5), ls=':', lw=1)
    ax[1].set(xlabel='coupling J', ylabel='ECE', title='(b) calibration error vs J')
    ax[1].legend(frameon=False)

    for k, md in method_keys:
        ax[2].errorbar(J_list, over[k], yerr=over_se[k], fmt=md['ls'],
                       color=md['c'], marker='o', ms=6, lw=2.2, capsize=2,
                       label=md['lab'])
    ax[2].axhline(0, color='0.6', lw=0.8)
    ax[2].axvline(JstarMF, color='r', ls=':', lw=1, label=r'$J^*_{MF}$')
    ax[2].axvline(JstarLBP, color=plt.cm.Blues(0.5), ls=':', lw=1, label=r'$J^*_{LBP}$')
    ax[2].set(xlabel='coupling J', ylabel='mean confidence - mean accuracy',
              title='(c) over-confidence')
    ax[2].legend(frameon=False)

    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.suptitle('Prediction 1: confidence miscalibration, present below the bifurcation')
    fig.tight_layout()
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=200, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Testable prediction #3 : duration-dependence of the apparent threshold
# ----------------------------------------------------------------------------
def _bimod_coeff(x):
    """Sarle's bimodality coefficient; > 5/9 indicates a bimodal (two-percept) distribution."""
    x = np.asarray(x, float); n = len(x)
    if n < 4 or np.std(x) < 1e-9:
        return 0.0
    z = (x - x.mean()) / x.std()
    g1 = np.mean(z ** 3); g2 = np.mean(z ** 4) - 3.0
    den = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1.0) / den) if den > 0 else 0.0


def _threshold_cross(J_grid, bc, crit=5.0 / 9.0):
    """First J where the (cumulative-max) bimodality coefficient crosses crit; linear interp."""
    bc = np.maximum.accumulate(np.asarray(bc, float))
    hit = np.flatnonzero(bc >= crit)
    if len(hit) == 0:
        return np.nan
    k = hit[0]
    if k == 0:
        return float(J_grid[0])
    x0, x1, y0, y1 = J_grid[k - 1], J_grid[k], bc[k - 1], bc[k]
    return float(x0 + (x1 - x0) * (crit - y0) / (y1 - y0)) if y1 != y0 else float(x1)


def _gibbs_reports(Jm, Bv, T_checks, n_trials, burn=1000):
    """Per-trial time-averaged magnetisation of Gibbs chains, evaluated at each T in
    T_checks (one chain per trial run to max(T_checks), read at every checkpoint)."""
    Tmax = int(T_checks[-1]); out = np.empty((len(T_checks), n_trials))
    for tr in range(n_trials):
        trace = _gibbs_traj(Jm, Bv, int(burn + Tmax), int(burn))    # length Tmax
        for ti, T in enumerate(T_checks):
            out[ti, tr] = trace[:int(T)].mean()
    return out


def _var_reports(kind, J, B, alpha, T_checks, n_trials, sigma, N=3, dt=0.05, tau=1.0):
    """Per-trial time-averaged report q for a variational Langevin scheme, at each duration
    in T_checks (one trajectory per trial to max duration, read at every checkpoint; the
    first 10% is discarded as burn-in)."""
    Tmax = float(T_checks[-1]); out = np.empty((len(T_checks), n_trials))
    for tr in range(n_trials):
        q = langevin_1d(kind, J, B, N=N, alpha=alpha, sigma=sigma, dt=dt, T=Tmax, seed=tr)
        for ti, T in enumerate(T_checks):
            n_end = int(T / dt); n0 = int(0.1 * n_end)
            out[ti, tr] = q[n0:n_end].mean()
    return out


def plot_duration_threshold(J_grid=np.round(np.linspace(0.30, 1.30, 18), 3),
                            T_gibbs=(200, 1000, 5000, 20000, 100000),
                            T_var=(40, 100, 250, 600, 1500),
                            B=0.0, n_trials=24, sigma_var=0.05, c=10.0,
                            theta=THETA_NECKER, seed=0, save=True,
                            fname='duration_threshold'):
    """Testable prediction #3: sampling has no true threshold, the variational schemes do.
    The apparent critical coupling is defined identically for every scheme -- the coupling at
    which the across-trial distribution of the time-averaged report turns bimodal (Sarle
    BC>5/9). Sweeping the observation length T:
      Gibbs J*(T) grows as (ln T)/c (barrier-limited mixing, no bifurcation);
      MF and LBP J*(T) stay flat at their bifurcation (1/N and 1/2 log[N/(N-2)]).
    Panels: (a) J*(T) per scheme + analytic Gibbs line and variational bifurcation lines;
    (b) fitted slope dJ*/d ln T per scheme (0 for variational, 1/c for Gibbs);
    (c) example Gibbs report distributions at short vs long T (bimodal -> unimodal).
    Note: the Langevin noise sigma_var is set so crossing times exceed the tested durations,
    i.e. the variational threshold reflects the bifurcation, not a timescale (sigma-independent
    position; only the sharpness changes)."""
    n_nodes = theta.shape[0]; Bv = np.full(n_nodes, float(B))
    deg = int(round(theta.sum(1)[0]))                # node degree (N=3 for the cube)
    JstarMF = 1.0 / deg; JstarLBP = 0.5 * np.log(deg / (deg - 2))
    np.random.seed(seed)

    def _jstar_curve(reports_fn, T_checks):
        Js = np.full(len(T_checks), np.nan)
        bc_all = np.zeros((len(T_checks), len(J_grid)))
        for jj, J in enumerate(tqdm(J_grid, desc='J sweep', leave=False)):
            R = reports_fn(J, T_checks)                      # (len(T_checks), n_trials)
            for ti in range(len(T_checks)):
                bc_all[ti, jj] = _bimod_coeff(R[ti])
        for ti in range(len(T_checks)):
            Js[ti] = _threshold_cross(J_grid, bc_all[ti])
        return Js, bc_all

    gib_fn = lambda J, Tc: _gibbs_reports(J * theta, Bv, Tc, n_trials)
    mf_fn = lambda J, Tc: _var_reports('mf', J, B, 1.0, Tc, n_trials, sigma_var, N=deg)
    lbp_fn = lambda J, Tc: _var_reports('fbp', J, B, 1.0, Tc, n_trials, sigma_var, N=deg)

    Jg, bcg = _jstar_curve(gib_fn, np.array(T_gibbs, int))
    Jmf, _ = _jstar_curve(mf_fn, np.array(T_var, float))
    Jlbp, _ = _jstar_curve(lbp_fn, np.array(T_var, float))

    def _slope(T, J):
        ok = np.isfinite(J)
        return np.polyfit(np.log(np.asarray(T)[ok]), np.asarray(J)[ok], 1)[0] if ok.sum() >= 2 else np.nan
    slopes = {'Gibbs': _slope(T_gibbs, Jg), 'MF': _slope(T_var, Jmf), 'LBP': _slope(T_var, Jlbp)}

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(T_gibbs, Jg, 'o-', color='k', label='Gibbs (empirical)')
    ax[0].plot(T_gibbs, (np.log(T_gibbs)) / c + (Jg[0] - np.log(T_gibbs[0]) / c),
               'k--', lw=1, label=r'$(\ln T)/c$')
    ax[0].plot(T_var, Jmf, 's-', color='r', label='MF')
    ax[0].plot(T_var, Jlbp, '^-', color='C0', label='LBP')
    ax[0].axhline(JstarMF, color='r', ls=':', lw=1); ax[0].axhline(JstarLBP, color='C0', ls=':', lw=1)
    ax[0].set_xscale('log')
    ax[0].set(xlabel='observation length T', ylabel=r'apparent $J^\ast$',
              title='(a) apparent threshold vs duration')
    ax[0].legend(frameon=False)
    # (b) slopes
    names = list(slopes); xs = np.arange(len(names)); col = {'Gibbs': 'k', 'MF': 'r', 'LBP': 'C0'}
    ax[1].bar(xs, [slopes[n] for n in names], color=[col[n] for n in names], alpha=0.75)
    ax[1].axhline(1.0 / c, color='green', ls='--', label=r'$1/c$ (Gibbs pred.)')
    ax[1].axhline(0, color='0.6', lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(names)
    ax[1].set(ylabel=r'slope $dJ^\ast/d\ln T$', title='(b) duration slope')
    ax[1].legend(frameon=False)
    # (c) example Gibbs report distributions short vs long T
    Jc = J_grid[np.argmin(np.abs(J_grid - 0.6))]
    R = _gibbs_reports(Jc * theta, Bv, np.array([T_gibbs[0], T_gibbs[-1]], int), max(n_trials, 200))
    ax[2].hist(R[0], bins=25, density=True, histtype='step', color='0.6',
               label=f'T={T_gibbs[0]} (bimodal)')
    ax[2].hist(R[1], bins=25, density=True, histtype='step', color='k',
               label=f'T={T_gibbs[-1]} (unimodal)')
    ax[2].set(xlabel='time-averaged report', ylabel='density',
              title=f'(c) Gibbs reports at J={Jc}')
    ax[2].legend(frameon=False)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.suptitle('Prediction 3: duration-dependent threshold for sampling, fixed for variational')
    fig.tight_layout()
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=200, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Own-covariance of each scheme (for the true FDT ratio rho = chi / C^own)
# ----------------------------------------------------------------------------
def _exact_cov(J, B, theta=THETA_NECKER):
    """Exact connected covariance C_ij = <x_i x_j> - <x_i><x_j> by enumeration."""
    n = theta.shape[0]; Jm = J * theta; Bv = np.full(n, float(B))
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = 0.5 * np.einsum('si,ij,sj->s', S, Jm, S) + S @ Bv
    w = np.exp(E - E.max()); w /= w.sum()
    m = (S * w[:, None]).sum(0)
    C = (S.T * w) @ S - np.outer(m, m)
    return C


def _bp_pair_cov(J, B, theta=THETA_NECKER, alpha=1.0):
    """(Fractional) belief-propagation own covariance on the edges, from the fractional-Bethe
    pairwise belief
        b_ij(x_i,x_j) ~ exp( alpha*Jmat_ij x_i x_j + (Q_i - alpha M[j,i]) x_i + (Q_j - alpha M[i,j]) x_j ),
    the unique belief that marginalises back to the FBP singleton belief q_i (verified in
    _check_bp_pair_cov). alpha=1 is loopy BP. Returns an n x n matrix with the neighbour
    covariances on the edges; the diagonal is 1 - <x_i>^2."""
    n = theta.shape[0]; Jm = J * theta; Bv = np.full(n, float(B))
    q, M = fractional_bp(Jm, Bv, alpha=alpha, max_iter=4000, return_messages=True)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    Q = 0.5 * np.log(q / (1 - q))
    m = 2 * q - 1
    C = np.diag(1.0 - m ** 2)
    xs = np.array([-1.0, 1.0])
    for i in range(n):
        for j in range(i + 1, n):
            if theta[i, j] == 0:
                continue
            a_i = Q[i] - alpha * M[j, i]; a_j = Q[j] - alpha * M[i, j]
            logw = np.array([[alpha * Jm[i, j] * xi * xj + a_i * xi + a_j * xj
                              for xj in xs] for xi in xs])
            w = np.exp(logw - logw.max()); w /= w.sum()
            exx = np.array([[xi * xj for xj in xs] for xi in xs])
            mi = (xs[:, None] * w).sum(); mj = (xs[None, :] * w).sum()
            C[i, j] = C[j, i] = float((exx * w).sum() - mi * mj)
    return C


# ----------------------------------------------------------------------------
# Section 6 figure: experimentally identifiable differences, one figure
#   (a) fluctuation-dissipation ratio, (b) confidence calibration, (c) r_d vs q
# ----------------------------------------------------------------------------
def plot_testable_differences(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                              q_grid=np.round(np.linspace(0.55, 0.93, 9), 3),
                              J_grid=np.round(np.arange(0.0, 6.0, 0.02), 3),
                              cal_J=(0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8), cal_trials=1200,
                              gibbs=(120000, 12000), gibbs_q=None,
                              theta=THETA_NECKER, seed=0, save=True,
                              fname='testable_differences'):
    """Single figure for experimentally identifiable differences.

    Uses the same algorithm ordering, colours, line styles, markers and sizes as
    the susceptibility plots: exact (black solid), MF (red dashed), and the full
    FBP family (blue shades; alpha=1 is LBP).
    """
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    deg = int(round(theta.sum(1).mean()))
    JMF = 1.0 / deg; JLBP = 0.5 * np.log(deg / (deg - 2))
    if gibbs_q is None:
        gibbs_q = q_grid

    methods = _susc_methods(alphas)
    method_keys = [('exact', md) if md['kind'] == 'exact' else
                   ('mf', md) if md['kind'] == 'mf' else
                   (f"fbp_{md['alpha']:g}", md) for md in methods]
    # orig figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))

    # ---- (a) FDT ratio vs matched q: rho = chi^alg / C^alg, EACH SCHEME'S OWN covariance ----
    # exact:  chi = Cov (FDT) -> rho = 1.   LBP: chi (Bethe response) vs Bethe pairwise cov -> !=1.
    # MF:     off-diagonal Cov is exactly 0 (factorised belief) -> rho -> inf (annotated).
    # Gibbs:  the sampler's covariance vs the true response (both estimate the posterior) -> rho ~ 1.
    def _r1(C):
        return _rd(C, dist, dvals)[1]

    def _rho_own(kind, a, q):
        """rho = r_1(chi^alg) / r_1(Cov^alg_own) at matched confidence q, each scheme's OWN Cov."""
        Jm = _fit_J_for_q(kind, q, B, a, theta, J_grid)
        if not np.isfinite(Jm):
            return np.nan
        if kind == 'exact':
            return _r1(linear_response_cov('exact', Jm, B, theta=theta)[0]) / _r1(_exact_cov(Jm, B, theta))
        # FBP / LBP: fractional-Bethe response over fractional-Bethe pairwise covariance
        chi = linear_response_cov('lbp', Jm, B, theta=theta, alpha=a)[0]
        return _r1(chi) / _r1(_bp_pair_cov(Jm, B, theta, alpha=a))

    for k, md in method_keys:
        if md['kind'] == 'mf':
            continue                                  # rho -> inf (annotated below)
        rho = [_rho_own(md['kind'], md['alpha'], q) for q in q_grid]
        ax[0].plot(q_grid, rho, md['ls'], color=md['c'], marker='o', ms=6, lw=2.2, label=md['lab'])
    gx, gy = [], []
    for q in gibbs_q:
        Jm = _fit_J_for_q('exact', q, B, 1.0, theta, J_grid)
        if not np.isfinite(Jm):
            continue
        chi_e = linear_response_cov('exact', Jm, B, theta=theta)[0]
        gx.append(q); gy.append(_r1(gibbs_susceptibility(Jm, B, theta, *gibbs)) / _r1(chi_e))
    ax[0].plot(gx, gy, ':', color='0.6', marker='D', ms=6, lw=2.0, label='Gibbs')
    ax[0].axhline(1.0, color='k', lw=1.0, ls=':')
    ax[0].text(0.03, 0.06, r'MF: $\mathrm{Cov}_{ij}{=}0\ (i{\neq}j)\Rightarrow\rho\to\infty$',
               transform=ax[0].transAxes, color='r', fontsize=9)
    ax[0].set(xlabel='matched confidence q',
              ylabel=r'$\rho=\chi_{ij}/\mathrm{Cov}^{\,\mathrm{own}}_{ij}$',
              title='(a) fluctuation--dissipation ratio')
    ax[0].legend(frameon=False)

    # ---- (b) confidence calibration -------------------------------------------
    over = {k: [] for k, _ in method_keys}
    for J in tqdm(cal_J, desc='calibration'):
        conf, corr = _calibration_trials(J, cal_trials, alphas=alphas,
                                         theta=theta, seed=seed)
        for k, _ in method_keys:
            over[k].append(abs(conf[k].mean() - corr[k].mean()))
    for k, md in method_keys:
        ax[1].plot(cal_J, over[k], md['ls'], color=md['c'], marker='o', ms=6,
                   lw=2.2, label=md['lab'])
    ax[1].axhline(0, color='0.6', lw=0.8)
    ax[1].axvline(JMF, color='r', ls=':', lw=1)
    ax[1].axvline(JLBP, color=plt.cm.Blues(0.5), ls=':', lw=1)
    ax[1].set(xlabel='coupling J', ylabel='mean confidence - mean accuracy',
              title='(b) confidence calibration')
    ax[1].legend(frameon=False)

    # ---- (c) susceptibility r_d vs matched q ----------------------------------
    for k, md in method_keys:
        r1, r2 = [], []
        for q in q_grid:
            Jm = _fit_J_for_q(md['kind'], q, B, md['alpha'], theta, J_grid)
            if not np.isfinite(Jm):
                r1.append(np.nan); r2.append(np.nan); continue
            rd = _rd(_chi(md['kind'], Jm, B, md['alpha'], theta), dist, dvals)
            r1.append(rd[1]); r2.append(rd[2] if len(rd) > 2 else np.nan)
        ax[2].plot(q_grid, r1, md['ls'], color=md['c'], marker='o', ms=6,
                   lw=2.2, label=md['lab'])
        ax[2].plot(q_grid, r2, md['ls'], color=md['c'], marker='o', ms=6,
                   lw=2.2, alpha=0.55)
    ax[2].set(xlabel='matched confidence q', ylabel=r'response $r_d$',
              title='(c) evidence spread (solid $d{=}1$, dashed $d{=}2$)')
    # Keep distance encoded by line style, as in the existing susceptibility plots.
    # Replot d=2 with dashed lines without changing the algorithm colour.
    for line in ax[2].lines[-len(method_keys)*2:]:
        if line.get_alpha() == 0.55:
            line.set_linestyle('--')
    ax[2].legend(frameon=False)

    for a_ in ax:
        a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    fig.suptitle('Experimentally identifiable differences (coupling matched out)')
    fig.tight_layout()
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=200, bbox_inches='tight')
    return fig


def plot_fdt_decomposition(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                           q_grid=np.round(np.linspace(0.55, 0.9, 10), 3),
                           J_grid=np.round(np.arange(0.0, 6.0, 0.02), 3),
                           gibbs=(150000, 15000), gibbs_q=None,
                           theta=THETA_NECKER, save=True, fname='fdt_decomposition'):
    """Decompose the fluctuation-dissipation ratio into its parts, for ALL algorithms, at
    matched confidence q (neighbour, d=1 values):
      (a) response  chi   = r_1( d<x_i>/dB_j ),
      (b) own covariance C = r_1( Cov^own(x_i,x_j) ),
      (c) ratio rho = chi / C.
    Exact/sampling have chi = C (FDT) so rho = 1; MF has C = 0 off-diagonal (factorised belief)
    so it sits at 0 in (b) and rho -> inf (annotated) in (c); FBP/LBP use the fractional-Bethe
    pairwise covariance (_bp_pair_cov). Same colours/styles as the susceptibility plots."""
    dist = _dist_matrix(theta); dvals = np.arange(0, int(dist.max()) + 1)
    if gibbs_q is None:
        gibbs_q = q_grid

    def r1(C):
        return _rd(C, dist, dvals)[1]

    def chi_C(kind, a, Jm):
        """Neighbour response and neighbour OWN covariance at coupling Jm."""
        if kind == 'exact':
            chi = linear_response_cov('exact', Jm, B, theta=theta)[0]; C = _exact_cov(Jm, B, theta)
        elif kind == 'mf':
            n = theta.shape[0]
            chi = linear_response_cov('mf', Jm, B, theta=theta)[0]
            m = _mf_magnetization(Jm * theta, np.full(n, B), np.zeros(n))  # Jm*theta = matrix
            C = np.diag(1.0 - m ** 2)                       # factorised: 0 off-diagonal
        else:                                              # fbp / lbp / fbp_opt
            a_eff = _alpha_hat(Jm, B, theta) if kind == 'fbp_opt' else a
            chi = linear_response_cov('lbp', Jm, B, theta=theta, alpha=a_eff)[0]
            C = _bp_pair_cov(Jm, B, theta, alpha=a_eff)
        return r1(chi), r1(C)

    methods = _susc_methods(alphas) + [_opt_method()]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for md in tqdm(methods):
        kind, a = md['kind'], md['alpha']
        chi_v, C_v, rho_v = [], [], []
        for q in q_grid:
            Jm = _fit_J_for_q(kind, q, B, a, theta, J_grid)
            if not np.isfinite(Jm):
                chi_v.append(np.nan); C_v.append(np.nan); rho_v.append(np.nan); continue
            c, cc = chi_C(kind, a, Jm)
            chi_v.append(c); C_v.append(cc)
            rho_v.append(c / cc if abs(cc) > 1e-12 else np.nan)   # MF: C=0 -> skip (inf)
        ax[0].plot(q_grid, chi_v, md['ls'], color=md['c'], marker='o', ms=5, lw=2.0, label=md['lab'])
        ax[1].plot(q_grid, C_v, md['ls'], color=md['c'], marker='o', ms=5, lw=2.0, label=md['lab'])
        ax[2].plot(q_grid, rho_v, md['ls'], color=md['c'], marker='o', ms=5, lw=2.0, label=md['lab'])

    # Gibbs (sampling): chi = C = sample covariance -> rho = 1
    gx, gchi, gC = [], [], []
    for q in gibbs_q:
        Jm = _fit_J_for_q('exact', q, B, 1.0, theta, J_grid)
        if not np.isfinite(Jm):
            continue
        Cg = gibbs_susceptibility(Jm, B, theta, *gibbs)
        gx.append(q); gchi.append(r1(Cg)); gC.append(r1(Cg))
    ax[0].plot(gx, gchi, ':', color='0.5', marker='D', ms=6, lw=1.8, label='Gibbs')
    ax[1].plot(gx, gC, ':', color='0.5', marker='D', ms=6, lw=1.8, label='Gibbs')
    ax[2].plot(gx, np.ones_like(gx), ':', color='0.5', marker='D', ms=6, lw=1.8, label='Gibbs')

    ax[2].axhline(1.0, color='k', lw=1.0, ls=':')
    ax[2].text(0.03, 0.06, r'MF: $C_{ij}=0\Rightarrow\rho\to\infty$', transform=ax[2].transAxes,
               color='r', fontsize=9)
    ax[0].set(xlabel='matched confidence q', ylabel=r'$r_1(\chi_{ij})$',
              title=r'(a) response $\chi$ (d=1)')
    ax[1].set(xlabel='matched confidence q', ylabel=r'$r_1(\mathrm{Cov}^{\,\mathrm{own}}_{ij})$',
              title='(b) own covariance $C$ (d=1)')
    ax[2].set(xlabel='matched confidence q', ylabel=r'$\rho=\chi/C$',
              title=r'(c) ratio $\rho$ (d=1)')
    ax[0].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    fig.suptitle('Fluctuation--dissipation decomposition: response, own covariance, and ratio')
    fig.tight_layout()
    if save:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ext_ in ('png', 'svg'):
            fig.savefig(DATA_FOLDER + f'{fname}.{ext_}', dpi=200, bbox_inches='tight')
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
    #                               methods=None, gibbs_steps=100000,
    #                               load_data=True, data_path=None, save=True)
    # plot_error_vs_complexity(n=9, levels=tuple(range(0, 12)), n_graphs=50,
    #                           J_list=(0.2, 0.4, 0.6, 0.8),
    #                           B_list=np.round(np.linspace(-0.5, 0.5, 7), 3),
    #                           metric='kl', xaxis='L', methods=None, gibbs_steps=100000,
    #                           load_data=True, data_path=None, save=True)
    
    # plot_susc_Bq_grid(B_grid=np.round(np.linspace(0, 0.5, 20), 3),
    #                   q_grid=np.round(np.linspace(0.55, 0.95, 20), 3),
    #                   alphas=(0.5, 1.0, 1.5, 2.0),
    #                   J_grid=np.round(np.arange(0, 6, 0.02), 3),
    #                   theta=THETA_NECKER, save=True, recompute=False)

    # plot_susc_ratios(q_star=0.8, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
    #                 J_grid=np.round(np.arange(0.0, 2.0, 0.01), 3), include_gibbs=True,
    #                 gibbs=(400000, 30000), theta=THETA_NECKER, save=True)
    # plot_susc_vs_J(d=1, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
    #                 J_grid=np.round(np.arange(0.05, 1.0, 0.02), 3), include_gibbs=True,
    #                 gibbs=(200000, 10000), theta=THETA_NECKER, save=True)
    # plot_susc_vs_J(d=2, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
    #                 J_grid=np.round(np.arange(0.05, 1.0, 0.02), 3), include_gibbs=True,
    #                 gibbs=(200000, 10000), theta=THETA_NECKER, save=True)
    # plot_susc_vs_J(d=3, B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
    #                 J_grid=np.round(np.arange(0.05, 1.0, 0.02), 3), include_gibbs=True,
    #                 gibbs=(200000, 10000), theta=THETA_NECKER, save=True)
    # plot_susc_overview(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
    #                       q_grid=np.round(np.linspace(0.55, 0.9, 25), 3),
    #                       J_grid_q=np.round(np.arange(0.0, 2.0, 0.01), 3),
    #                       J_grid=np.round(np.arange(0.05, 2.0, 0.025), 3),
    #                       theta=THETA_NECKER, save=True)
    # plot_posterior_matrices(j_list=np.round(np.arange(0.0, 1.0001, 0.005), 4),
    #                         b_list=np.round(np.arange(-0.5, 0.5001, 0.005), 4),
    #                         alphas=(0.5, 1.0, 1.5, 2.0), include_opt=True,
    #                         gibbs=(1000, 10000, 100000), node=0, show_jstar=True,
    #                         theta=THETA_NECKER, save=True, recompute=False)
    
    # plot_overconfidence_vs_J(j_list=np.round(np.arange(0.0, 1.0001, 0.02), 3),
    #                               b_list=np.round(np.arange(-0.5, 0.5001, 0.02), 3),
    #                               alphas=(0.5, 1.0, 1.5, 2.0), gibbs=(1000, 10000, 100000),
    #                               node=0, theta=THETA_NECKER,
    #                               recompute=False, save=True, include_opt=True)

    # main figure: exact - MF - LBP - FBP(alpha-hat) - Gibbs 1e3/1e4/1e5
    # plot_posterior_matrices(
    #     j_list=np.round(np.arange(0.0, 1.0001, 0.005), 4),
    #     b_list=np.round(np.arange(-0.5, 0.5001, 0.005), 4),
    #     alphas=(1.0,), include_opt=True, gibbs=(1000, 10000, 100000),
    #     theta=THETA_NECKER, save=True, fname='posterior_matrices_main')

    # supp figure: full FBP alpha sweep
    # plot_posterior_matrices(
    #     j_list=np.round(np.arange(0.0, 1.0001, 0.005), 4),
    #     b_list=np.round(np.arange(-0.5, 0.5001, 0.005), 4),
    #     alphas=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    #     include_opt=True, gibbs=(), theta=THETA_NECKER, save=True,
    #     fname='posterior_matrices_supp_alpha')
    
    # plot_mse_vs_alpha(j_list=np.round(np.arange(0.0, 1.0001, 0.02), 3),
    #                   b_list=np.round(np.arange(-0.5, 0.5001, 0.02), 3),
    #                   alpha_grid=np.round(np.arange(0.025, 2.501, 0.025), 2),
    #                   J_show=(0.1, 0.25, 0.5, 0.75, 1), init='det', steps=100, node=0,
    #                   theta=THETA_NECKER, recompute=False, save=True, fname='mse_vs_alpha')

    # plot_paper_figure(j_list=np.round(np.arange(0.0, 1.0001, 0.005), 4),
    #                       b_list=np.round(np.arange(-0.5, 0.5001, 0.005), 4),
    #                       gibbs=(1000, 10000, 100000),
    #                       susc_alphas=(0.5, 1.0, 1.5, 2.0), susc_B=0.1,
    #                       susc_q_grid=np.round(np.linspace(0.55, 0.99, 12), 3),
    #                       susc_J_grid=np.round(np.arange(0.0, 6.0, 0.02), 3),
    #                       rdJ_d=1, rdJ_J_grid=np.round(np.arange(0.05, 1.0, 0.05), 3),
    #                       steps=100, node=0, gibbs_c=10.0, theta=THETA_NECKER,
    #                       recompute=False, save=True, fname='paper_figure')
    # plot_gibbs_jstar_overconfidence(
    #         T_grid=np.round(np.logspace(2, 6, 9)).astype(int),
    #         J_list=(0.5, 0.7, 0.9, 1.1),
    #         B_grid=np.round(np.linspace(-0.5, 0.5, 21), 3),
    #         Bstar_list=(0.0, 0.2, 0.4),
    #         n_seeds=10, c=10.0, tilt=6.0, burn=1000, node_mean=True,
    #         theta=THETA_NECKER, save=True, fname='gibbs_jstar_overconfidence')
    # plot_confidence_calibration(J_list=np.arange(0, 0.85, 0.05),
    #                             J_show=0.4, n_trials=1500, alphas=(0.5, 1.0, 1.5, 2.0),
    #                             theta=THETA_NECKER, seed=0, save=True,
    #                             fname='confidence_calibration')
    # plot_testable_differences(B=0.2, alphas=(0.5, 1.0, 1.5, 2.0),
    #                           q_grid=np.round(np.linspace(0.55, 0.93, 15), 3),
    #                           J_grid=np.round(np.arange(0.0, 8.0, 0.01), 3),
    #                           cal_J=np.arange(0, 0.85, 0.05), cal_trials=1200,
    #                           gibbs=(1000000, 12000), gibbs_q=None,
    #                           theta=THETA_NECKER, seed=0, save=True,
    #                           fname='testable_differences_B_02')
    # plot_testable_differences(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
    #                           q_grid=np.round(np.linspace(0.55, 0.93, 15), 3),
    #                           J_grid=np.round(np.arange(0.0, 8.0, 0.01), 3),
    #                           cal_J=np.arange(0, 0.85, 0.05), cal_trials=1200,
    #                           gibbs=(1000000, 12000), gibbs_q=None,
    #                           theta=THETA_NECKER, seed=0, save=True,
    #                           fname='testable_differences_B_01')
    plot_fdt_decomposition(B=0.1, alphas=(0.5, 1.0, 1.5, 2.0),
                               q_grid=np.round(np.linspace(0.55, 0.95, 15), 3),
                               J_grid=np.round(np.arange(0.0, 6.0, 0.01), 3),
                               gibbs=(150000, 15000), gibbs_q=None,
                               theta=THETA_NECKER, save=True, fname='fdt_decomposition_01')
    plot_fdt_decomposition(B=0.2, alphas=(0.5, 1.0, 1.5, 2.0),
                               q_grid=np.round(np.linspace(0.55, 0.95, 15), 3),
                               J_grid=np.round(np.arange(0.0, 6.0, 0.01), 3),
                               gibbs=(150000, 15000), gibbs_q=None,
                               theta=THETA_NECKER, save=True, fname='fdt_decomposition_02')
