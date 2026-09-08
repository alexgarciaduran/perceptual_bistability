# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:30:09 2026

@author: alexg
"""

import numpy as np
import itertools
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy.optimize


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
# Gibbs sampling
# ----------------------------
def gibbs_sampling(J, B, steps=10000, burn_in=1000):
    n = len(B)
    s = np.random.choice([-1, 1], size=n)

    samples = []
    for t in range(steps):
        for i in range(n):
            h = B[i] + np.dot(J[i], s)
            prob = 1 / (1 + np.exp(-2 * h))
            s[i] = 1 if np.random.rand() < prob else -1

        if t >= burn_in:
            samples.append(s.copy())

    samples = np.array(samples)
    marginals = (samples == 1).mean(axis=0)
    return marginals

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
        print(f"\n=== Running for p = {p} ===")
        results = []

        for graph_id in range(N):
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

            # Fractional BP
            for alpha in [0.5, 0.75, 1.25, 1.5, 2, 2.5, 3]:
                res[f"fbp_{alpha}"] = loopy_bp(J, B, alpha=alpha, max_iter=200)

            # Fractional BP at the degree-aware optimal (inference) alpha. The
            # homogeneous theory needs scalars: j = mean nonzero |J|, b = mean B,
            # n = mean degree, true marginal p = mean(exact).
            Jvals = np.abs(J)[J != 0]
            j_scalar = Jvals.mean() if Jvals.size else 0.0
            n_mean = np.mean(np.sum(A != 0, axis=1))
            res["alpha_opt"] = optimal_alpha(j_scalar, np.mean(B),
                                             float(np.mean(res["exact"])), n_mean)
            res["fbp_opt"] = loopy_bp(J, B, alpha=res["alpha_opt"], max_iter=200)

            results.append(res)

            print(f"Graph {graph_id+1}/{N} done")

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
                   "fbp_1.5", "fbp_2.5", "fbp_opt"]

    method_names = ['Gibbs\nsampling', 'Mean-Field',
                    'LBP', r'FBP ($\alpha=1.5$)',
                    r'FBP ($\alpha=2.5$)', r'FBP ($\hat{\alpha}$)']
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
        for J_val in J_list:
            print(f"\n=== d={d}, J={J_val} ===")
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

                # Fractional BP
                for alpha in [0.5, 0.75, 1.25, 1.5, 2, 2.5, 3]:
                    res[f"fbp_{alpha}"] = loopy_bp(J_mat, B, alpha=alpha, max_iter=200)

                # Fractional BP at the degree-aware optimal (inference) alpha.
                # (J_val, B_val) scalar, every node has degree d -> n = d,
                # true marginal p = mean(exact).
                res["alpha_opt"] = optimal_alpha(J_val, B_val,
                                                 float(np.mean(res["exact"])), d)
                res["fbp_opt"] = loopy_bp(J_mat, B, alpha=res["alpha_opt"], max_iter=200)

                results[d][J_val].append(res)

                print(f"Graph {graph_id+1}/{N} done")

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


if __name__ == "__main__":
    p_list = np.round(np.arange(0.2, 1.01, 0.1), 2)
    d_list = list(range(2, 7))   # degrees 2-6
    J_list = np.round(np.arange(0., 1.01, 0.1), 2) # J = 0.0, 0.05, ..., 0.5

    # --- (re)generate the data ONCE: this now stores res["fbp_opt"] too -----
    results_regular = run_regular_graph_experiment(d_list, J_list, n=8, N=30)

    with open(DATA_FOLDER + "/ising_results_multi_J_d.pkl", "wb") as f:
        pickle.dump(results_regular, f)

    results = run_experiment_multi_p(p_list, N=30, n=8)

    with open(DATA_FOLDER + "/ising_results_multi_p.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"Saved to {DATA_FOLDER}/ising_results_multi_J_d.pkl")

    # --- afterwards: just load + plot (fbp_opt is already in the pkl) --------
    with open(DATA_FOLDER + "/ising_results_multi_J_d.pkl", "rb") as f:
        all_results = pickle.load(f)

    plot_regular_results_dcolor(all_results, d_list, N=30)

    with open(DATA_FOLDER + "/ising_results_multi_p.pkl", "rb") as f:
        all_results = pickle.load(f)

    plot_grid_by_method_and_p(all_results, methods=None)
