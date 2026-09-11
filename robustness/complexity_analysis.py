# -*- coding: utf-8 -*-
"""
Robustness of inference algorithms across (coupling J) x (PGM complexity), on the
IMAGE denoising task. Complexity = pixel-lattice CONNECTIVITY (4->8->24 neighbours;
more neighbours = more loops). One imposed ferromagnetic prior per (complexity, J,
seed). Score = per-pixel accuracy (or MSE) vs the clean GT field.

Four analyses, each a J_SWEEP x COMPLEXITY panel figure (one line per algorithm):
  1. exact           : accuracy vs exact-inference reference   (ONLY if N<=EXACT_MAX,
                       otherwise silently skipped -- exact is 2^N)
  2. blackbox_flip   : accuracy vs GT under random label flips (sweep strength)
  3. blackbox_<name> : accuracy vs GT under other black-box corruptions (gaussian, impulse, ...)
  4. whitebox_l0     : accuracy vs GT under greedy/gradient L0 targeted flips

Results are CACHED to results/complexity_<analysis>_<hash>.pkl -> re-plotting or
adding seeds is cheap. Run as many seeds/graphs as you like via N_SEEDS.

Colors: MF firebrick; FBP Blues (darker=higher alpha); exact black, Gibbs dimgray.

Usage:
    python complexity_analysis.py                 # compute+plot all analyses
    python complexity_analysis.py --plot          # re-plot from cache only
    python complexity_analysis.py --seeds 20      # more graphs/seeds
    python complexity_analysis.py --metric mse
"""
import argparse, os, pickle, hashlib, itertools
import numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from binary_mrf_denoise import (make_fields, flip, infer_mf, infer_fbp_sparse,
                                infer_sampling, infer_gibbs_discrete, l0_flip,
                                G, N, BETA, ITERS, N_SAMPLES)
from mrf_inference_nets import CORRUPTIONS


pc = 'CRM'
if pc == 'Alex':
    SAVE_ROOT = r"C:\Users\alexg\OneDrive\Escritorio\phd\folder_save\robustness_analysis\depth_denoise"
if pc == 'CRM':
    SAVE_ROOT = r"C:\Users\agarcia\Desktop\phd\necker\data_folder"

# ------------------------------------------------------------------ config ---
CONN = (4, 8, 12)                        # complexity levels (neighbours)
J_SWEEP = (0.3, 0.6, 0.9, 1.2)           # coupling scales
N_SEEDS = 10                             # imposed-prior draws per cell (graphs)
N_TEST = 20                              # clean fields per seed
BASE_P = 0.15                            # base observation flip noise
FLIP_STR = (0.0, 0.1, 0.2, 0.3, 0.4)     # black-box flip sweep
NAT_NAMES = ('gaussian', 'impulse')      # other black-box corruptions to sweep
NAT_STR = (0.0, 0.2, 0.4, 0.6, 0.8)
EPS_L0 = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2)    # white-box L0 budget (fraction pixels)
EXACT_MAX = 20                           # exact only if N <= this (2^N enumeration)
ALGOS = [('mf', None, 'MF'), ('fbp', 0.5, 'FBP0.5'), ('fbp', 1.0, 'LBP'),
         ('fbp', 1.5, 'FBP1.5'), ('fbp', 2.0, 'FBP2.0'), ('gibbs', None, 'Gibbs')]
if N <= EXACT_MAX:
    ALGOS = [('exact', None, 'exact')] + ALGOS


def col(kind, alpha):
    if kind == 'mf':    return 'firebrick'
    if kind == 'exact': return 'black'
    if kind == 'gibbs': return 'dimgray'
    return plt.cm.Blues({0.5: 0.42, 1.0: 0.60, 1.5: 0.78, 2.0: 0.97}[alpha])


# --------------------------------------------------------------- graph/prior -
def grid_adjacency_k(g, k):
    """k = number of neighbours per interior pixel:
       4  = von Neumann  (N,S,E,W)
       8  = Moore r1     (+ 4 diagonals)
       12 = Manhattan<=2 (8 + the 4 straight-2 offsets)
       24 = Chebyshev r2 (full 5x5 block minus centre)."""
    d = {(dx, dy): max(abs(dx), abs(dy)) for dx in range(-2, 3) for dy in range(-2, 3)}
    man = {(dx, dy): abs(dx) + abs(dy) for dx in range(-2, 3) for dy in range(-2, 3)}
    if k == 4:
        offs = [o for o in d if man[o] == 1]
    elif k == 8:
        offs = [o for o in d if d[o] == 1]
    elif k == 12:
        offs = [o for o in d if 1 <= man[o] <= 2]
    elif k == 24:
        offs = [o for o in d if d[o] in (1, 2)]
    else:
        raise ValueError(f'k must be 4, 8, 12 or 24 (got {k})')
    A = torch.zeros(g * g, g * g)
    for r in range(g):
        for c in range(g):
            i = r * g + c
            for dx, dy in offs:
                rr, cc = r + dx, c + dy
                if 0 <= rr < g and 0 <= cc < g:
                    A[i, rr * g + cc] = 1.0
    return A


def ferro_J(A, seed):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(N, N, generator=gen) * A            # only on-graph edges
    Jm = (W.abs() + W.abs().t()) / 2
    return Jm


# --------------------------------------------------------------- inference ---
def _exact_marg(Jm, B):
    """Exact P(x_i=1) by enumeration (small N only)."""
    Jn = Jm.numpy(); Bn = B.numpy()
    n = Bn.shape[1]; states = np.array(list(itertools.product([-1, 1], repeat=n)), float)
    E = 0.5 * np.einsum('si,ij,sj->s', states, Jn, states)[None] + Bn @ states.T   # [b,S]
    w = np.exp(E - E.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    q = w @ ((states + 1) / 2)                                                      # [b,n]
    return torch.tensor(q * 2 - 1, dtype=torch.float32)                             # spins


@torch.no_grad()
def infer(o, Jm, kind, alpha, discrete=True):
    B = BETA * (2 * o.view(o.shape[0], -1) - 1)
    if kind == 'mf':    return infer_mf(B, Jm, ITERS)
    if kind == 'exact': return _exact_marg(Jm, B)
    if kind == 'gibbs':
        return infer_gibbs_discrete(B, Jm, iters=ITERS, n_samples=N_SAMPLES) if discrete \
               else infer_sampling(B, Jm, iters=ITERS, n_samples=N_SAMPLES)
    return infer_fbp_sparse(B, Jm, alpha=alpha, iters=ITERS)


def _diff_forward(kind, alpha, Jm):
    """A differentiable o->marginals map for white-box L0 (relaxed gibbs)."""
    def f(o, sampler=None):
        B = BETA * (2 * o.view(o.shape[0], -1) - 1)
        if kind == 'mf':    return infer_mf(B, Jm, ITERS)
        if kind == 'gibbs': return infer_sampling(B, Jm, iters=ITERS, n_samples=N_SAMPLES)
        return infer_fbp_sparse(B, Jm, alpha=alpha, iters=ITERS)
    return f


def score(m, s, metric):
    if metric == 'mse':
        return (((m + 1) / 2 - s.view(m.shape)) ** 2).mean().item()
    return ((m > 0).float() == s.view(m.shape)).float().mean().item()


# --------------------------------------------------------------- evaluation --
def _cache(analysis, n_seeds, metric):
    key = hashlib.md5(repr((analysis, CONN, J_SWEEP, n_seeds, N_TEST, G, metric,
                            FLIP_STR, NAT_STR, EPS_L0, BASE_P)).encode()).hexdigest()[:8]
    return os.path.join(SAVE_ROOT, 'results', f'complexity_{analysis}_{key}.pkl')


def evaluate(analysis, n_seeds=N_SEEDS, metric='acc', recompute=False):
    """Compute curves[(conn,J,label)] = array [n_seeds*N_TEST, len(xs)]. Cached."""
    os.makedirs(os.path.join(SAVE_ROOT, 'results'), exist_ok=True)
    cp = _cache(analysis, n_seeds, metric)
    if os.path.exists(cp) and not recompute:
        return pickle.load(open(cp, 'rb'))
    xs = (EPS_L0 if analysis == 'whitebox_l0'
          else FLIP_STR if analysis in ('blackbox_flip', 'exact')
          else NAT_STR)
    S = make_fields(N_TEST, seed=123)
    curves = {}
    pbar = tqdm(total=len(CONN) * len(J_SWEEP) * n_seeds, desc=analysis)
    for k in CONN:
        A = grid_adjacency_k(G, k)
        for J in J_SWEEP:
            for seed in range(n_seeds):
                pbar.set_postfix(conn=k, J=J, seed=seed); pbar.update(1)
                Jm = ferro_J(A, seed) * J
                o0 = flip(S, BASE_P, torch.Generator().manual_seed(700 + seed))
                for kind, alpha, lab in ALGOS:
                    if analysis == 'whitebox_l0' and kind == 'exact':
                        continue                              # exact has no gradients
                    row = []
                    for st in xs:
                        if analysis == 'whitebox_l0':
                            model = type('M', (), {'__call__': staticmethod(_diff_forward(kind, alpha, Jm))})()
                            oa = o0 if st == 0 else _l0(model, o0, S, int(round(st * N)))
                            m = infer(oa, Jm, kind, alpha)
                        elif analysis in ('blackbox_flip', 'exact'):
                            oc = flip(o0, st, torch.Generator().manual_seed(int(st * 100) + 3))
                            m = infer(oc, Jm, kind, alpha)
                        else:                                   # blackbox_<name>
                            name = analysis.split('blackbox_')[1]
                            gen = torch.Generator().manual_seed(int(st * 100) + 5)
                            oc = CORRUPTIONS[name](o0, st, gen) if st > 0 else o0
                            m = infer(oc, Jm, kind, alpha)
                        if analysis == 'exact':
                            me = infer(oc if 'oc' in dir() else o0, Jm, 'exact', None)
                            row.append(score(m, (me > 0).float(), metric))   # vs exact labels
                        else:
                            row.append(score(m, S, metric))
                    curves.setdefault((k, J, lab), []).append(row)
    pbar.close()
    curves = {kk: np.array(v) for kk, v in curves.items()}
    curves['_xs'] = np.array(xs); curves['_metric'] = metric
    pickle.dump(curves, open(cp, 'wb')); print('cached', cp)
    return curves


def _l0(model, o, s, k):
    return l0_flip(model, o, s, k, steps=min(8, max(1, k)))


# --------------------------------------------------------------- plotting ----
def plot(analysis, n_seeds=N_SEEDS, metric='acc'):
    C = evaluate(analysis, n_seeds, metric)
    xs = C['_xs']; ylab = 'MSE vs GT' if metric == 'mse' else 'accuracy vs GT'
    if analysis == 'exact':
        ylab = 'MSE vs exact' if metric == 'mse' else 'agreement w/ exact'
    fig, axes = plt.subplots(len(CONN), len(J_SWEEP),
                             figsize=(3.0 * len(J_SWEEP), 2.7 * len(CONN)),
                             squeeze=False, sharex=True, sharey=True)
    for ri, k in enumerate(CONN):
        for ci, J in enumerate(J_SWEEP):
            ax = axes[ri][ci]
            for kind, alpha, lab in ALGOS:
                if (k, J, lab) not in C:
                    continue
                arr = C[(k, J, lab)]; mu, sd = arr.mean(0), arr.std(0)
                ax.plot(xs, mu, '-o', ms=3, color=col(kind, alpha), label=lab)
                ax.fill_between(xs, mu - sd, mu + sd, color=col(kind, alpha), alpha=0.12)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if ri == 0: ax.set_title(f'J={J}', fontsize=10)
            if ci == 0: ax.set_ylabel(f'{k}-neigh\n{ylab}', fontsize=8)
            if ri == len(CONN) - 1: ax.set_xlabel('corruption strength', fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.suptitle(f'{analysis}: {ylab}  (rows=complexity/connectivity, cols=coupling J; {n_seeds} seeds +/-1 std)')
    fig.tight_layout()
    resdir = os.path.join(SAVE_ROOT, 'results')
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(resdir, f'complexity_{analysis}.{ext}'), dpi=160, bbox_inches='tight')
    print('saved', os.path.join(resdir, f'complexity_{analysis}.png'))
    return fig


ANALYSES = (['exact'] if N <= EXACT_MAX else []) + \
           ['blackbox_flip'] + [f'blackbox_{n}' for n in NAT_NAMES] + ['whitebox_l0']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='re-plot from cache only')
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--metric', choices=['acc', 'mse'], default='acc')
    ap.add_argument('--only', default=None, help='run one analysis, e.g. blackbox_flip')
    args, _ = ap.parse_known_args()
    todo = [args.only] if args.only else ANALYSES
    for a in todo:
        if not args.plot:
            evaluate(a, args.seeds, args.metric)
        plot(a, args.seeds, args.metric)
