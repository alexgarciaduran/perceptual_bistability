# -*- coding: utf-8 -*-
"""
Per-pixel binary-MRF DENOISING: isolate the inference ALGORITHM's robustness.

    noisy observation o (GxG, in [0,1])
        --> evidence field  B = beta*(2o - 1)  over a LOCAL pixel-lattice Ising graph
        --[inference: gibbs | mf | fbp(alpha) | lbp;  IMPOSED ferromagnetic J]-->
        marginals  m_i = <s_i>            (identity readout: the marginals ARE the output)
        --> per-pixel label  s_hat_i = 1[m_i > 0]

There is NO encoder and NO decoder: the observation is the evidence field and the
marginals are the answer, so the inference is the whole computation and nothing can
absorb or mask algorithm differences.  The prior J is IMPOSED (frozen, ferromagnetic
smoothness) -> ZERO trainable parameters.  A network = one random draw of the ferro
prior at a given coupling scale.

Two design constraints, both enforced here:
  (1) networks differ ONLY by the inference algorithm  (same observation, same J draw,
      same iters/samples = matched compute, identity readout);
  (2) conclusions must not depend on the PGM coupling -> we SWEEP J (J_SWEEP), run
      N_SEEDS networks per J, and provide a clean-accuracy-matching driver so
      robustness can be compared at a matched operating point, not a matched J.

Task ground truth (the true field) is set by the data, never by J: J is only a prior.
Score = per-pixel accuracy.  `linear` = no-inference control (threshold the raw obs).

Run:  python binary_mrf_denoise.py            (eval sweep + plot, resumable per net)
      python binary_mrf_denoise.py --fast     (tiny smoke test)
      python binary_mrf_denoise.py --plot     (re-plot from saved results)
"""
import argparse, json, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mrf_inference_nets import (grid_adjacency, infer_mf, infer_sampling,
                                infer_gibbs_discrete, VARIANTS, CORRUPTIONS)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 15
plt.rcParams["axes.grid"] = False

SAVE_ROOT = r"C:\Users\alexg\OneDrive\Escritorio\phd\folder_save\robustness_analysis\depth_denoise"

G = 28                      # GxG lattice
N = G * G                   # nodes
BETA = 1.0                  # evidence strength (fixed)
ITERS = 30                  # inference sweeps (matched across variants)
N_SAMPLES = 20              # chains for sampling variants (matched)
BLOB_SIGMA = 3.0            # spatial scale of the synthetic regions (piecewise-smooth)
BASE_P = 0.2                # base observation flip-noise (the nominal denoising task)
N_TEST = 40                 # images the whole robustness eval runs on (one batch)

# ---- the coupling sweep: constraint (2) ----
J_SWEEP = (0.15, 0.25, 0.35, 0.5, 0.7)   # 5 coupling scales
N_SEEDS = 10                              # networks (random ferro-J draws) per J

LINEAR = 'linear'          # no-inference control: threshold the raw observation
RUN_VARIANTS = ['linear', 'gibbs20', 'mf', 'fbp0.5', 'lbp', 'fbp1.5', 'fbp2.0']
EPS_LINF = (0, 0.05, 0.1, 0.15, 0.2, 0.3)
EPS_L2 = (0, 0.5, 1.0, 1.5, 2.0, 3.0)
NAT_STR = (0, 0.1, 0.2, 0.3, 0.4)         # EXTRA corruption on top of the base obs
NORMS = (('linf', EPS_LINF), ('l2', EPS_L2))


def _is_sampling(v):
    return v in VARIANTS and VARIANTS[v][0] == 'sampling'


def _eval_modes(v):
    return ['relaxed_sampling', 'gibbs_sampling'] if _is_sampling(v) else ['det']


def _color(label):
    base = {'linear': '0.6', 'gibbs20': 'k', 'mf': 'r', 'fbp0.5': 'green',
            'lbp': 'C0', 'fbp1.5': 'purple', 'fbp2.0': 'orange'}
    return base.get(label.split('::')[0], 'gray')


def _style(label):
    return '--' if label.endswith('relaxed_sampling') else '-'


# ----------------------------------------------------- sparse fractional BP -
def infer_fbp_sparse(B, Jmat, alpha=1.0, iters=15, damping=0.5):
    """Fractional BP with messages on the graph's directed edges only (O(E))."""
    bsz, n = B.shape
    idx = (Jmat != 0).nonzero(as_tuple=False)
    src, tgt = idx[:, 0], idx[:, 1]
    key = src * n + tgt
    order = torch.argsort(key)
    rev = order[torch.searchsorted(key[order], tgt * n + src)]
    Jt = torch.tanh(alpha * Jmat[src, tgt])
    M = torch.zeros(bsz, src.shape[0])
    for _ in range(iters):
        Q = B.index_add(1, tgt, M)
        h = Q[:, src] - alpha * M[:, rev]
        arg = (Jt * torch.tanh(h)).clamp(-0.999, 0.999)
        newM = (1.0 / alpha) * torch.atanh(arg)
        M = damping * newM + (1 - damping) * M
    return torch.tanh(B.index_add(1, tgt, M))


# --------------------------------------------------------------- data -------
def _gauss_blur(x, s):
    r = max(1, int(3 * s)); xs = torch.arange(-r, r + 1).float()
    k = torch.exp(-xs**2 / (2 * s * s)); k /= k.sum()
    x = F.pad(x, (r, r, r, r), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1)); return F.conv2d(x, k.view(1, 1, -1, 1))


def make_fields(n, seed=0):
    """n piecewise-smooth binary fields s in {0,1}, [n,1,G,G] (thresholded blurred
    noise -> organic regions that match the ferromagnetic smoothness prior)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 1, G, G, generator=g)
    return (_gauss_blur(z, BLOB_SIGMA) > 0.0).float()


def flip(x, p, gen=None):
    if p <= 0:
        return x
    m = (torch.rand(x.shape, generator=gen) < p)
    return torch.where(m, 1.0 - x, x)


def corrupt_flip(x, s, gen):
    return flip(x, s, gen)


CORR = {**CORRUPTIONS, 'flip': corrupt_flip}


# --------------------------------------------------------------- model ------
class BinaryMRF(nn.Module):
    """Identity-readout MRF denoiser. IMPOSED ferromagnetic prior (frozen), so the
    module has NO trainable parameters -- a 'network' is a (coupling j_sigma, seed)
    draw. forward returns the per-pixel marginals m (spins in [-1,1])."""
    def __init__(self, variant, j_sigma, seed=0, beta=BETA, iters=ITERS):
        super().__init__()
        self.variant = variant
        self.algo = 'linear' if variant == LINEAR else VARIANTS[variant][0]
        self.alpha = float(VARIANTS[variant][1]) if self.algo == 'fbp' else 1.0
        self.beta, self.iters = beta, iters
        A = grid_adjacency(G)
        self.register_buffer('mask', A)
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(N, N, generator=gen) * j_sigma
        J0 = A * (W.abs() + W.abs().t()) / 2         # ferromagnetic, imposed, frozen
        self.register_buffer('Jraw', J0)

    def Jmat(self):
        J = self.Jraw * self.mask
        return (J + J.t()) / 2

    def _infer(self, B, Jm, sampler):
        if self.algo == 'linear':
            return B                                 # no inference: raw evidence
        if self.algo == 'mf':
            return infer_mf(B, Jm, self.iters)
        if self.algo == 'sampling':
            if sampler == 'gibbs':
                return infer_gibbs_discrete(B, Jm, iters=self.iters, n_samples=N_SAMPLES)
            return infer_sampling(B, Jm, iters=self.iters, n_samples=N_SAMPLES)
        return infer_fbp_sparse(B, Jm, alpha=self.alpha, iters=self.iters)

    def forward(self, o, sampler=None):
        B = self.beta * (2 * o.view(o.shape[0], -1) - 1)     # observation = evidence
        return self._infer(B, self.Jmat(), sampler)          # marginals (identity readout)


# --------------------------------------------------------------- metrics ----
@torch.no_grad()
def acc_px(model, o, s, sampler=None):
    """Per-pixel labelling accuracy: 1[m>0] vs the true field s."""
    m = model(o, sampler=sampler)
    return ((m > 0).float() == s.view(m.shape)).float().mean().item()


def _bce(model, o, s):
    m = model(o)                                     # differentiable (relaxed for sampling)
    p = ((m + 1) / 2).clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(p, s.view(m.shape))


def pgd_px(model, o, s, eps, norm='linf', steps=20):
    """Per-pixel white-box PGD on the observation (maximise the denoiser's BCE)."""
    if eps == 0:
        return o
    a = 2.5 * eps / steps
    b = o.shape[0]
    if norm == 'linf':
        oa = (o + torch.empty_like(o).uniform_(-eps, eps)).clamp(0, 1).detach()
        for _ in range(steps):
            oa.requires_grad_(True)
            g, = torch.autograd.grad(_bce(model, oa, s), oa)
            oa = (oa + a * g.sign()).clamp(o - eps, o + eps).clamp(0, 1).detach()
        return oa
    oa = o.clone().detach()
    for _ in range(steps):
        oa.requires_grad_(True)
        g, = torch.autograd.grad(_bce(model, oa, s), oa)
        gn = g / (g.view(b, -1).norm(dim=1).view(b, 1, 1, 1) + 1e-12)
        oa = oa + a * gn
        delta = (oa - o).view(b, -1)
        f = (eps / delta.norm(dim=1).clamp(min=1e-12)).clamp(max=1.0)
        oa = (o + (delta * f.view(b, 1)).view_as(o)).clamp(0, 1).detach()
    return oa


def corrupt_curve(model, o0, s, fn, strengths, sampler=None, seed=0):
    """Per-pixel accuracy as an EXTRA corruption is swept on top of the base obs."""
    out = []
    for st in strengths:
        gen = torch.Generator().manual_seed(seed)
        oc = fn(o0, st, gen) if st > 0 else o0
        out.append(acc_px(model, oc, s, sampler))
    return out


# --------------------------------------------------------------- orchestrate
def _paths(j_sigma, variant, seed):
    d = os.path.join(SAVE_ROOT, f'J{j_sigma}', variant, f'seed{seed}')
    return d, os.path.join(d, 'robustness.json')


def run(fast=False, steps=20, re_compute=False):
    """J-sweep x seeds x variants, per-pixel robustness, resumable per network.
    No training (imposed prior). Aggregates to results/robustness.pkl."""
    js = [0.35] if fast else list(J_SWEEP)
    variants = ['linear', 'mf', 'lbp', 'gibbs20'] if fast else RUN_VARIANTS
    nseed = 1 if fast else N_SEEDS
    S = make_fields(8 if fast else N_TEST, seed=12345)          # fixed test fields
    results = {'linf': {}, 'l2': {}, 'nat': {}, 'transfer': {}, 'clean': {},
               'eps': {'linf': EPS_LINF, 'l2': EPS_L2}, 'nat_strengths': NAT_STR,
               'J_sweep': js}
    for j_sigma in js:
        for seed in range(nseed):
            print(f'J = {j_sigma}, seed = {seed}')
            o0 = flip(S, BASE_P, torch.Generator().manual_seed(1000 + seed))   # base obs
            todo = [v for v in variants if re_compute or
                    not os.path.exists(_paths(j_sigma, v, seed)[1])]
            models, adv = {}, {}
            print('Load or save models')
            if todo:
                for v in variants:
                    models[v] = BinaryMRF(v, j_sigma, seed); models[v].eval()
                print('Run PGD')
                for v in tqdm(models, leave=False): # craft PGD once per source
                    print(f'Model {v}')
                    adv[v] = {nm: {e: (pgd_px(models[v], o0, S, e, nm, steps) if e else o0)
                                   for e in epl} for nm, epl in NORMS}
            print('Start attacks')
            for v in tqdm(variants, desc=f'J={j_sigma} seed{seed}', leave=False):
                d, rp = _paths(j_sigma, v, seed); os.makedirs(d, exist_ok=True)
                if os.path.exists(rp) and not re_compute:
                    rec = json.load(open(rp))
                else:
                    model = models[v]; rec = {'modes': _eval_modes(v)}
                    for mode in rec['modes']:
                        smp = 'gibbs' if mode == 'gibbs_sampling' else None
                        r = {}
                        for nm, epl in NORMS:
                            r[nm] = {'eps': list(epl),
                                     'acc': [acc_px(model, adv[v][nm][e], S, smp) for e in epl]}
                        r['transfer'] = {src: {nm: {'eps': list(epl),
                            'acc': [acc_px(model, adv[src][nm][e], S, smp) for e in epl]}
                            for nm, epl in NORMS} for src in models}
                        r['nat'] = {name: {'strength': list(NAT_STR),
                            'acc': corrupt_curve(model, o0, S, fn, NAT_STR, smp)}
                            for name, fn in CORR.items()}
                        r['clean'] = acc_px(model, o0, S, smp)   # nominal denoising acc
                        rec[mode] = r
                    json.dump(rec, open(rp, 'w'), indent=2)
                    print(f"  J={j_sigma} {v}/seed{seed}: clean={rec[rec['modes'][0]]['clean']:.3f}")
                for mode in rec['modes']:
                    lab = v if mode == 'det' else f'{v}::{mode}'
                    key = (j_sigma, lab)
                    results['clean'].setdefault(key, []).append(rec[mode]['clean'])
                    results['linf'].setdefault(key, []).append(rec[mode]['linf']['acc'])
                    results['l2'].setdefault(key, []).append(rec[mode]['l2']['acc'])
                    for name in CORR:
                        results['nat'].setdefault((j_sigma, lab, name), []).append(rec[mode]['nat'][name]['acc'])
                    for nm, _ in NORMS:
                        for src in rec[mode]['transfer']:
                            results['transfer'].setdefault((j_sigma, nm, lab, src), []).append(
                                rec[mode]['transfer'][src][nm]['acc'])
    os.makedirs(os.path.join(SAVE_ROOT, 'results'), exist_ok=True)
    pickle.dump(results, open(os.path.join(SAVE_ROOT, 'results', 'robustness.pkl'), 'wb'))
    return results


# --------------------------------------------------------------- plotting ---
def _eps50(eps, acc):
    """Critical budget: eps where accuracy/clean falls to 0.5 (linear interp)."""
    acc = np.asarray(acc, float); c = acc[0]
    if c <= 0:
        return np.nan
    rel = acc / c
    for i in range(1, len(rel)):
        if rel[i] <= 0.5:
            x0, x1, y0, y1 = eps[i-1], eps[i], rel[i-1], rel[i]
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
    return eps[-1]


def plot(results=None, normalize=True):
    if results is None:
        results = pickle.load(open(os.path.join(SAVE_ROOT, 'results', 'robustness.pkl'), 'rb'))
    resdir = os.path.join(SAVE_ROOT, 'results'); os.makedirs(resdir, exist_ok=True)
    js = results['J_sweep']
    labs = sorted({k[1] for k in results['linf']})

    # (A) robustness curves: one panel per J, per norm (raw + normalized)
    for nm in ('linf', 'l2'):
        eps = results['eps'][nm]
        for norm_flag, sfx, ylab in [(False, '', 'per-pixel acc'), (True, '_norm', 'acc / clean')]:
            fig, axes = plt.subplots(1, len(js), figsize=(3.6*len(js), 3.8), sharey=True, squeeze=False)
            for ax, j in zip(axes[0], js):
                for lab in labs:
                    if (j, lab) not in results[nm]:
                        continue
                    arr = np.array(results[nm][(j, lab)])
                    if norm_flag:
                        arr = arr / np.clip(arr[:, :1], 1e-6, None)
                    mu, sd = arr.mean(0), arr.std(0)
                    ax.plot(eps, mu, _style(lab), color=_color(lab), marker='o', ms=3, label=lab)
                    ax.fill_between(eps, mu-sd, mu+sd, color=_color(lab), alpha=0.1)
                ax.set(title=f'J={j}', xlabel=f'PGD {nm} eps')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            axes[0][0].set_ylabel(ylab); axes[0][-1].legend(frameon=False, fontsize=7)
            fig.suptitle(f'PGD-{nm} vs coupling J  ({ylab})'); fig.tight_layout()
            fig.savefig(os.path.join(resdir, f'robustness_{nm}{sfx}.png'), dpi=160, bbox_inches='tight')

    # (B) clean-accuracy-matching: robustness (eps50) vs clean acc, one point per J
    for nm in ('linf', 'l2'):
        eps = results['eps'][nm]
        fig, ax = plt.subplots(figsize=(6.4, 5))
        for lab in labs:
            xs, ys = [], []
            for j in js:
                if (j, lab) not in results[nm]:
                    continue
                clean = np.mean([c[0] if isinstance(c, list) else c for c in results['clean'][(j, lab)]])
                e50 = np.mean([_eps50(eps, a) for a in results[nm][(j, lab)]])
                xs.append(clean); ys.append(e50)
            if xs:
                ax.plot(xs, ys, _style(lab), color=_color(lab), marker='o', ms=6, label=lab)
        ax.set(xlabel='clean per-pixel accuracy (operating point, set by J)',
               ylabel=f'robustness  eps50  (PGD {nm})',
               title=f'Matched-operating-point view (PGD {nm})\nread a VERTICAL slice = compare at equal clean acc')
        ax.legend(frameon=False, fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(resdir, f'matched_{nm}.png'), dpi=160, bbox_inches='tight')

    # (C) a couple of natural-corruption curves per J (flip, gaussian)
    for name in ('flip', 'gaussian'):
        fig, axes = plt.subplots(1, len(js), figsize=(3.6*len(js), 3.8), sharey=True, squeeze=False)
        for ax, j in zip(axes[0], js):
            for lab in labs:
                k = (j, lab, name)
                if k not in results['nat']:
                    continue
                arr = np.array(results['nat'][k]); mu = arr.mean(0)
                ax.plot(NAT_STR, mu, _style(lab), color=_color(lab), marker='o', ms=3, label=lab)
            ax.set(title=f'J={j}', xlabel=f'{name} strength')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[0][0].set_ylabel('per-pixel acc'); axes[0][-1].legend(frameon=False, fontsize=7)
        fig.suptitle(f'Corruption {name} vs coupling J'); fig.tight_layout()
        fig.savefig(os.path.join(resdir, f'corruption_{name}.png'), dpi=160, bbox_inches='tight')
    print('plots written to', resdir)
    return results


def plot_denoise_examples(js=(0.15, 0.35, 0.7), ps=(0.1, 0.2, 0.3),
                          variants=('linear', 'mf', 'lbp', 'fbp1.5', 'gibbs20'),
                          seed=0, field_seed=7, beta=BETA, save=True):
    """Ready-to-run qualitative panel. One figure per coupling J; within it rows =
    observation flip-prob p, columns = [clean field, noisy obs, <each algorithm's
    denoised marginals>]. Each inference panel is annotated with its per-pixel
    accuracy. Sampling variants use true Gibbs. Saves results/denoise_examples_J*.png."""
    resdir = os.path.join(SAVE_ROOT, 'results'); os.makedirs(resdir, exist_ok=True)
    s = make_fields(1, seed=field_seed)                       # one clean field
    paths = []
    for j in js:
        models = {v: BinaryMRF(v, j, seed, beta=beta) for v in variants}
        for m in models.values():
            m.eval()
        cols = ['clean', 'noisy'] + list(variants)
        fig, ax = plt.subplots(len(ps), len(cols), figsize=(2.0*len(cols), 2.0*len(ps)),
                               squeeze=False)
        for r, p in enumerate(ps):
            o = flip(s, p, torch.Generator().manual_seed(100 + r))
            ax[r][0].imshow(s[0, 0], cmap='gray', vmin=0, vmax=1)
            ax[r][1].imshow(o[0, 0], cmap='gray', vmin=0, vmax=1)
            for c, v in enumerate(variants):
                smp = 'gibbs' if _is_sampling(v) else None
                with torch.no_grad():
                    mm = models[v](o, sampler=smp).view(G, G)
                acc = (((mm > 0).float() == s[0, 0]).float().mean().item())
                ax[r][2+c].imshow(mm, cmap='coolwarm', vmin=-1, vmax=1)
                ax[r][2+c].set_xlabel(f'acc {acc:.2f}', fontsize=8)
            ax[r][0].set_ylabel(f'p={p}', fontsize=11)
            if r == 0:
                for c, name in enumerate(cols):
                    ax[0][c].set_title(name, fontsize=10)
        for row in ax:
            for a in row:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f'Denoising examples  (J_sigma={j}, beta={beta}, ferro prior, seed={seed})', y=1.0)
        fig.tight_layout()
        out = os.path.join(resdir, f'denoise_examples_J{j}_beta{beta}.png')
        if save:
            fig.savefig(out, dpi=150, bbox_inches='tight'); paths.append(out)
    print('wrote:', *paths, sep='\n  ')
    return paths


@torch.no_grad()
def mse_matrix(betas=(0.25, 0.5, 1.0, 2.0), j_sigmas=J_SWEEP, p=BASE_P, n_seeds=50,
               variants=RUN_VARIANTS, field_seed=2024, save=True):
    """Per algorithm, a (beta x J_sigma) heatmap of denoising MSE, averaged over
    `n_seeds` clean-field images (one shared noisy observation per cell). MSE is
    between the output P(pixel=1)=(m+1)/2 and the clean field s. Sampling uses
    true Gibbs. Saves results/mse_matrix.png and returns {variant: matrix}."""
    S = make_fields(n_seeds, seed=field_seed)                 # [n,1,G,G] clean fields
    O = flip(S, p, torch.Generator().manual_seed(777))        # one noisy observation set
    St = S.view(n_seeds, N)
    mats = {v: np.full((len(betas), len(j_sigmas)), np.nan) for v in variants}
    for v in tqdm(variants, desc='mse_matrix'):
        smp = 'gibbs' if _is_sampling(v) else None
        for bi, beta in enumerate(betas):
            for ji, js in enumerate(j_sigmas):
                model = BinaryMRF(v, js, seed=0, beta=beta); model.eval()
                ppred = ((model(O, sampler=smp) + 1) / 2).clamp(0, 1)   # [n,N] P(pixel=1)
                mats[v][bi, ji] = ((ppred - St) ** 2).mean().item()
    vmax = max(np.nanmax(m) for m in mats.values())
    fig, axes = plt.subplots(1, len(variants), figsize=(2.6*len(variants), 3.2), squeeze=False)
    for ax, v in zip(axes[0], variants):
        M = mats[v]
        im = ax.imshow(M, cmap='viridis_r', vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(j_sigmas))); ax.set_xticklabels(j_sigmas, fontsize=8, rotation=90)
        ax.set_yticks(range(len(betas))); ax.set_yticklabels(betas, fontsize=8)
        ax.set_title(v, fontsize=10); ax.set_xlabel('J_sigma', fontsize=9)
        for bi in range(len(betas)):
            for ji in range(len(j_sigmas)):
                ax.text(ji, bi, f'{M[bi,ji]:.3f}', ha='center', va='center',
                        color='w' if M[bi, ji] > 0.5*vmax else 'k', fontsize=6)
    axes[0][0].set_ylabel('beta', fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.02, label='denoising MSE')
    fig.suptitle(f'Denoising MSE (beta x J_sigma), mean over {n_seeds} images, p={p}', y=1.02)
    if save:
        resdir = os.path.join(SAVE_ROOT, 'results'); os.makedirs(resdir, exist_ok=True)
        out = os.path.join(resdir, 'mse_matrix.png')
        fig.savefig(out, dpi=160, bbox_inches='tight'); print('saved', out)
        pickle.dump({'mats': mats, 'betas': betas, 'j_sigmas': j_sigmas, 'p': p},
                    open(os.path.join(resdir, 'mse_matrix.pkl'), 'wb'))
    return mats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--plot', action='store_true', help='re-plot from saved results')
    ap.add_argument('--examples', action='store_true', help='qualitative denoise panels')
    ap.add_argument('--mse', action='store_true', help='beta x J_sigma MSE heatmaps')
    ap.add_argument('--recompute', action='store_true')
    args, _ = ap.parse_known_args()
    if args.examples:
        plot_denoise_examples()
    elif args.mse:
        mse_matrix()
    elif args.plot:
        plot()
    else:
        res = run(fast=args.fast, re_compute=args.recompute)
        plot(res)
