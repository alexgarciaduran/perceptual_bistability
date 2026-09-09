# -*- coding: utf-8 -*-
"""
Binary-MRF denoising classifier -- SAME pipeline structure as mrf_inference_nets
(train_all / attack_all / plot_results, resumable per network, transfer + both
sampling eval modes), specialised to a load-bearing binary task:

    binary image x (GxG) --[flip noise]--> pooled evidence B = beta*(2*pool(x)-1)
        over a LOCAL pixel-lattice Ising graph
        --[inference: gibbs | mf | fbp(alpha) | lbp, shared J, ITERS sweeps]--> m
        --[linear readout]--> class logits

The image IS the evidence field (identity encoder + a fixed shared mean-pool),
so there is no CNN to bypass and inference is load-bearing by construction.
Only the inference algorithm differs across variants.

    Type A (imposed) : FERROMAGNETIC local couplings J_ij = |.| >= 0, FIXED (a
                       genuine smoothing/denoising prior; frozen, shared by all
                       algorithms). Random positive magnitudes per seed give PGM
                       diversity.  N_SEEDS_A control networks.
    Type B (learned) : signed local J_ij trained end to end.  N_SEEDS_B networks.

Robustness axes (same artefacts as the other pipeline): white-box PGD L2/Linf,
transfer (both norms), naturalistic corruptions incl. pixel-flip, J-off ablation.

All leaf functions (inference engines, PGD, corruptions, accuracy, plotting
helpers) are imported from mrf_inference_nets -- no duplication.

Run:
    python binary_mrf_denoise.py --mode train            # train all (resumable)
    python binary_mrf_denoise.py --mode attack           # attacks on trained nets
    python binary_mrf_denoise.py --mode plot             # figures
    python binary_mrf_denoise.py --mode diag             # load-bearing report
    python binary_mrf_denoise.py --mode all              # train + attack + plot
    python binary_mrf_denoise.py --mode all --fast       # tiny smoke test
"""
import argparse, json, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mrf_inference_nets import (grid_adjacency, infer_mf, infer_fbp,
                                infer_sampling, infer_gibbs_discrete, VARIANTS,
                                pgd_linf, pgd_l2, ATTACKS, CORRUPTIONS,
                                _acc, _corrupt_curve, _color, _style,
                                _eval_modes, _mode_tag)

SAVE_ROOT = r"C:\Users\alexg\OneDrive\Escritorio\phd\folder_save\robustness_analysis\binary_denoise"
DATA_DIR = r"C:\Users\alexg\OneDrive\Escritorio\phd\folder_save\robustness_analysis"

G = 28                      # GxG lattice
N = G * G                   # nodes
BETA = 1.0                  # evidence strength (bounded field); kept FIXED
ITERS = 15                  # inference sweeps (matched across variants)
N_SAMPLES = 20              # chains for sampling variants (matched)
J_SIGMA = 0.4               # coupling scale (|.| for ferro Type A; signed Type B)
EPOCHS = 20
N_SEEDS_B = 20              # learned-J networks (Type B, primary)
N_SEEDS_A = 10              # fixed ferromagnetic-J control networks (Type A)
RUN_VARIANTS = ['gibbs20', 'mf', 'fbp0.5', 'lbp', 'fbp1.5', 'fbp2.0']
FLIP_PROBS = (0.0, 0.1, 0.2, 0.3, 0.4)   # flip-noise sweep (diagnostic report)
TRAIN_FLIP = 0.25          # flip noise applied during (denoising) training

TYPES = {'typeA_ferro': False, 'typeB_learned': True}     # learn_J flag
EPS_LINF = (0, 0.05, 0.1, 0.15, 0.2, 0.3)
EPS_L2 = (0, 0.5, 1.0, 1.5, 2.0, 3.0)
NAT_STR = (0, 0.2, 0.4, 0.6, 0.8, 1.0)
NORMS = (('linf', EPS_LINF), ('l2', EPS_L2))


# --------------------------------------------------------------- data -------
def flip(x, p, gen=None):
    """Flip each pixel (0<->1) independently with probability p."""
    if p <= 0:
        return x
    m = (torch.rand(x.shape, generator=gen) < p)
    return torch.where(m, 1.0 - x, x)


def get_binary_data(fake=False, noisy=False, flip_p=TRAIN_FLIP, seed=0,
                    n_train=20000, n_test=2000):
    """MNIST -> resize GxG (area) -> binarize {0,1}, as float [N,1,G,G].

    noisy=True bakes a fixed flip(flip_p) into BOTH splits (a ready-made noisy
    dataset for quick experiments / eval); training still applies fresh on-the-
    fly flips each epoch for denoising. noisy=False returns the clean binary
    images."""
    if fake:
        gen = torch.Generator().manual_seed(seed)
        Xtr = (torch.rand(512, 1, G, G, generator=gen) > 0.5).float()
        Ytr = torch.randint(0, 10, (512,), generator=gen)
        Xte = (torch.rand(256, 1, G, G, generator=gen) > 0.5).float()
        Yte = torch.randint(0, 10, (256,), generator=gen)
    else:
        from torchvision import datasets, transforms
        tf = transforms.ToTensor()
        tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
        te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

        def pack(ds, n):
            X = torch.stack([ds[i][0] for i in range(min(n, len(ds)))])
            Y = torch.tensor([ds[i][1] for i in range(min(n, len(ds)))])
            # X = F.interpolate(X, size=(G, G), mode='area')
            return (X > 0.5).float(), Y
        Xtr, Ytr = pack(tr, n_train)
        Xte, Yte = pack(te, n_test)
    if noisy:
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        Xtr, Xte = flip(Xtr, flip_p, g1), flip(Xte, flip_p, g2)
    return Xtr, Ytr, Xte, Yte


def corrupt_flip(x, s, gen):                 # pixel-flip as a corruption family
    return flip(x, s, gen)


CORR = {**CORRUPTIONS, 'flip': corrupt_flip}


# ----------------------------------------------------- sparse fractional BP -
def infer_fbp_sparse(B, Jmat, alpha=1.0, iters=15, damping=0.5):
    """Fractional BP with messages stored ONLY on the graph's directed edges.
    Mathematically identical to mrf_inference_nets.infer_fbp but O(E) memory
    and compute instead of O(n^2) -- the grid is ~99.5% non-edges at 28x28.
    Differentiable in B and Jmat (training / white-box PGD)."""
    bsz, n = B.shape
    idx = (Jmat != 0).nonzero(as_tuple=False)
    src, tgt = idx[:, 0], idx[:, 1]                      # directed edge i->j
    key = src * n + tgt
    order = torch.argsort(key)
    rev = order[torch.searchsorted(key[order], tgt * n + src)]   # (j->i) index
    Jt = torch.tanh(alpha * Jmat[src, tgt])              # [E]
    M = torch.zeros(bsz, src.shape[0])
    for _ in range(iters):
        Q = B.index_add(1, tgt, M)                       # Q_j = B_j + sum_i m_{i->j}
        h = Q[:, src] - alpha * M[:, rev]                # cavity field for i->j
        arg = (Jt * torch.tanh(h)).clamp(-0.999, 0.999)
        newM = (1.0 / alpha) * torch.atanh(arg)
        M = damping * newM + (1 - damping) * M
    return torch.tanh(B.index_add(1, tgt, M))


# --------------------------------------------------------------- model ------
class BinaryMRF(nn.Module):
    """Identity encoder over a LOCAL pixel-lattice Ising graph (B = beta*(2x-1)).
    learn_J=False -> Type A: frozen FERROMAGNETIC (positive) couplings, a real
    smoothing prior. learn_J=True -> Type B: trained signed couplings."""
    def __init__(self, variant, learn_J=True, seed=0, beta=BETA, iters=ITERS):
        super().__init__()
        self.variant = variant
        self.algo, param = VARIANTS[variant]
        self.alpha = float(param) if self.algo == 'fbp' else 1.0
        self.beta, self.iters = beta, iters
        A = grid_adjacency(G)                       # 4-neighbour LOCAL lattice
        self.register_buffer('mask', A)
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(N, N, generator=gen) * J_SIGMA
        if learn_J:
            J0 = A * (W + W.t()) / 2                 # signed, symmetric, on-graph
            self.Jraw = nn.Parameter(J0)
        else:
            J0 = A * (W.abs() + W.abs().t()) / 2     # FERROMAGNETIC (>=0), frozen
            self.register_buffer('Jraw', J0)
        self.readout = nn.Linear(N, 10)

    def Jmat(self):
        J = self.Jraw * self.mask
        return (J + J.t()) / 2

    def _infer(self, B, Jm, sampler):
        if self.algo == 'mf':
            return infer_mf(B, Jm, self.iters)
        if self.algo == 'sampling':
            if sampler == 'gibbs':
                return infer_gibbs_discrete(B, Jm, iters=self.iters, n_samples=N_SAMPLES)
            return infer_sampling(B, Jm, iters=self.iters, n_samples=N_SAMPLES)
        return infer_fbp_sparse(B, Jm, alpha=self.alpha, iters=self.iters)

    def forward(self, x, sampler=None):
        B = self.beta * (2 * x.view(x.shape[0], -1) - 1)   # evidence field
        m = self._infer(B, self.Jmat(), sampler)
        return self.readout(m)


def make_model(variant, type_, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    return BinaryMRF(variant, learn_J=TYPES[type_], seed=seed)


# --------------------------------------------------------------- train/eval -
@torch.no_grad()
def acc_at_flip(model, X, Y, p, sampler=None, bs=500, seed=0):
    model.eval(); correct = 0
    for i in range(0, len(X), bs):
        gen = torch.Generator().manual_seed(seed + i)
        xb = flip(X[i:i+bs], p, gen)
        correct += (model(xb, sampler=sampler).argmax(1) == Y[i:i+bs]).sum().item()
    return correct / len(X)


def train_model(model, data, epochs=EPOCHS, lr=1e-3, bs=128):
    """Denoising training: fresh flip(TRAIN_FLIP) each batch. Returns
    (final_acc, best_acc, history) mirroring mrf_inference_nets.train_model."""
    Xtr, Ytr, Xte, Yte = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = acc = 0.0
    hist = {'step_loss': [], 'epoch_loss': [], 'epoch_acc': []}
    ebar = tqdm(range(epochs), desc='  epochs', leave=False)
    for ep in ebar:
        model.train(); perm = torch.randperm(len(Xtr)); elosses = []
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            xb = flip(Xtr[idx], TRAIN_FLIP)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), Ytr[idx])
            loss.backward(); opt.step()
            hist['step_loss'].append(loss.item()); elosses.append(loss.item())
        acc = acc_at_flip(model, Xte, Yte, TRAIN_FLIP)      # test acc @ train noise
        best = max(best, acc)
        hist['epoch_loss'].append(float(np.mean(elosses))); hist['epoch_acc'].append(acc)
        ebar.set_postfix(loss=f"{hist['epoch_loss'][-1]:.3f}", acc=f'{acc:.3f}')
        tqdm.write(f"    epoch {ep+1}/{epochs}: loss={hist['epoch_loss'][-1]:.4f} "
                   f"test_acc@p={TRAIN_FLIP}={acc:.4f}")
    return acc, best, hist


# --------------------------------------------------------------- orchestrate
def _paths(type_, variant, seed):
    d = os.path.join(SAVE_ROOT, type_, variant, f'seed{seed}')
    return d, os.path.join(d, 'model.pt'), os.path.join(d, 'meta.json')


def _seeds(type_, fast):
    if fast:
        return [0]
    return list(range(N_SEEDS_B if type_ == 'typeB_learned' else N_SEEDS_A))


def train_all(variants, types, data, epochs=EPOCHS, fast=False):
    for type_ in types:
        for variant in variants:
            for s in _seeds(type_, fast):
                d, mp, meta = _paths(type_, variant, s)
                if os.path.exists(meta):
                    print(f"skip {type_}/{variant}/seed{s} (done)"); continue
                os.makedirs(d, exist_ok=True)
                model = make_model(variant, type_, s)
                print(f"training {type_}/{variant}/seed{s} ...")
                acc, best, hist = train_model(model, data, epochs=(1 if fast else epochs))
                torch.save(model.state_dict(), mp)
                info = {'type': type_, 'variant': variant, 'seed': s,
                        'test_acc': acc, 'best_acc': best, 'epochs': epochs,
                        'epoch_acc': hist['epoch_acc'], 'epoch_loss': hist['epoch_loss']}
                gmsg = ''
                if VARIANTS[variant][0] == 'sampling':
                    acc_g = acc_at_flip(model, data[2], data[3], TRAIN_FLIP, sampler='gibbs')
                    info['test_acc_relaxed'] = acc; info['test_acc_gibbs'] = acc_g
                    gmsg = f"  [relaxed={acc:.3f} gibbs={acc_g:.3f}]"
                json.dump(info, open(meta, 'w'), indent=2)
                json.dump(hist, open(os.path.join(d, 'history.json'), 'w'))
                print(f"{type_}/{variant}/seed{s}: acc={acc:.3f} (best {best:.3f}){gmsg}")


def attack_all(variants, types, data, n_imgs=50, steps=20, fast=False,
               re_compute=False):
    """All perturbation families, resumable PER NETWORK. Same schema as
    mrf_inference_nets.attack_all: per (type, seed) the trained variants are
    loaded together so TRANSFER reuses one source-crafted PGD set (both norms);
    adversarials crafted on the differentiable forward, each net evaluated under
    its eval mode(s) (sampling: relaxed + true Gibbs; else det). Adds the pixel-
    flip corruption and a clean J-off ablation. -> each net's robustness.json;
    aggregated to results/robustness.pkl."""
    _, _, Xte, Yte = data
    X, Y = Xte[:n_imgs], Yte[:n_imgs]
    results = {'linf': {}, 'l2': {}, 'nat': {}, 'transfer': {},
               'eps': {'linf': EPS_LINF, 'l2': EPS_L2}, 'nat_strengths': NAT_STR}
    for type_ in types:
        for s in _seeds(type_, fast):
            avail = {v: _paths(type_, v, s)[1] for v in variants
                     if os.path.exists(_paths(type_, v, s)[1])}
            if not avail:
                continue
            todo = [v for v in avail
                    if re_compute or not os.path.exists(
                        os.path.join(_paths(type_, v, s)[0], 'robustness.json'))]
            models, adv = {}, {}
            if todo:                                    # craft PGD sets once per source
                for v, mp in avail.items():
                    m = make_model(v, type_, s)
                    m.load_state_dict(torch.load(mp)); m.eval(); models[v] = m
                for v, m in models.items():
                    adv[v] = {nm: {e: (ATTACKS[nm](m, X, Y, e, steps=steps) if e else X)
                                   for e in epl} for nm, epl in NORMS}
            for v in avail:
                d, mp, meta = _paths(type_, v, s)
                rp = os.path.join(d, 'robustness.json')
                if os.path.exists(rp) and not re_compute:
                    rec = json.load(open(rp))
                else:
                    model = models[v]
                    rec = {'modes': _eval_modes(v)}
                    for mode in rec['modes']:
                        tag = _mode_tag(mode); r = {}
                        for nm, epl in NORMS:
                            r[nm] = {'eps': list(epl),
                                     'acc': [_acc(model, adv[v][nm][e], Y, tag) for e in epl]}
                        r['transfer'] = {src: {nm: {'eps': list(epl),
                            'acc': [_acc(model, adv[src][nm][e], Y, tag) for e in epl]}
                            for nm, epl in NORMS} for src in models}
                        r['nat'] = {name: {'strength': list(NAT_STR),
                            'acc': _corrupt_curve(model, X, Y, fn, NAT_STR, tag)}
                            for name, fn in CORR.items()}
                        rec[mode] = r
                    # clean J-off ablation (default forward)
                    Jsave = model.Jraw.detach().clone()
                    rec['acc_clean'] = _acc(model, X, Y, 'relaxed')
                    with torch.no_grad(): model.Jraw.zero_()
                    rec['acc_J0'] = _acc(model, X, Y, 'relaxed')
                    with torch.no_grad(): model.Jraw.copy_(Jsave)
                    rec['J_norm'] = float(model.Jraw.detach().abs().sum())
                    json.dump(rec, open(rp, 'w'), indent=2)
                    print(f"  {type_}/{v}/seed{s}: clean={rec['acc_clean']:.3f} "
                          f"J0={rec['acc_J0']:.3f} |J|={rec['J_norm']:.1f}")
                    try:
                        from torchvision.utils import save_image
                        for nm, epl in NORMS:
                            xa = adv[v][nm][epl[-1]]
                            torch.save(xa, os.path.join(d, f'adv_{nm}.pt'))
                            save_image(xa, os.path.join(d, f'adv_{nm}.png'), nrow=10)
                    except Exception as ex:
                        print('adv image save skipped:', ex)
                for mode in rec['modes']:
                    lab = v if mode == 'det' else f'{v}::{mode}'
                    results['linf'].setdefault((type_, lab), []).append(rec[mode]['linf']['acc'])
                    results['l2'].setdefault((type_, lab), []).append(rec[mode]['l2']['acc'])
                    for name in CORR:
                        results['nat'].setdefault((type_, lab, name), []).append(rec[mode]['nat'][name]['acc'])
                    for nm, _ in NORMS:
                        for src in rec[mode]['transfer']:
                            results['transfer'].setdefault((type_, nm, lab, src), []).append(
                                rec[mode]['transfer'][src][nm]['acc'])
    os.makedirs(os.path.join(SAVE_ROOT, 'results'), exist_ok=True)
    pickle.dump(results, open(os.path.join(SAVE_ROOT, 'results', 'robustness.pkl'), 'wb'))
    return results


def plot_results(results=None, transfer_summary='max'):
    """PGD-L2/Linf + per-corruption (incl flip) + transfer heatmaps (summary +
    per-epsilon grid), Type A (imposed ferro) vs Type B (learned). Mirrors
    mrf_inference_nets.plot_results; reuses its _color/_style."""
    import matplotlib.pyplot as plt
    if results is None:
        results = pickle.load(open(os.path.join(SAVE_ROOT, 'results', 'robustness.pkl'), 'rb'))
    resdir = os.path.join(SAVE_ROOT, 'results'); os.makedirs(resdir, exist_ok=True)
    types = list(TYPES)

    def curves(store, xs, key3=None):
        fig, axes = plt.subplots(1, len(types), figsize=(6*len(types), 4.6), sharey=True)
        for ax, t in zip(axes, types):
            labs = sorted({k[1] for k in store if k[0] == t and (key3 is None or k[-1] == key3)})
            for lab in labs:
                k = (t, lab) if key3 is None else (t, lab, key3)
                arr = np.array(store[k]); mu, sd = arr.mean(0), arr.std(0)
                ax.plot(xs, mu, _style(lab), marker='o', ms=3, color=_color(lab), label=lab)
                ax.fill_between(xs, mu - sd, mu + sd, color=_color(lab), alpha=0.12)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.set_title(t.replace('_', ' '))
        axes[0].set_ylabel('accuracy'); axes[0].legend(frameon=False, fontsize=8)
        return fig, axes

    for nm in ('linf', 'l2'):
        fig, axes = curves(results[nm], results['eps'][nm])
        for ax in axes:
            ax.set_xlabel(f'PGD {nm} eps')
        fig.suptitle(f'PGD-{nm}: imposed(ferro) vs learned J'); fig.tight_layout()
        fig.savefig(os.path.join(resdir, f'robustness_{nm}.png'), dpi=180, bbox_inches='tight')

    for name in sorted({k[2] for k in results['nat']}):
        fig, axes = curves(results['nat'], NAT_STR, key3=name)
        for ax in axes:
            ax.set_xlabel(f'{name} strength')
        fig.suptitle(f'Corruption {name}: imposed(ferro) vs learned J'); fig.tight_layout()
        fig.savefig(os.path.join(resdir, f'corruption_{name}.png'), dpi=180, bbox_inches='tight')

    tr = results.get('transfer', {})

    def _matrix(type_, nm, reduce_):
        ks = [k for k in tr if k[0] == type_ and k[1] == nm]
        if not ks:
            return None, None, None
        tgts = sorted({k[2] for k in ks}); srcs = sorted({k[3] for k in ks})
        M = np.full((len(tgts), len(srcs)), np.nan)
        for i, tg in enumerate(tgts):
            for j, sc in enumerate(srcs):
                k = (type_, nm, tg, sc)
                if k in tr:
                    curve = np.array(tr[k]).mean(0)
                    M[i, j] = curve.mean() if reduce_ == 'auc' else curve[reduce_]
        return M, tgts, srcs

    def _draw(ax, M, tgts, srcs, title):
        im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(srcs))); ax.set_xticklabels(srcs, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tgts))); ax.set_yticklabels(tgts, fontsize=6)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                            color='w' if M[i, j] < 0.5 else 'k', fontsize=5)
        ax.set_title(title, fontsize=9); return im

    for nm in ('linf', 'l2'):
        eps = results['eps'][nm]
        for t in types:
            red = len(eps) - 1 if transfer_summary == 'max' else transfer_summary
            M, tgts, srcs = _matrix(t, nm, red)
            if M is None:
                continue
            fig, ax = plt.subplots(figsize=(1.4 + 0.7*len(srcs), 1.4 + 0.6*len(tgts)))
            tag = 'AUC' if red == 'auc' else f'eps={eps[red]}'
            im = _draw(ax, M, tgts, srcs, f'Transfer {nm} ({tag}): {t.replace("_"," ")}')
            ax.set_xlabel('source (crafted on)'); ax.set_ylabel('target (evaluated)')
            fig.colorbar(im, ax=ax, fraction=0.046, label='accuracy'); fig.tight_layout()
            fig.savefig(os.path.join(resdir, f'transfer_{t}_{nm}.png'), dpi=180, bbox_inches='tight')
            idxs = [k for k in range(len(eps)) if eps[k] != 0]
            fig, axes = plt.subplots(1, len(idxs), figsize=(3.2*len(idxs), 3.0), squeeze=False)
            for ax, k in zip(axes[0], idxs):
                Mk, tg2, sc2 = _matrix(t, nm, k)
                _draw(ax, Mk, tg2, sc2, f'{nm} eps={eps[k]}')
            fig.suptitle(f'Transfer vs epsilon: {t.replace("_"," ")} ({nm})'); fig.tight_layout()
            fig.savefig(os.path.join(resdir, f'transfer_{t}_{nm}_byeps.png'), dpi=170, bbox_inches='tight')
    return results


def diagnose(n_eval=1000, seeds_max=3, sampler='relaxed', save=True):
    """Load-bearing report: accuracy vs flip probability with J ON vs J OFF
    (ablated), per trained net. If the prior J denoises, acc(p) sits well ABOVE
    acc_J0(p) at high p; the gap acc-acc_J0 is the graphical model's contribution.
    Reads trained model.pt (no retraining); plots to results/."""
    import matplotlib.pyplot as plt
    _, _, Xte, Yte = get_binary_data()
    Xte, Yte = Xte[:n_eval], Yte[:n_eval]
    ps = list(FLIP_PROBS); out = {}
    for type_ in TYPES:
        for v in RUN_VARIANTS:
            accs, acc0s, nseed = np.zeros(len(ps)), np.zeros(len(ps)), 0
            for s in range(seeds_max):
                mp = _paths(type_, v, s)[1]
                if not os.path.exists(mp):
                    continue
                model = make_model(v, type_, s)
                model.load_state_dict(torch.load(mp)); model.eval()
                smp = sampler if VARIANTS[v][0] == 'sampling' else None
                a = [acc_at_flip(model, Xte, Yte, p, sampler=smp) for p in ps]
                Jsave = model.Jraw.detach().clone()
                with torch.no_grad(): model.Jraw.zero_()
                a0 = [acc_at_flip(model, Xte, Yte, p, sampler=smp) for p in ps]
                with torch.no_grad(): model.Jraw.copy_(Jsave)
                accs += np.array(a); acc0s += np.array(a0); nseed += 1
            if nseed:
                out[(type_, v)] = (accs / nseed, acc0s / nseed, nseed)
                print(f'{type_}/{v} (n={nseed}): '
                      + ' '.join(f'p{p}:{accs[i]/nseed:.2f}/{acc0s[i]/nseed:.2f}'
                                 for i, p in enumerate(ps)))
    types = [t for t in TYPES if any(k[0] == t for k in out)]
    if not types:
        print('no trained nets found for diagnose()'); return out
    fig, axes = plt.subplots(2, len(types), figsize=(6*len(types), 8.5), squeeze=False)
    for j, t in enumerate(types):
        for v in RUN_VARIANTS:
            if (t, v) not in out:
                continue
            a, a0, _ = out[(t, v)]
            axes[0][j].plot(ps, a, '-o', ms=4, color=_color(v), label=v)
            axes[0][j].plot(ps, a0, ':', color=_color(v), alpha=0.6)
            axes[1][j].plot(ps, a - a0, '-o', ms=4, color=_color(v), label=v)
        axes[0][j].set(title=t.replace('_', ' '), xlabel='flip prob p', ylabel='accuracy')
        axes[1][j].set(xlabel='flip prob p', ylabel='acc(J) - acc(J=0)')
        axes[1][j].axhline(0, color='0.7', lw=1)
        for ax in (axes[0][j], axes[1][j]):
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[0][j].legend(frameon=False, fontsize=8)
    axes[0][0].plot([], [], 'k-', label='J on'); axes[0][0].plot([], [], 'k:', label='J=0')
    fig.suptitle('Load-bearing report: accuracy & J-ablation gap vs flip noise')
    fig.tight_layout()
    if save:
        resdir = os.path.join(SAVE_ROOT, 'results'); os.makedirs(resdir, exist_ok=True)
        fig.savefig(os.path.join(resdir, 'denoising_diagnostic.png'), dpi=180, bbox_inches='tight')
        pickle.dump(out, open(os.path.join(resdir, 'denoising_diag.pkl'), 'wb'))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'attack', 'plot', 'diag', 'all'], default='all')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--variants', nargs='*', default=list(RUN_VARIANTS))
    ap.add_argument('--types', nargs='*', default=list(TYPES))
    args, _ = ap.parse_known_args()             # robust to Spyder's injected argv

    variants = ['mf', 'lbp', 'gibbs20'] if args.fast else args.variants
    types = args.types
    data = get_binary_data(fake=args.fast, n_train=(512 if args.fast else 20000))
    if args.mode == 'diag':
        diagnose()
    else:
        if args.mode in ('train', 'all'):
            train_all(variants, types, data, epochs=args.epochs, fast=args.fast)
        if args.mode in ('attack', 'all'):
            attack_all(variants, types, data,
                       n_imgs=(8 if args.fast else 50), fast=args.fast)
        if args.mode in ('plot', 'all'):
            plot_results()
