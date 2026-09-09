# -*- coding: utf-8 -*-
"""
Robustness benchmark pipeline: image classifiers that differ ONLY in the
approximate-inference algorithm applied over a latent Ising graph.

    image --[shared CNN encoder]--> evidence B over an n-node lattice MRF
          --[inference: Gibbs | MF | FBP(alpha) | LBP]--> marginals m
          --[linear readout]--> class logits

Six inference variants: gibbs (sampling), mf, fbp0.5, lbp (=FBP alpha=1),
fbp1.5, fbp2.0. Two graph types:
    Type A (imposed) : per-edge couplings J_ij fixed (random per seed).
    Type B (learned) : J_ij trained (random init per seed).
Default: 20 seeds x 6 variants x 2 types = 240 networks, each trained to
>= target test accuracy. Robustness by PGD in L2 and Linf.

Artifacts saved under SAVE_ROOT:
    {typeA_imposed|typeB_learned}/{variant}/seed{k}/model.pt + meta.json
    results/robustness_{l2|linf}.pkl , results/*.png

Usage:
    python mrf_inference_nets.py --mode train           # train all (resumable)
    python mrf_inference_nets.py --mode attack          # PGD on trained models
    python mrf_inference_nets.py --mode plot            # figures
    python mrf_inference_nets.py --mode all
    python mrf_inference_nets.py --mode all --fast      # tiny smoke test
"""
import argparse
import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SAVE_ROOT = r"C:\Users\alexg\OneDrive\Escritorio\phd\folder_save\robustness_analysis"
DATA_DIR = SAVE_ROOT               # MNIST lives here (torchvision makes DATA_DIR/MNIST)
# variant -> (inference algo, param): param = sampling iters for gibbs, alpha for fbp
VARIANTS = {'gibbs10': ('sampling', 10), 'gibbs20': ('sampling', 20),
            'gibbs30': ('sampling', 30), 'mf': ('mf', None),
            'fbp0.5': ('fbp', 0.5), 'lbp': ('fbp', 1.0),
            'fbp1.5': ('fbp', 1.5), 'fbp2.0': ('fbp', 2.0)}
TYPES = {'typeA_imposed': False, 'typeB_learned': True}   # learn_J flag

# ---- defaults for a plain run (Spyder F5, no CLI args) ----
LEAN = True
EPOCHS = 10
N_SEEDS = 10
RUN_VARIANTS = ['gibbs20', 'mf', 'fbp0.5', 'lbp', 'fbp1.5', 'fbp2.0']  # gibbs20 only from the gibbs family

# torch.set_num_threads(16)

# --------------------------------------------------------------- graph ------
def grid_adjacency(g):
    n = g * g
    A = torch.zeros(n, n)
    for r in range(g):
        for c in range(g):
            i = r * g + c
            if c + 1 < g:
                A[i, i + 1] = A[i + 1, i] = 1.0
            if r + 1 < g:
                A[i, i + g] = A[i + g, i] = 1.0
    return A


# --------------------------------------------------------------- encoder ----
class Encoder(nn.Module):
    """Maps image -> evidence B over the n latents.
    lean=False: high-capacity CNN (encoder dominates, MRF ~ pass-through).
    lean=True : tiny conv stack pooled to a g x g evidence map, so each latent
                is a local patch and the classifier must rely on the MRF."""
    def __init__(self, n_latent, g, in_ch=1, lean=False):
        super().__init__()
        self.lean = lean
        if lean:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.AdaptiveAvgPool2d(g), nn.Flatten())     # -> g*g evidence
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(64 * 49, 256), nn.ReLU(),
                nn.Linear(256, n_latent))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------ inference -----
def infer_mf(B, Jmat, iters=10):
    m = torch.zeros_like(B)
    for _ in range(iters):
        m = torch.tanh(m @ Jmat + B)
    return m


def infer_fbp(B, Jmat, alpha=1.0, iters=10, damping=0.5):
    bsz, n = B.shape
    mask = (Jmat != 0).float()
    M = torch.zeros(bsz, n, n)
    for _ in range(iters):
        Q = B + (M * mask).sum(1)                       # Q_j = B_j + sum_i M_ij
        h = Q.unsqueeze(2) - alpha * M.transpose(1, 2)  # cavity i->j
        arg = torch.tanh(alpha * Jmat).unsqueeze(0) * torch.tanh(h)
        newM = (1.0 / alpha) * torch.atanh(arg.clamp(-0.999, 0.999)) * mask
        M = damping * newM + (1 - damping) * M
    return torch.tanh(B + (M * mask).sum(1))


def infer_sampling(B, Jmat, iters=12, n_samples=20, noise=0.6):
    """RELAXED sampling: differentiable Gumbel-sigmoid (Concrete) surrogate,
    parallel synchronous sweeps. Used for training and white-box gradients."""
    bsz, n = B.shape
    x = torch.zeros(bsz, n_samples, n)
    for _ in range(iters):
        field = x @ Jmat + B.unsqueeze(1)
        p = torch.sigmoid(2 * field).clamp(1e-6, 1 - 1e-6)
        u = torch.rand_like(p).clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p) - torch.log(1 - p) + noise * (torch.log(u) - torch.log(1 - u))
        x = torch.tanh(logit / 2)
    return x.mean(1)


@torch.no_grad()
def infer_gibbs_discrete(B, Jmat, iters=20, n_samples=20, burn=None):
    """TRUE discrete Gibbs: sequential hard +/-1 resampling of each node from its
    conditional; marginals = spin mean over post-burn sweeps and chains. Not
    differentiable -- for evaluation only."""
    bsz, n = B.shape
    burn = iters // 2 if burn is None else burn
    x = torch.where(torch.rand(bsz, n_samples, n) < 0.5, 1.0, -1.0)
    acc = torch.zeros(bsz, n_samples, n); cnt = 0
    for t in range(iters):
        for i in range(n):
            field = torch.einsum('bsn,n->bs', x, Jmat[:, i]) + B[:, i:i+1]
            p = torch.sigmoid(2 * field)
            x[:, :, i] = torch.where(torch.rand(bsz, n_samples) < p, 1.0, -1.0)
        if t >= burn:
            acc += x; cnt += 1
    return (acc / cnt).mean(1)


class MRFClassifier(nn.Module):
    def __init__(self, variant, g=7, n_classes=10, learn_J=False,
                 J_sigma=0.15, iters=10, seed=0, lean=False):
        super().__init__()
        self.variant = variant
        self.algo, param = VARIANTS[variant]
        self.alpha = float(param) if self.algo == 'fbp' else 1.0
        self.samp_iters = int(param) if self.algo == 'sampling' else 12
        self.iters = iters
        self.n = g * g
        A = grid_adjacency(g)
        self.register_buffer('mask', A)
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(self.n, self.n, generator=gen) * J_sigma
        J0 = A * (W + W.t()) / 2                          # symmetric, on-graph
        if learn_J:
            self.Jraw = nn.Parameter(J0)
        else:
            self.register_buffer('Jraw', J0)
        self.encoder = Encoder(self.n, g, lean=lean)
        self.readout = nn.Linear(self.n, n_classes)

    def Jmat(self):
        J = self.Jraw * self.mask
        return (J + J.t()) / 2                             # keep symmetric

    def marginals(self, x, sampler=None):
        """sampler only affects the 'sampling' variant: None/'relaxed' -> the
        differentiable Gumbel surrogate (default, used for training + PGD craft);
        'gibbs' -> true discrete Gibbs (eval only, non-differentiable)."""
        B = self.encoder(x)
        Jm = self.Jmat()
        if self.algo == 'mf':
            return infer_mf(B, Jm, self.iters)
        if self.algo == 'sampling':
            if sampler == 'gibbs':
                return infer_gibbs_discrete(B, Jm, iters=self.samp_iters)
            return infer_sampling(B, Jm, iters=self.samp_iters)
        return infer_fbp(B, Jm, alpha=self.alpha, iters=self.iters)

    def forward(self, x, sampler=None):
        return self.readout(self.marginals(x, sampler=sampler))


# --------------------------------------------------------------- attacks ----
def pgd_linf(model, x, y, eps, steps=20):
    if eps == 0:
        return x
    a = 2.5 * eps / steps
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = (xa + a * g.sign()).clamp(x - eps, x + eps).clamp(0, 1).detach()
    return xa


def pgd_l2(model, x, y, eps, steps=20):
    if eps == 0:
        return x
    a = 2.5 * eps / steps
    b = x.shape[0]
    xa = x.clone().detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        gn = g / (g.view(b, -1).norm(dim=1).view(b, 1, 1, 1) + 1e-12)
        xa = xa + a * gn
        delta = (xa - x).view(b, -1)
        dn = delta.norm(dim=1).clamp(min=1e-12)
        factor = (eps / dn).clamp(max=1.0)
        xa = (x + (delta * factor.view(b, 1)).view_as(x)).clamp(0, 1).detach()
    return xa


ATTACKS = {'linf': pgd_linf, 'l2': pgd_l2}


# ------------------------------------------------- naturalistic corruptions -
# Each fn(x, s, gen) maps images in [0,1] and strength s in [0,1] -> corrupted
# images in [0,1]. Same paradigm as the attacks: sweep s. Stochastic ones take
# a torch.Generator so the SAME corrupted images are used for every model.
def _gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    xs = torch.arange(-r, r + 1).float()
    k = torch.exp(-xs ** 2 / (2 * sigma ** 2)); k /= k.sum()
    return k, r


def corrupt_gaussian(x, s, gen):                 # white noise
    return (x + s * 0.6 * torch.randn(x.shape, generator=gen)).clamp(0, 1)


def corrupt_shot(x, s, gen):                     # Poisson / photon noise
    lam = max(1.0, (1 - s) * 60.0 + 1.0)
    return (torch.poisson(x * lam, generator=gen) / lam).clamp(0, 1)


def corrupt_impulse(x, s, gen):                  # salt & pepper
    m = torch.rand(x.shape, generator=gen)
    out = x.clone()
    out[m < s / 2] = 0.0
    out[m > 1 - s / 2] = 1.0
    return out


def corrupt_fog(x, s, gen):                      # low-frequency haze overlay
    low = torch.rand((x.shape[0], 1, 7, 7), generator=gen)
    fog = F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)
    return (x * (1 - 0.8 * s) + 0.8 * s * fog).clamp(0, 1)


def corrupt_blur(x, s, gen=None):                # defocus / gaussian blur
    if s <= 0:
        return x
    sigma = 0.3 + 2.5 * s
    k, r = _gauss_kernel(sigma)
    xp = F.pad(x, (r, r, r, r), mode='reflect')
    xb = F.conv2d(xp, k.view(1, 1, 1, -1))
    xb = F.conv2d(xb, k.view(1, 1, -1, 1))
    return xb.clamp(0, 1)


def corrupt_contrast(x, s, gen=None):            # contrast reduction
    return ((x - 0.5) * (1 - 0.9 * s) + 0.5).clamp(0, 1)


CORRUPTIONS = {'gaussian': corrupt_gaussian, 'shot': corrupt_shot,
               'impulse': corrupt_impulse, 'fog': corrupt_fog,
               'blur': corrupt_blur, 'contrast': corrupt_contrast}


def _acc(model, x, y, mode='relaxed'):
    """Accuracy under an eval mode: 'gibbs' uses true discrete Gibbs, anything
    else ('relaxed'/'det') uses the model's default differentiable forward."""
    with torch.no_grad():
        out = model(x, sampler=('gibbs' if mode == 'gibbs' else None))
        return (out.argmax(1) == y).float().mean().item()


def _corrupt_curve(model, X, Y, fn, strengths, mode='relaxed', seed=0):
    curve = []
    for s in strengths:
        gen = torch.Generator().manual_seed(seed)      # same noise for every model
        xc = fn(X, s, gen) if s > 0 else X
        curve.append(_acc(model, xc, Y, mode))
    return curve


# --------------------------------------------------------------- data -------
def get_data(fake=False, n_train=20000, n_test=2000):
    if fake:
        return (torch.rand(512, 1, 28, 28), torch.randint(0, 10, (512,)),
                torch.rand(256, 1, 28, 28), torch.randint(0, 10, (256,)))
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(min(n_train, len(tr)))])
    Ytr = torch.tensor([tr[i][1] for i in range(min(n_train, len(tr)))])
    Xte = torch.stack([te[i][0] for i in range(min(n_test, len(te)))])
    Yte = torch.tensor([te[i][1] for i in range(min(n_test, len(te)))])
    return Xtr, Ytr, Xte, Yte


@torch.no_grad()
def test_acc(model, Xte, Yte, bs=500):
    model.eval(); correct = 0
    for i in range(0, len(Xte), bs):
        correct += (model(Xte[i:i+bs]).argmax(1) == Yte[i:i+bs]).sum().item()
    return correct / len(Xte)


def train_model(model, data, epochs=30, lr=1e-3, bs=128):
    """Train for a FIXED number of epochs. Returns (final acc, best acc, history)
    where history logs per-step training loss and per-epoch train loss + test acc."""
    Xtr, Ytr, Xte, Yte = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = acc = 0.0
    hist = {'step_loss': [], 'epoch_loss': [], 'epoch_acc': []}
    ebar = tqdm(range(epochs), desc='  epochs', leave=False)
    for ep in ebar:
        model.train()
        perm = torch.randperm(len(Xtr))
        elosses = []
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
            hist['step_loss'].append(float(loss)); elosses.append(float(loss))
        acc = test_acc(model, Xte, Yte)
        best = max(best, acc)
        hist['epoch_loss'].append(float(np.mean(elosses)))
        hist['epoch_acc'].append(acc)
        ebar.set_postfix(loss=f"{hist['epoch_loss'][-1]:.3f}", acc=f'{acc:.3f}')
        tqdm.write(f"    epoch {ep+1}/{epochs}: loss={hist['epoch_loss'][-1]:.4f}  test_acc={acc:.4f}")
    return acc, best, hist


# --------------------------------------------------------------- orchestrate
def _paths(type_, variant, seed, lean=False):
    tag = 'lean' if lean else 'full'
    d = os.path.join(SAVE_ROOT, type_, variant, f'seed{seed}_{tag}')
    return d, os.path.join(d, 'model.pt'), os.path.join(d, 'meta.json')


def train_all(seeds, variants, types, data, epochs=30, g=7, lean=False):
    for type_ in types:
        for variant in variants:
            for s in seeds:
                d, mp, meta = _paths(type_, variant, s, lean)
                if os.path.exists(meta):
                    print(f"skip {type_}/{variant}/seed{s} (done)"); continue
                os.makedirs(d, exist_ok=True)
                torch.manual_seed(s); np.random.seed(s)
                model = MRFClassifier(variant, g=g, learn_J=TYPES[type_], seed=s, lean=lean)
                print(f"training {type_}/{variant}/seed{s} ...")
                acc, best, hist = train_model(model, data, epochs=epochs)
                torch.save(model.state_dict(), mp)
                json.dump({'type': type_, 'variant': variant, 'seed': s,
                           'test_acc': acc, 'best_acc': best, 'epochs': epochs,
                           'epoch_acc': hist['epoch_acc'], 'epoch_loss': hist['epoch_loss']},
                          open(meta, 'w'), indent=2)
                json.dump(hist, open(os.path.join(d, 'history.json'), 'w'))  # full curves incl per-step loss
                print(f"{type_}/{variant}/seed{s}: acc={acc:.3f} (best {best:.3f}, {epochs} ep)")


def attack_all(seeds, variants, types, data, g=7, n_imgs=50, lean=False, steps=20,
               eps_linf=(0, 0.05, 0.1, 0.15, 0.2, 0.3),
               eps_l2=(0, 0.5, 1.0, 1.5, 2.0, 3.0),
               eps_transfer=(0, 0.1, 0.2, 0.3),
               nat_strengths=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """All perturbation families, resumable PER NETWORK. For each (type, seed)
    the trained variants are loaded together so TRANSFER attacks can reuse one
    source-crafted adversarial set. Each net's full record is saved to its own
    robustness.json (skipped if present):
        rec['linf'|'l2']  : white-box PGD accuracy vs epsilon
        rec['transfer']   : {source_variant: acc vs eps} on Linf adv from others
        rec['nat']        : {corruption: acc vs strength} (gaussian/shot/... )
    Per-net files are aggregated into results/robustness.pkl for plotting."""
    _, _, Xte, Yte = data
    X, Y = Xte[:n_imgs], Yte[:n_imgs]
    results = {'linf': {}, 'l2': {}, 'nat': {},
               'eps': {'linf': eps_linf, 'l2': eps_l2, 'transfer': eps_transfer},
               'nat_strengths': nat_strengths}
    for type_ in types:
        for s in seeds:
            avail = {v: _paths(type_, v, s, lean)[1] for v in variants
                     if os.path.exists(_paths(type_, v, s, lean)[1])}
            if not avail:
                continue
            todo = [v for v in avail
                    if not os.path.exists(os.path.join(_paths(type_, v, s, lean)[0], 'robustness.json'))]
            models, adv = {}, {}
            if todo:                                    # load all variants + craft transfer sets once
                for v, mp in avail.items():
                    m = MRFClassifier(v, g=g, learn_J=TYPES[type_], seed=s, lean=lean)
                    m.load_state_dict(torch.load(mp)); m.eval(); models[v] = m
                for v, m in models.items():
                    adv[v] = {e: (pgd_linf(m, X, Y, e, steps=steps) if e else X)
                              for e in eps_transfer}
            for v in avail:
                d, mp, meta = _paths(type_, v, s, lean)
                rp = os.path.join(d, 'robustness.json')
                if os.path.exists(rp):
                    rec = json.load(open(rp))
                else:
                    model = models[v]
                    rec = {}
                    for norm, epslist in (('linf', eps_linf), ('l2', eps_l2)):
                        rec[norm] = {'eps': list(epslist), 'acc': [
                            _acc(model, ATTACKS[norm](model, X, Y, e, steps=steps) if e else X, Y)
                            for e in tqdm(epslist, desc=f'{v} s{s} {norm}', leave=False)]}
                    rec['transfer'] = {src: {'eps': list(eps_transfer),
                        'acc': [_acc(model, adv[src][e], Y) for e in eps_transfer]}
                        for src in models}
                    rec['nat'] = {name: {'strength': list(nat_strengths),
                        'acc': _corrupt_curve(model, X, Y, fn, nat_strengths)}
                        for name, fn in CORRUPTIONS.items()}
                    json.dump(rec, open(rp, 'w'), indent=2)
                    print(f"attacked {type_}/{v}/seed{s}")
                results['linf'].setdefault((type_, v), []).append(rec['linf']['acc'])
                results['l2'].setdefault((type_, v), []).append(rec['l2']['acc'])
                for name in CORRUPTIONS:
                    if name in rec.get('nat', {}):
                        results['nat'].setdefault((type_, v, name), []).append(rec['nat'][name]['acc'])
    os.makedirs(os.path.join(SAVE_ROOT, 'results'), exist_ok=True)
    pickle.dump(results, open(os.path.join(SAVE_ROOT, 'results', 'robustness.pkl'), 'wb'))
    return results


def plot_results(results=None):
    import matplotlib.pyplot as plt
    if results is None:
        results = pickle.load(open(os.path.join(SAVE_ROOT, 'results', 'robustness.pkl'), 'rb'))
    cmap = {'gibbs10': '0.55', 'gibbs20': '0.3', 'gibbs30': 'k', 'mf': 'r',
            'fbp0.5': 'green', 'lbp': 'C0', 'fbp1.5': 'purple', 'fbp2.0': 'orange'}
    for norm in ('linf', 'l2'):
        eps = results['eps'][norm]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, type_ in zip(axes, ['typeA_imposed', 'typeB_learned']):
            for variant in VARIANTS:
                key = (type_, variant)
                if key not in results[norm]:
                    continue
                arr = np.array(results[norm][key])            # [seeds, eps]
                mu, sd = arr.mean(0), arr.std(0)
                ax.plot(eps, mu, 'o-', color=cmap[variant], label=variant)
                ax.fill_between(eps, mu - sd, mu + sd, color=cmap[variant], alpha=0.15)
            ax.set(xlabel=f'PGD {norm} epsilon', title=type_.replace('_', ' '))
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[0].set_ylabel('accuracy'); axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f'Adversarial robustness (PGD-{norm}): imposed vs learned PGM')
        fig.tight_layout()
        fig.savefig(os.path.join(SAVE_ROOT, 'results', f'robustness_{norm}.png'),
                    dpi=180, bbox_inches='tight')

    # naturalistic corruptions: one figure per corruption
    strengths = results.get('nat_strengths')
    corruptions = sorted({name for (_, _, name) in results.get('nat', {})})
    for name in corruptions:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, type_ in zip(axes, ['typeA_imposed', 'typeB_learned']):
            for variant in VARIANTS:
                key = (type_, variant, name)
                if key not in results['nat']:
                    continue
                arr = np.array(results['nat'][key])
                mu, sd = arr.mean(0), arr.std(0)
                ax.plot(strengths, mu, 'o-', color=cmap[variant], label=variant)
                ax.fill_between(strengths, mu - sd, mu + sd, color=cmap[variant], alpha=0.15)
            ax.set(xlabel=f'{name} strength', title=type_.replace('_', ' '))
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[0].set_ylabel('accuracy'); axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f'Naturalistic corruption ({name}): imposed vs learned PGM')
        fig.tight_layout()
        fig.savefig(os.path.join(SAVE_ROOT, 'results', f'corruption_{name}.png'),
                    dpi=180, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'attack', 'plot', 'all'], default='all')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--n_seeds', type=int, default=N_SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS, help='fixed epochs for every net')
    ap.add_argument('--variants', nargs='*', default=list(RUN_VARIANTS))
    ap.add_argument('--types', nargs='*', default=list(TYPES))
    ap.add_argument('--lean', dest='lean', action='store_true', default=LEAN,
                    help='lean encoder: MRF carries the representation')
    ap.add_argument('--no-lean', dest='lean', action='store_false')
    args, _ = ap.parse_known_args()          # robust to Spyder's injected argv

    if args.fast:
        args.n_seeds, args.epochs = 1, 1
        args.variants, args.types = ['mf', 'lbp', 'gibbs20'], ['typeA_imposed']
    data = get_data(fake=args.fast, n_train=20000)
    seeds = list(range(args.n_seeds))
    if args.mode in ('train', 'all'):
        train_all(seeds, args.variants, args.types, data, epochs=args.epochs,
                  lean=args.lean)
    if args.mode in ('attack', 'all'):
        attack_all(seeds, args.variants, args.types, data,
                   n_imgs=(8 if args.fast else 50), lean=args.lean)
    if args.mode in ('plot', 'all'):
        plot_results()

