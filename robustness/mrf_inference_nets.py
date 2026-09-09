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

SAVE_ROOT = r"C:\Users\alexg\OneDrive\Escritorio\phd\folder_save\robustness_analysis"
DATA_DIR = SAVE_ROOT               # MNIST lives here (torchvision makes DATA_DIR/MNIST)
# variant -> (inference algo, param): param = sampling iters for gibbs, alpha for fbp
VARIANTS = {'gibbs10': ('sampling', 10), 'gibbs20': ('sampling', 20),
            'gibbs30': ('sampling', 30), 'mf': ('mf', None),
            'fbp0.5': ('fbp', 0.5), 'lbp': ('fbp', 1.0),
            'fbp1.5': ('fbp', 1.5), 'fbp2.0': ('fbp', 2.0)}
TYPES = {'typeA_imposed': False, 'typeB_learned': True}   # learn_J flag


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
    def __init__(self, n_latent, in_ch=1):
        super().__init__()
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


def infer_sampling(B, Jmat, iters=12, n_samples=8, noise=0.6):
    bsz, n = B.shape
    x = torch.zeros(bsz, n_samples, n)
    for _ in range(iters):
        field = x @ Jmat + B.unsqueeze(1)
        p = torch.sigmoid(2 * field).clamp(1e-6, 1 - 1e-6)
        u = torch.rand_like(p).clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p) - torch.log(1 - p) + noise * (torch.log(u) - torch.log(1 - u))
        x = torch.tanh(logit / 2)
    return x.mean(1)


class MRFClassifier(nn.Module):
    def __init__(self, variant, g=7, n_classes=10, learn_J=False,
                 J_sigma=0.15, iters=10, seed=0):
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
        self.encoder = Encoder(self.n)
        self.readout = nn.Linear(self.n, n_classes)

    def Jmat(self):
        J = self.Jraw * self.mask
        return (J + J.t()) / 2                             # keep symmetric

    def marginals(self, x):
        B = self.encoder(x)
        Jm = self.Jmat()
        if self.algo == 'mf':
            return infer_mf(B, Jm, self.iters)
        if self.algo == 'sampling':
            return infer_sampling(B, Jm, iters=self.samp_iters)
        return infer_fbp(B, Jm, alpha=self.alpha, iters=self.iters)

    def forward(self, x):
        return self.readout(self.marginals(x))


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


def train_model(model, data, target=0.85, max_epochs=30, lr=1e-3, bs=128):
    Xtr, Ytr, Xte, Yte = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        acc = test_acc(model, Xte, Yte)
        best = max(best, acc)
        if acc >= target:
            return acc, ep + 1
    return best, max_epochs


# --------------------------------------------------------------- orchestrate
def _paths(type_, variant, seed):
    d = os.path.join(SAVE_ROOT, type_, variant, f'seed{seed}')
    return d, os.path.join(d, 'model.pt'), os.path.join(d, 'meta.json')


def train_all(seeds, variants, types, data, target=0.85, max_epochs=30, g=7):
    for type_ in types:
        for variant in variants:
            for s in seeds:
                d, mp, meta = _paths(type_, variant, s)
                if os.path.exists(meta):
                    print(f"skip {type_}/{variant}/seed{s} (done)"); continue
                os.makedirs(d, exist_ok=True)
                torch.manual_seed(s); np.random.seed(s)
                model = MRFClassifier(variant, g=g, learn_J=TYPES[type_], seed=s)
                acc, ep = train_model(model, data, target=target, max_epochs=max_epochs)
                torch.save(model.state_dict(), mp)
                json.dump({'type': type_, 'variant': variant, 'seed': s,
                           'test_acc': acc, 'epochs': ep, 'reached_target': acc >= target},
                          open(meta, 'w'), indent=2)
                print(f"{type_}/{variant}/seed{s}: acc={acc:.3f} ({ep} ep)")


def attack_all(seeds, variants, types, data, g=7, n_imgs=50,
               eps_linf=(0, 0.05, 0.1, 0.15, 0.2, 0.3),
               eps_l2=(0, 0.5, 1.0, 1.5, 2.0, 3.0)):
    _, _, Xte, Yte = data
    X, Y = Xte[:n_imgs], Yte[:n_imgs]
    results = {'linf': {}, 'l2': {}, 'eps': {'linf': eps_linf, 'l2': eps_l2}}
    for type_ in types:
        for variant in variants:
            for s in seeds:
                d, mp, meta = _paths(type_, variant, s)
                if not os.path.exists(mp):
                    continue
                model = MRFClassifier(variant, g=g, learn_J=TYPES[type_], seed=s)
                model.load_state_dict(torch.load(mp)); model.eval()
                for norm, epslist in (('linf', eps_linf), ('l2', eps_l2)):
                    curve = []
                    for e in epslist:
                        xa = ATTACKS[norm](model, X, Y, e) if e else X
                        with torch.no_grad():
                            curve.append((model(xa).argmax(1) == Y).float().mean().item())
                    results[norm].setdefault((type_, variant), []).append(curve)
                print(f"attacked {type_}/{variant}/seed{s}")
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
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'attack', 'plot', 'all'], default='all')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--n_seeds', type=int, default=20)
    ap.add_argument('--target', type=float, default=0.85)
    ap.add_argument('--max_epochs', type=int, default=30)
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--types', nargs='*', default=list(TYPES))
    args = ap.parse_args()

    if args.fast:
        args.n_seeds, args.target, args.max_epochs = 1, 0.5, 1
        args.variants, args.types = ['mf', 'lbp', 'gibbs10'], ['typeA_imposed']
    data = get_data(fake=args.fast, n_train=20000)
    seeds = list(range(args.n_seeds))
    if args.mode in ('train', 'all'):
        train_all(seeds, args.variants, args.types, data, args.target, args.max_epochs)
    if args.mode in ('attack', 'all'):
        attack_all(seeds, args.variants, args.types, data,
                   n_imgs=(8 if args.fast else 50))
    if args.mode in ('plot', 'all'):
        plot_results()
