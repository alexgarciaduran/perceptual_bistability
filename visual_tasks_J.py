# -*- coding: utf-8 -*-
"""How coupling J helps perception: four visual tasks on a binary MRF.

The same ferromagnetic prior that produces bistability (strong coupling J binding neighbouring
features to the same state) also does useful perceptual work. Here we show it on four tasks, each a
different kind of degraded evidence that the coupling repairs:

  1. denoising            -- random pixel flips; J averages out the noise.
  2. contour integration  -- a smooth contour of oriented elements hidden among random ones;
                             an orientation-tuned (association-field) coupling makes it pop out.
  3. occlusion            -- a blanked region with no evidence; J fills it in from the context.
  4. deblurring           -- soft/greyscale evidence; J sharpens it back to a crisp binary figure.

Inference engines: mean field (MF) and loopy BP (LBP) on the grid, versus the J=0 independent
baseline (per-pixel MAP), which cannot use the prior. Each task reports reconstruction accuracy as a
function of J, so the benefit of coupling is quantitative, not just visual.

Outputs one MAIN figure (one example per task) and one SUPPLEMENTARY figure per task (several
examples + accuracy-vs-J). Run:  python visual_tasks_J.py
"""
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SEED = 20240915

# ----------------------------------------------------------------- output location
_CANDIDATES = [
    r"C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/visual_tasks",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_tasks_figures"),
]
OUT = next((c for c in _CANDIDATES if os.path.isdir(os.path.dirname(c))), _CANDIDATES[-1])
FIGS = os.path.normpath(os.path.join(OUT, "figures"))
os.makedirs(FIGS, exist_ok=True)

# palette (front = +1 = warm, back = -1 = cool), matching the paper's depth colouring
CMAP = plt.get_cmap('RdBu_r')


# ============================================================ inference on a 2D grid
def _neighbour_sum(m):
    """Sum of the 4-neighbourhood with zero-padded borders (no wraparound)."""
    s = np.zeros_like(m)
    s[1:, :] += m[:-1, :]; s[:-1, :] += m[1:, :]
    s[:, 1:] += m[:, :-1]; s[:, :-1] += m[:, 1:]
    return s


def mf_grid(h, J, n_iter=300, tol=1e-6):
    """Mean-field on an Ising grid p(x) ~ exp(J sum_<ij> x_i x_j + sum_i h_i x_i). Returns m in [-1,1]."""
    m = np.tanh(h)
    for _ in range(n_iter):
        m_new = np.tanh(J * _neighbour_sum(m) + h)
        if np.max(np.abs(m_new - m)) < tol:
            m = m_new; break
        m = m_new
    return m


def fbp_grid(h, J, alpha=1.0, n_iter=300, damping=0.5, tol=1e-6):
    """Fractional BP on the Ising grid (alpha=1 = loopy BP), log-ratio messages on the 4 edges."""
    tJa = np.tanh(J * alpha)
    Iu = np.zeros_like(h); Id = np.zeros_like(h); Il = np.zeros_like(h); Ir = np.zeros_like(h)

    def _msg(cav):
        return (1.0 / alpha) * np.arctanh(np.clip(tJa * np.tanh(cav), -1 + 1e-9, 1 - 1e-9))

    for _ in range(n_iter):
        total = h + Iu + Id + Il + Ir
        out_down = _msg(total - alpha * Id)     # this node -> node below; becomes below node's Iu
        out_up = _msg(total - alpha * Iu)       # -> node above; becomes above node's Id
        out_right = _msg(total - alpha * Ir)    # -> right node; becomes right node's Il
        out_left = _msg(total - alpha * Il)     # -> left node; becomes left node's Ir
        nIu = np.zeros_like(h); nId = np.zeros_like(h); nIl = np.zeros_like(h); nIr = np.zeros_like(h)
        nIu[1:, :] = out_down[:-1, :]
        nId[:-1, :] = out_up[1:, :]
        nIl[:, 1:] = out_right[:, :-1]
        nIr[:, :-1] = out_left[:, 1:]
        nIu = damping * nIu + (1 - damping) * Iu
        nId = damping * nId + (1 - damping) * Id
        nIl = damping * nIl + (1 - damping) * Il
        nIr = damping * nIr + (1 - damping) * Ir
        d = max(np.max(np.abs(nIu - Iu)), np.max(np.abs(nId - Id)),
                np.max(np.abs(nIl - Il)), np.max(np.abs(nIr - Ir)))
        Iu, Id, Il, Ir = nIu, nId, nIl, nIr
        if d < tol:
            break
    return np.tanh(h + Iu + Id + Il + Ir)


def lbp_grid(h, J, **kw):
    return fbp_grid(h, J, alpha=1.0, **kw)


def independent(h):
    """J = 0 baseline: per-pixel MAP, sign of the local evidence."""
    return np.tanh(h)


def _acc(recon_m, clean):
    """Pixel accuracy of sign(recon) against the clean +/-1 image."""
    return float(np.mean(np.sign(recon_m) == clean))


# ============================================================ stimuli (clean shapes)
def shape(kind, H=48, W=48):
    """Return a clean binary image in {-1,+1} (+1 = front/figure)."""
    img = -np.ones((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    if kind == 'square':
        img[(np.abs(xx - W / 2) < W / 4) & (np.abs(yy - H / 2) < H / 4)] = 1
    elif kind == 'disk':
        img[((xx - W / 2) ** 2 + (yy - H / 2) ** 2) < (W / 3.5) ** 2] = 1
    elif kind == 'triangle':
        img[(yy > H * 0.25) & (yy < H * 0.8) &
            (np.abs(xx - W / 2) < (yy - H * 0.25) * 0.7)] = 1
    elif kind == 'ring':
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        img[(r < W / 3) & (r > W / 5)] = 1
    elif kind == 'cross':
        img[(np.abs(xx - W / 2) < W / 8) | (np.abs(yy - H / 2) < H / 8)] = 1
    elif kind == 'letter':          # a blocky 'T'
        img[(yy > H * 0.25) & (yy < H * 0.4)] = 1
        img[(np.abs(xx - W / 2) < W / 10) & (yy > H * 0.25) & (yy < H * 0.75)] = 1
    return img


# ============================================================ tasks (grid-based)
def make_denoise(clean, p=0.2, beta=0.6, rng=None):
    obs = clean.copy()
    flip = rng.random(clean.shape) < p
    obs[flip] *= -1
    return obs, beta * obs                      # (shown image, evidence field h)


def make_occlusion(clean, beta=0.9, frac=0.32, rng=None):
    H, W = clean.shape
    obs = clean.copy()
    y0 = int(H * (0.5 - frac / 2)); y1 = int(H * (0.5 + frac / 2))
    x0 = int(W * (0.28)); x1 = int(W * (0.72))
    h = beta * clean.copy()
    h[y0:y1, x0:x1] = 0.0                        # no evidence in the occluded band
    obs = clean.copy(); obs[y0:y1, x0:x1] = 0.0  # shown with a grey occluder
    return obs, h


def make_deblur(clean, sigma=2.2, beta=1.1, noise=0.5, rng=None):
    """Blur plus sensor noise: the blur softens the edges and the noise flips signs, so the
    per-pixel evidence is often wrong; J restores a crisp figure."""
    if rng is None:
        rng = np.random.default_rng(0)
    blur = gaussian_filter(clean.astype(float), sigma)
    blur = blur / np.max(np.abs(blur) + 1e-9)          # soft evidence in [-1,1]
    obs = blur + noise * rng.standard_normal(clean.shape)
    return np.clip(obs, -1, 1), beta * obs


# ============================================================ contour integration
def _contour_curve(curve, H, W, rng):
    """Return (xs, ys) for a smooth contour of one of several shapes."""
    t = np.linspace(0.0, 1.0, 60)
    if curve == 'sine':
        xs = 4 + t * (W - 8); ys = H / 2 + 0.30 * H * np.sin(2 * np.pi * xs / (W - 8))
    elif curve == 'sine_hi':
        xs = 4 + t * (W - 8); ys = H / 2 + 0.22 * H * np.sin(3.2 * np.pi * xs / (W - 8))
    elif curve == 'circle':
        a = np.pi * (0.15 + 1.7 * t); R = 0.34 * min(H, W)
        xs = W / 2 + R * np.cos(a); ys = H / 2 + R * np.sin(a)
    elif curve == 'parabola':
        xs = 4 + t * (W - 8); ys = 6 + 0.9 * (xs - W / 2) ** 2 / (W / 2)
    elif curve == 'sdiag':
        xs = 4 + t * (W - 8); ys = H / 2 + 0.32 * H * np.tanh(4 * (t - 0.5))
    else:  # 'line' at a random angle
        ang = rng.uniform(0.2, np.pi - 0.2); L = 0.42 * min(H, W)
        cx, cy = W / 2, H / 2
        xs = cx + L * (t - 0.5) * 2 * np.cos(ang); ys = cy + L * (t - 0.5) * 2 * np.sin(ang)
    return xs, ys


def make_contour(H=44, W=44, spacing=2, n_noise=150, jitter=0.35, h0=0.2, seed=0,
                 curve='sine'):
    """Oriented-element association-field model. Elements sit on a jittered grid; a smooth
    contour (shape given by `curve`) carries elements whose orientation is the local tangent,
    the rest random. Local evidence h0 is identical for every element, so only the
    good-continuation coupling can separate the contour from the background."""
    rng = np.random.default_rng(seed)
    pos = []; ori = []; on = []
    xs, ys = _contour_curve(curve, H, W, rng)
    for k in range(len(xs) - 1):
        px, py = xs[k], ys[k]
        tang = np.arctan2(ys[k + 1] - ys[k], xs[k + 1] - xs[k])
        pos.append((px + rng.normal(0, jitter), py + rng.normal(0, jitter)))
        ori.append(tang); on.append(1)
    # background clutter: random positions and orientations
    for _ in range(n_noise):
        pos.append((rng.uniform(2, W - 2), rng.uniform(2, H - 2)))
        ori.append(rng.uniform(0, np.pi)); on.append(0)
    pos = np.array(pos); ori = np.array(ori); on = np.array(on)
    h = np.full(len(pos), h0)                     # identical local evidence
    return pos, ori, on, h


def _association(pos, ori, radius=5.0, thr=0.85):
    """Sparse association-field weights: only neighbours within radius that are strongly
    co-circular (aligned to the link on both ends). The strict threshold is the prior that lets
    a genuine contour survive while sparse background alignments do not."""
    n = len(pos)
    edges = []
    for i in range(n):
        d = pos - pos[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        cand = np.where((dist < radius) & (dist > 1e-6))[0]
        for j in cand:
            if j <= i:
                continue
            phi = np.arctan2(d[j, 1], d[j, 0])    # direction i->j
            a = abs(np.cos(ori[i] - phi)) * abs(np.cos(ori[j] - phi))  # both aligned to the link
            if a > thr:
                edges.append((i, j, a * np.exp(-(dist[j] ** 2) / (2 * (radius / 2) ** 2))))
    return edges


def mf_graph(h, edges, J, n_iter=400, tol=1e-6):
    """Mean field on a general graph given a sparse weighted edge list."""
    n = len(h)
    m = np.tanh(h)
    ii = np.array([e[0] for e in edges]); jj = np.array([e[1] for e in edges])
    ww = np.array([e[2] for e in edges])
    for _ in range(n_iter):
        field = np.zeros(n)
        np.add.at(field, ii, ww * m[jj])
        np.add.at(field, jj, ww * m[ii])
        m_new = np.tanh(J * field + h)
        if np.max(np.abs(m_new - m)) < tol:
            m = m_new; break
        m = m_new
    return m


def bp_graph(h, edges, J, alpha=1.0, n_iter=400, damping=0.5, tol=1e-6):
    """Fractional/loopy BP on a general weighted graph (alpha=1 = LBP). Coupling per edge = J*w."""
    E = len(edges)
    src = np.empty(2 * E, int); dst = np.empty(2 * E, int); w = np.empty(2 * E)
    for k, (i, j, ww) in enumerate(edges):
        src[2 * k], dst[2 * k], w[2 * k] = i, j, ww
        src[2 * k + 1], dst[2 * k + 1], w[2 * k + 1] = j, i, ww
    rev = np.arange(2 * E) ^ 1
    tJa = np.tanh(alpha * J * w)
    M = np.zeros(2 * E); n = len(h)
    for _ in range(n_iter):
        incoming = np.zeros(n); np.add.at(incoming, dst, M)
        cav = h[src] + incoming[src] - M[rev]
        Mnew = (1.0 / alpha) * np.arctanh(np.clip(tJa * np.tanh(cav), -1 + 1e-9, 1 - 1e-9))
        Mnew = damping * Mnew + (1 - damping) * M
        if np.max(np.abs(Mnew - M)) < tol:
            M = Mnew; break
        M = Mnew
    incoming = np.zeros(n); np.add.at(incoming, dst, M)
    return np.tanh(h + incoming)


def _auc(score, label):
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        ix = allv == v; ranks[ix] = ranks[ix].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ============================================================ J sweeps
def sweep_grid(make, kind, J_grid, engine=mf_grid, reps=6, base_seed=SEED, **kw):
    """Mean accuracy vs J for a grid task over several shapes/noise draws."""
    kinds = ['square', 'disk', 'triangle', 'ring', 'cross', 'letter']
    acc = np.zeros((reps, len(J_grid)))
    for r in range(reps):
        rng = np.random.default_rng(base_seed + r)
        clean = shape(kinds[r % len(kinds)])
        _, h = make(clean, rng=rng, **kw)
        for k, J in enumerate(J_grid):
            acc[r, k] = _acc(engine(h, J), clean)
    return acc.mean(0), acc.std(0) / np.sqrt(reps)


def sweep_contour(J_grid, reps=6, base_seed=SEED):
    auc = np.zeros((reps, len(J_grid)))
    for r in range(reps):
        pos, ori, on, h = make_contour(seed=base_seed + r)
        edges = _association(pos, ori)
        for k, J in enumerate(J_grid):
            m = mf_graph(h, edges, J)
            auc[r, k] = _auc((m + 1) / 2, on)
    return auc.mean(0), auc.std(0) / np.sqrt(reps)


# ============================================================ figures
def _imshow(ax, img, title=None):
    ax.imshow(img, cmap=CMAP, vmin=-1, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def _best_J(J_grid, mean_acc):
    return float(J_grid[int(np.argmax(mean_acc))])


def main_figure(J_denoise=0.7, J_occl=0.7, J_deblur=0.8, J_contour=0.5):
    """Rows = tasks, columns = algorithms (clean, observed, J=0, MF, LBP, FBP alpha=0.5),
    all coupled algorithms evaluated at a fixed representative J per task. The reconstructions
    are nearly identical across algorithms: the jump is from J=0 to J>0, not between schemes."""
    COLS = ['clean', 'observed', 'J = 0', 'MF', 'LBP', r'FBP $\alpha$=0.5']
    fig, ax = plt.subplots(4, 6, figsize=(17, 11), constrained_layout=True)

    # ---- row 0: denoising (grid)
    rng = np.random.default_rng(SEED)
    clean = shape('disk'); obs, h = make_denoise(clean, p=0.22, beta=0.6, rng=rng)
    recons = [independent(h), mf_grid(h, J_denoise), lbp_grid(h, J_denoise), fbp_grid(h, J_denoise, 0.5)]
    _imshow(ax[0, 0], clean); _imshow(ax[0, 1], obs)
    for c, r in enumerate(recons):
        _imshow(ax[0, 2 + c], r); ax[0, 2 + c].set_xlabel(f'acc {_acc(r, clean):.2f}', fontsize=8)
    ax[0, 0].set_ylabel(f'denoising\n(22% noise)', fontsize=10)

    # ---- row 1: contour integration (association-field graph)
    pos, ori, on, hc = make_contour(seed=SEED); edges = _association(pos, ori)
    gres = [mf_graph(hc, edges, 0.0), mf_graph(hc, edges, J_contour),
            bp_graph(hc, edges, J_contour, 1.0), bp_graph(hc, edges, J_contour, 0.5)]
    _plot_elements(ax[1, 0], pos, ori, on.astype(float), truth=True)
    _plot_elements(ax[1, 1], pos, ori, on.astype(float), truth=True)
    # _plot_elements(ax[1, 1], pos, ori, np.full(len(pos), 0.5))
    for c, m in enumerate(gres):
        _plot_elements(ax[1, 2 + c], pos, ori, (m + 1) / 2)
        ax[1, 2 + c].set_xlabel(f'AUC {_auc((m + 1) / 2, on):.2f}', fontsize=8)
    ax[1, 0].set_ylabel(f'contour\nintegration', fontsize=10)

    # ---- row 2: occlusion (grid)
    clean = shape('square'); obs, h = make_occlusion(clean, beta=0.9)
    recons = [independent(h), mf_grid(h, J_occl), lbp_grid(h, J_occl), fbp_grid(h, J_occl, 0.5)]
    _imshow(ax[2, 0], clean); _imshow(ax[2, 1], obs)
    for c, r in enumerate(recons):
        _imshow(ax[2, 2 + c], r); ax[2, 2 + c].set_xlabel(f'acc {_acc(r, clean):.2f}', fontsize=8)
    ax[2, 0].set_ylabel('occlusion', fontsize=10)

    # ---- row 3: deblurring (grid)
    clean = shape('triangle'); obs, h = make_deblur(clean, sigma=2.2, rng=np.random.default_rng(SEED + 3))
    recons = [independent(h), mf_grid(h, J_deblur), lbp_grid(h, J_deblur), fbp_grid(h, J_deblur, 0.5)]
    _imshow(ax[3, 0], clean); _imshow(ax[3, 1], obs)
    for c, r in enumerate(recons):
        _imshow(ax[3, 2 + c], r); ax[3, 2 + c].set_xlabel(f'acc {_acc(r, clean):.2f}', fontsize=8)
    ax[3, 0].set_ylabel('deblurring', fontsize=10)

    for c, name in enumerate(COLS):
        ax[0, c].set_title(name, fontsize=11)
    # one shared colorbar per row (same RdBu_r scale for every panel: red=front +1, blue=back -1)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP)
    for r in range(4):
        cb = fig.colorbar(sm, ax=ax[r, :].tolist(), fraction=0.018, pad=0.01, ticks=[-1, 0, 1])
        cb.ax.set_yticklabels(['back\n(-1)', '0.5', 'front\n(+1)'], fontsize=7)
        cb.set_label('posterior  q(x=1)', fontsize=8)
    fig.savefig(os.path.join(FIGS, 'fig_visual_tasks_main.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'fig_visual_tasks_main.svg'), bbox_inches='tight')
    plt.close(fig)


def _plot_elements(ax, pos, ori, val, title=None, truth=False, L=1.6):
    """Each oriented segment is one edge element; colour = posterior belief q(x=1) via the shared
    RdBu_r map (red=+1 on-contour, blue=-1 off). For inferred panels, opacity/width also grow with
    belief so the detected contour pops out; ground-truth panels use full opacity."""
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    ax.set_xlim(0, 44); ax.set_ylim(0, 44)
    for (x, y), th, v in zip(pos, ori, val):
        dx, dy = L * np.cos(th), L * np.sin(th)
        c = CMAP(float(np.clip(v, 0, 1)))                # same colormap as the grid tasks
        if truth:
            lw = 1.7; al = 0.95
        else:
            vv = float(np.clip((v - 0.5) * 2, 0, 1))     # stretch: baseline->0, detected->1
            lw = 0.5 + 2.6 * vv; al = 0.25 + 0.75 * vv
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=c, lw=lw, alpha=al, solid_capstyle='round')
    if title:
        ax.set_title(title, fontsize=9)


def supp_figure(name, make, J_grid, color, reps_show=5, **kw):
    """One supplementary figure per grid task: several examples (clean/observed/J=0/coupled) + acc vs J."""
    kinds = ['disk', 'square', 'triangle', 'ring', 'cross', 'letter']
    mean, sem = sweep_grid(make, name, J_grid, reps=8, **kw)
    Jb = _best_J(J_grid, mean)
    fig, ax = plt.subplots(reps_show, 5, figsize=(13, 2.5 * reps_show))
    for r in range(reps_show):
        rng = np.random.default_rng(SEED + 100 + r)
        clean = shape(kinds[r % len(kinds)])
        obs, h = make(clean, rng=rng, **kw)
        _imshow(ax[r, 0], clean, 'clean' if r == 0 else None)
        _imshow(ax[r, 1], obs, 'observed' if r == 0 else None)
        _imshow(ax[r, 2], independent(h),
                (f'J=0 ({_acc(independent(h), clean):.2f})'))
        _imshow(ax[r, 3], mf_grid(h, Jb),
                (f'J={Jb:.1f} ({_acc(mf_grid(h, Jb), clean):.2f})'))
        ax[r, 4].plot(J_grid, [ _acc(mf_grid(make(clean, rng=np.random.default_rng(SEED+100+r), **kw)[1], J), clean) for J in J_grid],
                      'o-', color=color, ms=3)
        ax[r, 4].axvline(Jb, color='0.6', ls=':'); ax[r, 4].set_ylim(0.45, 1.02)
        ax[r, 4].spines['top'].set_visible(False); ax[r, 4].spines['right'].set_visible(False)
        if r == 0:
            ax[r, 4].set_title('accuracy vs J', fontsize=9)
    ax[-1, 4].set_xlabel('coupling J', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f'supp_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return mean, sem, Jb


def supp_contour(J_grid, reps_show=5):
    """One row per DIFFERENT contour shape, each with its own clutter density/jitter, so every
    example is a fresh stimulus. The coupled column uses that row's own best J."""
    # (curve shape, n_noise clutter, jitter) -- different contour and different line noise per row
    variants = [('sine', 150, 0.35), ('circle', 170, 0.30), ('sine_hi', 130, 0.45),
                ('parabola', 160, 0.35), ('sdiag', 190, 0.40)][:reps_show]
    fig, ax = plt.subplots(reps_show, 4, figsize=(11, 2.6 * reps_show))
    for r, (curve, nn, jit) in enumerate(variants):
        pos, ori, on, h = make_contour(seed=SEED + 200 + r, curve=curve, n_noise=nn, jitter=jit)
        edges = _association(pos, ori)
        aucs = np.array([_auc((mf_graph(h, edges, J) + 1) / 2, on) for J in J_grid])
        Jc = _best_J(J_grid, aucs)
        m0 = mf_graph(h, edges, 0.0); mJ = mf_graph(h, edges, Jc)
        _plot_elements(ax[r, 0], pos, ori, on.astype(float),
                       'contour+clutter' if r == 0 else None, truth=True)
        ax[r, 0].set_ylabel(curve, fontsize=9)
        _plot_elements(ax[r, 1], pos, ori, (m0 + 1) / 2, f'J=0 (AUC {_auc((m0+1)/2, on):.2f})')
        _plot_elements(ax[r, 2], pos, ori, (mJ + 1) / 2, f'J={Jc:.1f} (AUC {_auc((mJ+1)/2, on):.2f})')
        ax[r, 3].plot(J_grid, aucs, 'o-', color='tab:blue', ms=3)
        ax[r, 3].axvline(Jc, color='0.6', ls=':'); ax[r, 3].set_ylim(0.45, 1.02)
        ax[r, 3].spines['top'].set_visible(False); ax[r, 3].spines['right'].set_visible(False)
        if r == 0:
            ax[r, 3].set_title('AUC vs J', fontsize=9)
    ax[-1, 3].set_xlabel('coupling J', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'supp_contour.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return Jc


def main():
    J_grid = np.round(np.linspace(0.0, 1.2, 20), 3)
    Jc_grid = np.round(np.linspace(0.0, 2.0, 20), 3)   # contour needs larger J (weak local evidence)
    print('output ->', FIGS)
    main_figure()
    print('main figure done')
    supp_figure('denoise', make_denoise, J_grid, 'firebrick', p=0.22, beta=0.6)
    supp_figure('occlusion', make_occlusion, J_grid, 'seagreen', beta=0.9)
    supp_figure('deblur', make_deblur, J_grid, 'darkorange', sigma=2.2)
    supp_contour(Jc_grid)
    print('supplementary figures done')


if __name__ == '__main__':
    main()
