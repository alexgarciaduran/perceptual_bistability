"""Reproduction of Wasmuht, Parker & Krug (2019), Sci Rep 9:11449,
"Interneuronal correlations at longer time scales predict decision signals
for bistable structure-from-motion perception" -- in the inference-network
model, with NO feedback connections.

Their three stimuli map onto two model knobs, the coupling J and the field B:

    random 2D motion      J < J*  (MONOSTABLE), B = 0      green
    ambiguous cylinder    J > J*  (BISTABLE),   B = 0      purple
    unambiguous cylinder  J > J*  (BISTABLE),   B > 0      blue

So the regime itself is the stimulus manipulation: random motion carries no
structure-from-motion constraint between local motion signals, hence weak
coupling; the cylinder does, hence strong coupling; and disparity supplies the
disambiguating evidence B. Their disparities (0 to 0.03 deg) map linearly onto
B through B_PER_DEG, whose scale is free and not identifiable from their data.

They record a single unit and multi-unit activity at the SAME electrode, so
both read one node; their r_SC measures how much of each neuron's variance is
shared rate fluctuation rather than private Poisson noise.

Two noise sources, as their data require: a fast COMMON input giving the
short-timescale correlation floor present in all three conditions, and an
INDEPENDENT sensory fluctuation per node whose recurrent amplification
produces the slow component seen only for the ambiguous stimulus.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, pearsonr

from posterior_computation_comparison import (
    THETA_NECKER, DATA_FOLDER, ising_params_heterogeneous, _beliefs, _map_step)

# ---------------------------------------------------------------- constants
C_AMB, C_UNAMB, C_RAND = '#6a3d9a', '#1f78b4', '#33a02c'   # their colours
TAU_MS, DT = 200.0, 0.05                 # tau = 200 ms, bin = dt*tau = 10 ms
BIN_MS = DT * TAU_MS
T_TRIAL = int(round(2000.0 / BIN_MS))    # 2 s trial
# Encoding matched to their recordings. At their example site (Fig. 1a) the
# single unit fires near 65 spks/s and the multi-unit near 145, and their
# Methods set the Schmitt-trigger level so that multi-unit rates for the 0 deg
# cylinder lay between 10 and 200 spks/s. The multi-unit is therefore about
# two neurons' worth of spikes above threshold, not a large pool. GAIN is the
# half-depth of percept modulation: rate = BASE +- GAIN as the belief flips.
K, N_MU = 10, 2                          # pool size, neurons in the multi-unit
GAIN, BASE = 20.0, 65.0                  # single unit 45-85, multi-unit 90-170
TAU_S = 1.0
# No injected sensory noise. A common input shared across all nodes would be
# indistinguishable from the top-down signal this reproduction exists to show
# is unnecessary, so adding one to match their random-motion baseline would
# assume the conclusion. The only variability is the spiking itself, acting
# through the random initial state.
SIG_IND, SIG_COM = 0.0, 0.0
JSTAR = {'mf': 0.33, 'fbp': 0.545}
# J_MONO is fitted, not assumed: random motion still shows r_SC = 0.23 in their
# data, and only a coupling close to criticality gives slow enough fluctuations
# to produce any shared variance at all without a bistable percept.
J_MONO, J_BI = 0.85, 1.50                # J = ratio * J*
DISPARITIES = (0.0, 0.01, 0.02, 0.03)    # degrees, as in their Fig. 2a
# fitted: their r_SC falls from 0.42 to 0.28 over 0 to 0.03 deg, which needs a
# field of only ~0.06 at full disparity. The earlier value of 10 was five times
# too steep and drove the unambiguous condition to zero correlation.
B_PER_DEG = 2.0
MAXLAG_MS = 1000.0                       # their Figs. 3b and 4b reach 10^3 ms
N_CHOICE = 15                            # steps averaged for the reported percept
# How the multi-unit is composed. True: its N_MU neurons are scattered at
# random over ALL units, as a real electrode would pick up a mixture that need
# not share the single unit's tuning. False: they all come from the single
# unit's own pool. A GLOBAL signal (the percept) survives mixing; a LOCAL one
# (coupling between neighbours) is diluted by it, so comparing the two
# separates those two sources without needing a distance metric.
MIXED_MU = True


def _trial(kind, al, Jm, Bvec, rng):
    """One trial. Returns pool counts (T_TRIAL, n) and the reported percept."""
    mask = Jm != 0.0
    n = len(Bvec)
    size = n if kind == 'mf' else int(mask.sum())
    scale = np.ones(n) if kind == 'mf' else np.abs(Jm[mask])
    x = np.clip(0.05 * rng.standard_normal(size), -scale, scale)
    lam = DT * TAU_MS * 1e-3 * K
    s = GAIN * (x / scale) + BASE
    pool = np.empty((T_TRIAL, n), np.int64)
    sgn = np.empty(T_TRIAL)
    q = _beliefs(kind, x, Jm, Bvec, mask)
    for t in range(T_TRIAL):
        xh = (s - BASE) / GAIN * scale
        Bt = (Bvec + SIG_IND * rng.standard_normal(n) / np.sqrt(DT)
              + SIG_COM * rng.standard_normal() / np.sqrt(DT))
        q = _beliefs(kind, xh, Jm, Bt, mask)
        f = _map_step(kind, xh, Jm, Bt, al, 1.0, mask)
        x = x + DT * (f - x)
        node = x if kind == 'mf' else (2 * q - 1)        # recorded at the node
        rate = np.clip(GAIN * node + BASE, 0.0, GAIN + BASE)
        pool[t] = rng.poisson(rate * lam)
        sgn[t] = np.mean(q) - 0.5
        urate = np.clip(GAIN * (x / scale) + BASE, 0.0, GAIN + BASE)
        s = s + (DT / TAU_S) * (rng.poisson(urate * lam) / lam - s)
    # the reported percept is the signed average over the last N_CHOICE steps,
    # not the single final sample, which is noisy
    return pool, float(np.mean(sgn[-N_CHOICE:]))


def _ccg(a, b, maxlag):
    """Mean-subtracted spike-time cross-correlogram, trials x time."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    T = a.shape[1]
    out = np.zeros(2 * maxlag + 1)
    for k in range(-maxlag, maxlag + 1):
        i0, i1 = max(0, -k), T - max(0, k)
        out[k + maxlag] = np.mean(a[:, i0:i1] * b[:, i0 + k:i1 + k])
    return out


def _auc(a, b):
    r = rankdata(np.r_[a, b])
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def _compose(pool, alloc, rng):
    """Single unit and multi-unit counts from the pool counts of every node.

    alloc[i, j] is how many of site i's multi-unit neurons sit in node j; the
    single unit is one further neuron of node i, excluded from its own pool so
    the two never share a cell."""
    n = pool.shape[1]
    su = np.stack([rng.binomial(pool[:, i], 1.0 / K) for i in range(n)], 1)
    mu = np.zeros_like(su)
    for i in range(n):
        for j in range(n):
            if alloc[i, j] == 0:
                continue
            avail = pool[:, j] - (su[:, j] if j == i else 0)
            mu[:, i] += rng.binomial(np.maximum(avail, 0),
                                     alloc[i, j] / (K - (1 if j == i else 0)))
    return su, mu


def run_condition(kind, al, Jv, Bv, n_trials=120, n_nets=5, seed0=0,
                  sigma_J=0.25, sigma_B=0.05, mixed_mu=None):
    """All sites of all network realisations for one stimulus condition.

    Heterogeneous couplings and fields make the sites differ, which is what
    gives the spread across sites that their Figs. 2 and 3 plot.
    Returns dict with per-site rSC, CP and rCCG, plus one example site."""
    n = len(THETA_NECKER)
    rsc, cp, ccg_all, example = [], [], [], None
    ml = int(round(MAXLAG_MS / BIN_MS))
    w = np.array([T_TRIAL - abs(k) for k in range(-ml, ml + 1)], float)
    for r in range(n_nets):
        rng = np.random.default_rng(seed0 + 1000 * r)
        Jm, Bvec = ising_params_heterogeneous(
            THETA_NECKER, Jv, Bv, sigma_J=sigma_J * Jv, sigma_B=sigma_B,
            seed=seed0 + r)
        # the electrode's composition is fixed for the site, not redrawn
        alloc = (rng.multinomial(N_MU, np.ones(n) / n, size=n)
                 if (MIXED_MU if mixed_mu is None else mixed_mu)
                 else np.eye(n, dtype=int) * N_MU)
        SU, MU, PER = [], [], []
        for _ in range(n_trials):
            pool, p = _trial(kind, al, Jm, Bvec, rng)
            a, b = _compose(pool, alloc, rng)
            SU.append(a); MU.append(b); PER.append(p)
        SU, MU, PER = np.array(SU), np.array(MU), np.array(PER)
        cs, cu = SU.sum(1), MU.sum(1)
        for i in range(n):
            if cs[:, i].std() <= 0 or cu[:, i].std() <= 0:
                continue
            rsc.append(np.corrcoef(cs[:, i], cu[:, i])[0, 1])
            cw = _ccg(SU[:, :, i], MU[:, :, i], ml) * w
            ccg_all.append(np.array([cw[ml - k:ml + k + 1].sum()
                                     for k in range(ml + 1)])
                           / (cs[:, i].std() * cu[:, i].std()))
            g1, g2 = cs[PER > 0, i], cs[PER <= 0, i]
            cp.append(max(_auc(g1, g2), 1 - _auc(g1, g2))
                      if len(g1) > 5 and len(g2) > 5 else np.nan)
            if example is None and r == 0 and i == 0:
                example = (cs[:, i].copy(), cu[:, i].copy())
    return dict(rsc=np.array(rsc), cp=np.array(cp),
                ccg=np.array(ccg_all), example=example)


def run_all(kind='mf', al=1.0, n_trials=120, n_nets=5, recompute=False,
            folder=DATA_FOLDER, verbose=True, mixed_mu=None):
    """Every stimulus condition for one algorithm, cached to disk."""
    mix = MIXED_MU if mixed_mu is None else mixed_mu
    f = folder + 'wasmuht_%s_a%g_T%d_N%d_%s.npz' % (
        kind, al, n_trials, n_nets, 'mix' if mix else 'same')
    if not recompute and os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return {k: z[k].item() for k in z.files}
    out = {}
    out['random'] = run_condition(kind, al, J_MONO * JSTAR[kind], 0.0,
                                  n_trials, n_nets, mixed_mu=mix)
    if verbose:
        print('  random motion done', flush=True)
    for d in DISPARITIES:
        out['disp%.2f' % d] = run_condition(kind, al, J_BI * JSTAR[kind],
                                            B_PER_DEG * d, n_trials, n_nets,
                                            mixed_mu=mix)
        if verbose:
            print('  disparity %.2f done' % d, flush=True)
    np.savez(f, **{k: np.array(v, dtype=object) for k, v in out.items()})
    return out


# ------------------------------------------------------------------ figures
def plot_figure1(res, fname='wasmuht_fig1'):
    """Their Fig. 1: one example site, ambiguous versus random motion."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, (lab, col) in enumerate((('disp0.00', C_AMB), ('random', C_RAND))):
        cs, cu = res[lab]['example']
        zs = (cs - cs.mean()) / cs.std()
        zu = (cu - cu.mean()) / cu.std()
        t = np.arange(len(zs))
        ax[k].plot(t, zs, color=col, lw=.6, alpha=.5)
        ax[k].plot(t, zu, color='0.4', lw=.6, alpha=.5)
        rm = lambda v: np.convolve(v, np.ones(11) / 11, 'same')
        ax[k].plot(t, rm(zs), color=col, lw=2, label='single unit')
        ax[k].plot(t, rm(zu), color='0.2', lw=2, label='multi unit')
        ax[k].set_title('%s   $r_{SC}$ = %.2f'
                        % ('ambiguous cylinder' if k == 0 else 'random motion',
                           np.corrcoef(cs, cu)[0, 1]), fontsize=11)
        ax[k].set_xlabel('trial'); ax[k].set_ylabel('z-scored spike count')
        ax[k].legend(frameon=False, fontsize=8)
    ax[2].hist([res['disp0.00']['rsc'], res['random']['rsc']], bins=12,
               color=[C_AMB, C_RAND], label=['ambiguous', 'random motion'])
    ax[2].set_xlabel(r'$r_{SC}$'); ax[2].set_ylabel('sites')
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    fig.tight_layout()
    for e in ('png', 'svg'):
        fig.savefig(DATA_FOLDER + '%s.%s' % (fname, e), dpi=300,
                    bbox_inches='tight')
    return fig


def plot_figure2(res, fname='wasmuht_fig2'):
    """Their Fig. 2: rSC against disparity (violins), and rCCG against tau."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    data = [res['random']['rsc']] + [res['disp%.2f' % d]['rsc']
                                     for d in DISPARITIES]
    # categorical positions: the disparities are 0.01 apart, so plotting them
    # on a linear axis would pile every violin on top of the next
    pos = list(range(len(data)))
    v = ax[0].violinplot(data, positions=pos, widths=.75, showmeans=True)
    for b, c in zip(v['bodies'], [C_RAND, C_AMB, C_UNAMB, C_UNAMB, C_UNAMB]):
        b.set_facecolor(c); b.set_alpha(.65)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        v[key].set_color('0.3'); v[key].set_linewidth(1.1)
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels(['random\nmotion'] + ['%.2f' % d for d in DISPARITIES])
    ax[0].set_xlabel('disparity (deg)'); ax[0].set_ylabel(r'$r_{SC}$')
    ax[0].axhline(0, color='0.7', lw=.8)
    lag = np.arange(res['random']['ccg'].shape[1]) * BIN_MS
    for lab, col, nm in (('disp0.00', C_AMB, 'ambiguous cylinder'),
                         ('disp%.2f' % DISPARITIES[-1], C_UNAMB,
                          'unambiguous cylinder'),
                         ('random', C_RAND, 'random motion')):
        m = res[lab]['ccg'].mean(0)
        e = res[lab]['ccg'].std(0) / np.sqrt(len(res[lab]['ccg']))
        ax[1].plot(lag, m, color=col, lw=2, label=nm)
        ax[1].fill_between(lag, m - e, m + e, color=col, alpha=.2, lw=0)
    ax[1].set_xlabel(r'integration window $\tau$ (ms)')
    ax[1].set_xscale('log')          # their Figs. 3b and 4b use a log tau axis
    ax[1].set_xlim(BIN_MS, MAXLAG_MS)
    ax[1].set_ylabel(r'$r_{CCG}$'); ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    fig.tight_layout()
    for e in ('png', 'svg'):
        fig.savefig(DATA_FOLDER + '%s.%s' % (fname, e), dpi=300,
                    bbox_inches='tight')
    return fig


def plot_figure3(res, fname='wasmuht_fig3'):
    """Their Fig. 3: CP against rSC, and rCCG split at the median CP."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    r_, c_, g_ = (res['disp0.00']['rsc'], res['disp0.00']['cp'],
                  res['disp0.00']['ccg'])
    ok = np.isfinite(c_) & np.isfinite(r_)
    ax[0].scatter(r_[ok], c_[ok], s=26, color=C_AMB, alpha=.75)
    if ok.sum() > 3:
        rr, pp = pearsonr(r_[ok], c_[ok])
        b = np.polyfit(r_[ok], c_[ok], 1)
        xs = np.linspace(r_[ok].min(), r_[ok].max(), 20)
        ax[0].plot(xs, np.polyval(b, xs), color='0.3', lw=1.4)
        ax[0].set_title('r = %.2f, p = %.3f, n = %d' % (rr, pp, ok.sum()),
                        fontsize=11)
    ax[0].axhline(0.5, color='0.7', lw=.8)
    ax[0].set_xlabel(r'$r_{SC}$'); ax[0].set_ylabel('choice probability')
    med = np.nanmedian(c_[ok])
    lag = np.arange(g_.shape[1]) * BIN_MS
    for sel, col, nm in ((c_ >= med, '#b2182b', 'CP $\\geq$ median'),
                         (c_ < med, '#2166ac', 'CP < median')):
        s = sel & ok
        if s.sum() < 2:
            continue
        m = g_[s].mean(0); e = g_[s].std(0) / np.sqrt(s.sum())
        ax[1].plot(lag, m, color=col, lw=2, label='%s (n=%d)' % (nm, s.sum()))
        ax[1].fill_between(lag, m - e, m + e, color=col, alpha=.2, lw=0)
    ax[1].set_xlabel(r'integration window $\tau$ (ms)')
    ax[1].set_xscale('log')          # their Figs. 3b and 4b use a log tau axis
    ax[1].set_xlim(BIN_MS, MAXLAG_MS)
    ax[1].set_ylabel(r'$r_{CCG}$')
    ax[1].set_title('median CP = %.2f' % med, fontsize=11)
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    fig.tight_layout()
    for e in ('png', 'svg'):
        fig.savefig(DATA_FOLDER + '%s.%s' % (fname, e), dpi=300,
                    bbox_inches='tight')
    return fig


if __name__ == '__main__':
    for kind, al, tag in (('mf', 1.0, 'MF'), ('fbp', 1.0, 'LBP')):
        print(tag, flush=True)
        res = run_all(kind, al)
        sfx = '%s_%s' % (tag, 'mixed' if MIXED_MU else 'same')
        plot_figure1(res, 'wasmuht_fig1_%s' % sfx)
        plot_figure2(res, 'wasmuht_fig2_%s' % sfx)
        plot_figure3(res, 'wasmuht_fig3_%s' % sfx)
        print('  rSC  ambiguous %.3f | unambiguous %.3f | random %.3f'
              % (np.nanmean(res['disp0.00']['rsc']),
                 np.nanmean(res['disp%.2f' % DISPARITIES[-1]]['rsc']),
                 np.nanmean(res['random']['rsc'])), flush=True)
        print('  CP   ambiguous %.3f | random %.3f'
              % (np.nanmean(res['disp0.00']['cp']),
                 np.nanmean(res['random']['cp'])), flush=True)
