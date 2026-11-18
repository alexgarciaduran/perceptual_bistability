# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:49:47 2026

@author: alexg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import rankdata, pearsonr
import posterior_computation_comparison as pc

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["axes.grid"] = False


"""
Reproduction of Wasmuht, Parker & Krug (2019), Sci Rep 9:11449,
"Interneuronal correlations at longer time scales predict decision signals
for bistable structure-from-motion perception", in the inference-network
model and with NO feedback connections.

Their three stimuli map onto two model knobs, the coupling J and the field B:

    random 2D motion      J < J*  (monostable), B = 0      green
    ambiguous cylinder    J > J*  (bistable),   B = 0      purple
    unambiguous cylinder  J > J*  (bistable),   B > 0      blue

So the regime itself is the stimulus manipulation: random motion carries no
structure-from-motion constraint between local motion signals, hence weak
coupling; the cylinder does, hence strong coupling; and disparity supplies the
disambiguating evidence B.

They record a single unit and multi-unit activity at the SAME electrode, so
both read one node, and their r_SC measures how much of each neuron's variance
is shared rate fluctuation rather than private Poisson noise.
"""

# ---GLOBAL VARIABLES
THETA = pc.THETA_NECKER
N = THETA.shape[0]
DATA_FOLDER = pc.DATA_FOLDER

C_AMB, C_UNAMB, C_RAND = '#6a3d9a', '#1f78b4', '#33a02c'  # their colours

TAU_MS, DT = 200.0, 0.05                 # tau = 200 ms, bin = dt*tau = 10 ms
BIN_MS = DT * TAU_MS
T_TRIAL = int(round(2000.0 / BIN_MS))    # 2 s trial
TAU_S = 1.0
MAXLAG_MS = 1000.0                       # their Figs. 3b and 4b reach 10^3 ms
N_CHOICE = 15                            # steps averaged for the reported percept
BIN_LONG = 100.0                         # independent bin of their Fig. 6
N_LAG = 4                                # +-50 to 450 ms, their r_SC^tau=4

# Tuning strength. Their sites differ in how strongly the cylinder modulates
# firing (F_cylinder), which correlates with both CP and r_SC, so each site
# draws its own readout gain. It scales only the RECORDED rate: the channel
# between units keeps GAIN, so the inference is untouched and the diversity is
# purely an encoding property. Without it every site is statistically
# identical, CP has no across-site variance, and their Fig. 6, which is an
# across-site correlation, has nothing to measure.
SIGMA_G = 0.30

# Encoding matched to their recordings. At their example site (Fig. 1a) the
# single unit fires near 65 spks/s and the multi-unit near 145, and their
# Methods set the Schmitt-trigger level so that multi-unit rates for the 0 deg
# cylinder lay between 10 and 200 spks/s. The multi-unit is therefore about two
# neurons' worth of spikes above threshold, not a large pool. GAIN is the
# half-depth of percept modulation: rate = BASE +- GAIN as the belief flips.
K, N_MU = 50, 2                          # pool size, neurons in the multi-unit
GAIN, BASE = 20.0, 65.0                  # single unit 45-85, multi-unit 90-170

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
# field of only ~0.06 at full disparity. A value of 10 is five times too steep
# and drives the unambiguous condition to zero correlation.
B_PER_DEG = 2.0

# How the multi-unit is composed. True: its N_MU neurons are scattered at
# random over ALL units, as a real electrode would pick up a mixture that need
# not share the single unit's tuning. False: they all come from the single
# unit's own pool. A GLOBAL signal (the percept) survives mixing, a LOCAL one
# (coupling between neighbours) is diluted by it, so comparing the two
# separates those two sources without needing a distance metric.
MIXED_MU = True


def simulate_trial(kind, alpha, j_mat, b_vec, rng, gain_site=None):
    """
    Simulate one 2 s trial of the spiking inference network.

    Parameters
    ----------
    kind : str
        'mf' for mean field, 'fbp' for fractional belief propagation.
    alpha : float
        Reciprocal (cavity) gain; 1 is loopy belief propagation.
    j_mat : array [N, N]
        Coupling matrix.
    b_vec : array [N]
        External field at each node.
    rng : numpy Generator
        Random source for the Poisson spiking.
    gain_site : array [N] or None
        Per-site readout gain. None gives every site the global GAIN.

    Returns
    -------
    pool : array [T_TRIAL, N]
        Spike counts of the whole pool at each node and time bin.
    percept : float
        Signed average of the global belief over the last N_CHOICE steps.
    """
    mask = j_mat != 0.0
    gain_site = np.full(N, GAIN) if gain_site is None else gain_site
    size = N if kind == 'mf' else int(mask.sum())
    scale = np.ones(N) if kind == 'mf' else np.abs(j_mat[mask])
    x = np.clip(0.05 * rng.standard_normal(size), -scale, scale)
    lam = DT * TAU_MS * 1e-3 * K
    synapse = GAIN * (x / scale) + BASE
    pool = np.empty((T_TRIAL, N), np.int64)
    sgn = np.empty(T_TRIAL)
    for t in range(T_TRIAL):
        x_hat = (synapse - BASE) / GAIN * scale
        b_now = (b_vec + SIG_IND * rng.standard_normal(N) / np.sqrt(DT)
                 + SIG_COM * rng.standard_normal() / np.sqrt(DT))
        q = pc._beliefs(kind, x_hat, j_mat, b_now, mask)
        f = pc._map_step(kind, x_hat, j_mat, b_now, alpha, 1.0, mask)
        x = x + DT * (f - x)
        node = x if kind == 'mf' else (2 * q - 1)       # recorded at the node
        rate = np.clip(BASE + gain_site * node, 0.0, BASE + gain_site)
        pool[t] = rng.poisson(rate * lam)
        sgn[t] = np.mean(q) - 0.5
        # the message/magnetisation units carry their own spiking channel
        unit_rate = np.clip(GAIN * (x / scale) + BASE, 0.0, GAIN + BASE)
        synapse = synapse + (DT / TAU_S) * (rng.poisson(unit_rate * lam) / lam
                                            - synapse)
    # the reported percept averages the last N_CHOICE steps, not the single
    # final sample, which is noisy
    return pool, float(np.mean(sgn[-N_CHOICE:]))


def cross_correlogram(a, b, max_lag):
    """Mean-subtracted spike-time cross-correlogram of two [trials, T] arrays."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    n_time = a.shape[1]
    ccg = np.zeros(2 * max_lag + 1)
    for k in range(-max_lag, max_lag + 1):
        i0, i1 = max(0, -k), n_time - max(0, k)
        ccg[k + max_lag] = np.mean(a[:, i0:i1] * b[:, i0 + k:i1 + k])
    return ccg


def roc_auc(a, b):
    """Area under the ROC curve separating samples a from samples b."""
    ranks = rankdata(np.r_[a, b])
    return (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def compose_electrode(pool, alloc, rng):
    """
    Draw the single-unit and multi-unit spike trains from the pool counts.

    Parameters
    ----------
    pool : array [T_TRIAL, N]
        Spike counts of the whole pool at each node.
    alloc : array [N, N]
        alloc[i, j] is how many of site i's multi-unit neurons sit in node j.
    rng : numpy Generator

    Returns
    -------
    su, mu : array [T_TRIAL, N]
        Counts of the single unit and of the multi-unit at each site. The
        single unit is one further neuron of node i, excluded from its own
        pool so that the two never share a cell.
    """
    su = np.stack([rng.binomial(pool[:, i], 1.0 / K) for i in range(N)], 1)
    mu = np.zeros_like(su)
    for i in range(N):
        for j in range(N):
            if alloc[i, j] == 0:
                continue
            available = pool[:, j] - (su[:, j] if j == i else 0)
            mu[:, i] += rng.binomial(np.maximum(available, 0),
                                     alloc[i, j] / (K - (1 if j == i else 0)))
    return su, mu


def rsc_matrix(su, mu):
    """
    Spike count correlation between every pair of long bins, for one site.

    Parameters
    ----------
    su, mu : array [trials, bins]
        Counts of the single unit and the multi-unit in independent bins.

    Returns
    -------
    array [bins, bins]
        Entry (i, j) is the correlation across trials between the single unit
        in bin i and the multi-unit in bin j.
    """
    a = su - su.mean(0, keepdims=True)
    b = mu - mu.mean(0, keepdims=True)
    sd_a, sd_b = su.std(0), mu.std(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (a.T @ b) / su.shape[0] / np.outer(sd_a, sd_b)
    out[sd_a <= 0, :] = np.nan
    out[:, sd_b <= 0] = np.nan
    return out


def rsc_components(mat, n_lag=N_LAG):
    """Instantaneous (the diagonal) and long-timescale (the off-diagonals)
    parts of the correlation matrix, as functions of trial time."""
    n = mat.shape[0]
    long = np.full(n, np.nan)
    for i in range(n):
        idx = [i + d for d in range(-n_lag, n_lag + 1)
               if d != 0 and 0 <= i + d < n]
        long[i] = np.nanmean(mat[i, idx])
    return np.diag(mat).copy(), long


def cp_timecourse(su, percept):
    """Choice probability per bin, from the single unit alone."""
    n = su.shape[1]
    out = np.full(n, np.nan)
    pos, neg = percept > 0, percept <= 0
    if pos.sum() > 5 and neg.sum() > 5:
        for i in range(n):
            if su[:, i].std() > 0:
                auc = roc_auc(su[pos, i], su[neg, i])
                out[i] = max(auc, 1 - auc)
    return out


def run_condition(kind, alpha, j_val, b_val, n_trials=120, n_nets=5, seed0=0,
                  sigma_J=0.25, sigma_B=0.05, mixed_mu=None):
    """
    Every site of every network realisation for one stimulus condition.

    Heterogeneous couplings and fields make the sites differ, which is what
    gives the spread across sites that their Figs. 2 and 3 plot.

    Returns
    -------
    dict with per-site r_SC and CP, the cumulative r_CCG of each site, and the
    trial-summed counts of one example site.
    """
    rsc, cp, ccg_all, example = [], [], [], None
    tau0_all, tau4_all, cp_time, gain_all = [], [], [], []
    per_bin = int(round(BIN_LONG / BIN_MS))
    n_long = T_TRIAL // per_bin
    n_lag = int(round(MAXLAG_MS / BIN_MS))
    weights = np.array([T_TRIAL - abs(k)
                        for k in range(-n_lag, n_lag + 1)], float)
    for r in range(n_nets):
        rng = np.random.default_rng(seed0 + 1000 * r)
        j_mat, b_vec = pc.ising_params_heterogeneous(
            THETA, j_val, b_val, sigma_J=sigma_J * j_val, sigma_B=sigma_B,
            seed=seed0 + r)
        # the electrode's composition is fixed for the site, not redrawn
        alloc = (rng.multinomial(N_MU, np.ones(N) / N, size=N)
                 if (MIXED_MU if mixed_mu is None else mixed_mu)
                 else np.eye(N, dtype=int) * N_MU)
        # tuning strength is a property of the site, drawn once
        gain_site = GAIN * np.clip(
            np.exp(SIGMA_G * rng.standard_normal(N)), 0.3, 2.2)
        su_all, mu_all, percept = [], [], []
        for _ in range(n_trials):
            pool, p = simulate_trial(kind, alpha, j_mat, b_vec, rng,
                                     gain_site)
            su, mu = compose_electrode(pool, alloc, rng)
            su_all.append(su)
            mu_all.append(mu)
            percept.append(p)
        su_all, mu_all = np.array(su_all), np.array(mu_all)
        percept = np.array(percept)
        count_su, count_mu = su_all.sum(1), mu_all.sum(1)
        # the same counts in the long bins of their Fig. 6
        su_long = su_all[:, :n_long * per_bin].reshape(
            n_trials, n_long, per_bin, N).sum(2)
        mu_long = mu_all[:, :n_long * per_bin].reshape(
            n_trials, n_long, per_bin, N).sum(2)
        for i in range(N):
            if count_su[:, i].std() <= 0 or count_mu[:, i].std() <= 0:
                continue
            rsc.append(np.corrcoef(count_su[:, i], count_mu[:, i])[0, 1])
            # Cov(sum_t a, sum_t b) = sum_k (T - |k|) ccg(k), and r_SC divides
            # by the sd of the trial-summed counts, so the full cumulative
            # area equals r_SC
            ccg = cross_correlogram(su_all[:, :, i], mu_all[:, :, i],
                                    n_lag) * weights
            ccg_all.append(np.array([ccg[n_lag - k:n_lag + k + 1].sum()
                                     for k in range(n_lag + 1)])
                           / (count_su[:, i].std() * count_mu[:, i].std()))
            count_pref = count_su[percept > 0, i]
            count_null = count_su[percept <= 0, i]
            if len(count_pref) > 5 and len(count_null) > 5:
                auc = roc_auc(count_pref, count_null)
                cp.append(max(auc, 1 - auc))
            else:
                cp.append(np.nan)
            inst, long = rsc_components(
                rsc_matrix(su_long[:, :, i], mu_long[:, :, i]))
            tau0_all.append(inst)
            tau4_all.append(long)
            cp_time.append(cp_timecourse(su_long[:, :, i], percept))
            gain_all.append(gain_site[i])
            if example is None and r == 0 and i == 0:
                example = (count_su[:, i].copy(), count_mu[:, i].copy())
    return dict(rsc=np.array(rsc), cp=np.array(cp),
                ccg=np.array(ccg_all), example=example,
                tau0=np.array(tau0_all), tau4=np.array(tau4_all),
                cp_time=np.array(cp_time), gain=np.array(gain_all))


def save_conditions(filename, res):
    """
    Write every condition to one npz as flat arrays.

    Storing the dicts as dtype=object would pickle them, and a pickle written
    by numpy 2.x cannot be read by numpy 1.x (it refers to numpy._core), so
    the cache would only load in the environment that produced it.
    """
    flat = {}
    for condition, out in res.items():
        for key in ('rsc', 'cp', 'ccg', 'tau0', 'tau4', 'cp_time',
                    'gain'):
            flat['%s__%s' % (condition, key)] = np.asarray(out[key])
        if out['example'] is not None:
            flat['%s__su' % condition] = np.asarray(out['example'][0])
            flat['%s__mu' % condition] = np.asarray(out['example'][1])
    np.savez(filename, **flat)


def load_conditions(filename):
    """Read back what save_conditions wrote, without unpickling anything."""
    data = np.load(filename, allow_pickle=False)
    res = {}
    for name in data.files:
        condition, key = name.split('__')
        res.setdefault(condition, {})[key] = data[name]
    for condition, out in res.items():
        example = ((out.pop('su'), out.pop('mu'))
                   if 'su' in out else None)
        out['example'] = example
    return res


def run_all(kind='mf', alpha=1.0, n_trials=120, n_nets=5, recompute=False,
            folder=None, verbose=True, mixed_mu=None):
    """Every stimulus condition for one algorithm, cached to disk."""
    folder = DATA_FOLDER if folder is None else folder
    mixed = MIXED_MU if mixed_mu is None else mixed_mu
    filename = folder + 'wasmuht_%s_a%g_T%d_N%d_%s.npz' % (
        kind, alpha, n_trials, n_nets, 'mix' if mixed else 'same')
    if not recompute and os.path.exists(filename):
        return load_conditions(filename)
    res = {}
    jstar = JSTAR[kind] if kind in ('mf', 'fbp') else 1/(2*alpha)*np.log(3/(3-2*alpha))
    res['random'] = run_condition(kind, alpha, J_MONO * jstar, 0.0,
                                  n_trials, n_nets, mixed_mu=mixed)
    if verbose:
        print('  random motion done', flush=True)
    for disparity in DISPARITIES:
        res['disp%.2f' % disparity] = run_condition(
            kind, alpha, J_BI * JSTAR[kind], B_PER_DEG * disparity,
            n_trials, n_nets, mixed_mu=mixed)
        if verbose:
            print('  disparity %.2f done' % disparity, flush=True)
    save_conditions(filename, res)
    return res


def save_figure(fig, name):
    """Save in both formats, as everywhere else in the project."""
    for extension in ('png', 'pdf'):
        fig.savefig(DATA_FOLDER + '%s.%s' % (name, extension), dpi=400,
                    bbox_inches='tight')


def plot_figure_1(res, name='wasmuht_fig1'):
    """Their Fig. 1: one example site, ambiguous cylinder versus random motion."""
    fig, ax = plt.subplots(ncols=3, figsize=(12, 3.4))
    for k, (label, colour) in enumerate((('disp0.00', C_AMB),
                                         ('random', C_RAND))):
        count_su, count_mu = res[label]['example']
        z_su = (count_su - count_su.mean()) / count_su.std()
        z_mu = (count_mu - count_mu.mean()) / count_mu.std()
        trials = np.arange(len(z_su))
        running = lambda v: np.convolve(v, np.ones(11) / 11, 'same')
        ax[k].plot(trials, z_su, color=colour, lw=.6, alpha=.5)
        ax[k].plot(trials, z_mu, color='0.4', lw=.6, alpha=.5)
        ax[k].plot(trials, running(z_su), color=colour, lw=2,
                   label='single unit')
        ax[k].plot(trials, running(z_mu), color='0.2', lw=2,
                   label='multi unit')
        ax[k].set_title('%s, $r_{SC}$ = %.2f'
                        % ('ambiguous cylinder' if k == 0 else 'random motion',
                           np.corrcoef(count_su, count_mu)[0, 1]), fontsize=14)
        ax[k].set_xlabel('trial')
        ax[k].set_ylabel('z-scored spike count')
        ax[k].legend(frameon=False)
    ax[2].hist([res['disp0.00']['rsc'], res['random']['rsc']], bins=12,
               color=[C_AMB, C_RAND], label=['ambiguous', 'random motion'])
    ax[2].set_xlabel(r'$r_{SC}$')
    ax[2].set_ylabel('sites')
    ax[2].legend(frameon=False)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def plot_figure_2(res, name='wasmuht_fig2'):
    """Their Fig. 2: r_SC against disparity (violins), and r_CCG against tau."""
    fig, ax = plt.subplots(ncols=2, figsize=(11, 3.8))
    data = [res['random']['rsc']] + [res['disp%.2f' % d]['rsc']
                                     for d in DISPARITIES]
    # categorical positions: the disparities are 0.01 apart, so plotting them
    # on a linear axis would pile every violin on top of the next
    positions = list(range(len(data)))
    violins = ax[0].violinplot(data, positions=positions, widths=.75,
                               showmeans=True)
    for body, colour in zip(violins['bodies'],
                            [C_RAND, C_AMB, C_UNAMB, C_UNAMB, C_UNAMB]):
        body.set_facecolor(colour)
        body.set_alpha(.65)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        violins[key].set_color('0.3')
        violins[key].set_linewidth(1.1)
    ax[0].set_xticks(positions)
    ax[0].set_xticklabels(['random\nmotion']
                          + ['%.2f' % d for d in DISPARITIES])
    ax[0].set_xlabel('disparity (deg)')
    ax[0].set_ylabel(r'$r_{SC}$')
    ax[0].axhline(0, color='0.7', lw=.8)
    lag = np.arange(res['random']['ccg'].shape[1]) * BIN_MS
    for label, colour, legend in (('disp0.00', C_AMB, 'ambiguous cylinder'),
                                  ('disp%.2f' % DISPARITIES[-1], C_UNAMB,
                                   'unambiguous cylinder'),
                                  ('random', C_RAND, 'random motion')):
        ccg = res[label]['ccg']
        mean = ccg.mean(0)
        err = ccg.std(0) / np.sqrt(len(ccg))
        ax[1].plot(lag, mean, color=colour, lw=2, label=legend)
        ax[1].fill_between(lag, mean - err, mean + err, color=colour,
                           alpha=.2, lw=0)
    ax[1].set_xlabel(r'integration window $\tau$ (ms)')
    ax[1].set_xscale('log')          # their Figs. 3b and 4b use a log tau axis
    ax[1].set_xlim(BIN_MS, MAXLAG_MS)
    ax[1].set_ylabel(r'$r_{CCG}$')
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def plot_figure_3(res, name='wasmuht_fig3'):
    """Their Figs. 4a and 4b: CP against r_SC, and r_CCG split at the median CP."""
    fig, ax = plt.subplots(ncols=2, figsize=(11, 3.8))
    rsc, cp, ccg = (res['disp0.00']['rsc'], res['disp0.00']['cp'],
                    res['disp0.00']['ccg'])
    ok = np.isfinite(cp) & np.isfinite(rsc)
    ax[0].scatter(rsc[ok], cp[ok], s=26, color=C_AMB, alpha=.75)
    if ok.sum() > 3:
        r_val, p_val = pearsonr(rsc[ok], cp[ok])
        fit = np.polyfit(rsc[ok], cp[ok], 1)
        grid = np.linspace(rsc[ok].min(), rsc[ok].max(), 20)
        ax[0].plot(grid, np.polyval(fit, grid), color='0.3', lw=1.4)
        ax[0].set_title('r = %.2f, p = %.3f, n = %d'
                        % (r_val, p_val, ok.sum()), fontsize=14)
    ax[0].axhline(0.5, color='0.7', lw=.8)
    ax[0].set_xlabel(r'$r_{SC}$')
    ax[0].set_ylabel('choice probability')
    cp_median = np.nanmedian(cp[ok])
    lag = np.arange(ccg.shape[1]) * BIN_MS
    for group, colour, legend in ((cp >= cp_median, '#b2182b',
                                   'CP $\\geq$ median'),
                                  (cp < cp_median, '#2166ac', 'CP < median')):
        group = group & ok
        if group.sum() < 2:
            continue
        mean = ccg[group].mean(0)
        err = ccg[group].std(0) / np.sqrt(group.sum())
        ax[1].plot(lag, mean, color=colour, lw=2,
                   label='%s (n=%d)' % (legend, group.sum()))
        ax[1].fill_between(lag, mean - err, mean + err, color=colour,
                           alpha=.2, lw=0)
    ax[1].set_xlabel(r'integration window $\tau$ (ms)')
    ax[1].set_xscale('log')
    ax[1].set_xlim(BIN_MS, MAXLAG_MS)
    ax[1].set_ylabel(r'$r_{CCG}$')
    ax[1].set_title('median CP = %.2f' % cp_median, fontsize=14)
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def plot_figure_4(res, name='wasmuht_fig4'):
    """Their Fig. 6: does the long-timescale correlation predict CP?

    For each site the long-lag correlation is fitted with a parabola in trial
    time; the sign of its derivative says whether that correlation is rising
    or decaying. Their result is a dissociation: a rising long-lag correlation
    in the first half goes with a larger CP in the second, while the
    instantaneous correlation predicts nothing.
    """
    out = res['disp0.00']
    tau0, tau4, cps = out['tau0'], out['tau4'], out['cp_time']
    n_long = tau4.shape[1]
    t = (np.arange(n_long) + .5) * BIN_LONG / 1000.0
    half = n_long // 2
    fig, ax = plt.subplots(ncols=3, figsize=(15, 3.9))

    ax[0].plot(t, np.nanmean(tau0, 0), color='0.4', lw=2,
               label='instantaneous')
    ax[0].plot(t, np.nanmean(tau4, 0), color=C_AMB, lw=2, label='long lag')
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel(r'$r_{SC}$')
    ax[0].legend(frameon=False)
    ax[0].set_title('the two components', fontsize=14)

    for curves, colour, lab in ((tau4, C_AMB, 'long lag'),
                                (tau0, '0.4', 'instantaneous')):
        slope = []
        for c in curves:
            ok = np.isfinite(c)
            if ok.sum() < 5:
                slope.append(np.nan)
                continue
            a, b, _ = np.polyfit(t[ok], c[ok], 2)
            slope.append(np.mean(2 * a * t[:half] + b))
        slope = np.array(slope)
        late = np.nanmean(cps[:, half:], 1)
        ok = np.isfinite(slope) & np.isfinite(late)
        r_val, p_val = pearsonr(slope[ok], late[ok])
        ax[1].scatter(slope[ok], late[ok], s=26, color=colour, alpha=.75,
                      label='%s: r = %+.2f, p = %.3f' % (lab, r_val, p_val))
        rising = ok & (slope > 0)
        decaying = ok & (slope <= 0)
        for sel, style in ((rising, '-'), (decaying, '--')):
            if sel.sum() > 1:
                ax[2].plot(t, np.nanmean(cps[sel], 0), style, color=colour,
                           lw=2, label='%s, slope %s (n=%d)'
                           % (lab, '>0' if style == '-' else '<0', sel.sum()))
    ax[1].set_xlabel('slope of the correlation, first half')
    ax[1].set_ylabel('CP, second half')
    ax[1].legend(frameon=False)
    ax[1].set_title('does the slope predict CP?', fontsize=14)
    ax[2].axhline(0.5, color='0.7', lw=.8)
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('CP')
    ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_title('CP split by the sign of the slope', fontsize=14)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


if __name__ == '__main__':
    for kind, alpha, tag in (('mf', 1.0, 'MF'), ('fbp', 1.0, 'LBP'),
                             ('fbp', 0.5, 'FBP_05'), ('fbp', 1.45, 'FBP_145')):
        print(tag, flush=True)
        res = run_all(kind, alpha, recompute=True)
        suffix = '%s_%s' % (tag, 'mixed' if MIXED_MU else 'same')
        plot_figure_1(res, 'wasmuht_fig1_%s' % suffix)
        plot_figure_2(res, 'wasmuht_fig2_%s' % suffix)
        plot_figure_3(res, 'wasmuht_fig3_%s' % suffix)
        plot_figure_4(res, 'wasmuht_fig4_%s' % suffix)
        print('  rSC  ambiguous %.3f | unambiguous %.3f | random %.3f'
              % (np.nanmean(res['disp0.00']['rsc']),
                 np.nanmean(res['disp%.2f' % DISPARITIES[-1]]['rsc']),
                 np.nanmean(res['random']['rsc'])), flush=True)
        print('  CP   ambiguous %.3f | random %.3f'
              % (np.nanmean(res['disp0.00']['cp']),
                 np.nanmean(res['random']['cp'])), flush=True)
