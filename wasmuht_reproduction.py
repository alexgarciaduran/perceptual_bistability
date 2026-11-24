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

# sampled from their figures: magenta for the ambiguous cylinder, grey
# for the unambiguous one, green for random motion
C_AMB, C_UNAMB, C_RAND = '#a21d66', '#7f7f7f', '#047d37'
C_REF = '#444444'                        # markers for their values

TAU_MS, DT = 200.0, 0.05                 # tau = 200 ms, bin = dt*tau = 10 ms
BIN_MS = DT * TAU_MS
T_TRIAL = int(round(2000.0 / BIN_MS))    # 2 s trial
TAU_S = 1.0
MAXLAG_MS = 1000.0
# their Figs. 3b and 4b draw the integration window from 1 ms, although the
# first point any binned correlogram can carry is one bin wide
TAU_MIN = 1.0                       # their Figs. 3b and 4b reach 10^3 ms
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
SIGMA_B = 0.05                           # field heterogeneity, set by the circuit

# Their analysis choices, followed here. Trials whose count is further than
# N_OUTLIER sd from the site's mean are dropped; the rest are z-scored against
# the N_ZWIN trials on either side, which removes slow drift in excitability;
# and r_SC values are averaged through Fisher's z. Only sites whose single
# unit and multi-unit are both strongly tuned, and similarly so, are analysed,
# as they select for strong and matching tuning.
N_OUTLIER, N_ZWIN = 3.0, 5
TUNE_MIN, TUNE_MATCH = 0.5, 1.5

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

# Shared input to the neurons recorded at one site, in spikes/s. It modulates
# their firing rate but never enters the inference, so it correlates the two
# recordings without carrying any percept: the feed-forward common input that
# their control conditions are usually explained by. TAU_FAST is its
# correlation time in units of TAU_MS, so 0.15 is 30 ms.
SIG_FAST, TAU_FAST = 0.0, 0.15

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
MIXED_MU = False

# Whether the network state is carried from one trial to the next instead of
# being reset. A perceptual system inherits its state at stimulus onset, and
# the reset is what makes the correlation climb through the trial, but it also
# introduces drift across trials and lowers the correlations the circuit was
# fitted to, so it is off by default.
CARRY_STATE = False

# Their published values, for reference in the figures.
WASMUHT = dict(rsc_amb=0.42, rsc_unamb=0.28, rsc_rand=0.23, cp_amb=0.69,
               cp_rand=0.56, r_cp_rsc=0.43, cp_high=0.48, cp_low=0.30,
               t90_amb=400.0, t90_rand=32.0)

# Named circuits. Each fixes everything the four inference schemes share; the
# couplings are ratios of that scheme's own J*(alpha), so the stimulus sets
# the regime for all of them. 'best' is the circuit on which a single scheme
# fits as well as it can, and is the one quoted; 'total' is the circuit
# fairest to all four at once, used to rank the schemes against each other.
CIRCUITS = {
    'best':  dict(gain=20.2, K=3, tau_s=0.20, r_mono=0.923, r_bi=2.15,
                  bpd=7.55, sigma_B=0.05),
    'total': dict(gain=28.0, K=3, tau_s=0.35, r_mono=0.793, r_bi=1.18,
                  bpd=1.45, sigma_B=0.05),
    # refitted with the pool size held at 20. A larger pool is quieter, so the
    # coupling must sit nearer threshold for the network to switch at all, and
    # the disparity effect and the Fig. 6 dissociation both weaken.
    'k20':   dict(gain=20.0, K=20, tau_s=0.20, r_mono=0.840, r_bi=1.43,
                  bpd=2.33, sigma_B=0.15),
    'k20_total': dict(gain=20.7, K=20, tau_s=0.50, r_mono=0.933, r_bi=1.28,
                      bpd=1.29, sigma_B=0.15),
    # the only circuit that reaches all four of their reported values at once,
    # found with the multi-unit drawn from the single unit's own pool. Those
    # four are all measured at zero disparity and so say nothing about bpd,
    # which is set separately by their unambiguous cylinder.
    'match': dict(gain=27.6, K=3, tau_s=0.20, r_mono=0.78, r_bi=1.14,
                  bpd=4.33, sigma_B=0.15),
}
CIRCUIT = 'match'                         # the one currently loaded


def critical_coupling(kind, alpha=1.0, neighs=3.0):
    """
    Coupling at which the scheme acquires a second fixed point.

    For mean field this is 1 / lambda_max; for fractional belief propagation
    it is log(neighs / (neighs - 2 alpha)) / (2 alpha), which diverges at
    alpha = neighs / 2, so no coupling makes those schemes bistable.
    """
    if kind == 'mf':
        return 1.0 / neighs
    return (np.inf if 2 * alpha >= neighs
            else float(np.log(neighs / (neighs - 2 * alpha)) / (2 * alpha)))


def use_circuit(circuit='best', kind='fbp', alpha=1.0):
    """
    Point the module at one circuit and one inference scheme.

    Pass a name from CIRCUITS or a dictionary of the same shape. The coupling
    ratios are resolved through this scheme's own critical coupling, so the
    same circuit puts every scheme at a matched distance from its bifurcation
    rather than at the same absolute coupling.
    """
    global GAIN, K, TAU_S, J_BI, J_MONO, B_PER_DEG, SIGMA_B, CIRCUIT
    c = CIRCUITS[circuit] if isinstance(circuit, str) else dict(circuit)
    GAIN, K, TAU_S = c['gain'], int(c['K']), c['tau_s']
    J_BI, J_MONO, B_PER_DEG = c['r_bi'], c['r_mono'], c['bpd']
    SIGMA_B = c.get('sigma_B', 0.05)
    CIRCUIT = circuit if isinstance(circuit, str) else 'custom'
    JSTAR[kind] = critical_coupling(kind, alpha)
    return c


def simulate_trial(kind, alpha, j_mat, b_vec, rng, gain_site=None,
                   state=None):
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
    state : tuple or None
        The previous trial's end state, or None to start from rest.

    Returns
    -------
    pool : array [T_TRIAL, N]
        Spike counts of the whole pool at each node and time bin.
    percept : float
        Signed average of the global belief over the last N_CHOICE steps.
    state : tuple
        The unit states and synaptic traces at the end of the trial, to be
        passed back in as `state` when trials are run without a reset.
    """
    mask = j_mat != 0.0
    gain_site = np.full(N, GAIN) if gain_site is None else gain_site
    size = N if kind == 'mf' else int(mask.sum())
    scale = np.ones(N) if kind == 'mf' else np.abs(j_mat[mask])
    if state is None:
        x = np.clip(0.05 * rng.standard_normal(size), -scale, scale)
        synapse = GAIN * (x / scale) + BASE
    else:
        x, synapse = state[0].copy(), state[1].copy()
    lam = DT * TAU_MS * 1e-3 * K
    # stationary Ornstein-Uhlenbeck, one per node, shared by every neuron
    # recorded there and independent of the network state
    fast = rng.standard_normal(N)
    decay = DT / TAU_FAST
    kick = np.sqrt(2 * decay)
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
        fast = fast - decay * fast + kick * rng.standard_normal(N)
        rate = np.clip(BASE + gain_site * node + SIG_FAST * fast, 0.0, None)
        pool[t] = rng.poisson(rate * lam)
        sgn[t] = np.mean(q) - 0.5
        # the message/magnetisation units carry their own spiking channel
        unit_rate = np.clip(GAIN * (x / scale) + BASE, 0.0, GAIN + BASE)
        synapse = synapse + (DT / TAU_S) * (rng.poisson(unit_rate * lam) / lam
                                            - synapse)
    # the reported percept averages the last N_CHOICE steps, not the single
    # final sample, which is noisy
    return pool, float(np.mean(sgn[-N_CHOICE:])), (x, synapse)


def cross_correlogram(a, b, max_lag):
    """
    Mean-subtracted correlogram of two [trials, T] count arrays, per time bin.

    Subtracting the across-trial mean at each bin removes the stimulus-locked
    response, which is what their shift predictor does. Passing the same array
    twice gives the auto-correlogram. Computed by FFT, which matters once the
    bin is small enough that the lag range runs to thousands of bins.
    """
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    n_trial, n_time = a.shape
    n_fft = 1 << int(np.ceil(np.log2(2 * n_time)))
    product = (np.conj(np.fft.rfft(a, n_fft, axis=1))
               * np.fft.rfft(b, n_fft, axis=1))
    full = np.fft.irfft(product, n_fft, axis=1).sum(0)
    lags = np.arange(-max_lag, max_lag + 1)
    return full[lags % n_fft] / (n_trial * (n_time - np.abs(lags)))


def fisher_mean(values):
    """Average correlations through Fisher's z, as all their statistics do."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return np.tanh(np.mean(np.arctanh(np.clip(v, -0.999999, 0.999999)))) \
        if v.size else np.nan


def zscore_trials(counts, n_side=N_ZWIN):
    """
    Normalise each trial against its neighbours, their equation (2).

    The mean and standard deviation come from the n_side trials before and
    after the current one, so slow drift in excitability does not inflate the
    correlation. The window shrinks at the ends of the session.
    """
    x = np.asarray(counts, float)
    n = x.size
    k = np.arange(n)
    lo = np.maximum(0, k - n_side)
    hi = np.minimum(n, k + n_side + 1)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    cum2 = np.concatenate([[0.0], np.cumsum(x * x)])
    # window totals with the trial itself taken out
    total = cum[hi] - cum[lo] - x
    total2 = cum2[hi] - cum2[lo] - x * x
    count = (hi - lo - 1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = total / count
        var = total2 / count - mean ** 2
        sd = np.sqrt(np.maximum(var, 0.0))
        out = np.where(sd > 0, (x - mean) / sd, np.nan)
    return out


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
    rsc, cp, ccg_all, rccg_all, example = [], [], [], [], None
    tau0_all, tau4_all, cp_time, gain_all, mat_all = [], [], [], [], []
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
        su_all, mu_all, percept, state = [], [], [], None
        for _ in range(n_trials):
            pool, p, end = simulate_trial(kind, alpha, j_mat, b_vec, rng,
                                          gain_site, state)
            state = end if CARRY_STATE else None
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
        # their selection: both units strongly and similarly tuned. The
        # multi-unit's tuning is the mean over the nodes its neurons occupy.
        mu_gain = alloc @ gain_site / max(N_MU, 1)
        for i in range(N):
            if count_su[:, i].std() <= 0 or count_mu[:, i].std() <= 0:
                continue
            ratio = gain_site[i] / mu_gain[i] if mu_gain[i] > 0 else np.inf
            if (gain_site[i] < TUNE_MIN * GAIN
                    or mu_gain[i] < TUNE_MIN * GAIN
                    or not 1 / TUNE_MATCH <= ratio <= TUNE_MATCH):
                continue
            # drop outlying trials, then z-score against neighbouring trials
            keep = ((np.abs(count_su[:, i] - count_su[:, i].mean())
                     <= N_OUTLIER * count_su[:, i].std())
                    & (np.abs(count_mu[:, i] - count_mu[:, i].mean())
                       <= N_OUTLIER * count_mu[:, i].std()))
            if keep.sum() < 20:
                continue
            z_su = zscore_trials(count_su[keep, i])
            z_mu = zscore_trials(count_mu[keep, i])
            good = np.isfinite(z_su) & np.isfinite(z_mu)
            if good.sum() < 20 or z_su[good].std() <= 0:
                continue
            rsc.append(np.corrcoef(z_su[good], z_mu[good])[0, 1])
            # Cov(sum_t a, sum_t b) = sum_k (T - |k|) ccg(k), and r_SC divides
            # by the sd of the trial-summed counts, so the full cumulative
            # area equals r_SC
            train_su, train_mu = su_all[keep, :, i], mu_all[keep, :, i]
            c_xy = cross_correlogram(train_su, train_mu, n_lag)
            # our cumulative form: the share of the count covariance within
            # +-tau, so the full area is r_SC by construction
            ccg = c_xy * weights
            ccg_all.append(np.array([ccg[n_lag - k:n_lag + k + 1].sum()
                                     for k in range(n_lag + 1)])
                           / (count_su[keep, i].std()
                              * count_mu[keep, i].std()))
            # their form, equation (7): the cross-correlogram integral over
            # the geometric mean of the two auto-correlogram integrals taken
            # over the same window, so the denominator grows with tau. The
            # rate normalisation they apply cancels between the two.
            c_xx = cross_correlogram(train_su, train_su, n_lag)
            c_yy = cross_correlogram(train_mu, train_mu, n_lag)
            cum = lambda c: np.array([c[n_lag - k:n_lag + k + 1].sum()
                                      for k in range(n_lag + 1)])
            den = np.sqrt(np.clip(cum(c_xx) * cum(c_yy), 1e-30, None))
            rccg_all.append(cum(c_xy) / den)
            count_pref = count_su[keep, i][percept[keep] > 0]
            count_null = count_su[keep, i][percept[keep] <= 0]
            if len(count_pref) > 5 and len(count_null) > 5:
                auc = roc_auc(count_pref, count_null)
                cp.append(max(auc, 1 - auc))
            else:
                cp.append(np.nan)
            mat = rsc_matrix(su_long[keep, :, i], mu_long[keep, :, i])
            inst, long = rsc_components(mat)
            mat_all.append(mat)
            tau0_all.append(inst)
            tau4_all.append(long)
            cp_time.append(cp_timecourse(su_long[keep, :, i], percept[keep]))
            gain_all.append(gain_site[i])
            # the first site that passes selection, not necessarily the first
            if example is None:
                example = (count_su[:, i].copy(), count_mu[:, i].copy())
    return dict(rsc=np.array(rsc), cp=np.array(cp),
                ccg=np.array(ccg_all), rccg=np.array(rccg_all),
                example=example,
                tau0=np.array(tau0_all), tau4=np.array(tau4_all),
                cp_time=np.array(cp_time), gain=np.array(gain_all),
                rsc_mat=np.array(mat_all))


def save_conditions(filename, res):
    """
    Write every condition to one npz as flat arrays.

    Storing the dicts as dtype=object would pickle them, and a pickle written
    by numpy 2.x cannot be read by numpy 1.x (it refers to numpy._core), so
    the cache would only load in the environment that produced it.
    """
    flat = {}
    for condition, out in res.items():
        for key in ('rsc', 'cp', 'ccg', 'rccg', 'tau0', 'tau4',
                    'cp_time', 'gain', 'rsc_mat'):
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


def bin_width(out):
    """
    The bin a saved condition was run at, read back from the data.

    The correlograms carry MAXLAG_MS / BIN_MS + 1 points, so the bin follows
    from their length. Taking it from the data rather than from the module
    keeps a figure honest when a cache is loaded under other settings.

    Parameters
    ----------
    out : dict
        One condition, as run_condition returns it.

    Returns
    -------
    float
        Bin width in ms.
    """
    return MAXLAG_MS / (np.asarray(out['rccg']).shape[-1] - 1)


def run_all(kind='mf', alpha=1.0, n_trials=120, n_nets=5, recompute=False,
            folder=None, verbose=True, mixed_mu=None, bin_ms=None):
    """
    Every stimulus condition for one algorithm, cached to disk.

    bin_ms selects the run: it sets the step, the bin and the trial length
    together, so one argument picks between a coarse and a fine simulation
    instead of three globals that have to agree.
    """
    global DT, BIN_MS, T_TRIAL
    if bin_ms is not None:
        BIN_MS = float(bin_ms)
        DT = BIN_MS / TAU_MS
        T_TRIAL = int(round(2000.0 / BIN_MS))
    folder = DATA_FOLDER if folder is None else folder
    mixed = MIXED_MU if mixed_mu is None else mixed_mu
    # the circuit belongs in the name: the same scheme on two circuits is
    # two different simulations and must not share a cache entry
    # the bin width belongs in the name too: a finer step is a different
    # simulation, and its correlograms reach lags the coarse one cannot
    fast_tag = '' if SIG_FAST == 0 else '_f%g_tf%g' % (SIG_FAST, TAU_FAST)
    filename = folder + 'wasmuht_%s_a%g_T%d_N%d_%s_b%g_bin%g%s_%s_%s.npz' % (
        kind, alpha, n_trials, n_nets, CIRCUIT, B_PER_DEG, BIN_MS, fast_tag,
        'carry' if CARRY_STATE else 'reset', 'mix' if mixed else 'same')
    if not recompute and os.path.exists(filename):
        cached = load_conditions(filename)
        got = bin_width(cached['random'])
        if abs(got - BIN_MS) < 1e-9:
            return cached
        print('  cache holds %g ms bins, %g ms asked for: recomputing'
              % (got, BIN_MS), flush=True)
    res = {}
    jstar = JSTAR[kind] if kind in ('mf', 'fbp') else 1/(2*alpha)*np.log(3/(3-2*alpha))
    res['random'] = run_condition(kind, alpha, J_MONO * jstar, 0.0,
                                  n_trials, n_nets, sigma_B=SIGMA_B,
                                  mixed_mu=mixed)
    if verbose:
        print('  random motion done', flush=True)
    for disparity in DISPARITIES:
        res['disp%.2f' % disparity] = run_condition(
            kind, alpha, J_BI * jstar, B_PER_DEG * disparity,
            n_trials, n_nets, sigma_B=SIGMA_B, mixed_mu=mixed)
        if verbose:
            print('  disparity %.2f done' % disparity, flush=True)
    save_conditions(filename, res)
    return res


def save_sweep(filename, sweep):
    """
    Write a coupling sweep to one npz as flat arrays.

    Same reason as save_conditions: the sweep is a list of dicts holding a
    nested dict, which numpy would only store by pickling, so it is flattened
    into one array per field and the disparities are kept as three parallel
    arrays rather than as keys.
    """
    flat = {}
    for n, entry in enumerate(sweep):
        flat['s%d__ratio' % n] = np.asarray(entry['ratio'])
        flat['s%d__circuit_ratio' % n] = np.asarray(
            entry.get('circuit_ratio', np.nan))
        for key in ('rsc', 'cp', 'rccg'):
            flat['s%d__%s' % (n, key)] = np.asarray(entry[key])
        disparities = sorted(entry['by_disp'])
        flat['s%d__disp' % n] = np.asarray(disparities)
        flat['s%d__disp_rsc' % n] = np.asarray(
            [entry['by_disp'][d]['rsc'] for d in disparities])
        flat['s%d__disp_cp' % n] = np.asarray(
            [entry['by_disp'][d]['cp'] for d in disparities])
        for k, d in enumerate(disparities):
            flat['s%d__sites%d' % (n, k)] = np.asarray(
                entry['by_disp'][d]['rsc_sites'])
    np.savez(filename, **flat)


def load_sweep(filename):
    """Read back what save_sweep wrote, without unpickling anything."""
    data = np.load(filename, allow_pickle=False)
    entries = {}
    for name in data.files:
        index, key = name.split('__')
        entries.setdefault(int(index[1:]), {})[key] = data[name]
    sweep = []
    for n in sorted(entries):
        out = entries[n]
        by_disp = {float(d): dict(rsc=float(r), cp=float(c),
                                  rsc_sites=out['sites%d' % k])
                   for k, (d, r, c) in enumerate(zip(out['disp'],
                                                     out['disp_rsc'],
                                                     out['disp_cp']))}
        sweep.append(dict(ratio=float(out['ratio']),
                          circuit_ratio=float(out.get('circuit_ratio',
                                                      np.nan)),
                          rsc=out['rsc'],
                          cp=out['cp'], rccg=out['rccg'], by_disp=by_disp))
    return sweep


def save_figure(fig, name):
    """Save in both formats, as everywhere else in the project."""
    for extension in ('png', 'pdf', 'svg'):
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
    for key, colour in (('rsc_amb', C_AMB), ('rsc_rand', C_RAND)):
        ax[2].axvline(WASMUHT[key], color=colour, ls='--', lw=1.8)
    ax[2].plot([], [], ls='--', color=C_REF, label='Wasmuht et al.')
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
    fig, ax = plt.subplots(ncols=2, figsize=(9, 3.8))
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
    for pos, key, col in ((0, 'rsc_rand', C_RAND), (1, 'rsc_amb', C_AMB),
                          (4, 'rsc_unamb', C_UNAMB)):
        ax[0].plot(pos, WASMUHT[key], marker='_', ms=26, mew=3, color=col,
                   zorder=6)
    ax[0].plot([], [], marker='_', ms=14, mew=3, color='0.3', ls='none',
               label='Wasmuht et al.')
    ax[0].legend(frameon=False, loc='lower left')
    lag = (np.arange(res['random']['ccg'].shape[1])
           * bin_width(res['random']))
    for label, colour, legend in (('disp0.00', C_AMB, 'ambiguous cylinder'),
                                  ('disp%.2f' % DISPARITIES[-1], C_UNAMB,
                                   'unambiguous cylinder'),
                                  ('random', C_RAND, 'random motion')):
        ccg = res[label]['rccg']        # their equation (7)
        mean = ccg.mean(0)
        err = ccg.std(0) / np.sqrt(len(ccg))
        ax[1].plot(lag, mean, color=colour, lw=2, label=legend)
        ax[1].fill_between(lag, mean - err, mean + err, color=colour,
                           alpha=.2, lw=0)
    ax[1].set_xlabel(r'integration window $\tau$ (ms)')
    ax[1].set_xscale('log')          # their Figs. 3b and 4b use a log tau axis
    ax[1].set_xlim(TAU_MIN, MAXLAG_MS)
    ax[1].set_ylabel(r'$r_{CCG}$')
    # where their curves sit: the plateau lag of each, at 90% of its r_SC
    for key, lagkey, col in (('rsc_amb', 't90_amb', C_AMB),
                             ('rsc_rand', 't90_rand', C_RAND)):
        ax[1].plot(WASMUHT[lagkey], .9 * WASMUHT[key], marker='*', ms=16,
                   color=col, mec='k', mew=.6, ls='none', zorder=6)
    ax[1].plot([], [], marker='*', ms=12, color='0.3', mec='k', mew=.6,
               ls='none', label='Wasmuht et al.')
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def plot_figure_3(res, name='wasmuht_fig3'):
    """Their Figs. 4a and 4b: CP against r_SC, and r_CCG split at the median CP."""
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3.8))
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
    ax[0].plot(WASMUHT['rsc_amb'], WASMUHT['cp_amb'], marker='*', ms=18,
               color='crimson', mec='k', mew=.6, ls='none', zorder=6,
               label='Wasmuht et al.')
    ax[0].legend(frameon=False, loc='lower right')
    ax[0].set_xlabel(r'$r_{SC}$')
    ax[0].set_ylabel('choice probability')
    cp_median = np.nanmedian(cp[ok])
    lag = np.arange(ccg.shape[1]) * bin_width(res['disp0.00'])
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
    ax[1].set_xlim(TAU_MIN, MAXLAG_MS)
    ax[1].set_ylabel(r'$r_{CCG}$')
    for key, colour in (('cp_high', '#a21d66'), ('cp_low', '0.25')):
        ax[1].axhline(WASMUHT[key], color=colour, ls='--', lw=1.4)
    ax[1].plot([], [], ls='--', color=C_REF, label='Wasmuht et al.')
    ax[1].set_title('median CP = %.2f' % cp_median, fontsize=14)
    ax[1].legend(frameon=False, fontsize=12)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def site_slopes(curves, t):
    """
    Per-site parabola through the long-lag correlation, and its derivative.

    Returns the fitted curves, the derivative evaluated on the same grid, and
    the vertex of each parabola. A site whose vertex falls inside the trial
    has a peak there; theirs peak near 0.8 s.
    """
    fits = np.full((len(curves), t.size), np.nan)
    deriv = np.full((len(curves), t.size), np.nan)
    vertex = np.full(len(curves), np.nan)
    for i, c in enumerate(curves):
        ok = np.isfinite(c)
        if ok.sum() < 5:
            continue
        a, b, c0 = np.polyfit(t[ok], c[ok], 2)
        fits[i] = np.polyval([a, b, c0], t)
        deriv[i] = 2 * a * t + b
        if a != 0:
            vertex[i] = -b / (2 * a)
    return fits, deriv, vertex


def plot_figure_4(res, name='wasmuht_fig4'):
    """
    Their Fig. 6: do the dynamics of the long-lag correlation predict CP?

    (a) the correlation between every pair of long bins, averaged over sites;
    (b) its long-lag component with the parabola they fit; (c) the across-site
    correlation between the slope at one time and CP at another; (d) CP split
    by the sign of the slope; (e) the per-site parabolas behind (b); and (f)
    where their vertices fall.
    """
    out = res['disp0.00']
    tau0, tau4, cps = out['tau0'], out['tau4'], out['cp_time']
    n_long = tau4.shape[1]
    t = (np.arange(n_long) + .5) * BIN_LONG / 1000.0
    half = n_long // 2
    fits, deriv, vertex = site_slopes(tau4, t)
    fits0, deriv0, _ = site_slopes(tau0, t)
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8))

    # (a) the correlation matrix, with the band used for the long-lag part
    mat = np.nanmean(out['rsc_mat'], 0)
    lim = np.nanmax(np.abs(mat))
    im = ax[0, 0].imshow(mat, origin='lower', cmap='viridis', vmin=-lim,
                         vmax=lim, extent=[0, 2, 0, 2])
    for d in (-N_LAG, -1, 1, N_LAG):
        ax[0, 0].plot([0, 2], [0 - d * BIN_LONG / 1000, 2 - d * BIN_LONG
                               / 1000], color='w', lw=.8)
    ax[0, 0].set_xlim(0, 2)
    ax[0, 0].set_ylim(0, 2)
    ax[0, 0].set_xlabel('multi-unit bin $t_j$ (s)')
    ax[0, 0].set_ylabel('single-unit bin $t_i$ (s)')
    ax[0, 0].set_title(r'(a) $r_{SC}(t_i, t_j)$', fontsize=13)
    fig.colorbar(im, ax=ax[0, 0], fraction=.046)

    # (b) the long-lag component and the parabola they fit
    mean = np.nanmean(tau4, 0)
    err = np.nanstd(tau4, 0) / np.sqrt(np.sum(np.isfinite(tau4), 0))
    ax[0, 1].errorbar(t, mean, yerr=err, color=C_AMB, lw=1.4, marker='o',
                      ms=4, label=r'$r_{SC}^{\tau=4}$')
    coef = np.polyfit(t, mean, 2)
    ax[0, 1].plot(t, np.polyval(coef, t), 'k-', lw=2,
                  label='parabola, curvature %+.4f' % coef[0])
    ax[0, 1].axvline(0.8, color=C_REF, ls=':', lw=1.4)
    ax[0, 1].annotate('their peak', xy=(0.82, ax[0, 1].get_ylim()[0]),
                      fontsize=8, color=C_REF)
    ax[0, 1].set_xlabel('time (s)')
    ax[0, 1].set_ylabel(r'$r_{SC}$')
    ax[0, 1].legend(frameon=False, fontsize=9)
    ax[0, 1].set_title('(b) long-lag component', fontsize=13)

    # (c) slope at one time against CP at another, across sites
    corr = np.full((n_long, n_long), np.nan)
    for i in range(n_long):
        for j in range(n_long):
            ok = np.isfinite(deriv[:, i]) & np.isfinite(cps[:, j])
            if ok.sum() > 5:
                corr[i, j] = pearsonr(deriv[ok, i], cps[ok, j])[0]
    im = ax[0, 2].imshow(corr, origin='lower', cmap='viridis', vmin=-.6,
                         vmax=.6, extent=[0, 2, 0, 2])
    ax[0, 2].axhline(1, color='k', lw=.9, ls='--')
    ax[0, 2].axvline(1, color='k', lw=.9, ls='--')
    ax[0, 2].set_xlabel('CP time (s)')
    ax[0, 2].set_ylabel('slope time (s)')
    ax[0, 2].set_title('(c) corr(slope, CP) across sites', fontsize=13)
    fig.colorbar(im, ax=ax[0, 2], fraction=.046)

    # (d) CP split by the sign of the slope, for both components
    late = np.nanmean(cps[:, half:], 1)
    for d, colour, lab in ((deriv, C_AMB, 'long lag'),
                           (deriv0, 'k', 'instantaneous')):
        early = np.nanmean(d[:, :half], 1)
        ok = np.isfinite(early) & np.isfinite(late)
        r_val, p_val = pearsonr(early[ok], late[ok])
        for sel, style in (((early > 0) & ok, '-'), ((early <= 0) & ok, '--')):
            if sel.sum() > 1:
                ax[1, 0].plot(t, np.nanmean(cps[sel], 0), style, color=colour,
                              lw=2, label='%s, slope %s (n=%d)'
                              % (lab, '>0' if style == '-' else '<0',
                                 sel.sum()))
        print('  %-14s slope -> late CP: r = %+.2f, p = %.3f'
              % (lab, r_val, p_val))
    ax[1, 0].axhline(0.5, color='0.7', lw=.8)
    ax[1, 0].axhline(WASMUHT['cp_amb'], color=C_REF, ls='--', lw=1.3)
    ax[1, 0].set_xlabel('time (s)')
    ax[1, 0].set_ylabel('CP')
    ax[1, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].set_title('(d) CP split by the sign of the slope', fontsize=13)

    # (e) every site's parabola
    peak = np.array([np.isfinite(v) and 0 < v < t[-1] for v in vertex])
    for i, f in enumerate(fits):
        if not np.isfinite(f).any():
            continue
        ax[1, 1].plot(t, f, lw=.9, alpha=.55,
                      color=C_AMB if peak[i] else '0.45')
    ax[1, 1].plot(t, np.polyval(coef, t), 'k-', lw=2.5, label='mean fit')
    ax[1, 1].plot([], [], color=C_AMB, lw=1.2,
                  label='peak inside the trial (n=%d)' % peak.sum())
    ax[1, 1].plot([], [], color='0.45', lw=1.2,
                  label='no peak inside (n=%d)' % (len(fits) - peak.sum()))
    ax[1, 1].set_xlabel('time (s)')
    ax[1, 1].set_ylabel(r'$r_{SC}^{\tau=4}$')
    ax[1, 1].legend(frameon=False, fontsize=8)
    ax[1, 1].set_title('(e) the parabola of every site', fontsize=13)

    # (f) where those peaks sit
    inside = vertex[np.isfinite(vertex) & (vertex > 0) & (vertex < t[-1])]
    ax[1, 2].hist(inside, bins=np.linspace(0, 2, 11), color=C_AMB, alpha=.85)
    ax[1, 2].axvline(0.8, color=C_REF, ls=':', lw=2, label='their peak')
    if inside.size:
        ax[1, 2].axvline(np.median(inside), color='k', lw=2,
                         label='our median %.2f s' % np.median(inside))
    ax[1, 2].set_xlabel('vertex of the parabola (s)')
    ax[1, 2].set_ylabel('sites')
    ax[1, 2].legend(frameon=False, fontsize=8)
    ax[1, 2].set_title('(f) where each site peaks', fontsize=13)

    for a in ax.ravel():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def plot_figure_5(res, name='wasmuht_fig5', n_boot=2000,
                  sizes=(5, 10, 15, 20, 30, 40, 60, 90)):
    """
    How the CP--r_SC correlation depends on how many sites were recorded.

    Sites are drawn without replacement, so the horizontal axis reads as the
    size of a hypothetical recording rather than as a resampling of this one.
    Left, the fitted line from each draw, darker the more sites it is fitted
    to. Right, the correlation itself against the number of sites.
    """
    out = res['disp0.00']
    cp = np.nanmean(out['cp_time'], 1)
    rsc = out['rsc']
    ok = np.isfinite(cp) & np.isfinite(rsc)
    cp, rsc = cp[ok], rsc[ok]
    n_site = cp.size
    rng = np.random.default_rng(0)
    sizes = [n for n in sizes if n <= n_site]

    boot = {}
    for n in sizes:
        vals, fits = [], []
        for _ in range(n_boot):
            idx = (np.arange(n_site) if n == n_site
                   else rng.choice(n_site, n, replace=False))
            if rsc[idx].std() <= 0 or cp[idx].std() <= 0:
                continue
            vals.append(pearsonr(rsc[idx], cp[idx])[0])
            fits.append(np.polyfit(rsc[idx], cp[idx], 1))
        boot[n] = (np.array(vals), np.array(fits))

    fig, ax = plt.subplots(ncols=2, figsize=(12.5, 4.6))
    grid = np.linspace(rsc.min(), rsc.max(), 20)
    shown = [n for n in (sizes[1], sizes[len(sizes) // 2], sizes[-1])]
    ax[0].scatter(rsc, cp, s=16, color='0.7', zorder=1)
    for k, n in enumerate(shown):
        shade = 0.3 + 0.7 * k / max(len(shown) - 1, 1)
        for coef in boot[n][1][:60]:
            ax[0].plot(grid, np.polyval(coef, grid), color=C_AMB,
                       alpha=.1 * shade, lw=1, zorder=2)
        ax[0].plot([], [], color=C_AMB, alpha=shade, lw=2,
                   label='%d sites' % n)
    ax[0].plot(grid, np.polyval(np.polyfit(rsc, cp, 1), grid), 'k-', lw=2,
               zorder=3)
    ax[0].set_xlabel(r'$r_{SC}$')
    ax[0].set_ylabel('choice probability')
    ax[0].legend(frameon=False, title='fits drawn from')

    med = [np.median(boot[n][0]) for n in sizes]
    lo = [np.percentile(boot[n][0], 5) for n in sizes]
    hi = [np.percentile(boot[n][0], 95) for n in sizes]
    ax[1].fill_between(sizes, lo, hi, color=C_AMB, alpha=.25,
                       label='5--95% of draws')
    ax[1].plot(sizes, med, color=C_AMB, lw=2, marker='o', label='median')
    ax[1].axhline(WASMUHT['r_cp_rsc'], color=C_REF, ls='--', lw=1.6,
                  label='Wasmuht et al.')
    ax[1].axvline(39, color='0.6', ls=':', lw=1.4)
    ax[1].set_xscale('log')
    ax[1].set_xticks(sizes)
    ax[1].set_xticklabels([str(n) for n in sizes])
    ax[1].set_xlabel('number of sites')
    ax[1].set_ylabel(r'corr(CP, $r_{SC}$)')
    ax[1].legend(frameon=False, loc='lower right')
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


# one parameter varied at a time; the baseline value is in every list, so the
# rows of the sweep figure share a common point
SWEEPS = (('gain', 'encoding gain (Hz)', (10.0, 15.0, 20.2, 30.0)),
          ('K', 'pool size', (3, 6, 12, 25)),
          ('tau_s', r'synapse $\tau_s$ (units of $\tau$)',
           (0.1, 0.2, 0.5, 1.0)),
          ('r_bi', r'$J_{bi}/J^\star$', (1.3, 1.7, 2.15, 2.6)),
          ('bpd', 'disparity scale', (2.0, 4.0, 7.55, 12.0)))


def run_parameter_sweep(kind='fbp', alpha=1.0, circuit='best', sweeps=SWEEPS,
                        n_trials=100, n_nets=4, verbose=True):
    """
    Vary each parameter in turn around a circuit, holding the others.

    Every setting is a full set of stimulus conditions and gets its own cache
    entry, so the sweep figure can be rebuilt without simulating again.

    Returns
    -------
    dict
        One entry per parameter, holding its axis label, the values tried,
        and for each value the mean r_SC, CP and cumulative correlogram of
        every stimulus condition.
    """
    global CIRCUIT
    base = dict(CIRCUITS[circuit])
    conds = ['random'] + ['disp%.2f' % d for d in DISPARITIES]
    sweep = {}
    for pname, label, values in sweeps:
        # the circuit's own value is always tried, so every row has a star
        values = sorted(set(list(values) + [base[pname]]))
        rows = []
        for v in values:
            use_circuit(dict(base, **{pname: v}), kind, alpha)
            # the rest of the circuit must be in the key too, or two
            # sweeps that happen to try the same value collide
            CIRCUIT = 'sweep_%s_%s_%g' % (
                circuit if isinstance(circuit, str) else 'custom',
                pname, v)
            res = run_all(kind, alpha, n_trials=n_trials, n_nets=n_nets,
                          verbose=True)
            rows.append({c: dict(rsc=fisher_mean(res[c]['rsc']),
                                 cp=float(np.nanmean(res[c]['cp'])),
                                 rccg=np.nanmean(res[c]['rccg'], 0))
                         for c in conds})
            if verbose:
                print('  %s = %g done' % (pname, v), flush=True)
        sweep[pname] = dict(label=label, values=list(values), rows=rows,
                            base=base[pname])
    use_circuit(circuit, kind, alpha)              # put the circuit back
    return sweep


def plot_figure_6(sweep, name='wasmuht_fig6_sweep'):
    """
    The sweep: one row per parameter, one colour per value.

    Columns are r_SC against the parameter, CP against the parameter, r_SC
    against disparity, and the cumulative correlogram for each of the three
    stimuli. Dashed lines mark the values they report, and a star marks the
    value the circuit itself uses.
    """
    rows = list(sweep)
    fig, ax = plt.subplots(len(rows), 6, figsize=(22, 3.2 * len(rows)),
                           squeeze=False)
    titles = (r'$r_{SC}$ against parameter', 'CP against parameter',
              r'$r_{SC}$ against disparity', 'ambiguous', 'unambiguous',
              'random motion')
    stim = (('disp0.00', 'ambiguous', C_AMB),
            ('disp0.03', 'unambiguous', C_UNAMB),
            ('random', 'random motion', C_RAND))
    for r, pname in enumerate(rows):
        d = sweep[pname]
        values, data = d['values'], d['rows']
        shades = plt.cm.viridis(np.linspace(.15, .9, len(values)))
        used = int(np.argmin([abs(v - d['base']) for v in values]))

        for cond, lab, colour in stim:
            y = [x[cond]['rsc'] for x in data]
            ax[r, 0].plot(values, y, marker='o', ms=5, color=colour, lw=2,
                          label=lab)
            ax[r, 0].plot(values[used], y[used], marker='*', ms=17,
                          color=colour, mec='k', mew=.6, ls='none', zorder=5)
            ax[r, 0].axhline(WASMUHT['rsc_' + {'disp0.00': 'amb',
                                               'disp0.03': 'unamb',
                                               'random': 'rand'}[cond]],
                             color=colour, ls='--', lw=1, alpha=.7)
        for cond, lab, colour in (stim[0], stim[2]):
            y = [x[cond]['cp'] for x in data]
            ax[r, 1].plot(values, y, marker='o', ms=5, color=colour, lw=2,
                          label=lab)
            ax[r, 1].plot(values[used], y[used], marker='*', ms=17,
                          color=colour, mec='k', mew=.6, ls='none', zorder=5)
            ax[r, 1].axhline(WASMUHT['cp_' + ('amb' if cond == 'disp0.00'
                                              else 'rand')],
                             color=colour, ls='--', lw=1, alpha=.7)

        for k, (v, x) in enumerate(zip(values, data)):
            wide = 3.2 if k == used else 1.8
            tag = ('%g (used)' % v) if k == used else '%g' % v
            ax[r, 2].plot(DISPARITIES,
                          [x['disp%.2f' % dd]['rsc'] for dd in DISPARITIES],
                          marker='*' if k == used else 'o',
                          ms=15 if k == used else 5, color=shades[k],
                          lw=wide, label=tag)
            lag = np.arange(x['disp0.00']['rccg'].size) * BIN_MS
            for c, (cond, lab, _) in enumerate(stim):
                ax[r, 3 + c].plot(lag, x[cond]['rccg'], color=shades[k],
                                  lw=wide, label=tag)
        for c in range(3, 6):
            ax[r, c].set_xscale('log')
            ax[r, c].set_xlim(TAU_MIN, MAXLAG_MS)
            ax[r, c].set_xlabel(r'$\tau$ (ms)')
            ax[r, c].set_ylabel(r'$r_{CCG}$')
        ax[r, 0].set_xlabel(d['label'])
        ax[r, 0].set_ylabel(r'$r_{SC}$')
        ax[r, 1].set_xlabel(d['label'])
        ax[r, 1].set_ylabel('CP')
        ax[r, 2].set_xlabel('disparity (deg)')
        ax[r, 2].set_ylabel(r'$r_{SC}$')
        ax[r, 2].legend(frameon=False, fontsize=8, title=d['label'],
                        title_fontsize=8)
        if r == 0:
            for c in range(6):
                ax[r, c].set_title(titles[c], fontsize=13)
            ax[r, 0].legend(frameon=False, fontsize=8)
    for a in ax.ravel():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


def run_coupling_sweep(kind='fbp', alpha=1.0, circuit='match',
                       ratios=np.arange(0.5, 1.5, 1e-1),
                       disparities=DISPARITIES, n_trials=120, n_nets=8,
                       sigma_J=0.0, mixed_mu=MIXED_MU, recompute=False,
                       folder=None, verbose=True):
    """
    Choice-related activity as the coupling is taken through its threshold.

    The stimulus sets the coupling, so sweeping J is the manipulation an
    experiment can make: their cylinder and their random dot motion differ in
    how strongly the local motion signals constrain one another, not in any
    decision variable. Every other parameter of the circuit is held fixed and
    nothing is fed back from a later stage.

    Each coupling is run at each disparity, so the sweep also gives r_SC as a
    function of the disambiguating evidence. The whole sweep is cached to disk
    under a name carrying everything that changes it, so the couplings are
    simulated once and later calls load in under a second.

    Returns
    -------
    list of dict
        One entry per coupling ratio, each holding the per-site r_SC and CP at
        zero disparity and the mean r_SC and CP at every disparity.
    """
    folder = DATA_FOLDER if folder is None else folder
    # always sample the circuit's own coupling, so the violins below are the
    # same simulation Fig. 2 shows rather than the nearest ratio on the grid
    ratios = tuple(sorted(set(ratios) | {CIRCUITS[circuit]['r_bi']}))
    # every argument that changes the result belongs in the name, or two
    # sweeps that differ only in one of them would share a cache entry
    filename = folder + ('wasmuht_coupling_%s_a%g_T%d_N%d_%s_b%g_sJ%g_%s_%s'
                         '.npz') % (
        kind, alpha, n_trials, n_nets, circuit, CIRCUITS[circuit]['bpd'],
        sigma_J, 'mix' if mixed_mu else 'same',
        '-'.join('%g' % ratio for ratio in ratios))
    if not recompute and os.path.exists(filename):
        # load the circuit even on a cache hit: the figure reads J_BI, and a
        # stale one would draw a different coupling from the same saved data
        use_circuit(circuit, kind, alpha)
        return load_sweep(filename)
    base = dict(CIRCUITS[circuit])
    out = []
    for ratio in ratios:
        use_circuit(dict(base, r_bi=ratio), kind, alpha)
        by_disp = {}
        for d in disparities:
            res = run_condition(kind, alpha, J_BI * JSTAR[kind],
                                B_PER_DEG * d, n_trials, n_nets, 0,
                                sigma_J=sigma_J, sigma_B=SIGMA_B,
                                mixed_mu=mixed_mu)
            by_disp[d] = dict(rsc=fisher_mean(res['rsc']),
                              cp=float(np.nanmean(res['cp'])),
                              rsc_sites=res['rsc'])
            if d == 0.0:
                zero = res
        out.append(dict(ratio=ratio, circuit_ratio=base['r_bi'],
                        rsc=zero['rsc'], cp=zero['cp'],
                        rccg=np.nanmean(zero['rccg'], 0), by_disp=by_disp))
        if verbose:
            print('  J/J* = %.2f done (%d sites)' % (ratio, zero['rsc'].size),
                  flush=True)
    use_circuit(circuit, kind, alpha)
    save_sweep(filename, out)
    return out


def matched_coupling(ratios, values, target):
    """The coupling at which a measure first reaches a reported value."""
    ratios, values = np.asarray(ratios, float), np.asarray(values, float)
    if target < values.min() or target > values.max():
        return np.nan
    return float(np.interp(target, values, ratios))


def plot_figure_7(sweep, name='wasmuht_fig7_coupling'):
    """
    Both of their conditions placed on one coupling axis.

    The first two panels sweep the coupling and mark the values reported for
    the ambiguous cylinder and for random dot motion, with the coupling at
    which the model reaches each. If the mapping from stimulus to coupling
    holds, the random-motion value should be matched below threshold and the
    ambiguous value above it. The third panel adds the disambiguating
    evidence, and the fourth shows that the two measures move together across
    sites as well as across couplings.
    """
    ratios = np.array([d['ratio'] for d in sweep])
    rsc = np.array([fisher_mean(d['rsc']) for d in sweep])
    cp = np.array([np.nanmean(d['cp']) for d in sweep])
    fig, ax = plt.subplots(ncols=4, figsize=(15, 3.3))

    for a, curve, err, lab, keys in (
            (ax[0], rsc, [np.nanstd(d['rsc']) / np.sqrt(np.isfinite(d['rsc']).sum())
                          for d in sweep], r'$r_{SC}$',
             (('rsc_amb', 'ambiguous cylinder', C_AMB),
              ('rsc_rand', 'random dot motion', C_RAND))),
            (ax[1], cp, [np.nanstd(d['cp']) / np.sqrt(np.isfinite(d['cp']).sum())
                         for d in sweep], 'choice probability',
             (('cp_amb', 'ambiguous cylinder', C_AMB),
              ('cp_rand', 'random dot motion', C_RAND)))):
        a.errorbar(ratios, curve, yerr=err, color='0.25', lw=2, marker='o',
                   ms=5, zorder=3)
        a.axvline(1.0, color='k', ls=':', lw=1.6)
        for key, text, colour in keys:
            a.axhline(WASMUHT[key], color=colour, ls='--', lw=1.5, label=text)
            m = matched_coupling(ratios, curve, WASMUHT[key])
            if np.isfinite(m):
                a.plot([m, m], [-0.2, WASMUHT[key]], color=colour,
                       ls=':', lw=1.4)
                a.set_ylim(-0.05 if a == ax[0] else 0.495, a.get_ylim()[1])
                a.plot(m, WASMUHT[key], marker='o', ms=9, color=colour,
                       mec='k', mew=.7, zorder=4)
                a.annotate(r'%.2f' % m, xy=(m, WASMUHT[key]),
                           xytext=(m + .04, WASMUHT[key] - .035),
                           fontsize=9, color=colour)
        a.set_xlabel(r'Coupling $J/J^\star$')
        a.set_ylabel(lab)
        a.legend(frameon=False, fontsize=12, loc='upper left')

    shades = plt.cm.viridis(np.linspace(.12, .9, len(sweep)))
    disps = sorted(sweep[0]['by_disp'])
    # categorical positions, as in Fig. 2: the disparities are 0.01 apart, so
    # a linear axis would pile every violin on top of the next
    positions = list(range(len(disps)))
    # the violins are the circuit's own coupling, the one Fig. 2 also shows.
    # It travels with the data: reading it from J_BI would draw a different
    # coupling from the same saved sweep whenever that global had moved on.
    target = sweep[0].get('circuit_ratio', np.nan)
    target = J_BI if not np.isfinite(target) else target
    shown = min(sweep, key=lambda d: abs(d['ratio'] - target))
    data = [np.asarray(shown['by_disp'][x]['rsc_sites']) for x in disps]
    data = [v[np.isfinite(v)] for v in data]
    violins = ax[2].violinplot(data, positions=positions, widths=.75,
                               showmeans=True)
    for body, colour in zip(violins['bodies'],
                            [C_AMB] + [C_UNAMB] * (len(disps) - 1)):
        body.set_facecolor(colour)
        body.set_alpha(.65)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        violins[key].set_color('0.3')
        violins[key].set_linewidth(1.1)
    for pos, key, colour in ((0, 'rsc_amb', C_AMB),
                             (len(disps) - 1, 'rsc_unamb', C_UNAMB)):
        ax[2].plot(pos, WASMUHT[key], marker='_', ms=26, mew=3, color=colour,
                   zorder=6)
    ax[2].plot([], [], marker='_', ms=14, mew=3, color='0.3', ls='none',
               label='Wasmuht et al.')
    ax[2].set_xticks(positions)
    ax[2].set_xticklabels(['%.2f' % x for x in disps])
    ax[2].set_xlabel('Disparity (deg)')
    ax[2].set_ylabel(r'$r_{SC}$')
    ax[2].legend(frameon=False, fontsize=12, loc='lower left')
    # one row of small panels for the slope changing with the coupling, and
    # one row of two large panels for what that slope does. a2 keeps the two
    # large axes last, so a2[-2] and a2[-1] still name them.
    f2 = plt.figure(figsize=(len(sweep)*1.7, 4.))
    top_grid = f2.add_gridspec(1, len(sweep), left=.035, right=.995,
                               top=.95, bottom=.68, wspace=.25)
    a2 = [f2.add_subplot(top_grid[0, k]) for k in range(len(sweep))]
    # the two summary panels are square and centred under the row: the side
    # is set from the height so that both axes keep the same proportions
    height = .4
    side = height * f2.get_figheight() / f2.get_figwidth()
    half = side * (2 + .4) / 2
    low_grid = f2.add_gridspec(1, 2, left=.5 - half*1.2, right=.5 + half*1.2,
                               top=.56, bottom=.56 - height, wspace=.6)
    a2.append(f2.add_subplot(low_grid[0, 0]))
    a2.append(f2.add_subplot(low_grid[0, 1]))
    a_ind = 0
    slopevals = {'ratio': [],
                 'slope': [],
                 'pval': []}
    for c, d in zip(shades, sweep):
        ok = np.isfinite(d['rsc']) & np.isfinite(d['cp'])
        ratio = round(d['ratio'], 4)
        # color = 'r' if ratio == 1 else c
        color = c
        ax[3].scatter(d['rsc'][ok], d['cp'][ok], s=15, color=color, alpha=.7)
        a2[a_ind].scatter(d['rsc'][ok], d['cp'][ok], s=15, color=color, alpha=.7)
        a2[a_ind].spines['right'].set_visible(False)
        a2[a_ind].spines['top'].set_visible(False)
        r, p = pearsonr(d['rsc'][ok], d['cp'][ok])
        slopevals['slope'].append(r)
        slopevals['pval'].append(p)
        slopevals['ratio'].append(ratio)
        fit = np.polyfit(d['rsc'][ok], d['cp'][ok], 1)
        grid = np.linspace(d['rsc'][ok].min(), d['rsc'][ok].max(), 20)
        a2[a_ind].plot(grid, np.polyval(fit, grid), 'k-', lw=2)
        a2[a_ind].set_title(f'Ratio:{ratio}', fontsize=12)
        a_ind += 1
    # one pair of limits across the scatter panels, so they can be read
    # against each other. The last two carry other units and keep their own,
    # which is why the figure is not simply built with sharex and sharey.
    finite_rsc = np.concatenate([d['rsc'][np.isfinite(d['rsc'])]
                                 for d in sweep])
    finite_cp = np.concatenate([d['cp'][np.isfinite(d['cp'])] for d in sweep])
    pad_x = .05 * np.ptp(finite_rsc)
    pad_y = .05 * np.ptp(finite_cp)
    for k in range(len(sweep)):
        a2[k].set_xlim(finite_rsc.min() - pad_x, finite_rsc.max() + pad_x)
        a2[k].set_ylim(finite_cp.min() - pad_y, finite_cp.max() + pad_y)
        a2[k].locator_params(axis='x', nbins=3)
        a2[k].locator_params(axis='y', nbins=4)
        if k > 0:
            a2[k].set_yticklabels([])
    a2[0].set_xlabel('rSC')
    a2[0].set_ylabel('CP')
    a2[-2].set_ylabel('Correlation')
    a2[-1].set_ylabel('p-value')
    a2[-2].set_xlabel('Coupling ratio')
    a2[-1].set_xlabel('Coupling ratio')
    a2[-2].spines['right'].set_visible(False)
    a2[-2].spines['top'].set_visible(False)
    a2[-2].plot(slopevals['ratio'], slopevals['slope'],
                color='k', linestyle='', marker='o',
                markersize=5)
    a2[-1].spines['right'].set_visible(False)
    a2[-1].spines['top'].set_visible(False)
    a2[-1].axhline(0.01, color='firebrick', linestyle='--')
    a2[-1].axvline(1, color='firebrick', linestyle='--')
    a2[-1].plot(slopevals['ratio'], slopevals['pval'],
                color='k', linestyle='', marker='o',
                markersize=5)
    a2[-1].set_yscale('log')
    save_figure(f2, name + '_rsc_cp')
    allr = np.concatenate([d['rsc'] for d in sweep])
    allc = np.concatenate([d['cp'] for d in sweep])
    ok = np.isfinite(allr) & np.isfinite(allc)
    fit = np.polyfit(allr[ok], allc[ok], 1)
    grid = np.linspace(allr[ok].min(), allr[ok].max(), 20)
    ax[3].axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax[3].plot(grid, np.polyval(fit, grid), 'k-', lw=2)
    for rk, ck, colour, text in (('rsc_amb', 'cp_amb', C_AMB, 'ambiguous'),
                                 ('rsc_rand', 'cp_rand', C_RAND, 'random')):
        ax[3].plot(WASMUHT[rk], WASMUHT[ck], marker='*', ms=20, color=colour,
                   mec='k', mew=.7, ls='none', label=text)
    ax[3].set_xlabel(r'$r_{SC}$')
    ax[3].set_ylabel('Choice probability')
    r, p = pearsonr(allr[ok], allc[ok])
    ax[3].annotate(f'r = {r:.3f}\np = {p:.2f}', xy=(.1, 1.), xycoords=ax[3].transAxes,
                   fontsize=12)
    ax[3].set_ylim(0.45, 1.05)
    ax[3].legend(frameon=False, fontsize=11, title='Wasmuht et al.',
                 title_fontsize=12)
    # one block per coupling, so the bar shows the colours actually drawn
    edges = np.concatenate(([ratios[0] - (ratios[1] - ratios[0]) / 2],
                            (ratios[:-1] + ratios[1:]) / 2,
                            [ratios[-1] + (ratios[-1] - ratios[-2]) / 2]))
    bar = fig.colorbar(mpl.cm.ScalarMappable(
        norm=mpl.colors.BoundaryNorm(edges, len(shades)),
        cmap=mpl.colors.ListedColormap(shades)), ax=ax[3])
    bar.set_label(r'coupling $J/J^\star$')
    bar.set_ticks(ratios)
    bar.ax.set_yticklabels(['%.2f' % r for r in ratios], fontsize=8)

    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)
    return fig


# the schemes run when none is named on the command line
SCHEMES = (('mf', 1.0, 'MF'), ('fbp', 1.0, 'LBP'),
           ('fbp', 0.5, 'FBP_05'), ('fbp', 1.45, 'FBP_145'))

SCHEMES = (('fbp', 1.0, 'LBP'), )


if __name__ == '__main__':
    # usage: python wasmuht_reproduction.py [circuit] [scheme] [alpha]
    #   python wasmuht_reproduction.py                 every scheme, 'best'
    #   python wasmuht_reproduction.py total           every scheme, 'total'
    #   python wasmuht_reproduction.py best fbp 1.0    loopy BP alone
    # import sys
    # circuit = sys.argv[1] if len(sys.argv) > 1 else 'best'
    # circuit = 'match'
    # schemes = ([(sys.argv[2], float(sys.argv[3]),
    #               '%s_a%s' % (sys.argv[2].upper(), sys.argv[3]))]
    #             if len(sys.argv) > 3 else SCHEMES)
    # for kind, alpha, tag in schemes:
    #     use_circuit(circuit, kind, alpha)
    #     print('%s on circuit %s: J* = %.3f, J_bi = %.3f, J_mono = %.3f'
    #           % (tag, circuit, JSTAR[kind], J_BI * JSTAR[kind],
    #               J_MONO * JSTAR[kind]), flush=True)
    #     res = run_all(kind, alpha, recompute=False, verbose=True,
    #                   bin_ms=1.0, n_nets=8)
    #     suffix = '%s_%s_%s' % (tag, circuit, 'mixed' if MIXED_MU else 'same')
    #     plot_figure_1(res, 'wasmuht_fig1_%s' % suffix)
    #     plot_figure_2(res, 'wasmuht_fig2_%s' % suffix)
    #     plot_figure_3(res, 'wasmuht_fig3_%s' % suffix)
    #     plot_figure_4(res, 'wasmuht_fig4_%s' % suffix)
    #     print('  rSC  ambiguous %.3f | unambiguous %.3f | random %.3f'
    #           % (np.nanmean(res['disp0.00']['rsc']),
    #               np.nanmean(res['disp%.2f' % DISPARITIES[-1]]['rsc']),
    #               np.nanmean(res['random']['rsc'])), flush=True)
    #     print('  CP   ambiguous %.3f | random %.3f'
    #           % (np.nanmean(res['disp0.00']['cp']),
    #               np.nanmean(res['random']['cp'])), flush=True)
    # sweep = run_parameter_sweep(circuit='best')
    # sweep = run_parameter_sweep(circuit='k20')
    # plot_figure_6(sweep)
    # plot_figure_7(sweep, name='wasmuht_fig7_coupling')
    sweep = run_coupling_sweep(recompute=False, ratios=np.arange(0.5, 2.1, 0.1))
    plot_figure_7(sweep)
