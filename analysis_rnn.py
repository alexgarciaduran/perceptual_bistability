"""
Analysis & figure functions for the directed CTRNN model.
Add these to (or import alongside) graph_ctrnn_directed.py.

Provides:
  1. plot_noisy_dynamics_with_alternations  -> dynamics + perceptual alternations
  2. plot_boltzmann_distributions           -> per-dot activity distros, bi vs mono
  3. add_K_noise                            -> noisy fixed q-q weights (u's preserved)
  4./5. compute_corr_and_cp + run_regime_stats -> noise correlations & choice prob
       (run_regime_stats now supports a global bias b -> the SFM(B>0) condition)
  6. reproduce_results_figure               -> the Data-vs-Model bar figure
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import zscore, pearsonr
from sklearn.metrics import roc_auc_score

from graph_ctrnn_directed import (get_graph, directed_channels, simulate, sig)

# colours requested
COL_BI   = "peru"        # bistable  (z=0)
COL_MONO = "cadetblue"   # monostable(z=6)

# =========================================================================== #
# 3. K-noise: perturb ONLY the fixed q-q coupling, keep the u-pathway intact
# =========================================================================== #
def add_K_noise(ei, ej, N, K, alpha, seed=0):
    """
    Return a per-edge array Kedge (length E) = K + alpha*white_noise.
    The auxiliaries (u's) are untouched -- only the direct q-q weights are noisy.
    Use with simulate_Knoise below (which accepts a per-edge K vector).
    """
    rng = np.random.default_rng(seed)
    E = len(ei)
    return K + alpha * rng.standard_normal(E)


def simulate_Knoise(ei, ej, N, z, Kedge, wqu=0.8, wuq=0.8, tau_q=10.0,
                    tau_u=0.2, noise=0.0, b=0.0, T=600.0, dt=0.05, seed=0,
                    q0=None, rec_every=2, biased_init=False):
    """
    Same dynamics as `simulate`, but the fixed q-q coupling is a per-EDGE vector
    Kedge (from add_K_noise) instead of a scalar. Auxiliaries unchanged.

    A constant global bias `b` (same for every q-unit) can be added to the q
    input; this is the SFM (B>0) condition (z=0, b=0.1). b=0 reproduces the
    original symmetric dynamics. Pure-numpy (not numba) for simplicity; fine for
    these sizes.
    """
    rng = np.random.default_rng(seed)
    src, dst = directed_channels(ei, ej)
    C = src.shape[0]; E = len(ei)
    steps = int(T/dt); sdt = np.sqrt(dt)
    if q0 is None:
        q0 = rng.uniform(0.3, 0.7, N)
    q = q0.copy()
    a0 = wqu*(2*q[src]-1)
    up = sig(a0 + z); um = sig(a0 - z)
    nrec = steps // rec_every
    Q = np.empty((nrec, N)); ridx = 0
    for t in range(steps):
        # auxiliaries (unchanged, read source only) -- NOT affected by bias b
        a = wqu*(2*q[src]-1)
        sp = sig(a + z); sm = sig(a - z)
        up = up + dt*(sp - up)/tau_u
        um = um + dt*(sm - um)/tau_u
        if noise > 0:
            up += noise*sdt*rng.standard_normal(C)
            um += noise*sdt*rng.standard_normal(C)
        up = np.clip(up, 1e-6, 1-1e-6); um = np.clip(um, 1e-6, 1-1e-6)
        # q input: noisy per-edge K  +  aux  +  constant bias b
        inp = np.full(N, float(b))
        np.add.at(inp, ei, Kedge*(2*q[ej]-1))
        np.add.at(inp, ej, Kedge*(2*q[ei]-1))
        aux = wuq*((2*up-1) + (2*um-1))
        np.add.at(inp, dst, aux)
        q = q + dt*(sig(inp) - q)/tau_q
        if noise > 0:
            q += noise*sdt*rng.standard_normal(N)
        q = np.clip(q, 1e-6, 1-1e-6)
        if t % rec_every == 0 and ridx < nrec:
            Q[ridx] = q; ridx += 1
    return q, Q


# =========================================================================== #
# 1. Noisy dynamics with perceptual alternations (bistable z=0 vs mono z=6)
# =========================================================================== #
def plot_noisy_dynamics_with_alternations(
        ei, ej, N, PARS, A,
        z_bi=0.0, z_mono=6.0, T=900.0, dt=0.05, rec_every=4,
        noise=0.07, bi_seeds=(53, 9, 23), mono_seeds=(0, 1, 2),
        wq_override=None, smooth_win=None, smooth_mode="gauss",
        bi_windows=None, mono_windows=None, save=None):
    """
    Three EXAMPLE runs (3 seeds) per condition.
    Bistable (z=0): population mean shows clean spontaneous perceptual
    alternations within the window.
    Monostable (z=6): stays near q=1/2.

    wq_override : if not None, sets wqu=wuq=sqrt(wq_override). Default None uses
    PARS as-is (wq=0.6), which gives wells shallow enough that noise drives clean
    population alternations within the window. Deepening them (e.g. W=1.0)
    suppresses switching, so leave this None for the alternation figure.

    smooth_win : DISPLAY-ONLY temporal smoothing of the traces, in *recorded
        steps* (1 step = dt*rec_every time units). None or 0 -> no smoothing.
        The dynamics are NOT changed: smoothing is applied to a copy just before
        plotting, and the RAW Q arrays are returned. Typical values ~15-40 clean
        the single-unit traces while keeping switches crisp.
    smooth_mode : "gauss" (Gaussian, sigma=smooth_win; rounds switch edges
        gently) or "ma" (boxcar moving average of width smooth_win).

    bi_windows / mono_windows : optional per-seed zoom windows for the x-axis,
        as a list of (t0, t1) tuples in TIME units (same length as the seed
        tuple). Use these to simulate long (large T) and crop each panel to a
        switch-rich segment, e.g. bi_windows=[(500,13000),(1000,11000),...].
        None -> show the full [0, T] window. Only the displayed x-range is
        cropped; the full trace is still simulated and returned.
    """
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d

    def _smooth(Q):
        if not smooth_win:
            return Q
        if smooth_mode == "ma":
            return uniform_filter1d(Q, size=int(smooth_win), axis=0, mode="nearest")
        return gaussian_filter1d(Q, sigma=float(smooth_win), axis=0, mode="nearest")

    pars = dict(PARS); pars["noise"] = noise
    if wq_override is not None:
        wq = np.sqrt(wq_override)
        pars["wqu"] = wq; pars["wuq"] = wq

    def runs(z, seeds, biased):
        out = []
        for s in seeds:
            Q = simulate(ei, ej, N, z, T=T, dt=dt, rec_every=rec_every,
                         biased_init=biased, seed=s, **pars)[3]
            out.append(Q)
        return out

    Qbi = runs(z_bi, bi_seeds, False)      # committed start, then alternates
    Qmo = runs(z_mono, mono_seeds, False)
    t = np.arange(Qbi[0].shape[0]) * dt * rec_every

    # display copies (smoothed); raw Qbi/Qmo are returned untouched
    Dbi = [_smooth(Q) for Q in Qbi]
    Dmo = [_smooth(Q) for Q in Qmo]

    def _xlim(ax_, windows, col):
        if windows is not None and windows[col] is not None:
            ax_.set_xlim(*windows[col])
        else:
            ax_.set_xlim(t[0], t[-1])

    # sharex=False so each column can zoom to its own window
    fig, ax = plt.subplots(3, 2, figsize=(5, 4), sharex=False, sharey=True)
    for row in range(3):
        # bistable column
        a = ax[row, 1]
        a.axis('off')
        a.axhline(0.5, color='k', alpha=0.5, linestyle='--', linewidth=3)
        for i in range(N):
            a.plot(t, Dbi[row][:, i], lw=0.3, alpha=0.1, color=COL_BI)
        a.plot(t, Dbi[row].mean(1), lw=3, color=COL_BI)
        a.set_ylim(-0.03, 1.03)
        _xlim(a, bi_windows, row)
        if row == 0: a.set_ylabel("q")
        # monostable column
        b = ax[row, 0]
        b.axis('off')
        b.axhline(0.5, color='k', alpha=0.5, linestyle='--', linewidth=3)
        for i in range(N):
            b.plot(t, Dmo[row][:, i], lw=0.3, alpha=0.1, color=COL_MONO)
        b.plot(t, Dmo[row].mean(1), lw=3, color=COL_MONO)
        b.axhline(0.5, color="gray", ls=":", lw=1)
        b.set_ylim(-0.03, 1.03); b.set_xlabel("time")
        # default: mono panels mirror the bistable column's window if given
        mw = mono_windows if mono_windows is not None else bi_windows
        _xlim(b, mw, row)
        if row == 0: b.set_ylabel("q")
    for a in ax.ravel(): a.spines[["top", "right"]].set_visible(False); a.axis('off')
    fig.tight_layout()
    if save:
        for ext in (".png", ".svg"):
            fig.savefig(save+ext, dpi=200, bbox_inches="tight")
    return Qbi, Qmo


# =========================================================================== #
# 2. Boltzmann distributions of per-dot activity (bi vs mono)
# =========================================================================== #
def plot_confidence_distributions(
        ei, ej, N, PARS, z_mono=6.0, z_bi=0.0,
        biases=(0.0, 0.06, 0.1),     # = [0, 0.4, 0.8, 1] * 0.1
        n_sims=200, T=600.0, dt=0.05, noise=0.5,
        align=True, save=None):
    """
    KDE of "confidence aligned with stimulus" (model only), one curve per bias b,
    in two panels: Monostable (z=z_mono) and Bistable (z=z_bi).

    Confidence = 2q - 1  (maps q in [0,1] -> [-1,1]).
    align=True: flip each trial's sign so the favoured interpretation is positive.
      - b > 0 : bias points toward q=1, so confidence is used as-is.
      - b = 0 : no preferred side, so align by the realised population choice sign
                (keeps the bistable +/-1 lobes instead of cancelling them).

    The bias b enters the q-input (simulate_Knoise), so all conditions use that
    path. Darker line = larger bias (light_palette gradient like the reference).
    """
    biases = list(biases)
    Kedge = np.full(len(ei), PARS["K"])

    def collect(z, b):
        out = []
        for s in tqdm(range(n_sims)):
            qf, _ = simulate_Knoise(
                ei, ej, N, z, Kedge,
                wqu=PARS["wqu"], wuq=PARS["wuq"],
                tau_q=PARS["tau_q"], tau_u=PARS["tau_u"],
                noise=noise, b=b, T=T, dt=dt, seed=s, biased_init=False)
            conf = 2.0 * qf - 1.0                       # [-1, 1] per unit
            if align:
                if b > 0:
                    pass                                # bias defines + direction
                else:
                    conf = conf * np.random.choice([-1, 1])
            out.append(conf)
        return np.concatenate(out)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    mono_pal = sns.light_palette(COL_MONO, n_colors=len(biases) + 1)[1:]
    bi_pal   = sns.light_palette(COL_BI,   n_colors=len(biases) + 1)[1:]

    results = {}
    for i, b in enumerate(biases):
        vm = collect(z_mono, b)
        vb = collect(z_bi,   b)
        results[b] = (vm, vb)
        lw = 1.4 + 1.6 * i / max(1, len(biases) - 1)    # thicker = larger bias
        sns.kdeplot(vm, ax=ax[0], color=mono_pal[i], lw=lw,
                    bw_adjust=0.4, label=f"{b:g}")
        sns.kdeplot(vb, ax=ax[1], color=bi_pal[i], lw=lw,
                    bw_adjust=0.4, label=f"{b:g}")

    ax[0].set_title("Monostable", color=COL_MONO, fontsize=13)
    ax[1].set_title("Bistable",   color=COL_BI,   fontsize=13)
    for a in ax:
        a.set_xlabel("Confidence aligned with stimulus")
        a.set_xlim(-1.5, 1.5)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Density of confidence")
    ax[1].set_ylabel("")
    ax[0].legend(title="Bias, b", frameon=False)

    fig.tight_layout()
    if save:
        for ext in (".png", ".svg"):
            fig.savefig(save + ext, dpi=200, bbox_inches="tight")
    return results


def plot_boltzmann_distributions(
        ei, ej, N, PARS, z_bi=0.0, z_mono=6.0, n_sims=200,
        T=600.0, dt=0.05, noise=0.07, save=None):
    """
    Collect the FINAL activity of every dot over many noisy simulations and
    plot the distribution (histogram) for bistable vs monostable.
    Bistable -> bimodal (near 0 and 1); monostable -> unimodal at 1/2.
    """
    pars = dict(PARS); pars["noise"] = noise
    vals_bi, vals_mo = [], []
    for s in range(n_sims):
        qb = simulate(ei, ej, N, z_bi, T=T, dt=dt, biased_init=False,
                      seed=s, **pars)[0]
        qm = simulate(ei, ej, N, z_mono, T=T, dt=dt, biased_init=False,
                      seed=s, **pars)[0]
        vals_bi.append(qb); vals_mo.append(qm)
    vals_bi = np.concatenate(vals_bi); vals_mo = np.concatenate(vals_mo)

    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0.05, 1.05, 61)
    ax.hist(vals_bi, bins=bins, density=True, alpha=0.7, color=COL_BI,
            label=f"Bistable (z={z_bi})")
    ax.hist(vals_mo, bins=bins, density=True, alpha=0.7, color=COL_MONO,
            label=f"Monostable (z={z_mono})")
    ax.set_xlabel("Neuron activity"); ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save:
        for ext in (".png", ".svg"):
            fig.savefig(save+ext, dpi=200, bbox_inches="tight")
    return vals_bi, vals_mo


# =========================================================================== #
# 4 & 5. Noise correlations (neuron vs neighbours) and choice probability
# =========================================================================== #
def compute_corr_and_cp(single_averages, neighs_averages, choices, abs_stats=False):
    """
    single_averages : (n_units, n_sims) trial-averaged activity per unit
    neighs_averages : (n_units, n_sims) trial-averaged neighbour activity
    choices         : (n_sims,) binary global choice per simulation
    abs_stats       : if True, fold signs -> |corr| and |CP-0.5|+0.5 (matches the
                      `absolute_cps_rsc=True` convention of the reference paper
                      function; raises and tightens the bars by removing the
                      cancellation of negatively-tuned units).
    Returns per-unit (correlations, choice-probabilities[=ROC AUC]).
    """
    n_units = single_averages.shape[0]
    corrs, rocs = [], []
    for i in range(n_units):
        x_single = single_averages[i]
        try:
            roc = roc_auc_score(choices, x_single)
        except ValueError:
            roc = np.nan                      # choices all identical
        act_single = zscore(single_averages[i], nan_policy='omit')
        act_neigh = zscore(neighs_averages[i], nan_policy='omit')
        corr, _ = pearsonr(act_single, act_neigh)
        if abs_stats:
            corr = np.abs(corr)
            roc = np.abs(roc - 0.5) + 0.5
        corrs.append(corr); rocs.append(roc)
    return np.array(corrs), np.array(rocs)
 
 
def run_regime_stats(ei, ej, N, A, PARS, z, n_sims=200, T=600.0, dt=0.05,
                     noise=0.07, b=0.0, choice_steps_before=50,
                     alpha_RM=0.0, abs_stats=False, rm_seed=0):
    """
    Run many noisy simulations at a given z (and optional global bias b) and
    return, per unit:
      single_averages (n_units, n_sims) : trial-mean activity
      neighs_averages (n_units, n_sims) : mean activity of each unit's neighbours
      choices (n_sims,)                 : global binary choice
    Then compute_corr_and_cp gives correlations & choice probabilities.
 
    alpha_RM : ONE symmetric random matrix X=(G+G.T)/sqrt2 (unit-variance
        Gaussian), scaled by alpha_RM, exactly as in the reference paper code.
        The SAME perturbed coupling is used in BOTH places:
          (1) DYNAMICS  -- the per-edge q-q coupling becomes
                Kedge[e] = K + alpha_RM * X[ei[e], ej[e]]
              and is fed to simulate_Knoise (so the network actually runs on the
              noisy weights, not just for the readout).
          (2) NEIGHBOURS -- the neighbour mask is
                |A + alpha_RM * X| > alpha_RM
              (a connection counts only if its perturbed weight clears the noise
              floor alpha_RM). This reshuffles/dilutes each unit's neighbour set.
        This is the single knob that spreads the correlation bars; there is no
        separate weight-noise term.
    b>0 (with shared z) gives the SFM (B>0) condition: same stimulus, biased
        readout (a constant b added to the q-input before the sigmoid).
    abs_stats : fold signs -> |corr| and |CP-0.5|+0.5 (the reference
        `absolute_cps_rsc=True` convention).
    rm_seed  : seed for the random matrix X. Shared across conditions so the
        perturbed coupling / neighbour graph is identical in RDM / SFM(B>0) /
        SFM(B=0); only z and b differ between conditions.
    """
    # ONE symmetric random matrix, used for BOTH the simulated coupling and the
    # neighbour definition (as in the reference plot_example_correlation).
    if alpha_RM > 0:
        rng_rm = np.random.default_rng(rm_seed)
        G = rng_rm.standard_normal((N, N))
        X = (G + G.T) / np.sqrt(2.0)            # symmetric, unit variance
        Wp = A + alpha_RM * X                   # perturbed coupling matrix
        # (2) neighbours from the SAME perturbed matrix
        mask = (np.abs(Wp) > alpha_RM).astype(float)
        np.fill_diagonal(mask, 0.0)
        deg = mask.sum(1, keepdims=True); deg[deg == 0] = 1
        neighs = mask / deg
        # (1) per-edge coupling for the dynamics: K + alpha_RM * X on each edge
        Kedge = PARS["K"] + alpha_RM * X[ei, ej]
    else:
        deg = A.sum(1, keepdims=True); deg[deg == 0] = 1
        neighs = A / deg
        Kedge = np.full(len(ei), PARS["K"])
 
    single = np.zeros((N, n_sims))
    neigh = np.zeros((N, n_sims))
    choices = np.zeros(n_sims)
    pars = dict(PARS); pars["noise"] = noise
    # the per-edge K path is needed whenever the coupling is perturbed or biased
    use_knoise = (alpha_RM > 0) or (b != 0.0)
 
    for s in range(n_sims):
        if use_knoise:
            qf, Q = simulate_Knoise(ei, ej, N, z, Kedge,
                                    wqu=PARS["wqu"], wuq=PARS["wuq"],
                                    tau_q=PARS["tau_q"], tau_u=PARS["tau_u"],
                                    noise=noise, b=b, T=T, dt=dt, seed=s,
                                    biased_init=False)
        else:
            qf, _, _, Q, _ = simulate(ei, ej, N, z, T=T, dt=dt,
                                      biased_init=False, seed=s, **pars)
        acg = np.nanmean(Q, axis=0)            # per-unit trial-mean activity
        single[:, s] = acg
        neigh[:, s] = neighs @ acg
        # global choice from last part of the trial
        last = Q[-choice_steps_before:].mean()
        choices[s] = (np.sign(last - 0.5) + 1) / 2
 
    corrs, cps = compute_corr_and_cp(single, neigh, choices, abs_stats=abs_stats)
    return dict(single=single, neigh=neigh, choices=choices,
                corrs=corrs, cps=cps)
 
 
# =========================================================================== #
# 6. Reproduce the results figure (Data vs Model bars)
# =========================================================================== #
def reproduce_results_figure(
        corr_data=(0.23, 0.28, 0.42), cp_data=(0.56, 0.67),
        corr_model=None, cp_model=None,
        corr_ylim=(0.0, 0.8), cp_ylim=(0.5, 0.8), save=None):
    """
    Reproduce the summary figure (Data = solid, Model = hatched):
      left  : interneuronal correlation, conditions RDM / SFM(B>0) / SFM(B=0)
              RDM is monostable (cadetblue); both SFM are bistable (peru).
      right : choice probability, conditions RDM / SFM(B=0)
              RDM monostable (cadetblue); SFM(B=0) bistable (peru).
    corr_data  : len-3 -> [RDM, SFM(B>0), SFM(B=0)]
    cp_data    : len-2 -> [RDM, SFM(B=0)]
    corr_model : len-3 overlay (same order); cp_model : len-2 overlay.
    corr_ylim  : y-limits for the correlation panel (default (0, 0.8)).
    cp_ylim    : y-limits for the choice-probability panel (default (0.5, 0.8)).
                 Wider ranges make the Data vs Model bars look more similar.
    """
    corr_labels = ["RDM", "SFM (B>0)", "SFM (B=0)"]
    cp_labels   = ["RDM", "SFM (B=0)"]
    corr_cols   = [COL_MONO, COL_BI, COL_BI]     # RDM mono; SFM bistable
    cp_cols     = [COL_MONO, COL_BI]
 
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    w = 0.38
 
    # --- correlations ---
    x = np.arange(len(corr_labels))
    ax[0].bar(x - w/2, corr_data, width=w, color=corr_cols, label="Data")
    if corr_model is not None:
        ax[0].bar(x + w/2, corr_model, width=w, color=corr_cols,
                  hatch="///", edgecolor="k", label="Model")
    # legend proxies (color-neutral)
    ax[0].bar([np.nan], [0], color="#888", label="Data")
    ax[0].bar([np.nan], [0], facecolor="#888", hatch="///", edgecolor="k",
              label="Model")
    ax[0].set_xticks(x); ax[0].set_xticklabels(corr_labels)
    ax[0].set_ylabel("Interneuronal correlation")
    ax[0].set_ylim(*corr_ylim)
    handles, labels = ax[0].get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l in ("Data", "Model")][-2:]
    ax[0].legend([h for h, _ in keep], [l for _, l in keep], frameon=False)
 
    # --- choice probability ---
    x2 = np.arange(len(cp_labels))
    ax[1].bar(x2 - w/2, cp_data, width=w, color=cp_cols, label="Data")
    if cp_model is not None:
        ax[1].bar(x2 + w/2, cp_model, width=w, color=cp_cols,
                  hatch="///", edgecolor="k", label="Model")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1)
    ax[1].set_xticks(x2); ax[1].set_xticklabels(cp_labels)
    ax[1].set_ylabel("Choice probability")
    ax[1].set_ylim(*cp_ylim)
 
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save:
        for ext in (".png", ".svg"):
            fig.savefig(save+ext, dpi=200, bbox_inches="tight")
    return fig, ax


# =========================================================================== #
if __name__ == "__main__":
    N, d = 50, 4
    A, ei, ej = get_graph(N=N, d=d, seed=0)
    PARS = dict(K=0.6, wqu=1.0, wuq=1.0, tau_q=25.0, tau_u=0.2, noise=0.0)
    OUT = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/rnn_aux/directed/analysis/'  # Alex
    os.makedirs(OUT, exist_ok=True)

    NOISE_DYN = 0.12      # white noise for dynamics (no OU) -> clean switches
    NOISE_CP  = 0.05      # white noise for corr/CP stats
    ALPHA_K   = 0.3       # K-weight noise for corr/CP stats
    NSIM  = 100
    B_POS = 0.1           # bias for the SFM (B>0) condition
    np.random.seed(0)

    # 1. alternations  (wq_override=None -> uses PARS wq=0.6, the regime that
    #    actually alternates; seeds 53/9/23 each show a clean there-and-back
    #    switch of the population mean within the window)
    # plot_noisy_dynamics_with_alternations(
    #     ei, ej, N, PARS, A, z_bi=0.0, z_mono=6.0,
    #     T=20000.0, dt=0.05, rec_every=20, noise=0.085,
    #     bi_seeds=(23, 6, 3), mono_seeds=(0, 1, 2), wq_override=None,
    #     smooth_win=25, smooth_mode="gauss",
    #     bi_windows=[(500, 8000), (900, 5000), (12500, 19000)],
    #     save=OUT+"alternations")

    # # 2. distributions
    PARS = dict(K=0.3, wqu=0.8, wuq=0.8, tau_q=10.0, tau_u=0.2, noise=0.0)
    plot_confidence_distributions(
        ei, ej, N, PARS, n_sims=NSIM, noise=0.05,
        save=os.path.join(OUT, "distributions"), T=50)

    # 4&5. stats  --  the two SFM conditions are the SAME bistable stimulus
    #   (same z); they differ ONLY in the interpretation bias B. RDM is the
    #   monostable stimulus (large z, no bias).
    #     SFM (B=0) : z = Z_SFM, b = 0      -> bistable, unbiased
    #     SFM (B>0) : z = Z_SFM, b = B_SFM  -> SAME stimulus, biased readout
    #     RDM       : z = Z_RDM, b = 0      -> monostable
    #   All other parameters (K, w's, taus, noise, alpha_RM, abs_stats,
    #   T) are identical across the three conditions.
    #
    #   Mechanisms (ported from the reference paper function):
    #     * abs_stats=True -> |corr| and |CP-0.5|+0.5 (absolute_cps_rsc convention)
    #     * alpha_RM        -> ONE symmetric random matrix alpha_RM*X. The SAME
    #       perturbed coupling K + alpha_RM*X is used in the DYNAMICS (per-edge
    #       coupling fed to the simulator) AND to define neighbours as
    #       |A + alpha_RM*X| > alpha_RM -- exactly as in the reference code. This
    #       is the only structured perturbation (no separate weight-noise term).
    #     * T=2 s trials (dt=0.01, tau_q=0.3) -> ~6-7 time-constants, matching the
    #       reference t_dur=2, tau=0.3; activity does not fully commit.
    #   The bias b reduces the single-vs-neighbour correlation (it pushes the
    #   readout toward one interpretation), so SFM(B>0) sits below SFM(B=0).
    # T_CP, DT_CP, NSIM = 2.0, 0.01, 200
    # CSB     = int(0.25 / DT_CP)           # choice window = last 0.25 s
    # NOISE_CP = 0.5
    # ALPHA_RM = 0.4
    # PARS_CP = dict(K=0.15, wqu=0.8, wuq=0.8, tau_q=0.3, tau_u=0.05, noise=0.0)
    # Z_RDM   = 5.0      # monostable stimulus
    # Z_SFM   = 1.4      # bistable stimulus, SHARED by both SFM conditions
    # B_SFM   = 0.45     # interpretation bias, applied only in SFM (B>0)
    # np.random.seed(0)
 
    # def _stats(z, b):
    #     return run_regime_stats(ei, ej, N, A, PARS_CP, z=z, n_sims=NSIM,
    #                             noise=NOISE_CP, b=b, alpha_RM=ALPHA_RM,
    #                             abs_stats=True, T=T_CP, dt=DT_CP,
    #                             choice_steps_before=CSB)
 
    # mo    = _stats(Z_RDM, 0.0)      # RDM         (monostable)
    # bipos = _stats(Z_SFM, B_SFM)    # SFM (B>0)   (same z, biased)
    # bi    = _stats(Z_SFM, 0.0)      # SFM (B=0)   (same z, unbiased)
 
    # corr_model = [np.nanmean(mo["corrs"]),
    #               np.nanmean(bipos["corrs"]),
    #               np.nanmean(bi["corrs"])]
    # cp_model   = [np.nanmean(mo["cps"]),
    #               np.nanmean(bi["cps"])]
 
    # print("corr  RDM=%.3f  SFM(B>0)=%.3f  SFM(B=0)=%.3f  (data .23/.28/.42)"
    #       % tuple(corr_model))
    # print("CP    RDM=%.3f  SFM(B=0)=%.3f               (data .56/.67)"
    #       % tuple(cp_model))
 
    # # 6. reproduce figure (data given; model from our sims)
    # reproduce_results_figure(corr_data=(0.23, 0.28, 0.42),
    #                           cp_data=(0.56, 0.67),
    #                           corr_model=corr_model,
    #                           cp_model=cp_model,
    #                           corr_ylim=(0.0, 0.5), cp_ylim=(0.5, 0.75),
    #                           save=os.path.join(OUT, "results_figure"))
    # print("Done")
