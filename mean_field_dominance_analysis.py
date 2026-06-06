# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:35:47 2026

@author: alexg
"""

from scipy.stats import gamma as gamma_dist, lognorm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl


mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams["axes.grid"] = False


def simulate_mf(J=3.0, B=0.0, noise_amp=0.20, tau_ou=8.0,
                adapt=True, ga=1.3, tau_a=150.0,
                T=500_000.0, dt=0.05, q0=None, seed=None):
    rng = np.random.default_rng(seed)
    n   = int(T / dt)
    q0  = float(np.clip(q0 if q0 is not None else 0.5 + 0.1*rng.standard_normal(), 0, 1))
    q   = np.empty(n);  q[0] = q0
    a   = np.zeros(n)
    eta = np.zeros(n)
    ou_decay = np.exp(-dt / tau_ou)
    ou_sig   = noise_amp * np.sqrt(1.0 - ou_decay**2)
    time = np.arange(0, T, dt)
    noisyframes = 15 // dt // 60
    nFrame = len(time)
    time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
    noise_exp = np.random.randn(len(time_interp))
    B = scipy.interpolate.interp1d(time_interp, noise_exp)(time)*0.5
    
    
    if adapt:
        for i in range(n - 1):
            qi, ai, ei = q[i], a[i], eta[i]
            s        = 2.0*qi - 1.0
            dq       = scipy.special.expit(2.0*J*(s - ga*ai) + 2.0*B[i]) - qi
            q[i+1]   = np.clip(qi + dq*dt + ei*dt, 0, 1)
            a[i+1]   = ai + (s - ai)/tau_a * dt
            eta[i+1] = ou_decay*ei + ou_sig*rng.standard_normal()
    else:
        for i in range(n - 1):
            qi, ei   = q[i], eta[i]
            dq       = scipy.special.expit(2.0*J*(2.0*qi - 1.0) + 2.0*B) - qi
            q[i+1]   = np.clip(qi + dq*dt + ei*dt, 0, 1)
            eta[i+1] = ou_decay*ei + ou_sig*rng.standard_normal()
    t = np.arange(n) * dt
    return t, q, a, eta


def dominance_durations_mf(t, q, threshold_up=0.65, threshold_dn=0.35, min_dur=0.1):
    """
    Hysteresis-based state detector (Schmitt trigger).
    
    State = +1 (UP)   once q crosses threshold_up from below.
    State = -1 (DOWN) once q crosses threshold_dn from above.
    
    A dominance epoch is the time spent continuously in one state.
    The dead zone (threshold_dn < q < threshold_up) is transient — 
    state is held until the opposite threshold is crossed.
    """
    n = len(q)
    state  = np.zeros(n, dtype=np.int8)   # 0 = uninitialized
    
    # initialise: find first committed state
    if q[0] > threshold_up:
        current = 1
    elif q[0] < threshold_dn:
        current = -1
    else:
        current = 0   # start in dead zone, wait
    
    for i in range(n):
        if current == 0:
            if q[i] > threshold_up:
                current = 1
            elif q[i] < threshold_dn:
                current = -1
        else:
            if current == 1 and q[i] < threshold_dn:
                current = -1
            elif current == -1 and q[i] > threshold_up:
                current = 1
        state[i] = current
    
    # find transitions between +1 and -1 (ignore 0s at the start)
    committed = np.where(state != 0)[0]
    if len(committed) == 0:
        return np.array([])
    
    s = state[committed]
    t_c = t[committed]
    breaks = np.where(np.diff(s) != 0)[0] + 1          # indices into committed
    edges  = np.concatenate(([0], breaks, [len(s)]))
    durs   = np.array([t_c[min(edges[i+1]-1, len(t_c)-1)] - t_c[edges[i]]
                        for i in range(len(edges)-1)])
    return durs[durs >= min_dur]


def dominance_durations_mf_v0(t, q, threshold_up=0.5, threshold_dn=None, min_dur=0.01):
    if threshold_dn is None:
        threshold_dn = threshold_up
    if threshold_dn == threshold_up:
        # symmetric: every sample belongs to one side
        state  = (q > threshold_up).astype(np.int8)
        breaks = np.where(np.diff(state) != 0)[0] + 1
        edges  = np.concatenate(([0], breaks, [len(t) - 1]))
        durs   = np.diff(t[edges])
    else:
        # asymmetric: only count time spent outside the dead zone
        in_dom = (q > threshold_up) | (q < threshold_dn)
        enters = np.where(np.diff(np.concatenate(([0],   in_dom.astype(int)))) ==  1)[0]
        exits  = np.where(np.diff(np.concatenate((in_dom.astype(int), [0]))) == -1)[0]
        m      = min(len(enters), len(exits))
        if m == 0:
            return np.array([])
        durs = np.array([t[min(exits[i], len(t)-1)] - t[enters[i]] for i in range(m)])
    return durs[durs >= min_dur]


def plot_dominance_histogram(durations, color='peru', label=None, fit='gamma',
                             alpha=0.55, ax=None, title=None, xlim=None,
                             n_bins=50, show_fit=True, figsize=(6, 4)):
    # normalise inputs to lists so one or two distributions both work
    if isinstance(durations, np.ndarray):
        durations = [durations]
    n     = len(durations)
    color = [color]*n if isinstance(color, str) else list(color)
    label = [label]*n if (label is None or isinstance(label, str)) else list(label)
    fit   = [fit]*n   if (fit is None or isinstance(fit, str)) else list(fit)

    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    cap  = np.max(durations)+0.1
    bins = np.linspace(0, cap, n_bins + 1)

    for d, c, l, f in zip(durations, color, label, fit):
        counts, edges = np.histogram(d[d <= cap], bins=bins)
        widths  = np.diff(edges)
        density = counts / (counts.sum() * widths + 1e-12)
        ax.bar(edges[:-1], density, width=widths,
               color=c, alpha=alpha, edgecolor='none', align='edge')
        if show_fit:
            xs = np.linspace(1e-3, cap, 600)
            if f == 'gamma':
                k, loc, theta = gamma_dist.fit(d, floc=0)
                pdf  = gamma_dist.pdf(xs, k, loc, theta)
                mode = max((k-1)*theta, 0.0)
                lbl  = ((l+'\n') if l else '') + f'Gamma: k={k:.2f},\nθ={theta:.1f}'
            elif f == 'exp':
                lam = 1.0 / np.mean(d)
                pdf = lam * np.exp(-lam * xs)
                lbl = ((l+'\n') if l else '') + f'Exponential: λ={lam:.3f},\nmean={1/lam:.1f}'
            elif f == 'lognorm':
                sh, loc, sc = lognorm.fit(d, floc=0)
                pdf = lognorm.pdf(xs, sh, loc, sc)
                lbl = ((l+'\n') if l else '') + f'Log-normal: μ={np.mean(np.log(d)):.2f}, σ={np.std(np.log(d)):.2f}'
            else:
                show_fit = False
                continue
            ax.plot(xs, pdf, color=c, lw=3., label=lbl)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_xlabel('Dominance duration (s)')
    ax.set_ylabel('Density')
    ax.set_xlim(*(xlim if xlim else (0, cap)))
    color_title = 'peru' if title == 'Bistable' else 'cadetblue'
    if title:
        ax.set_title(title, color=color_title, fontsize=15)
    if show_fit:
        ax.legend(frameon=False)
    plt.tight_layout()
    return ax


def run_bistable_vs_monostable(J_high=3.0, noise_high=0.20,
                                J_low=0.7,  noise_low=0.20,
                                threshold_up=0.5, threshold_dn=0.5,
                                B=0.0, tau_ou=.20, ga=1.3, tau_a=50.0,
                                T=100_000.0, dt=0.05, seed=0,
                                figsize=(12, 5), save_path=None):
    print(f"Simulating T={T:,.0f} ({int(T/dt):,} steps) x 2 ...")

    t_hi, q_hi, a_hi, eta_hi = simulate_mf(J=J_high, B=B, noise_amp=noise_high,
                                            tau_ou=tau_ou, adapt=True, ga=ga, tau_a=tau_a,
                                            T=T, dt=dt, seed=seed)
    dur_hi = dominance_durations_mf(t_hi, q_hi, threshold_up=threshold_up, threshold_dn=threshold_dn)
    cv_hi  = np.std(dur_hi) / np.mean(dur_hi)
    print(f"  [1/2] bistable   J={J_high}: n={len(dur_hi):,}  median={np.median(dur_hi):.1f}  CV={cv_hi:.2f}")

    t_lo, q_lo, a_lo, eta_lo = simulate_mf(J=J_low, B=B, noise_amp=noise_low,
                                            tau_ou=tau_ou, adapt=True, ga=ga, tau_a=tau_a,
                                            T=T, dt=dt, seed=seed)
    dur_lo = dominance_durations_mf(t_lo, q_lo, threshold_up=threshold_up, threshold_dn=threshold_dn)
    cv_lo  = np.std(dur_lo) / np.mean(dur_lo)
    print(f"  [2/2] monostable J={J_low}:  n={len(dur_lo):,}  median={np.median(dur_lo):.1f}  CV={cv_lo:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_dominance_histogram(
        dur_hi, color='peru', fit='', ax=axes[1],
        title='Bistable',
        xlim=(-0.1, 20.1)
      )
    #   title=(f'High J = {J_high}  (bistable)\n'
    #            f'n={len(dur_hi):,}   median={np.median(dur_hi):.1f}   CV={cv_hi:.2f}')

    plot_dominance_histogram(
        dur_lo, color='cadetblue', fit='', ax=axes[0],
        title='Monostable',
        xlim=(-0.1, 20.1)
        )
    # title=(f'Low J = {J_low}  (monostable,  thr=[{threshold_dn},{threshold_up}])\n'
    #            f'n={len(dur_lo):,}   median={np.median(dur_lo):.1f}   CV={cv_lo:.2f}')

    # fig.suptitle(
    #     r'$\dot{q}=\sigma(2J[(2q-1)-g_a a]+2B)-q+\eta(t)$'
    #     f'   |   $g_a={ga},\\ \\tau_a={tau_a},\\ \\tau_{{\\rm ou}}={tau_ou}$',
    #     fontsize=14)
    axes[1].set_ylabel('')  # only one y-axis label for the whole figure
    axes[1].set_xlabel('')
    for a in axes:
        a.set_xticks([0, 10, 20])

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, {
        'bistable':   {'t': t_hi, 'q': q_hi, 'a': a_hi, 'eta': eta_hi, 'durations': dur_hi},
        'monostable': {'t': t_lo, 'q': q_lo, 'a': a_lo, 'eta': eta_lo, 'durations': dur_lo},
    }


if __name__ == '__main__':
    print('MF dominance')
    fig, results = run_bistable_vs_monostable(
        J_high       = 3,
        J_low        = 0.1,
    
        noise_high   = 0.35,
        noise_low    = 0.35,      # SAME — only J changes
    
        tau_ou       = 0.1,       # 200 ms  (Moreno-Bote 2007)
        ga           = 1.4,
        tau_a        = 11.0,
    
        threshold_up = 0.55,
        threshold_dn = 0.45,
    
        T            = 10_000.0,
        dt           = 0.01,
        seed         = 42,

        # save_path     = 'mf_dominance_histograms.svg',
        figsize=(5.2, 2.5)
    )
