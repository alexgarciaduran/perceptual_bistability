
# -*- coding: utf-8 -*-
"""
Levelt's propositions — 1D projection for Props 1–3,
full N-D system for Proposition 4 (Levelt IV).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# ─── Optional Numba JIT ────────────────────────────────────────────────────────
try:
    from numba import njit as _jit
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    _jit = lambda f: f

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["axes.grid"] = False

# ─── Parameters ────────────────────────────────────────────────────────────────
N        = 3          # half-population size → full system is 2N-dimensional
J        = 0.42
NOISE    = 0.1
TAU      = 0.008
DT       = 0.001
TIME_END = 5000 if _USE_NUMBA else 500
THRESH   = 0.5

B_LIST = np.arange(-0.1, 0.1, 0.0025)

TIME_CTE  = DT / TAU
NOISE_CTE = np.sqrt(TIME_CTE) * NOISE
COUP_CTE  = 2 * N * J
N_STEPS   = int(TIME_END / DT)

# ─── Seed ────────────────────────────────────────────────────────────────
np.random.seed(3)

# ─── 1D mean-field Euler–Maruyama (Props 1–3) ─────────────────────────────────
@_jit
def _run_euler(q0, n_steps, time_cte, coup_cte, b, noise_vals):
    q   = q0
    out = np.empty(n_steps)
    for t in range(n_steps):
        z   = coup_cte * (2.0 * q - 1.0) + 2.0 * b
        sig = 1.0 / (1.0 + np.exp(-z))
        q  += (sig - q) * time_cte + noise_vals[t]
        if   q > 1.0: q = 1.0
        elif q < 0.0: q = 0.0
        out[t] = q
    return out


# ─── Full N-D mean-field Euler–Maruyama (Prop 4 / Levelt IV) ──────────────────
# theta is the (n_units × n_units) coupling matrix; b_vec is per-unit drive.
@_jit
def _run_euler_full(x0, theta, b_vec, n_steps, time_cte, noise_vals):
    n_units = x0.shape[0]
    x       = x0.copy()
    out     = np.empty((n_steps, n_units))
    for t in range(n_steps):
        # coupling term: theta @ (2x − 1)
        coup = theta @ (2.0 * x - 1.0)
        z    = 2.0 * J * coup + 2.0 * b_vec
        sig  = 1.0 / (1.0 + np.exp(-z))
        x    = x + (sig - x) * time_cte + noise_vals[t]
        for k in range(n_units):
            if   x[k] > 1.0: x[k] = 1.0
            elif x[k] < 0.0: x[k] = 0.0
        out[t] = x
    return out


def rle(arr):
    idx = np.concatenate(([0], np.flatnonzero(np.diff(arr)) + 1, [len(arr)]))
    return np.diff(idx), arr[idx[:-1]]


# ─── Simulation (Props 1–3, unchanged) ────────────────────────────────────────
predom_q1, predom_q2 = [], []
mean_t1,   mean_t2   = [], []
alt_rate             = []

if _USE_NUMBA:
    _run_euler(0.5, 2, TIME_CTE, COUP_CTE, 0.0, np.zeros(2))

print(f"Simulating {len(B_LIST)} B values × {N_STEPS:,} steps "
      f"({'numba' if _USE_NUMBA else 'pure Python'}) …")

for i, b in enumerate(tqdm(B_LIST)):

    noise_vals = np.random.randn(N_STEPS) * NOISE_CTE
    states     = _run_euler(float(np.random.rand()), N_STEPS,
                            TIME_CTE, COUP_CTE, float(b), noise_vals)

    predom_q1.append(np.mean(states > 0.5))
    predom_q2.append(np.mean(states < 0.5))

    binary = np.where(states > THRESH, 1.0,
                      np.where(states < 1.0 - THRESH, -1.0, 0.0))
    clean  = binary[binary != 0.0]

    if len(clean) > 2:
        lens, vals = rle(clean)
        t1 = lens[vals ==  1.0] * DT
        t2 = lens[vals == -1.0] * DT
        mean_t1.append(np.nanmean(t1) if len(t1) else np.nan)
        mean_t2.append(np.nanmean(t2) if len(t2) else np.nan)
        alt_rate.append(int(np.sum(np.diff(clean) != 0)))
    else:
        mean_t1.append(np.nan)
        mean_t2.append(np.nan)
        alt_rate.append(0)

print("\nDone (Props 1–3).")

pq1  = np.array(predom_q1)
pq2  = np.array(predom_q2)
mt1  = np.array(mean_t1)
mt2  = np.array(mean_t2)
rate = np.array(alt_rate, dtype=float) / (N_STEPS * DT)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

# Panel 1 · Proposition 1
ax = axes[0]
ax.plot(B_LIST, pq1, color='green', lw=2.5, label='q(x=1)')
ax.plot(B_LIST, pq2, color='red',   lw=2.5, label='q(x=−1)')
ax.set_xlabel('Sensory evidence, B')
ax.set_ylabel('Perceptual predominance ⟨q(x=i)⟩')
ax.set_title('Proposition 1', fontsize=14)
ax.legend(frameon=False)

# Panel 2 · Proposition 2
ax = axes[1]
ax.plot(B_LIST, mt1, color='green', lw=2.5, label='T(x=1)')
ax.plot(B_LIST, mt2, color='red',   lw=2.5, label='T(x=−1)')
ax.set_yscale('log')
ax.set_xlabel('Sensory evidence, B')
ax.set_ylabel('Avg. perceptual dominance, T')
ax.set_title('Proposition 2', fontsize=14)
ax.legend(frameon=False)

# Panel 3 · Proposition 3 — alternation rate vs fraction + entropy
ax  = axes[2]
order = np.argsort(pq1)
ax.plot(pq1[order], rate[order], color='k', lw=2.5)
ax.set_xlabel('Fraction of q(x=1)')
ax.set_ylabel('Alternation rate (Hz)')
ax.set_title('Proposition 3', fontsize=14)
axt = ax.twinx()
xf  = np.linspace(1e-9, 1 - 1e-9, 2000)
axt.plot(xf, -(xf * np.log(xf) + (1 - xf) * np.log(1 - xf)),
         color='grey', ls='--', lw=1.5)
axt.set_ylabel('Binary entropy H(p)', color='grey')
axt.tick_params(axis='y', labelcolor='grey')
axt.spines['top'].set_visible(False)

# Panel 4 · Proposition 3 — log–log dominance
ax    = axes[3]
valid = ~(np.isnan(mt1) | np.isnan(mt2) | (mt1 <= 0) | (mt2 <= 0))
lt1, lt2 = np.log(mt1[valid]), np.log(mt2[valid])
reg  = stats.linregress(lt2, lt1)
xfit = np.linspace(lt2.min(), lt2.max(), 300)
ax.scatter(lt2, lt1, color='k', s=18, zorder=3, label='Simulation')
ax.plot(xfit, reg.slope * xfit + reg.intercept, 'b--', lw=1.8, alpha=0.7,
        label=f'y ~ log(x), slope = {reg.slope:.2f}')
ax.set_xlabel('log T(x=−1)')
ax.set_ylabel('log T(x=1)')
ax.set_title('Proposition 3', fontsize=14)
ax.legend(frameon=False)

# Panel 5 · Alternation rate vs B
ax = axes[4]
ax.plot(B_LIST, rate, color='k', lw=2.5)
ax.axvline(0, color='grey', ls=':', lw=1)
ax.set_xlabel('Sensory evidence, B')
ax.set_ylabel('Alternation rate')
ax.set_title('Prop. 3 — Rate vs B', fontsize=14)

# ─── Panel 6 · Proposition 4 (Levelt IV) — FULL N-D SYSTEM ─────────────────────
# Levelt IV: increasing common/symmetric input to BOTH interpretations
# raises the alternation rate. The two interpretations are the two halves
# of the population, driven by +b and −b respectively. The magnitude b is
# the "common input strength".
n_units = 8
theta = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]], dtype=np.float64)


# warm-up JIT for the full solver
if _USE_NUMBA:
    _run_euler_full(np.full(n_units, 0.5), theta, np.zeros(n_units),
                    2, TIME_CTE, np.zeros((2, n_units)))

b_common_list = np.arange(0.0, 0.2, 0.01)
alt_rate_iv   = []

print(f"Simulating Prop 4 (full {n_units}-D system) over "
      f"{len(b_common_list)} input strengths …")

for i, b in enumerate(tqdm(b_common_list)):

    # split drive: first half +b, second half −b
    b_vec = np.full(n_units, b)
    b_vec[n_units // 2:] = -b

    x0         = np.random.rand(n_units)
    noise_vals = np.random.randn(N_STEPS, n_units) * NOISE_CTE * 2
    vec        = _run_euler_full(x0, theta, b_vec, N_STEPS, TIME_CTE, noise_vals)

    # project to 1D: mean activity, then binarise
    mean_states = np.clip(np.mean(vec, axis=1), 0.0, 1.0)
    binary = np.where(mean_states > THRESH, 1.0,
                      np.where(mean_states < 1.0 - THRESH, -1.0, 0.0))
    clean  = binary[binary != 0.0]

    alt_rate_iv.append(int(np.sum(np.diff(clean) != 0)) if len(clean) > 2 else 0)

print("\nDone (Prop 4).")

ax = axes[5]
T_total = N_STEPS * DT          # total simulated time (model units)
ax.plot(b_common_list, np.array(alt_rate_iv, dtype=float) / T_total, color='k', lw=2.5)
ax.set_xlabel('Common sensory input strength')
ax.set_ylabel('Alternation rate (Hz)')
ax.set_title('Proposition 4', fontsize=14)

# plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.tight_layout()
plt.savefig('levelt_propositions.png', dpi=200, bbox_inches='tight')
plt.savefig('levelt_propositions.svg', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: levelt_propositions.png")

