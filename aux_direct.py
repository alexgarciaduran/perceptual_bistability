"""
Exact reduction of a triplewise term to a pairwise+unary auxiliary graph,
solved DIRECTLY from the 8 configurations (no intermediate parity 'c' step).

Model (p ~ exp(-E)):
    E_orig(s) = a0 + a1 x + a2 y + a3 z + a4 xy + a5 xz + a6 yz + a7 xyz
    E_aux(s,u)= b0 + b1 x + b2 y + b3 z + b4 xy + b5 xz + b6 yz          (visible)
                + b7 u x + b8 u y + b9 u z + b10 u                       (aux)

Marginalizing u exactly:
    sum_u exp(-E_aux) = 2 exp(-E_vis) cosh(b7 x + b8 y + b9 z + b10)

Matching  exp(-E_orig(s)) = sum_u exp(-E_aux(s,u))  at each of the 8 configs s,
and taking -log, gives 8 linear-in-(b0..b6) equations:

    E_vis(s) = E_orig(s) + ln 2 + ln cosh( b7 x + b8 y + b9 z + b10 )         (8 eqs)

Write E_vis(s) = P(s)·b_vis where P(s) are the 7 visible parity regressors
[1, x, y, z, xy, xz, yz]. Because the 8 parity monomials (those 7 PLUS xyz) are an
orthogonal basis on the cube, the right-hand side decomposes uniquely as

    RHS(s) = Σ_chi r_chi · chi(s),   r_chi = (1/8) Σ_s chi(s) RHS(s).

- The 7 visible components r_chi (chi ≠ xyz) directly GIVE b0..b6.
- The single xyz component r_xyz must VANISH for an exact visible fit — this is the
  one nonlinear constraint that fixes the aux couplings (b7..b10).

There are 11 unknowns and effectively 8 constraints -> a 3-parameter family of exact
solutions. We pick the fully-symmetric one b7=b8=b9=b10=w (w free, may be negative),
so a SINGLE scalar root-find in w handles both signs of a7 with no sign-flip / no
asymmetry.

DEFINITION.  "c_xyz" used earlier == r_xyz evaluated with a7 removed, i.e. the xyz
parity component of ln cosh(...). Here we never need it separately: we just require
the xyz component of the full RHS to be zero.
"""

import numpy as np
import itertools
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os

# ---- path to save imgs ----
DATA_DIR = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/pgm_aux/figures/'  # Alex

# ---- cube and the 8 parity regressors (orthogonal basis) ----
CONFIGS = np.array(list(itertools.product([-1, 1], repeat=3)))
X, Y, Z = CONFIGS[:, 0], CONFIGS[:, 1], CONFIGS[:, 2]
# design matrix columns: const, x, y, z, xy, xz, yz, xyz
DESIGN = np.column_stack([np.ones(8), X, Y, Z, X*Y, X*Z, Y*Z, X*Y*Z])
NAMES = ["const", "x", "y", "z", "xy", "xz", "yz", "xyz"]

def E_orig(a):
    """Energy of original model at all 8 configs. a = length-8 array (a0..a7)."""
    return (a[0] + a[1]*X + a[2]*Y + a[3]*Z
            + a[4]*X*Y + a[5]*X*Z + a[6]*Y*Z + a[7]*X*Y*Z)

def rhs_parity(a, b7, b8, b9, b10):
    """
    Parity components of RHS(s) = E_orig(s) + ln2 + ln cosh(b7 x+b8 y+b9 z+b10).
    Returns dict over NAMES. b0..b6 read off the visible components; the 'xyz'
    component must be zero for an exact fit.
    """
    A = b7 * X + b8 * Y + b9 * Z + b10
    aA = np.abs(A)
    lncosh = aA + np.log1p(np.exp(-2 * aA)) - np.log(2.0)   # stable ln cosh
    RHS = E_orig(a) + np.log(2.0) + lncosh
    coeff = (DESIGN.T @ RHS) / 8.0             # Hadamard inversion
    return dict(zip(NAMES, coeff))

def solve_b(a):
    """
    Direct solve of b0..b10 from a0..a7 using the 8-config matching.
    Symmetric family b7=b8=b9=b10=w; w found by driving the xyz residual to 0.
    """
    a = np.asarray(a, float)
    a7 = a[7]
    if a7 < 0:
        raise ValueError("This solver is restricted to a7 >= 0.")

    # Fully x<->y<->z symmetric family: b7 = b8 = b9 = b10 = t (all equal, t>=0).
    # Because the three couplings are identical, x, y, z are treated the same way,
    # so the auxiliary adds NO spurious asymmetry; any x/y/z differences in the
    # resulting b's come only from asymmetry already present in the a's.
    # The xyz-residual starts at a7 (t=0) and decreases through zero as t grows.
    def resid(t):
        return rhs_parity(a, t, t, t, t)["xyz"]

    if a7 < 1e-12:
        t = 0.0
    else:
        hi = 1.0
        while resid(1e-9) * resid(hi) > 0 and hi < 200:
            hi *= 1.4
        t = brentq(resid, 1e-9, hi, xtol=1e-12)

    coeff = rhs_parity(a, t, t, t, t)
    b = {
        "b0": coeff["const"], "b1": coeff["x"], "b2": coeff["y"], "b3": coeff["z"],
        "b4": coeff["xy"], "b5": coeff["xz"], "b6": coeff["yz"],
        "b7": t, "b8": t, "b9": t, "b10": t,
    }
    return b

# ---------------------------------------------------------------------------
# CLOSED-FORM analytic solver (no root-find).
# With b7=b8=b9=b10=t the ln cosh parity expansion has only two nonzero values:
#     p(t) = (1/8) ln cosh 4t      -> shifts ALL fields & pairwise terms equally
#     q(t) = (1/2) ln cosh 2t      -> extra piece for constant & triple residual
# Constraint q(t) - p(t) = a7  solves in closed form:
#     cosh^2(2t) = e^{4a7} ( e^{4a7} + sqrt(e^{8a7} - 1) )
# ---------------------------------------------------------------------------
def solve_b_analytic(a):
    a = np.asarray(a, float)
    a7 = a[7]
    if a7 < 0:
        raise ValueError("Analytic solver restricted to a7 >= 0.")
    if a7 < 1e-15:
        t = 0.0
    else:
        E4 = np.exp(4.0 * a7)
        cosh2_2t = E4 * (E4 + np.sqrt(np.expm1(8.0 * a7) + 1.0 - 1.0))  # e^{4a7}(e^{4a7}+sqrt(e^{8a7}-1))
        # guard tiny negative under sqrt from rounding
        val = np.exp(8.0 * a7) - 1.0
        cosh2_2t = E4 * (E4 + np.sqrt(max(val, 0.0)))
        t = 0.5 * np.arccosh(np.sqrt(cosh2_2t))
    p = np.log(np.cosh(4.0 * t)) / 8.0
    q = np.log(np.cosh(2.0 * t)) / 2.0
    b = {
        "b0": a[0] + q + p + np.log(2.0),
        "b1": a[1] + p, "b2": a[2] + p, "b3": a[3] + p,
        "b4": a[4] + p, "b5": a[5] + p, "b6": a[6] + p,
        "b7": t, "b8": t, "b9": t, "b10": t,
    }
    return b, t, p, q

def max_pdiff(a, b):
    """Max abs difference between original posterior and u-marginal of aux model."""
    a = np.asarray(a, float)
    def E_aux(s, u):
        x, y, z = s
        return (b["b0"] + b["b1"]*x + b["b2"]*y + b["b3"]*z
                + b["b4"]*x*y + b["b5"]*x*z + b["b6"]*y*z
                + b["b7"]*u*x + b["b8"]*u*y + b["b9"]*u*z + b["b10"]*u)
    po = np.exp(-E_orig(a)); po /= po.sum()
    pa = np.array([sum(np.exp(-E_aux(s, u)) for u in (-1, 1)) for s in CONFIGS]); pa /= pa.sum()
    return float(np.max(np.abs(po - pa)))

# ---------------------------------------------------------------------------
# Sweep J = a7, keep a1..a6 fixed, record all b's
# ---------------------------------------------------------------------------
A_FIXED = np.array([0.0, 0.30, -0.20, 0.10, 0.40, -0.15, 0.25, 0.0])  # a7 filled per J
J_vals = np.linspace(0.0, 1.2, 241)

rec = {f"b{i}": [] for i in range(11)}
err = 0.0
for J in J_vals:
    a = A_FIXED.copy(); a[7] = J
    b = solve_b(a)
    for i in range(11):
        rec[f"b{i}"].append(b[f"b{i}"])
    err = max(err, max_pdiff(a, b))
for k in rec:
    rec[k] = np.array(rec[k])

print(f"Max round-trip |Δp| over sweep: {err:.2e}")
a = A_FIXED.copy(); a[7] = 0.5
bs = solve_b(a)
print("Solution at J=0.5 (symmetric, w<0):")
for i in range(11):
    print(f"  b{i:<2d} = {bs[f'b{i}']:+.4f}")

# ---------------------------------------------------------------------------
# Plot: panels grouped by term type; each curve labelled by what it weighs
# ---------------------------------------------------------------------------
UNARY    = [("b1", r"$b_1\,(x)$"), ("b2", r"$b_2\,(y)$"), ("b3", r"$b_3\,(z)$")]
PAIRWISE = [("b4", r"$b_4\,(xy)$"), ("b5", r"$b_5\,(xz)$"), ("b6", r"$b_6\,(yz)$")]
AUX      = [("b7", r"$b_7\,(ux)$"), ("b8", r"$b_8\,(uy)$"), ("b9", r"$b_9\,(uz)$"),
            ("b10", r"$b_{10}\,(u)$"), ("b0", r"$b_0\,(\mathrm{const})$")]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

def panel(ax, items, title, cmap):
    cols = cmap(np.linspace(0.15, 0.85, len(items)))
    for (key, lab), col in zip(items, cols):
        ax.plot(J_vals, rec[key], label=lab, color=col, lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"triple coefficient  $J=a_7$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")

panel(axes[0], UNARY,    "Visible UNARY fields\n(weigh single sites)",             plt.cm.autumn)
panel(axes[1], PAIRWISE, "Visible PAIRWISE couplings\n(weigh site pairs)",        plt.cm.winter)
panel(axes[2], AUX,      "AUX: $u$–site couplings & $u$ field\n(create the triple)", plt.cm.viridis)
axes[0].set_ylabel("coefficient value")

fig.suptitle(
    r"$b_i(J)$ from the direct 8-configuration solve, $J\geq 0$  "
    r"(symmetric family $b_7=b_8=b_9=b_{10}=t$)",
    fontsize=13, y=1.04)
fig.tight_layout()
print("Saved: fig_b_direct_panels.png")

# ===========================================================================
# EXTRA VERIFICATION PLOTS for the analytic solution
# ===========================================================================
# Recompute over the sweep with BOTH solvers to compare.
t_numeric, t_analytic = [], []
p_of_t, q_of_t = [], []
resid_closed = []          # closed-form residual q-p-a7 (should be ~0 by construction)
b1_num, b1_ana = [], []    # spot-check one coefficient
err_ana = 0.0
for J in J_vals:
    a = A_FIXED.copy(); a[7] = J
    bn = solve_b(a)                       # numeric root-find
    ba, t, p, q = solve_b_analytic(a)     # closed form
    t_numeric.append(bn["b7"]); t_analytic.append(t)
    p_of_t.append(p); q_of_t.append(q)
    resid_closed.append((q - p) - J)      # constraint check
    b1_num.append(bn["b1"]); b1_ana.append(ba["b1"])
    err_ana = max(err_ana, max_pdiff(a, ba))

t_numeric = np.array(t_numeric); t_analytic = np.array(t_analytic)
p_of_t = np.array(p_of_t); q_of_t = np.array(q_of_t)
resid_closed = np.array(resid_closed)
b1_num = np.array(b1_num); b1_ana = np.array(b1_ana)

print(f"\nMax |Δp| using ANALYTIC b over sweep: {err_ana:.2e}")
print(f"Max |t_analytic - t_numeric|:         {np.max(np.abs(t_analytic-t_numeric)):.2e}")
print(f"Max |closed-form residual (q-p-a7)|:  {np.max(np.abs(resid_closed)):.2e}")

fig2, ax = plt.subplots(2, 2, figsize=(12, 8))

# (a) analytic vs numeric t
ax[0,0].plot(J_vals, t_numeric, lw=6, color="#cbd5e1", label="numeric root-find")
ax[0,0].plot(J_vals, t_analytic, lw=2, color="#e11d48", ls="--", label="closed form")
ax[0,0].set_title(r"(a) aux coupling $t(a_7)$: analytic vs numeric")
ax[0,0].set_xlabel(r"$a_7$"); ax[0,0].set_ylabel(r"$t$")
ax[0,0].legend(); ax[0,0].grid(alpha=0.25)

# (b) the two scalar functions p(t), q(t) and q-p=a7
ax[0,1].plot(J_vals, p_of_t, color="#2563eb", lw=2, label=r"$p(t)=\frac{1}{8}\ln\cosh 4t$")
ax[0,1].plot(J_vals, q_of_t, color="#16a34a", lw=2, label=r"$q(t)=\frac{1}{2}\ln\cosh 2t$")
ax[0,1].plot(J_vals, q_of_t - p_of_t, color="#f59e0b", lw=2, ls="--",
             label=r"$q-p$ (should $=a_7$)")
ax[0,1].plot(J_vals, J_vals, color="k", lw=0.8, ls=":", label=r"$y=a_7$")
ax[0,1].set_title(r"(b) the two scalar functions & the constraint")
ax[0,1].set_xlabel(r"$a_7$"); ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=0.25)

# (c) closed-form residual (machine-precision zero)
ax[1,0].plot(J_vals, resid_closed, color="#7c3aed", lw=2)
ax[1,0].set_title(r"(c) constraint residual $q(t)-p(t)-a_7$")
ax[1,0].set_xlabel(r"$a_7$"); ax[1,0].set_ylabel("residual")
ax[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax[1,0].grid(alpha=0.25)

# (d) spot check: analytic b1 vs numeric b1, and the diagonal law b1 = a1 + p
ax[1,1].plot(J_vals, b1_num, lw=6, color="#cbd5e1", label=r"$b_1$ numeric")
ax[1,1].plot(J_vals, b1_ana, lw=2, color="#e11d48", ls="--", label=r"$b_1=a_1+p(t)$")
ax[1,1].axhline(A_FIXED[1], color="grey", lw=0.8, ls=":", label=r"$a_1$ (baseline)")
ax[1,1].set_title(r"(d) single-coefficient law $b_1 = a_1 + p(t)$")
ax[1,1].set_xlabel(r"$a_7$"); ax[1,1].set_ylabel(r"$b_1$")
ax[1,1].legend(fontsize=9); ax[1,1].grid(alpha=0.25)

fig2.suptitle("Verification of the closed-form analytic transform", fontsize=14, y=1.00)
fig2.tight_layout()
print("Saved: fig_verify_analytic.png")

# ===========================================================================
# EXPLICIT MARGINALIZATION DEMO
# Prove: sum_u exp(-E_aux(x,y,z,u)) reproduces exp(-E_orig) (same joint).
# ===========================================================================
print("\n" + "="*70)
print("MARGINALIZATION DEMONSTRATION  (a7 = 0.5)")
print("="*70)
a_demo = A_FIXED.copy(); a_demo[7] = 0.5
b_demo, t_demo, _, _ = solve_b_analytic(a_demo)

def E_orig_s(a, s):
    x, y, z = s
    return (a[0] + a[1]*x + a[2]*y + a[3]*z
            + a[4]*x*y + a[5]*x*z + a[6]*y*z + a[7]*x*y*z)

def E_aux_s(b, s, u):
    x, y, z = s
    return (b["b0"] + b["b1"]*x + b["b2"]*y + b["b3"]*z
            + b["b4"]*x*y + b["b5"]*x*z + b["b6"]*y*z
            + b["b7"]*u*x + b["b8"]*u*y + b["b9"]*u*z + b["b10"]*u)

# Unnormalized weights
print(f"\n{'(x, y, z)':>12} | {'exp(-E_orig)':>13} | {'Σ_u exp(-E_aux)':>16} | {'ratio':>8}")
print("-"*60)
w_orig, w_marg = [], []
for s in CONFIGS:
    wo = np.exp(-E_orig_s(a_demo, s))
    wm = sum(np.exp(-E_aux_s(b_demo, s, u)) for u in (-1, 1))
    w_orig.append(wo); w_marg.append(wm)
    print(f"{str(tuple(int(v) for v in s)):>12} | {wo:13.6f} | {wm:16.6f} | {wm/wo:8.4f}")
w_orig = np.array(w_orig); w_marg = np.array(w_marg)

# The ratio is a CONSTANT (= 2 * exp(-const shift)); after normalization joints match.
po = w_orig / w_orig.sum()
pm = w_marg / w_marg.sum()
print("-"*60)
print(f"Unnormalized ratio Σ_u e^-E_aux / e^-E_orig is constant: "
      f"min={w_marg.min()/w_orig.min() if False else (w_marg/w_orig).min():.6f}, "
      f"max={(w_marg/w_orig).max():.6f}")
print(f"  (constant ratio -> identical normalized joint)")
print(f"\nMax |p_orig - p_marginalized| = {np.max(np.abs(po-pm)):.3e}")

# Figure: side-by-side normalized joint
labels = [str(tuple(int(v) for v in s)) for s in CONFIGS]
xidx = np.arange(8)
fig3, axm = plt.subplots(1, 2, figsize=(13, 4.6))
axm[0].bar(xidx-0.2, po, width=0.4, label=r"$p_{\mathrm{orig}}(x,y,z)$", color="#2563eb")
axm[0].bar(xidx+0.2, pm, width=0.4, label=r"$\sum_u p_{\mathrm{aux}}(x,y,z,u)$",
           color="#f59e0b", alpha=0.85)
axm[0].set_xticks(xidx); axm[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
axm[0].set_ylabel("probability"); axm[0].set_title("(a) joint over $(x,y,z)$: original vs $u$-marginal")
axm[0].legend(); axm[0].grid(alpha=0.25, axis="y")

axm[1].plot(xidx, po - pm, "o-", color="#7c3aed")
axm[1].set_xticks(xidx); axm[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
axm[1].set_title("(b) difference $p_{\\mathrm{orig}} - p_{\\mathrm{marg}}$ (machine zero)")
axm[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axm[1].grid(alpha=0.25)
fig3.suptitle(r"Marginalizing $u$ reproduces the original joint ($a_7=0.5$)",
              fontsize=13, y=1.02)
fig3.tight_layout()
print("Saved: fig_marginalization.png")

# ===========================================================================
# MULTI-CONFIGURATION ANALYSIS: vary the a-vector, fix a7, solve b's.
# Shows t depends ONLY on a7; visible b's = a_i + p (diagonal law).
# ===========================================================================
print("\n" + "="*70)
print("MULTI-CONFIGURATION ANALYSIS (all at a7 = 0.4)")
print("="*70)
cases = {
    "symmetric":     [0, 0.25, 0.25, 0.25,  0.30,  0.30, 0.30, 0.4],
    "unary-asym":    [0, 0.60, -0.30, 0.10,  0.30,  0.30, 0.30, 0.4],
    "pairwise-asym": [0, 0.25, 0.25, 0.25,  0.70, -0.20, 0.40, 0.4],
    "all-random":    [0, 0.50, -0.40, 0.20, -0.30,  0.60, 0.10, 0.4],
    "strong-fields": [0, 1.20, -1.00, 0.80,  0.10,  0.10, 0.10, 0.4],
}
case_b = {}
for name, a in cases.items():
    b, t, p, q = solve_b_analytic(a)
    case_b[name] = (a, b, t, p)
    print(f"  {name:<14} t={t:.4f}  p={p:.4f}  "
          f"b1..3=[{b['b1']:.3f},{b['b2']:.3f},{b['b3']:.3f}]  "
          f"b4..6=[{b['b4']:.3f},{b['b5']:.3f},{b['b6']:.3f}]  "
          f"|Δp|={max_pdiff(a, b):.1e}")

# Figure: each case, plot a_i and b_i to show the constant vertical shift p.
fig4, ax4 = plt.subplots(1, 2, figsize=(13, 5))
names_order = list(cases.keys())
colors4 = plt.cm.tab10(np.linspace(0, 1, len(names_order)))
xpos = np.arange(6)  # b1..b6
lab6 = ["b1(x)", "b2(y)", "b3(z)", "b4(xy)", "b5(xz)", "b6(yz)"]

for name, col in zip(names_order, colors4):
    a, b, t, p = case_b[name]
    a_vals = [a[i] for i in range(1, 7)]
    b_vals = [b[f"b{i}"] for i in range(1, 7)]
    ax4[0].plot(xpos, a_vals, "o--", color=col, alpha=0.5, lw=1)
    ax4[0].plot(xpos, b_vals, "o-", color=col, lw=2, label=name)
ax4[0].set_xticks(xpos); ax4[0].set_xticklabels(lab6, rotation=30, ha="right")
ax4[0].set_title(r"(a) $a_i$ (dashed) $\to$ $b_i$ (solid): constant lift by $p(t)$")
ax4[0].set_ylabel("coefficient"); ax4[0].legend(fontsize=8); ax4[0].grid(alpha=0.25)

# The lift b_i - a_i should equal p for ALL i and ALL cases (single number per case)
for name, col in zip(names_order, colors4):
    a, b, t, p = case_b[name]
    lifts = [b[f"b{i}"] - a[i] for i in range(1, 7)]
    ax4[1].plot(xpos, lifts, "o-", color=col, lw=2, label=f"{name} (p={p:.3f})")
ax4[1].set_xticks(xpos); ax4[1].set_xticklabels(lab6, rotation=30, ha="right")
ax4[1].set_title(r"(b) lift $b_i - a_i = p(t)$ — flat & identical across $i$")
ax4[1].set_ylabel(r"$b_i - a_i$"); ax4[1].legend(fontsize=8); ax4[1].grid(alpha=0.25)
fig4.suptitle(r"Different $a$-vectors (fixed $a_7=0.4$): same $t$, diagonal shift $p$",
              fontsize=13, y=1.01)
fig4.tight_layout()
print("Saved: fig_multiconfig.png")

# ===========================================================================
# FIXED-t TEST: pick t directly (not from a root-find), read off which a7 it
# realizes, and verify BOTH (i) analytic vs numeric b's and (ii) marginalization.
# ===========================================================================
print("\n" + "="*70)
print("FIXED-t TESTS (choose t, derive the a7 it produces, verify)")
print("="*70)
a_base = np.array([0.0, 0.30, -0.20, 0.10, 0.40, -0.15, 0.25, 0.0])
t_fixed_vals = [0.3, 0.7, 1.2, 2.0]

fig5, ax5 = plt.subplots(1, 2, figsize=(13, 5))

# (a) For each fixed t, the realized a7 = q(t)-p(t); build b analytically and by the
#     direct projection, compare, and record marginalization error.
realized_a7, ana_num_err, marg_err = [], [], []
for t in t_fixed_vals:
    p = np.log(np.cosh(4*t))/8
    q = np.log(np.cosh(2*t))/2
    a7 = q - p
    a = a_base.copy(); a[7] = a7
    # analytic b (should recover this exact t)
    b_ana, t_rec, _, _ = solve_b_analytic(a)
    # numeric b (independent root-find)
    b_num = solve_b(a)
    realized_a7.append(a7)
    ana_num_err.append(abs(t_rec - t))
    marg_err.append(max_pdiff(a, b_ana))

ax5[0].plot(t_fixed_vals, realized_a7, "o-", color="#2563eb", lw=2)
ax5[0].set_xlabel(r"fixed aux coupling $t$")
ax5[0].set_ylabel(r"realized triple $a_7 = q(t)-p(t)$")
ax5[0].set_title(r"(a) which $a_7$ a given $t$ produces")
ax5[0].grid(alpha=0.25)
for t, a7 in zip(t_fixed_vals, realized_a7):
    ax5[0].annotate(f"t={t}\n$a_7$={a7:.3f}", (t, a7),
                    textcoords="offset points", xytext=(6, -18), fontsize=8)

ax5[1].semilogy(t_fixed_vals, np.maximum(ana_num_err, 1e-18), "o-",
                color="#e11d48", lw=2, label="|t_analytic - t_numeric|")
ax5[1].semilogy(t_fixed_vals, np.maximum(marg_err, 1e-18), "s-",
                color="#16a34a", lw=2, label=r"marginalization $|\Delta p|$")
ax5[1].set_xlabel(r"fixed aux coupling $t$")
ax5[1].set_ylabel("error (log scale)")
ax5[1].set_title("(b) both errors at machine precision")
ax5[1].legend(fontsize=9); ax5[1].grid(alpha=0.25, which="both")
fig5.suptitle(r"Fixed-$t$ tests: analytic$\leftrightarrow$numeric and marginalization",
              fontsize=13, y=1.01)
fig5.tight_layout()
print("  fixed t:", t_fixed_vals)
print("  realized a7:", [f"{v:.4f}" for v in realized_a7])
print("  max |t_ana - t_num|:", f"{max(ana_num_err):.1e}")
print("  max marginalization |Δp|:", f"{max(marg_err):.1e}")
print("Saved: fig_fixed_t_tests.png")


# ===========================================================================
# PAPER-READY COMPOSITE FIGURE
# Self-contained function reusing the solvers above (E_orig, solve_b_analytic,
# solve_b, max_pdiff, CONFIGS). No titles/annotations inside panels: axis
# labels only. Coefficients shown as CURVES vs a7 (revealing the a_i + p(t)
# structure), not bars. Saves both PNG and SVG.
#   from aux_direct import make_paper_figure
#   make_paper_figure()                       # pops the figure
#   make_paper_figure(save="paper_figure")    # writes .png and .svg
# ===========================================================================
def make_paper_figure(save=None, show=True,
                      a_demo=(0.0, 0.30, -0.20, 0.10, 0.40, -0.15, 0.25, 0.5),
                      fs=14):
    from matplotlib.gridspec import GridSpec

    a_demo = np.asarray(a_demo, float)

    # ---------- (A) t vs a7 : analytic curve + numeric markers ----------
    a_base = np.array([0.0, 0.30, -0.20, 0.10, 0.40, -0.15, 0.25, 0.0])
    a7_grid = np.linspace(0.0, 1.2, 200)
    t_ana = np.array([solve_b_analytic(np.r_[a_base[:7], v])[1] for v in a7_grid])
    a7_mk = np.linspace(0.05, 1.2, 25)
    t_num = np.array([solve_b(np.r_[a_base[:7], v])["b7"] for v in a7_mk])

    # ---------- (B) probability match ----------
    b_demo = solve_b_analytic(a_demo)[0]
    def _joint(a, b):
        def Ea(s, u):
            x, y, z = s
            return (b["b0"]+b["b1"]*x+b["b2"]*y+b["b3"]*z
                    + b["b4"]*x*y+b["b5"]*x*z+b["b6"]*y*z
                    + b["b7"]*u*x+b["b8"]*u*y+b["b9"]*u*z+b["b10"]*u)
        po = np.exp(-E_orig(a)); po /= po.sum()
        pm = np.array([sum(np.exp(-Ea(s, u)) for u in (-1, 1)) for s in CONFIGS])
        pm /= pm.sum()
        return po, pm
    po, pm = _joint(a_demo, b_demo)
    conf_lab = ["".join("+" if v > 0 else "-" for v in s) for s in CONFIGS]

    # ---------- (C,D,E) coefficient curves vs a7 ----------
    # For a fixed visible base (a1..a6), sweep a7 and record b's. Each visible
    # b_i traces a_i + p(t(a7)); every field/pair shares the SAME lift p, so the
    # curves are vertical translates of one another -> visually "a_i + const(t)".
    a_vis = np.array([0.0, 0.30, -0.20, 0.10, 0.40, -0.15, 0.25])
    sweep = np.linspace(0.0, 1.2, 200)
    B = {f"b{i}": np.empty_like(sweep) for i in range(11)}
    lift = np.empty_like(sweep)                 # p(t) : the shared constant added to every a_i
    for k, v in enumerate(sweep):
        sol, tt, pp, qq = solve_b_analytic(np.r_[a_vis, v])
        for i in range(11):
            B[f"b{i}"][k] = sol[f"b{i}"]
        lift[k] = pp

    # ---------- reconstruction error |Δp| across the whole a7 range (for B inset) ----------
    err_sweep = np.array([max_pdiff(np.r_[a_vis, v], solve_b_analytic(np.r_[a_vis, v])[0])
                          for v in sweep])
    err_sweep = np.maximum(err_sweep, 1e-18)    # floor for log axis

    # ---------- scaffold ----------
    plt.rcParams.update({
        "font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs-2,
        "ytick.labelsize": fs-2, "legend.fontsize": fs-3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1:])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])

    base = "#1f3b73"; accent = "#e6194B"

    # (A)
    axA.plot(a7_grid, t_ana, color=base, lw=2.4, label="analytic")
    # axA.scatter(a7_mk, t_num, s=20, facecolor="white", edgecolor=accent,
    #             linewidth=1.8, zorder=3, label="numeric")
    axA.set_xlabel(r"$a_{xyz}$")
    axA.set_ylabel(r"Weight $b_7=b_8=b_9=b_{10}=t$")
    # axA.legend(loc="upper left", frameon=False)

    # (B)
    xi = np.arange(8); w = 0.4
    axB.bar(xi - w/2, po, width=w, color=base, label=r"$p_{\mathrm{orig}}(x,y,z)$")
    axB.bar(xi + w/2, pm, width=w, color=accent, alpha=0.85,
            label=r"$\sum_u p_{\mathrm{aux}}(x,y,z, u)$")
    axB.set_xticks(xi); axB.set_xticklabels(conf_lab)
    axB.set_xlabel(r"$(x,y,z)$")
    axB.set_ylabel(r"$p$")
    axB.legend(loc="upper right", frameon=False)
    # inset: reconstruction error |Δp| vs a_xyz across the whole range
    ins = axB.inset_axes([0.36, 0.65, 0.34, 0.45])
    ins.semilogy(sweep, err_sweep, color="#6a3d9a", lw=1.6)
    ins.set_xlabel(r"$a_{xyz}$", fontsize=fs-4, labelpad=1)
    ins.set_ylabel(r"$|\Delta p|$", fontsize=fs-4, labelpad=1)
    ins.set_ylim(1e-18, 1e-13)
    ins.tick_params(labelsize=fs-6)
    ins.grid(alpha=0.25, which="both")

    # (C) unary curves vs a7
    unary = [("b1", base), ("b2", "#3f7fbf"), ("b3", "#7cb5e8")]
    for key, col in unary:
        axC.plot(sweep, B[key], color=col, lw=2.2)
    axC.set_xlabel(r"$a_{xyz}$")
    axC.set_ylabel(r"$b_1,\,b_2,\,b_3$")

    # (D) pairwise curves vs a7
    pair = [("b4", "#2a7f62"), ("b5", "#3faf88"), ("b6", "#8fd6bd")]
    for key, col in pair:
        axD.plot(sweep, B[key], color=col, lw=2.2)
    axD.set_xlabel(r"$a_{xyz}$")
    axD.set_ylabel(r"$b_4,\,b_5,\,b_6$")

    # (E) the shared lift p(t): every visible b_i equals a_i + p(t(a_xyz)).
    # Plotting p makes the "a_i + constant" structure of panels C,D explicit.
    axE.plot(sweep, lift, color="#6a3d9a", lw=2.6, label=r"$p(t)$")
    axE.set_xlabel(r"$a_{xyz}$")
    axE.set_ylabel(r"lift  $p(t)=b_i-a_i$")
    axE.legend(loc="upper left", frameon=False)

    if save:
        fig.savefig(save + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(save + ".svg", bbox_inches="tight")
        print(f"Saved {save}.png and {save}.svg  (match err {max_pdiff(a_demo, b_demo):.1e})")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    plt.close('all')
    path_save = DATA_DIR
    os.makedirs(path_save, exist_ok=True)
    make_paper_figure(save=path_save, show=True,
                      a_demo=(0.0, 0.1, 0.1, 0.05, -0.8, 0.0, 0.0, 0.1))
