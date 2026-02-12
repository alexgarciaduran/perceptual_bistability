import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ---------- model params (change as needed) ----------
J = 0.5        # coupling
N = 4.0        # N factor in your formula
D = 0.05       # noise intensity (Kramers D)
alpha = 1/60   # sweep rate (Bias units per second)
x_lo, x_hi = 0.0, 1.0  # domain for x (sigma outputs in (0,1))

# ---------- model functions ----------
def sigma(u):
    return 1.0/(1.0 + np.exp(-u))

def sigma_prime(u):
    s = sigma(u)
    return s*(1-s)

def u_of(x, Bias):
    return 2*J*N*(2*x - 1) + 2*Bias

def f(x, Bias):
    return sigma(u_of(x, Bias)) - x

def fprime(x, Bias):
    return 4*J*N * sigma_prime(u_of(x, Bias)) - 1.0

# numerical primitive V(x) = integral (x - sigma(u(x))) dx
# we compute V by numerical quadrature from a reference x0
def V_of(x, Bias, x0=0.5):
    val = x*x/2 - np.log(1+np.exp(2*N*(J*(2*x-1))+Bias*2))/(4*N*J)
    return val

# ---------- helpers: find all roots of f(x)=0 in (0,1) ----------
def find_roots_bias(Bias, x_guesses=None):
    if x_guesses is None:
        x_guesses = np.linspace(0.01,0.99,41)
    roots = []
    for x0 in x_guesses:
        try:
            sol = fsolve(lambda x: f(x, Bias), x0, maxfev=200, xtol=1e-12)
            xsol = float(sol)
            if 0.0 < xsol < 1.0:
                # de-dup
                if all(abs(xsol - r) > 1e-6 for r in roots):
                    roots.append(xsol)
        except Exception:
            pass
    roots.sort()
    return np.array(roots)

# classify equilibria: stable if f'(x)<0
def classify_equilibria(Bias):
    roots = find_roots_bias(Bias)
    items = []
    for x in roots:
        items.append((x, fprime(x, Bias)))
    # sort by x
    items.sort(key=lambda t: t[0])
    return items  # list of (x, f'(x))

# ---------- compute barrier ΔV and prefactor A for a Bias where 3 equilibria exist ----------
def barrier_and_prefactor(Bias):
    items = classify_equilibria(Bias)
    if len(items) < 3:
        return None
    # assume stable-left, saddle, stable-right ordering
    x_left, fp_left = items[0]
    x_mid, fp_mid = items[1]
    x_right, fp_right = items[2]
    # choose the well of interest: typically left well when sweeping Bias upward;
    # choose the stable minimum that's on the side we're escaping from.
    # Here we'll compute for escaping left->right (forward): min = left, saddle = mid.
    x_min = x_left
    x_sad = x_mid
    # compute potential V at these
    V_min = V_of(x_min, Bias)
    V_sad = V_of(x_sad, Bias)
    deltaV = V_sad - V_min
    # prefactor (use magnitudes)
    A = np.sqrt(abs(fp_mid) * abs(fp_left)) / (2*np.pi)
    return dict(x_min=x_min, x_sad=x_sad, fp_min=fp_left, fp_sad=fp_mid, deltaV=deltaV, A=A)

# ---------- root-finding functions for switching Bias r_sw ----------
# full Kramers equation: A(Bias)*exp(-deltaV(Bias)/D) - alpha = 0
def rate_minus_alpha(Bias):
    val = barrier_and_prefactor(Bias)
    if val is None:
        # outside bistable region: return sign such that brentq can skip
        return -alpha if Bias < Bias_min_est else +1.0
    rate = val['A'] * np.exp(-val['deltaV']/D)
    return rate - alpha

# approximate equation: deltaV(Bias) - D*ln(1/alpha) = 0
def approx_eq(Bias):
    val = barrier_and_prefactor(Bias)
    if val is None:
        return 1.0
    return val['deltaV'] - D*np.log(1.0/alpha)

# ---------- scan Bias to find bistable interval ----------
Bias_vals = np.linspace(-3.0, 3.0, 801)
bistable_flags = []
for B in Bias_vals:
    roots = find_roots_bias(B)
    bistable_flags.append(len(roots) >= 3)
# rough bounds
if any(bistable_flags):
    idxs = np.where(bistable_flags)[0]
    Bias_min_est = Bias_vals[idxs[0]]
    Bias_max_est = Bias_vals[idxs[-1]]
else:
    Bias_min_est = None
    Bias_max_est = None

print("Estimated deterministic bistable Bias interval:", Bias_min_est, Bias_max_est)

# ---------- solve for r_sw (full equation) ----------
if Bias_min_est is None:
    print("No deterministic bistability for current J,N.")
else:
    # forward sweep: find Bias in [Bias_min_est, Bias_max_est] solving rate(Bias)=alpha
    try:
        # need to find bracket where rate_minus_alpha crosses zero
        # build sample of sign of rate_minus_alpha
        R = [rate_minus_alpha(B) for B in np.linspace(Bias_min_est, Bias_max_est, 201)]
        Bs = np.linspace(Bias_min_est, Bias_max_est, 201)
        sign_changes = []
        for i in range(len(R)-1):
            if R[i]*R[i+1] < 0:
                sign_changes.append((Bs[i], Bs[i+1]))
        if len(sign_changes)==0:
            print("No sign change for full Kramers eqn in interval.")
            rsw_full = None
        else:
            a,b = sign_changes[0]
            rsw_full = brentq(lambda B: rate_minus_alpha(B), a, b)
            print("Full Kramers forward switch Bias (approx):", rsw_full)
    except Exception as e:
        print("Error solving full eq:", e)
        rsw_full = None

    # approx eq
    try:
        R2 = [approx_eq(B) for B in np.linspace(Bias_min_est, Bias_max_est, 201)]
        Bs2 = np.linspace(Bias_min_est, Bias_max_est, 201)
        sign_changes2 = []
        for i in range(len(R2)-1):
            if R2[i]*R2[i+1] < 0:
                sign_changes2.append((Bs2[i], Bs2[i+1]))
        if len(sign_changes2)==0:
            print("No sign change for approx eq in interval.")
            rsw_approx = None
        else:
            a,b = sign_changes2[0]
            rsw_approx = brentq(lambda B: approx_eq(B), a, b)
            print("Approx rule switch Bias (deltaV = D ln 1/alpha):", rsw_approx)
    except Exception as e:
        print("Error solving approx eq:", e)
        rsw_approx = None


if __name__ == '__main__':
    # ---------- optional: SDE simulation to estimate empirical switching distribution ----------
    do_sim = True
    if do_sim:
        import tqdm
        dt = 0.001
        ntrials = 1000
        T = (Bias_max_est - Bias_min_est)/alpha  # total time needed for sweep
        nsteps = int(T/dt)
        # sweep Bias linearly from Bias_min_est to Bias_max_est over time T
        biases_time = np.linspace(Bias_min_est, Bias_max_est, nsteps)
        switch_times = []
        for tr in range(ntrials):
            x = 0.1  # initial
            for i in range(nsteps-1):
                B = biases_time[i]
                xi = np.sqrt(2*D/dt)*np.random.randn()
                x = x + (sigma(u_of(x,B)) - x)*dt + xi*np.sqrt(dt)
                # detect crossing to the right (x crosses 0.5)
                if x>0.5:
                    t_switch = i*dt
                    switch_times.append(t_switch)
                    break
        # convert times to Bias
        from collections import Counter
        Bias_switches = [Bias_min_est + (Bias_max_est-Bias_min_est)*(t/T) for t in switch_times]
        print("Empirical mean switch Bias (forward):", np.mean(Bias_switches), "N events:", len(Bias_switches))
    
    # ---------- plotting diagnostic ----------
    # plot deltaV and log-rate across Bias
    if Bias_min_est is not None:
        Bs_plot = np.linspace(Bias_min_est, Bias_max_est, 300)
        deltaVs = []
        rates = []
        As = []
        for B in Bs_plot:
            val = barrier_and_prefactor(B)
            if val is None:
                deltaVs.append(np.nan); rates.append(np.nan); As.append(np.nan)
            else:
                deltaVs.append(val['deltaV'])
                As.append(val['A'])
                rates.append(val['A']*np.exp(-val['deltaV']/D))
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(Bs_plot, deltaVs, label='DeltaV')
        if rsw_approx is not None: plt.axvline(rsw_approx, color='C3', linestyle='--', label='approx r_sw')
        if rsw_full is not None: plt.axvline(rsw_full, color='C4', linestyle='--', label='full r_sw')
        plt.legend(); plt.xlabel('Bias'); plt.title('Barrier ΔV')
        plt.subplot(1,2,2)
        plt.semilogy(Bs_plot, rates, label='Kramers rate')
        plt.axhline(alpha, color='k', linestyle=':', label='alpha')
        plt.legend(); plt.xlabel('Bias'); plt.title('Rate vs Bias')
        plt.tight_layout()
        plt.show()
