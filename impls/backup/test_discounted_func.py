# Minimalist visualization of several φ(h) schedules on h = 1..H.
# Requirements from the user/instructions:
# - Max horizon 1000 (H=1000 below)
# - One chart (no subplots), matplotlib (no seaborn), no explicit colors/styles.

import numpy as np
import matplotlib.pyplot as plt

# ---------------------- configurable knobs ----------------------
H = 1000
eps = 0.35  # epsilon floor in [eps, 1]
# extra parameters for families
pow_p1, pow_p2 = 0.7, 3.5
exp_alpha = 0.08
hyp_beta = 0.5
log_alpha = 0.2
two_lin_k = H // 2
two_lin_rho = 0.2
# ---------------------------------------------------------------

h = np.arange(1, H + 1, dtype=float)
t = (h - 1) / (H - 1 + 1e-12)  # normalized [0,1]

def geometric(g, h):
    return g**h 

def phi_linear(h):
    return eps + (1 - eps) * (H - h) / (H - 1)

def phi_power(h, p):
    return eps + (1 - eps) * ((H - h) / (H - 1)) ** p

def phi_exp(h, alpha):
    num = np.exp(-alpha * (h - 1)) - np.exp(-alpha * (H - 1))
    den = 1 - np.exp(-alpha * (H - 1))
    s = num / (den + 1e-12)
    return eps + (1 - eps) * s

def phi_hyperbolic(h, beta):
    num = h ** (-beta) - H ** (-beta)
    den = 1 - H ** (-beta)
    s = num / (den + 1e-12)
    return eps + (1 - eps) * s

def phi_cos(h):
    s = (1 + np.cos(np.pi * (h - 1) / (H - 1))) / 2.0
    return eps + (1 - eps) * s

def phi_smoothstep(h):
    tt = (h - 1) / (H - 1 + 1e-12)
    sm = tt**2 * (3 - 2 * tt)  # smoothstep
    s = 1 - sm
    return eps + (1 - eps) * s

def phi_logistic(h, alpha):
    c = (H + 1) / 2.0
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    num = sig(alpha * (1 - c)) - sig(alpha * (h - c))
    den = sig(alpha * (1 - c)) - sig(alpha * (H - c))
    s = num / (den + 1e-12)
    return eps + (1 - eps) * s

def phi_two_linear(h, k, rho):
    # Piecewise linear with knee at k
    s = np.empty_like(h, dtype=float)
    left_mask = h <= k
    right_mask = ~left_mask
    # left: 1 -> 1 - rho
    s[left_mask] = 1.0 - rho * (h[left_mask] - 1) / max(k - 1, 1)
    # right: 1 - rho -> 0
    s[right_mask] = (1 - rho) * (H - h[right_mask]) / max(H - k, 1)
    return eps + (1 - eps) * s

gamma = float(eps) ** (1.0 / float(H))

# Compute schedules
curves = {
    "linear": phi_linear(h),
    f"power(p={pow_p1})": phi_power(h, pow_p1),
    f"power(p={pow_p2})": phi_power(h, pow_p2),
    f"exp(alpha={exp_alpha})": phi_exp(h, exp_alpha),
    f"geometric={gamma}": geometric(gamma, h),
    f"hyperbolic(beta={hyp_beta})": phi_hyperbolic(h, hyp_beta),
    "cosine": phi_cos(h),
    "smoothstep": phi_smoothstep(h),
    f"logistic(alpha={log_alpha})": phi_logistic(h, log_alpha),
    f"two-linear(k={two_lin_k},rho={two_lin_rho})": phi_two_linear(h, two_lin_k, two_lin_rho),
}

# Plot (single chart, no specific colors/styles)
plt.figure(figsize=(8, 5))
for name, y in curves.items():
    plt.plot(h, y, label=name, linewidth=1.5)
plt.xlabel("h (steps)")
plt.ylabel("φ(h)")
plt.title(f"Step weighting schedules φ(h) with H={H}, ε={eps}")
plt.legend(loc="best", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
