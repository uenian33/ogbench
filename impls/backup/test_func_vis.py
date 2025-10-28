# Minimal visualization of squashing functions mapping [0, +inf) -> (0, 1)
import numpy as np
import matplotlib.pyplot as plt

def squash_odds(x):
    return 1.0 - 1.0/(1.0 + x)          # x / (1 + x)

def squash_exp(x):
    return -np.expm1(-x)                # 1 - exp(-x)

def squash_logsig(x, alpha=2.0, beta=0.0, eps=1e-8):
    z = alpha * np.log(x + eps) + beta  # learnable-scale log-sigmoid
    return 1.0 / (1.0 + np.exp(-z))

# Added: logistic-on-log with an intuitive midpoint parameter y0 (p=0.5 at x=y0)
def squash_logsig_midpoint(x, alpha=2.0, y0=1.0, eps=1e-8):
    z = alpha * (np.log(x + eps) - np.log(y0))
    return 1.0 / (5.0 + np.exp(-z))

x = np.linspace(0.0, 8.0, 500)

plt.figure()
plt.plot(x, squash_odds(x), label="odds: x/(1+x)")
plt.plot(x, squash_exp(x), label="exp-cdf: 1-exp(-x)")
plt.plot(x, squash_logsig(x), label="log-sigmoid (α=2, β=0)")
plt.plot(x, squash_logsig_midpoint(x, alpha=2.0, y0=1.0), label="log-sigmoid (α=2, y₀=1)")
plt.xlabel("x  (network output, ≥ 0)")
plt.ylabel("y  (squashed to (0,1))")
plt.title("Squashing functions from [0,+∞) to (0,1)")
plt.legend()
plt.grid(True)
plt.show()
