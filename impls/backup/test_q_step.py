import math

def steps_from_Q(Q, gamma):
    if gamma == 1.0:
        return -Q
    rhs = 1.0 + (1.0 - gamma) * Q  # must be in (0, 1]
    rhs = max(min(rhs, 1.0), 1e-12)
    return math.log(rhs) / math.log(gamma)  # exact if deterministic; upper bound otherwise

Q = -10000.0
gamma = 0.99
steps = steps_from_Q(Q, gamma)
print(steps)
