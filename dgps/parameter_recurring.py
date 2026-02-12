"""
Recurring/Markov-switching breaks: DGPs with probabilistic regime switches.
"""
import numpy as np

# =====================================================
# 1) DGP: Markov-switching AR(1), Gaussian
# =====================================================

def simulate_ms_ar1_phi_only(
    T=400,
    p00=0.97,
    p11=0.97,
    phi0=0.2,
    phi1=0.9,
    sigma=1.0,
    y0=0.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)
    s = np.zeros(T, dtype=int)

    y[0] = y0
    s[0] = rng.integers(0, 2)

    for t in range(1, T):
        if s[t - 1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0

        eps = rng.normal(0.0, sigma)
        phi = phi0 if s[t] == 0 else phi1
        y[t] = phi * y[t - 1] + eps

    return y, s

