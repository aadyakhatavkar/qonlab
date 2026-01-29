"""
Recurring/Markov-switching breaks: DGPs with probabilistic regime switches.
"""
import numpy as np


def simulate_markov_switching_ar1(
    T=400,
    p00=0.97,
    p11=0.97,
    phi0=0.2,
    phi1=0.9,
    sigma=1.0,
    seed=None
):
    """
    Simulate AR(1) with Markov-switching regimes.
    
    Parameters:
        T: Series length
        p00: Probability of staying in regime 0
        p11: Probability of staying in regime 1
        phi0: AR coefficient in regime 0
        phi1: AR coefficient in regime 1
        sigma: Innovation standard deviation
        seed: Random seed
    
    Returns:
        y: Time series
        s: Regime indicator (0 or 1)
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    s = np.zeros(T, dtype=int)

    y[0] = 0.0
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
