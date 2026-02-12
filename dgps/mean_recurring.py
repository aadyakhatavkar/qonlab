"""
Recurring/Markov-switching Mean Break DGP
=========================================
AR(1) with Markov-switching mean (recurring intercept changes).
"""
import numpy as np


def simulate_ms_ar1_mean_only(
    T=400,
    p00=0.95,
    p11=0.95,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    seed=None
):
    """
    Simulate AR(1) with Markov-switching mean.
    
    Parameters:
        T: Total length
        p00: Persistence of regime 0 (low mean)
        p11: Persistence of regime 1 (high mean)
        mu0: Mean in regime 0
        mu1: Mean in regime 1
        phi: AR(1) coefficient
        sigma: Innovation standard deviation
        seed: Random seed
    
    Returns:
        y: Simulated time series
        s: State/regime indicator (0 or 1)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    y = np.zeros(T)
    s = np.zeros(T, dtype=int)
    
    # Initial state and observation
    s[0] = rng.integers(0, 2)
    mu = mu0 if s[0] == 0 else mu1
    y[0] = mu
    
    # Generate time series
    for t in range(1, T):
        # Regime transition
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0
        
        # Generate observation
        mu_t = mu0 if s[t] == 0 else mu1
        eps = rng.normal(0.0, sigma)
        y[t] = mu_t + phi * (y[t-1] - mu_t) + eps
    
    return y, s
