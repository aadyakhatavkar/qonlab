"""
Recurring/Markov-switching Variance Break DGP
==============================================
AR(1) with Markov-switching variance (recurring regime changes).
"""
import numpy as np


def simulate_ms_ar1_variance_only(
    T=400,
    p00=0.95,
    p11=0.95,
    sigma1=1.0,
    sigma2=2.0,
    phi=0.6,
    mu=0.0,
    seed=None
):
    """
    Simulate AR(1) with Markov-switching variance.
    
    Parameters:
        T: Total length
        p00: Persistence of regime 0 (low variance)
        p11: Persistence of regime 1 (high variance)
        sigma1: Volatility in regime 0
        sigma2: Volatility in regime 1
        phi: AR(1) coefficient
        mu: Mean
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
    
    # Initial state
    s[0] = rng.integers(0, 2)
    y[0] = rng.normal(mu, sigma1 if s[0] == 0 else sigma2)
    
    # Generate time series
    for t in range(1, T):
        # Regime transition
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0
        
        # Generate observation
        sigma = sigma1 if s[t] == 0 else sigma2
        eps = rng.normal(0.0, sigma)
        y[t] = mu + phi * (y[t-1] - mu) + eps
    
    return y, s
