"""
Mean Break DGPs
===============
Data-generating processes for AR(1) with mean (intercept) breaks.
"""
import numpy as np


def simulate_mean_break_ar1(
    T=300,
    Tb=150,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    y0=0.0,
    rng=None,
    seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the intercept (mean).
    
    Parameters:
        T: Sample size
        Tb: Break point
        mu0: Pre-break mean
        mu1: Post-break mean
        phi: AR coefficient
        sigma: Innovation standard deviation
        y0: Initial value
        rng: Random number generator (optional)
        seed: Random seed (optional, used if rng not provided)
    
    Returns:
        y: Simulated time series
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    y = np.zeros(T, dtype=float)
    y[0] = y0
    
    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi * y[t-1] + rng.normal(0.0, sigma)
    
    return y
