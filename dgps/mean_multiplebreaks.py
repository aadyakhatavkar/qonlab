"""
Multiple mean break DGPs: AR(1) with multiple mean shifts.
"""
import numpy as np


def simulate_multiple_mean_breaks_ar1(
    T=300,
    breaks=None,
    means=None,
    phi=0.6,
    sigma=1.0,
    seed=None
):
    """
    Simulate AR(1) with multiple deterministic mean breaks.
    
    Parameters:
        T: Series length
        breaks: List of break points [b1, b2, ...] in increasing order
        means: List of means [mu0, mu1, mu2, ...] for each regime
               Length must be len(breaks) + 1
        phi: AR(1) coefficient
        sigma: Innovation standard deviation
        seed: Random seed
    
    Returns:
        y: Time series
    
    Example:
        y = simulate_multiple_mean_breaks_ar1(
            T=300,
            breaks=[100, 200],
            means=[0.0, 2.0, -2.0],
            phi=0.6,
            sigma=1.0
        )
    """
    if breaks is None:
        breaks = [T // 2]
    if means is None:
        means = [0.0, 2.0]
    
    breaks = sorted(breaks)
    if len(means) != len(breaks) + 1:
        raise ValueError(f"Need {len(breaks) + 1} means for {len(breaks)} breaks")
    
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    
    for t in range(1, T):
        # Determine current regime
        regime = 0
        for i, b in enumerate(breaks):
            if t > b:
                regime = i + 1
        
        mu = means[regime]
        y[t] = mu + phi * y[t - 1] + rng.normal(0.0, sigma)
    
    return y
