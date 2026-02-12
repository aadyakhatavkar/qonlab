"""
Multiple mean break DGPs: AR(1) with multiple mean shifts.
"""
import numpy as np
from .mean_singlebreaks import simulate_single_break_with_seasonality

# =========================================================
# 1) DGPs: Multiple breaks with seasonality
# =========================================================

def simulate_multiple_breaks_with_seasonality(
    T=300, b1=100, b2=200, mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, s=12, A=1.0, y0=0.0, rng=None
):
    """
    Simulate AR(1) with seasonality and multiple mean breaks.
    
    Parameters:
        T: Total length
        b1: First break point
        b2: Second break point
        mu0: Mean in regime 0 (t <= b1)
        mu1: Mean in regime 1 (b1 < t <= b2)
        mu2: Mean in regime 2 (t > b2)
        phi: AR(1) coefficient
        sigma: Noise standard deviation
        s: Seasonal period
        A: Seasonal amplitude
        y0: Initial value
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()
    y = np.zeros(T, dtype=float)
    y[0] = y0
    for t in range(1, T):
        if t <= b1:
            mu = mu0
        elif t <= b2:
            mu = mu1
        else:
            mu = mu2
        seasonal = A * np.sin(2*np.pi*t/s)
        y[t] = mu + seasonal + phi*y[t-1] + rng.normal(0.0, sigma)
    return y


__all__ = [
    "simulate_single_break_with_seasonality",
    "simulate_multiple_breaks_with_seasonality",
]
