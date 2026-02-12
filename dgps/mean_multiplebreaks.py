"""
Multiple mean break DGPs: AR(1) with multiple mean shifts.
"""
import numpy as np

# =========================================================
# 1) DGPs: Single break + seasonality, Multiple breaks + seasonality
# =========================================================

def simulate_single_break_with_seasonality(
    T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, s=12, A=1.0, y0=0.0, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    y = np.zeros(T, dtype=float)
    y[0] = y0
    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        seasonal = A * np.sin(2*np.pi*t/s)
        y[t] = mu + seasonal + phi*y[t-1] + rng.normal(0.0, sigma)
    return y

def simulate_multiple_breaks_with_seasonality(
    T=300, b1=100, b2=200, mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, s=12, A=1.0, y0=0.0, rng=None
):
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

