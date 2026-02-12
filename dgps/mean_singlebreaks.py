"""
Mean Break DGPs
===============
Data-generating processes for AR(1) with mean (intercept) breaks.
"""
import numpy as np

# =========================================================
# 1) DGP: AR(1) + SEASONALITY + ONE mean break
# =========================================================
def simulate_single_break_with_seasonality(
    T=300,
    Tb=150,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    s=12,            # seasonal period
    A=1.0,           # seasonal amplitude
    y0=0.0,
    rng=None
):
    """
    DGP:
      y_t = mu_t + seasonal_t + phi*y_{t-1} + eps_t
    mu_t = mu0 for t <= Tb, mu1 for t > Tb
      seasonal_t = A * sin(2π t / s)
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T, dtype=float)
    y[0] = y0

    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        seasonal = A * np.sin(2*np.pi*t/s)
        y[t] = mu + seasonal + phi*y[t-1] + rng.normal(0.0, sigma)

    return y
