"""
Parameter Break DGPs
====================
Data-generating processes for AR(1) with parameter (AR coefficient) breaks.
"""
import numpy as np

# =====================================================
# 1) DGP: AR(1) with SINGLE deterministic break in phi
# =====================================================
def simulate_single_break_ar1(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    innovation="normal",
    df=None,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2

        if innovation == "normal":
            eps = rng.normal(0.0, sigma)
        elif innovation == "student":
            if df is None or df <= 2:
                raise ValueError("df must be > 2 for finite variance")
            eps = rng.standard_t(df) * sigma / np.sqrt(df / (df - 2))
        else:
            raise ValueError(f"Unknown innovation type: {innovation}")

        y[t] = phi * y[t - 1] + eps

    return y

