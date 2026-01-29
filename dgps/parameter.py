"""
Parameter Break DGPs
====================
Data-generating processes for AR(1) with parameter (AR coefficient) breaks.
"""
import numpy as np


def simulate_parameter_break_ar1(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    innovation="normal",
    df=None,
    rng=None,
    seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the AR coefficient.
    
    Parameters:
        T: Sample size
        Tb: Break point
        phi1: Pre-break AR coefficient
        phi2: Post-break AR coefficient
        sigma: Innovation standard deviation
        innovation: 'normal' or 'student' for distribution type
        df: Degrees of freedom for Student-t (required if innovation='student')
        rng: Random number generator (optional)
        seed: Random seed (optional, used if rng not provided)
    
    Returns:
        y: Simulated time series
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    y = np.zeros(T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2

        if innovation == "normal":
            eps = rng.normal(0.0, sigma)
        elif innovation == "student":
            if df is None or df <= 2:
                raise ValueError("df must be > 2 for Student-t innovations")
            eps = rng.standard_t(df) * sigma / np.sqrt(df / (df - 2))
        else:
            raise ValueError(f"Unknown innovation type: {innovation}")

        y[t] = phi * y[t - 1] + eps

    return y
