"""
Mean Break DGPs
===============
Data-generating processes for AR(1) with mean (intercept) breaks.
"""
import numpy as np


def _generate_t_innovations(size, nu, scale=1.0, seed=None):
    """
    Generate standardized T-distributed innovations.
    
    The T-distribution with degrees of freedom nu has variance = nu/(nu-2).
    To compare with standard normal (variance=1), we standardize by dividing
    by the standard deviation: sqrt(nu/(nu-2)).
    
    Parameters:
        size: Number of innovations to generate
        nu: Degrees of freedom
        scale: Scaling factor (applied after standardization)
        seed: Random seed
    
    Returns:
        Standardized T-distributed innovations with variance = scale^2
    """
    rng = np.random.default_rng(seed)
    t_innovations = rng.standard_t(nu, size=size)
    
    # Standardize to unit variance
    if nu > 2:
        std_t = np.sqrt(nu / (nu - 2))
        t_innovations = t_innovations / std_t
    
    return t_innovations * scale


# =========================================================
# 1) DGP: AR(1) + SEASONALITY + ONE mean break
# =========================================================
def simulate_single_break_with_seasonality(
    T=400,
    Tb=200,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    s=12,            # seasonal period
    A=1.0,           # seasonal amplitude
    y0=0.0,
    innovation_type='gaussian',
    dof=None,
    rng=None
):
    """
    DGP:
      y_t = mu_t + seasonal_t + phi*y_{t-1} + eps_t
    mu_t = mu0 for t <= Tb, mu1 for t > Tb
      seasonal_t = A * sin(2π t / s)
    
    Parameters:
        T: Total length
        Tb: Break point (mean changes after time Tb)
        mu0: Mean before break
        mu1: Mean after break
        phi: AR(1) coefficient
        sigma: Std deviation of innovations
        s: Seasonal period
        A: Seasonal amplitude
        y0: Initial value
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T, dtype=float)
    y[0] = y0

    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        seasonal = A * np.sin(2*np.pi*t/s)
        
        if innovation_type.lower() == 'gaussian':
            eps = rng.normal(0.0, sigma)
        elif innovation_type.lower() == 'student':
            if dof is None:
                raise ValueError("dof must be specified for Student-t innovations")
            eps = _generate_t_innovations(1, dof, scale=sigma, seed=None)[0]
        else:
            raise ValueError(f"Unknown innovation_type: {innovation_type}")
        
        y[t] = mu + seasonal + phi*y[t-1] + eps

    return y
