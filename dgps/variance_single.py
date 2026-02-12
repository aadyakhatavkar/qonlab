"""
Single Variance Break DGP
=========================
AR(1) with a single shift in variance at time Tb.
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


def simulate_variance_break_ar1(
    T=400, Tb=200, phi=0.6, mu=0.0, sigma1=1.0, sigma2=2.0, 
    innovation_type='gaussian', dof=None, seed=None
):
    """
    Simulate AR(1) series with variance break.
    
    Parameters:
        T: Total length
        Tb: Break point (variance changes after time Tb)
        phi: AR(1) coefficient
        mu: Mean
        sigma1: Variance (or scale) in regime 1
        sigma2: Variance (or scale) in regime 2
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
        seed: Random seed
    
    Returns:
        y: Simulated AR(1) series
    """
    if not (1 <= Tb < T):
        raise ValueError(f"Tb must satisfy 1 <= Tb < T (got Tb={Tb}, T={T})")

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = np.zeros(T)

    if innovation_type.lower() == 'gaussian':
        eps[:Tb] = rng.normal(0.0, sigma1, size=Tb)
        eps[Tb:] = rng.normal(0.0, sigma2, size=T - Tb)
    elif innovation_type.lower() == 'student':
        if dof is None:
            raise ValueError("dof must be specified for Student-t innovations")
        # Generate standardized t-innovations
        seed1 = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
        seed2 = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
        eps[:Tb] = _generate_t_innovations(Tb, dof, scale=sigma1, seed=seed1)
        eps[Tb:] = _generate_t_innovations(T - Tb, dof, scale=sigma2, seed=seed2)
    else:
        raise ValueError(f"Unknown innovation_type: {innovation_type}")
    
    for t in range(1, T):
        y[t] = mu + phi * (y[t - 1] - mu) + eps[t]
    return y