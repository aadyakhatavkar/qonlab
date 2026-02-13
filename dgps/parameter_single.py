"""
Parameter Break DGPs
====================
Data-generating processes for AR(1) with parameter (AR coefficient) breaks.
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


# =====================================================
# 1) DGP: AR(1) with SINGLE deterministic break in phi
# =====================================================
def simulate_single_break_ar1(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    innovation_type='gaussian',
    dof=None,
    rng=None
):
    """
    Simulate AR(1) series with a single break in AR coefficient.
    
    Parameters:
        T: Total length
        Tb: Break point (AR coefficient changes after time Tb)
        phi1: AR coefficient before break
        phi2: AR coefficient after break
        sigma: Std deviation of innovations
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2

        if innovation_type.lower() == 'gaussian':
            eps = rng.normal(0.0, sigma)
        elif innovation_type.lower() == 'student':
            if dof is None or dof <= 2:
                raise ValueError("dof must be > 2 for finite variance")
            eps = rng.standard_t(dof) * sigma / np.sqrt(dof / (dof - 2))
        else:
            raise ValueError(f"Unknown innovation_type: {innovation_type}")

        y[t] = phi * y[t - 1] + eps

    return y

