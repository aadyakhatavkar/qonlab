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
    distribution='normal', nu=3, seed=None
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
        distribution: 'normal' or 't' (Student-t with nu degrees of freedom)
        nu: Degrees of freedom for t-distribution (ignored if distribution='normal')
        seed: Random seed
    
    Returns:
        y: Simulated AR(1) series
    """
    if not (1 <= Tb < T):
        raise ValueError(f"Tb must satisfy 1 <= Tb < T (got Tb={Tb}, T={T})")

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = np.zeros(T)

    if distribution.lower() == 'normal':
        eps[:Tb] = rng.normal(0.0, sigma1, size=Tb)
        eps[Tb:] = rng.normal(0.0, sigma2, size=T - Tb)
    elif distribution.lower() == 't':
        # Generate standardized t-innovations
        seed1 = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
        seed2 = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
        eps[:Tb] = _generate_t_innovations(Tb, nu, scale=sigma1, seed=seed1)
        eps[Tb:] = _generate_t_innovations(T - Tb, nu, scale=sigma2, seed=seed2)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    for t in range(1, T):
        y[t] = mu + phi * (y[t - 1] - mu) + eps[t]
    return y


def estimate_variance_break_point(y, trim=0.15):
    """
    Estimate the variance break point using grid search on squared residuals.
    
    Fits AR(1) in two regimes and finds the break point that minimizes
    sum of squared residuals.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    lo = max(int(np.floor(trim * T)), 10)
    hi = min(int(np.ceil((1 - trim) * T)) - 1, T - 11)
    
    best_Tb, best_sse = None, np.inf
    
    for Tb_cand in range(lo, hi):
        # Fit AR(1) in regime 1
        y1 = y[:Tb_cand+1]
        if len(y1) < 3:
            continue
        y1_dep = y1[1:]
        y1_lag = y1[:-1]
        X1 = np.column_stack([np.ones_like(y1_lag), y1_lag])
        try:
            beta1 = np.linalg.lstsq(X1, y1_dep, rcond=None)[0]
            resid1 = y1_dep - X1 @ beta1
            sse1 = float(np.sum(resid1**2))
        except Exception:
            continue
        
        # Fit AR(1) in regime 2
        y2 = y[Tb_cand+1:]
        if len(y2) < 3:
            continue
        y2_dep = y2[1:]
        y2_lag = y2[:-1]
        X2 = np.column_stack([np.ones_like(y2_lag), y2_lag])
        try:
            beta2 = np.linalg.lstsq(X2, y2_dep, rcond=None)[0]
            resid2 = y2_dep - X2 @ beta2
            sse2 = float(np.sum(resid2**2))
        except Exception:
            continue
        
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb_cand
    
    return int(best_Tb if best_Tb is not None else T // 2)


def simulate_realized_volatility(
    T=400, Tb=200, intervals_per_day=1, phi_rv=0.5, sigma1_rv=1.0, sigma2_rv=2.0, seed=None
):
    """
    Simulate realized volatility (RV) from high-frequency intra-day data.
    """
    rng = np.random.default_rng(seed)
    
    # Generate high-frequency returns
    n_hf = T * intervals_per_day
    hf_returns = np.zeros(n_hf)
    
    # Simple model: constant volatility with structural break
    for i in range(n_hf):
        # Determine which regime we're in
        day = i // intervals_per_day
        if day < Tb:
            vol = sigma1_rv / np.sqrt(intervals_per_day)
        else:
            vol = sigma2_rv / np.sqrt(intervals_per_day)
        hf_returns[i] = rng.normal(0.0, vol)
    
    # Compute realized volatility (sum of squared returns) for each day
    rv = np.zeros(T)
    for day in range(T):
        start_idx = day * intervals_per_day
        end_idx = (day + 1) * intervals_per_day
        rv[day] = np.sqrt(np.sum(hf_returns[start_idx:end_idx]**2))
    
    return rv, hf_returns


def calculate_rv_from_returns(returns, intervals_per_day=1):
    """
    Calculate realized volatility from high-frequency returns.
    """
    returns = np.asarray(returns, dtype=float)
    n_periods = len(returns) // intervals_per_day
    
    rv = np.zeros(n_periods)
    for period in range(n_periods):
        start_idx = period * intervals_per_day
        end_idx = (period + 1) * intervals_per_day
        rv[period] = np.sqrt(np.sum(returns[start_idx:end_idx]**2))
    
    return rv
