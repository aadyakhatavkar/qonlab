import numpy as np
from scipy import stats


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


def simulate_variance_break(
    T=400, variance_Tb=200, phi=0.6, mu=0.0, variance_sigma1=1.0, variance_sigma2=2.0, 
    distribution='normal', nu=3, seed=None
):
    """
    Simulate AR(1) series with variance break.
    
    Parameters:
        T: Total length
        variance_Tb: Break point (variance changes after time variance_Tb)
        phi: AR(1) coefficient
        mu: Mean
        variance_sigma1: Variance (or scale) in regime 1
        variance_sigma2: Variance (or scale) in regime 2
        distribution: 'normal' or 't' (Student-t with nu degrees of freedom)
        nu: Degrees of freedom for t-distribution (ignored if distribution='normal')
        seed: Random seed
    
    Returns:
        y: Simulated AR(1) series
    """
    if not (1 <= variance_Tb < T):
        raise ValueError(f"variance_Tb must satisfy 1 <= variance_Tb < T (got variance_Tb={variance_Tb}, T={T})")

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = np.zeros(T)

    if distribution.lower() == 'normal':
        eps[:variance_Tb] = rng.normal(0.0, variance_sigma1, size=variance_Tb)
        eps[variance_Tb:] = rng.normal(0.0, variance_sigma2, size=T - variance_Tb)
    elif distribution.lower() == 't':
        # Generate standardized t-innovations
        seed1 = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
        seed2 = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
        eps[:variance_Tb] = _generate_t_innovations(variance_Tb, nu, scale=variance_sigma1, seed=seed1)
        eps[variance_Tb:] = _generate_t_innovations(T - variance_Tb, nu, scale=variance_sigma2, seed=seed2)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    for t in range(1, T):
        y[t] = mu + phi * (y[t - 1] - mu) + eps[t]
    return y


def simulate_mean_break(
    T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the intercept (mean).
    y_t = mu_t + phi * y_{t-1} + eps_t
    mu_t = mu0 for t <= Tb, mu1 for t > Tb
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = rng.normal(0.0, sigma, size=T)

    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi * y[t-1] + eps[t]

    return y


def simulate_parameter_break(
    T=400, Tb=200, phi1=0.2, phi2=0.9, sigma=1.0, seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the autoregressive parameter phi.
    y_t = phi_t * y_{t-1} + eps_t
    phi_t = phi1 for t <= Tb, phi2 for t > Tb
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = rng.normal(0.0, sigma, size=T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2
        y[t] = phi * y[t-1] + eps[t]

    return y


def _validate_scenarios(scenarios, T):
    if scenarios is None:
        return [{"name": "Single variance break", "variance_Tb": max(1, T // 2), "variance_sigma1": 1.0, "variance_sigma2": 2.0, "task": "variance"}]

    validated = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            raise ValueError("Each scenario must be a dict")
        
        task = sc.get("task", "variance")
        
        if task == "variance":
            # Mapping old keys to new variance-prefix keys for backwards compatibility
            key_map = {"Tb": "variance_Tb", "sigma1": "variance_sigma1", "sigma2": "variance_sigma2"}
            for old_k, new_k in key_map.items():
                if new_k not in sc and old_k in sc:
                    sc[new_k] = sc[old_k]
                    
            for key in ("name", "variance_Tb", "variance_sigma1", "variance_sigma2"):
                if key not in sc:
                    raise ValueError(f"Scenario missing required key: {key} for variance task")
            
            v_Tb = int(sc["variance_Tb"])
            if v_Tb >= T: v_Tb = T - 1
            if v_Tb < 1: v_Tb = 1
            
            validated.append({
                "name": sc["name"], 
                "task": "variance",
                "variance_Tb": v_Tb, 
                "variance_sigma1": float(sc["variance_sigma1"]), 
                "variance_sigma2": float(sc["variance_sigma2"]),
                "distribution": sc.get("distribution", "normal"),
                "nu": sc.get("nu", 3)
            })
        elif task == "mean":
            for key in ("name", "Tb", "mu0", "mu1"):
                if key not in sc:
                    raise ValueError(f"Scenario missing required key: {key} for mean task")
            validated.append({
                "name": sc["name"],
                "task": "mean",
                "Tb": int(sc["Tb"]),
                "mu0": float(sc["mu0"]),
                "mu1": float(sc["mu1"]),
                "phi": float(sc.get("phi", 0.6)),
                "sigma": float(sc.get("sigma", 1.0))
            })
        elif task == "parameter":
            for key in ("name", "Tb", "phi1", "phi2"):
                if key not in sc:
                    raise ValueError(f"Scenario missing required key: {key} for parameter task")
            validated.append({
                "name": sc["name"],
                "task": "parameter",
                "Tb": int(sc["Tb"]),
                "phi1": float(sc["phi1"]),
                "phi2": float(sc["phi2"]),
                "sigma": float(sc.get("sigma", 1.0))
            })
        else:
            validated.append(sc)

    return validated


def simulate_realized_volatility(
    T=400, variance_Tb=200, intervals_per_day=1, phi_rv=0.5, variance_sigma1_rv=1.0, variance_sigma2_rv=2.0, seed=None
):
    """
    Simulate realized volatility (RV) from high-frequency intra-day data.
    
    This generates high-frequency returns and aggregates to compute realized volatility
    for each day/period. This is useful for modeling volatility dynamics that are 
    more directly observable from market microstructure data.
    
    Parameters:
        T: Number of daily realized volatility observations
        variance_Tb: Break point in realized volatility
        intervals_per_day: Number of high-frequency intervals per day (e.g., 78 for 5-min data)
        phi_rv: AR(1) coefficient for RV process
        variance_sigma1_rv: Volatility of the RV process before break
        variance_sigma2_rv: Volatility of the RV process after break
        seed: Random seed
    
    Returns:
        rv: Array of realized volatilities of length T
        hf_data: High-frequency returns (for reference/analysis) of shape (T * intervals_per_day, )
    """
    rng = np.random.default_rng(seed)
    
    # Generate high-frequency returns
    n_hf = T * intervals_per_day
    hf_returns = np.zeros(n_hf)
    
    # Simple model: constant volatility with structural break
    for i in range(n_hf):
        # Determine which regime we're in
        day = i // intervals_per_day
        if day < variance_Tb:
            vol = variance_sigma1_rv / np.sqrt(intervals_per_day)
        else:
            vol = variance_sigma2_rv / np.sqrt(intervals_per_day)
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
    
    Parameters:
        returns: Array of high-frequency returns
        intervals_per_day: Number of intervals per aggregation period
    
    Returns:
        rv: Realized volatility for each period
    """
    returns = np.asarray(returns, dtype=float)
    n_periods = len(returns) // intervals_per_day
    
    rv = np.zeros(n_periods)
    for period in range(n_periods):
        start_idx = period * intervals_per_day
        end_idx = (period + 1) * intervals_per_day
        rv[period] = np.sqrt(np.sum(returns[start_idx:end_idx]**2))
    
    return rv
