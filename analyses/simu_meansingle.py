import numpy as np
import pandas as pd
from pathlib import Path
from dgps.mean_singlebreaks import simulate_single_break_ar1
from estimators.mean_singlebreak import (
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_sarima_break_dummy_oracle,
    forecast_ses,
    forecast_holt_winters,
)
from analyses.metrics import rmse, mae, bias, var_error

# =========================================================
# 3) Monte Carlo evaluation
# =========================================================
def run_mc_single_break_sarima(
    n_sim=300,
    T=400,
    Tb=200,
    window=70,
    seed=123,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    order=(1,0,1),
    seasonal_order=(1,0,0,12),
    trim=0.15,
    innovation_type='gaussian',
    dof=None
):
    """
    Monte Carlo evaluation for single mean break with random forecast origin after Tb.
    
    Parameters:
        n_sim: Number of Monte Carlo replications
        T: Total time series length
        Tb: Structural break point
        window: Rolling window size
        seed: Random seed
        mu0: Mean before break
        mu1: Mean after break
        phi: AR(1) coefficient
        sigma: Std deviation of innovations
        s: Seasonal period
        A: Seasonal amplitude
        gap_after_break: Gap between break and forecast origin
        order: SARIMA order (p, d, q)
        seasonal_order: SARIMA seasonal order (P, D, Q, s)
        trim: Trimming fraction for break point estimation
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
    
    Returns:
        DataFrame with RMSE, MAE, Bias, Variance for each method
    """
    rng = np.random.default_rng(seed)
    
    # Validate that we have room for post-break forecasts
    if Tb + 2 >= T:
        raise ValueError("Not enough data after break point for forecasting.")

    methods = [
        ("SARIMA Global", lambda ytr: forecast_sarima_global(ytr)),
        ("SARIMA Rolling", lambda ytr: forecast_sarima_rolling(ytr, window=window)),
        ("SARIMA + Break Dummy (oracle Tb)", lambda ytr: forecast_sarima_break_dummy_oracle(ytr, Tb=Tb)),
        ("Simple Exp. Smoothing (SES)", lambda ytr: forecast_ses(ytr)),
        ("Holt-Winters (additive)", lambda ytr: forecast_holt_winters(ytr)),
    ]

    errors = {name: [] for name, _ in methods}
    fails  = {name: 0 for name, _ in methods}

    for _ in range(n_sim):
        # Choose random forecast origin between Tb and T
        t0 = rng.integers(Tb + 1, T - 1)
        
        y = simulate_single_break_ar1(
            T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, sigma=sigma,
            innovation_type=innovation_type, dof=dof,
            rng=rng
        )
        y_train = y[:t0]
        y_true  = float(y[t0])

        for name, func in methods:
            try:
                f = func(y_train)
                errors[name].append(y_true - f)
            except Exception:
                fails[name] += 1

    rows = []
    for name in errors:
        e = np.asarray(errors[name], dtype=float)
        n_success = len(e)
        n_fail = fails[name]
        
        rows.append({
            "Method": name,
            "RMSE": rmse(e),
            "MAE": mae(e),
            "Bias": bias(e),
            "Var(error)": var_error(e),
            "Successes": n_success,
            "Failures": n_fail,
            "N": n_success  # Total attempts
        })

    df = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    
    # Save to outputs/
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs_dir / "mean_single_results.csv", index=False)
    
    return df

