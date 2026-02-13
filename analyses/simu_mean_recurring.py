"""
Mean Recurring (Markov-Switching) Break: Monte Carlo Simulations
================================================================
Monte Carlo experiments for AR(1) with recurring (Markov-switching) mean.

Uses same forecasting methods as single breaks for consistency.
- DGPs: dgps.mean_recurring
- Estimators: estimators.mean_singlebreak (same as single breaks)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dgps.mean_recurring import simulate_ms_ar1_mean_only
from estimators.mean_singlebreak import (
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_sarima_break_dummy_oracle,
    forecast_ses,
    forecast_holt_winters,
)
from analyses.metrics import rmse, mae, bias, var_error


def mc_mean_recurring(
    n_sim=300,
    T=400,
    p=0.95,
    phi=0.6,
    mu0=0.0,
    mu1=2.0,
    sigma=1.0,
    window=70,
    horizon=1,
    seed=42,
    verbose=False,
    innovation_type='gaussian',
    dof=None
):
    """
    Monte Carlo for recurring (Markov-switching) mean breaks.
    Uses same forecasting methods as single breaks for consistency.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        p: Persistence (p00, p11)
        phi: AR(1) coefficient
        mu0: Mean in regime 0
        mu1: Mean in regime 1
        sigma: Standard deviation
        window: Rolling window size for rolling SARIMA
        horizon: Forecast horizon (1-step ahead)
        seed: Random seed
        verbose: Print progress
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
        
    Returns:
        DataFrame with RMSE, MAE, Bias, Variance for each method
    """
    rng = np.random.default_rng(seed)
    
    methods = [
        ("SARIMA Global", lambda ytr: forecast_sarima_global(ytr)),
        ("SARIMA Rolling", lambda ytr: forecast_sarima_rolling(ytr, window=window)),
        ("SARIMA + Break Dummy (oracle Tb)", lambda ytr: forecast_sarima_break_dummy_oracle(ytr, Tb=T//2)),
        ("Simple Exp. Smoothing (SES)", lambda ytr: forecast_ses(ytr)),
        ("Holt-Winters (additive)", lambda ytr: forecast_holt_winters(ytr)),
    ]
    
    errors = {name: [] for name, _ in methods}
    failures = {name: 0 for name, _ in methods}
    
    for sim in range(n_sim):
        if verbose and (sim + 1) % max(1, n_sim // 10) == 0:
            print(f"  MC iteration {sim+1}/{n_sim}")
        
        # Generate Markov-switching data
        y, _ = simulate_ms_ar1_mean_only(
            T=T, p00=p, p11=p, phi=phi, mu0=mu0, mu1=mu1, sigma=sigma,
            seed=rng.integers(0, 1_000_000)
        )
        
        # Choose random forecast origin (allow sufficient data for training)
        t_orig = rng.integers(max(T // 4, 50), T - horizon - 1)
        y_train = y[:t_orig]
        y_true = float(y[t_orig])
        
        # Generate forecasts using single break methods
        for method_name, method_func in methods:
            try:
                pred = method_func(y_train)
                if not np.isnan(pred):
                    errors[method_name].append(y_true - pred)
                else:
                    failures[method_name] += 1
            except Exception:
                failures[method_name] += 1
    
    # Compute metrics
    rows = []
    for method_name in [name for name, _ in methods]:
        e = np.asarray(errors[method_name], dtype=float)
        n_success = len(e)
        n_fail = failures[method_name]
        
        rows.append({
            "Method": method_name,
            "RMSE": rmse(e),
            "MAE": mae(e),
            "Bias": bias(e),
            "Var(error)": var_error(e),
            "Successes": n_success,
            "Failures": n_fail,
            "N": n_success
        })
    
    df = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    df["Break Type"] = f"Recurring (p={p})"
    
    # Save to outputs/
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs_dir / f"mean_recurring_p{p}_results.csv", index=False)
    
    return df


__all__ = ["mc_mean_recurring"]
