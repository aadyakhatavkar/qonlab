"""
Variance Recurring (Markov-Switching) Break: Monte Carlo Simulations
====================================================================
Monte Carlo experiments for AR(1) with recurring (Markov-switching) variance.

Uses:
- DGPs: dgps.variance_recurring
- Estimators: estimators.variance_recurring, estimators.variance_single
"""

import numpy as np
import pandas as pd
from dgps.variance_recurring import simulate_ms_ar1_variance_only
from estimators.variance_recurring import forecast_markov_switching
from estimators.variance_single import (
    forecast_variance_dist_sarima_global,
    forecast_variance_dist_sarima_rolling,
    forecast_variance_averaged_window,
)
from protocols import calculate_metrics


def mc_variance_recurring(
    n_sim=100,
    T=400,
    p=0.95,
    phi=0.6,
    sigma1=1.0,
    sigma2=2.0,
    window=100,
    horizon=1,
    seed=42,
    verbose=False
):
    """
    Monte Carlo for recurring (Markov-switching) variance breaks.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        p: Persistence (p00, p11)
        phi: AR(1) coefficient
        sigma1: Variance in regime 0
        sigma2: Variance in regime 1
        window: Rolling window size
        horizon: Forecast horizon
        seed: Random seed
        verbose: Print progress
        
    Returns:
        DataFrame with RMSE, MAE, Bias for each method
    """
    rng = np.random.default_rng(seed)
    
    methods = {
        "SARIMA Global": lambda ytr: forecast_variance_dist_sarima_global(ytr, horizon=horizon)[0],
        "SARIMA Rolling": lambda ytr: forecast_variance_dist_sarima_rolling(ytr, window=window, horizon=horizon)[0],
        "SARIMA Avg-Window": lambda ytr: forecast_variance_averaged_window(ytr, window_sizes=[50, 100, 200], horizon=horizon)[0],
        "MS AR(1)": lambda ytr: forecast_markov_switching(ytr, horizon=horizon)[0],
    }
    
    errors = {m: [] for m in methods}
    failures = {m: 0 for m in methods}
    
    for sim in range(n_sim):
        if verbose and (sim + 1) % max(1, n_sim // 10) == 0:
            print(f"  MC iteration {sim+1}/{n_sim}")
        
        # Generate Markov-switching data
        y = simulate_ms_ar1_variance_only(
            T=T, p00=p, p11=p, phi=phi, sigma1=sigma1, sigma2=sigma2, 
            seed=rng.integers(0, 1_000_000)
        )
        
        # Select forecast origin (need sufficient data)
        t_orig = min(T - horizon - 10, max(T // 2, 100))
        y_train = y[:t_orig]
        y_true = float(y[t_orig]) if horizon == 1 else y[t_orig:t_orig+horizon]
        
        # Generate forecasts
        for method_name, method_func in methods.items():
            try:
                pred = method_func(y_train)
                if isinstance(pred, np.ndarray):
                    pred = float(pred[0]) if len(pred) > 0 else np.nan
                else:
                    pred = float(pred)
                    
                if not np.isnan(pred):
                    errors[method_name].append(y_true - pred)
                else:
                    failures[method_name] += 1
            except Exception:
                failures[method_name] += 1
    
    # Compute metrics
    rows = []
    for method_name in methods:
        e = np.asarray(errors[method_name], dtype=float)
        e = e[~np.isnan(e)]
        
        if len(e) == 0:
            metrics = {"RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "Variance": np.nan}
        else:
            metrics = calculate_metrics(e)
        
        rows.append({
            "Method": method_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Bias": metrics["Bias"],
            "Variance": metrics["Variance"],
            "Successes": len(e),
            "Failures": failures[method_name],
        })
    
    df = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    df["Break Type"] = f"Recurring (p={p})"
    return df


__all__ = ["mc_variance_recurring"]
