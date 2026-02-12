"""
Variance Single Break: Monte Carlo Simulations
===============================================
Monte Carlo experiments for AR(1) with single variance break.

Uses:
- DGPs: dgps.variance_single.simulate_variance_break_ar1
- Estimators: estimators.variance_single
"""

import numpy as np
import pandas as pd
from dgps.variance_single import simulate_variance_break_ar1
from estimators.variance_single import (
    forecast_variance_dist_sarima_global,
    forecast_variance_dist_sarima_rolling,
    forecast_garch_variance,
    forecast_variance_sarima_post_break,
    forecast_variance_averaged_window,
)
from protocols import calculate_metrics


def mc_variance_single_break(
    n_sim=100,
    T=400,
    Tb=200,
    phi=0.6,
    sigma1=1.0,
    sigma2=2.0,
    window=100,
    horizon=1,
    seed=42,
    verbose=False
):
    """
    Monte Carlo for single variance break.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        Tb: Break point
        phi: AR(1) coefficient
        sigma1: Variance before break
        sigma2: Variance after break
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
        "GARCH": lambda ytr: _safe_forecast_garch(ytr, horizon=horizon),
        "SARIMA Post-Break": lambda ytr: forecast_variance_sarima_post_break(ytr, horizon=horizon)[0],
        "SARIMA Avg-Window": lambda ytr: forecast_variance_averaged_window(ytr, window_sizes=[50, 100, 200], horizon=horizon)[0],
    }
    
    errors = {m: [] for m in methods}
    failures = {m: 0 for m in methods}
    
    for sim in range(n_sim):
        if verbose and (sim + 1) % max(1, n_sim // 10) == 0:
            print(f"  MC iteration {sim+1}/{n_sim}")
        
        # Generate data
        y = simulate_variance_break_ar1(
            T=T, Tb=Tb, phi=phi, sigma1=sigma1, sigma2=sigma2, seed=rng.integers(0, 1_000_000)
        )
        
        # Select forecast origin (after break)
        t_orig = min(Tb + 20, T - horizon - 2)
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
    df["Break Type"] = "Single"
    return df


def _safe_forecast_garch(y_train, horizon=1):
    """Safe wrapper for GARCH that returns NaN on failure."""
    try:
        mean, var = forecast_garch_variance(y_train, horizon=horizon)
        return float(mean[0]) if isinstance(mean, np.ndarray) else float(mean)
    except Exception:
        return np.nan


__all__ = ["mc_variance_single_break"]
