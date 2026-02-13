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
    forecast_variance_averaged_window,
    variance_interval_coverage,
    variance_log_score_normal,
)
from protocols import calculate_metrics


def mc_variance_single_break(
    n_sim=300,
    T=400,
    Tb=200,
    phi=0.6,
    sigma1=1.0,
    sigma2=2.0,
    window=70,
    horizon=1,
    innovation_type='gaussian',
    dof=None,
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
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
        seed: Random seed
        verbose: Print progress
        
    Returns:
        DataFrame with RMSE, MAE, Bias for each method
    """
    rng = np.random.default_rng(seed)
    
    # Validate that we have room for post-break forecasts
    if Tb + 2 >= T:
        raise ValueError("Not enough data after break point for forecasting.")
    
    methods = {
        "SARIMA Global": lambda ytr: forecast_variance_dist_sarima_global(ytr, horizon=horizon),
        "SARIMA Rolling": lambda ytr: forecast_variance_dist_sarima_rolling(ytr, window=window, horizon=horizon),
        "GARCH": lambda ytr: _safe_forecast_garch(ytr, horizon=horizon),
        "SARIMA Avg-Window": lambda ytr: forecast_variance_averaged_window(ytr, window_sizes=[window], horizon=horizon),
    }
    
    errors = {m: [] for m in methods}
    failures = {m: 0 for m in methods}
    variance_preds = {m: [] for m in methods}  # (pred_mean, pred_var) tuples
    
    for sim in range(n_sim):
        if verbose and (sim + 1) % max(1, n_sim // 10) == 0:
            print(f"  MC iteration {sim+1}/{n_sim}")
        
        # Generate data
        y = simulate_variance_break_ar1(
            T=T, Tb=Tb, phi=phi, sigma1=sigma1, sigma2=sigma2, 
            innovation_type=innovation_type, dof=dof,
            seed=rng.integers(0, 1_000_000)
        )
        
        # Choose random forecast origin between Tb and T
        t_orig = rng.integers(Tb + 1, T - horizon - 1)
        y_train = y[:t_orig]
        y_true = float(y[t_orig]) if horizon == 1 else y[t_orig:t_orig+horizon]
        
        # Generate forecasts
        for method_name, method_func in methods.items():
            try:
                result = method_func(y_train)
                
                # Handle tuple (mean, var) or scalar predictions
                if isinstance(result, tuple) and len(result) == 2:
                    pred_mean, pred_var = result
                    pred_mean = float(np.asarray(pred_mean).flat[0])
                    pred_var = float(np.asarray(pred_var).flat[0])
                    variance_preds[method_name].append((pred_mean, pred_var))
                    pred = pred_mean
                elif isinstance(result, np.ndarray):
                    pred = float(result[0]) if len(result) > 0 else np.nan
                    variance_preds[method_name].append((pred, np.nan))
                else:
                    pred = float(result)
                    variance_preds[method_name].append((pred, np.nan))
                    
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
            cov80 = np.nan
            cov95 = np.nan
            logscore = np.nan
        else:
            metrics = calculate_metrics(e)
            
            # Compute coverage and log-score from variance predictions if available
            var_preds = variance_preds[method_name]
            if len(var_preds) > 0:
                means = np.array([m for m, v in var_preds])
                vars_ = np.array([v for m, v in var_preds])
                
                # Reconstruct y_true values for coverage calculation
                y_true_vals = means + e
                
                # Calculate coverage probabilities
                cov80 = variance_interval_coverage(y_true_vals, means, vars_, level=0.80)
                cov95 = variance_interval_coverage(y_true_vals, means, vars_, level=0.95)
                logscore = variance_log_score_normal(y_true_vals, means, vars_)
            else:
                cov80 = np.nan
                cov95 = np.nan
                logscore = np.nan
        
        rows.append({
            "Method": method_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Bias": metrics["Bias"],
            "Variance": metrics["Variance"],
            "Coverage80": cov80,
            "Coverage95": cov95,
            "LogScore": logscore,
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
