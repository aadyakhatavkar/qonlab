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
from pathlib import Path
from dgps.variance_recurring import simulate_ms_ar1_variance_only
from estimators.variance_recurring import forecast_markov_switching
from estimators.variance_single import (
    forecast_variance_dist_sarima_global,
    forecast_variance_dist_sarima_rolling,
    forecast_variance_averaged_window,
    variance_interval_coverage,
    variance_log_score_normal,
)
from analyses.metrics import rmse, mae, bias, var_error, coverage_from_errors, logscore_from_errors


def mc_variance_recurring(
    n_sim=300,
    T=400,
    p=0.95,
    phi=0.6,
    sigma1=1.0,
    sigma2=2.0,
    window=70,
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
        "SARIMA Global": lambda ytr: forecast_variance_dist_sarima_global(ytr, horizon=horizon),
        "SARIMA Rolling": lambda ytr: forecast_variance_dist_sarima_rolling(ytr, window=window, horizon=horizon),
        "SARIMA Avg-Window": lambda ytr: forecast_variance_averaged_window(ytr, window_sizes=[window], horizon=horizon),
        "MS AR(1)": lambda ytr: forecast_markov_switching(ytr, horizon=horizon),  # Return full tuple!
    }
    
    errors = {m: [] for m in methods}
    failures = {m: 0 for m in methods}
    variance_preds = {m: [] for m in methods}  # (pred_mean, pred_var) tuples
    
    for sim in range(n_sim):
        if verbose and (sim + 1) % max(1, n_sim // 10) == 0:
            print(f"  MC iteration {sim+1}/{n_sim}")
        
        # Generate Markov-switching data
        y, _ = simulate_ms_ar1_variance_only(
            T=T, p00=p, p11=p, phi=phi, sigma1=sigma1, sigma2=sigma2, 
            seed=rng.integers(0, 1_000_000)
        )
        
        # Choose random forecast origin (allow sufficient data for training)
        t_orig = rng.integers(max(T // 4, 50), T - horizon - 1)
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
            rmse_val = np.nan
            mae_val = np.nan
            bias_val = np.nan
            var_error_val = np.nan
            cov80 = np.nan
            cov95 = np.nan
            logscore = np.nan
        else:
            rmse_val = rmse(e)
            mae_val = mae(e)
            bias_val = bias(e)
            var_error_val = var_error(e)
            
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
            "RMSE": rmse_val,
            "MAE": mae_val,
            "Bias": bias_val,
            "Variance": var_error_val,
            "Coverage80": cov80,
            "Coverage95": cov95,
            "LogScore": logscore,
            "Successes": len(e),
            "Failures": failures[method_name],
            "N": n_sim,
        })
    
    df = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    df["Break Type"] = f"Recurring (p={p})"
    
    # Save to outputs/
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs_dir / f"variance_recurring_p{p}_results.csv", index=False)
    
    return df


__all__ = ["mc_variance_recurring"]
