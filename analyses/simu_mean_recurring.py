"""
Mean Recurring (Markov-Switching) Break: Monte Carlo Simulations
================================================================
Monte Carlo experiments for AR(1) with recurring (Markov-switching) mean.

Uses:
- DGPs: dgps.mean_recurring
- Estimators: estimators.mean_recurring, estimators.mean_singlebreak
"""

import numpy as np
import pandas as pd
from dgps.mean_recurring import simulate_ms_ar1_mean_only
from estimators.mean_recurring import (
    forecast_ms_ar1_mean,
    forecast_mean_arima_global,
)
from protocols import calculate_metrics


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
    verbose=False
):
    """
    Monte Carlo for recurring (Markov-switching) mean breaks.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        p: Persistence (p00, p11)
        phi: AR(1) coefficient
        mu0: Mean in regime 0
        mu1: Mean in regime 1
        sigma: Standard deviation
        window: Rolling window size
        horizon: Forecast horizon
        seed: Random seed
        verbose: Print progress
        
    Returns:
        DataFrame with RMSE, MAE, Bias for each method
    """
    rng = np.random.default_rng(seed)
    
    methods = {
        "ARIMA Global": lambda ytr: forecast_mean_arima_global(ytr, horizon=horizon),
        "MS AR(1)": lambda ytr: forecast_ms_ar1_mean(ytr, horizon=horizon)[0],
    }
    
    errors = {m: [] for m in methods}
    failures = {m: 0 for m in methods}
    
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


__all__ = ["mc_mean_recurring"]
