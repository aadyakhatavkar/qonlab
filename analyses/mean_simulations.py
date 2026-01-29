"""
Mean Break Monte Carlo Simulations
===================================
MC runners for mean break analysis.
Uses auto-selected ARMA models (Box-Jenkins methodology) for non-seasonal breaks.
Also includes SARIMA methods for seasonal mean breaks.
"""
import numpy as np
import pandas as pd

from dgps.mean import simulate_mean_break_ar1, simulate_mean_break_ar1_seasonal
from estimators.mean import (
    mean_forecast_global_arma,
    mean_forecast_rolling_arma,
    mean_forecast_ar1_with_break_dummy_oracle,
    mean_forecast_ar1_with_estimated_break,
    mean_forecast_markov_switching,
    mean_metrics,
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_sarima_with_break_dummy,
    forecast_sarima_with_estimated_break,
)


def mc_mean_breaks(
    n_sim=200,
    T=300,
    Tb=150,
    window=60,
    seed=123,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    gap_after_break=20,
    trim=0.15,
    verbose=True
):
    """
    Monte Carlo simulation for mean break forecasting.
    
    Uses auto-selected ARMA models (Box-Jenkins methodology) for forecasting.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        Tb: Break point
        window: Rolling window size
        seed: Random seed
        mu0, mu1: Pre/post-break means
        phi: AR coefficient
        sigma: Innovation std
        gap_after_break: Forecast origin is Tb + gap_after_break
        trim: Trimming for break detection
        verbose: Print progress
    
    Returns:
        pd.DataFrame: Results with RMSE, MAE, Bias per method
    """
    rng = np.random.default_rng(seed)
    t0 = Tb + gap_after_break
    
    if t0 >= T:
        raise ValueError("gap_after_break too large for T.")

    errors = {
        "ARMA Global (auto)": [],
        "ARMA Rolling (auto)": [],
        "ARMA + Break Dummy (oracle Tb)": [],
        "ARMA + Estimated Break (grid)": [],
        "Markov Switching (2 regimes)": [],
    }
    fails = {k: 0 for k in errors}

    if verbose:
        print(f"--- MC Mean Break START | n_sim={n_sim} ---")

    for i in range(n_sim):
        if verbose and i % 50 == 0:
            print(f"  Iteration {i}/{n_sim}")
            
        y = simulate_mean_break_ar1(T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, sigma=sigma, rng=rng)
        y_train = y[:t0]
        y_true = float(y[t0])

        try:
            f = mean_forecast_global_arma(y_train)
            errors["ARMA Global (auto)"].append(y_true - f)
        except Exception:
            fails["ARMA Global (auto)"] += 1

        try:
            f = mean_forecast_rolling_arma(y_train, window=window)
            errors["ARMA Rolling (auto)"].append(y_true - f)
        except Exception:
            fails["ARMA Rolling (auto)"] += 1

        try:
            f = mean_forecast_ar1_with_break_dummy_oracle(y_train, Tb=Tb)
            errors["ARMA + Break Dummy (oracle Tb)"].append(y_true - f)
        except Exception:
            fails["ARMA + Break Dummy (oracle Tb)"] += 1

        try:
            f = mean_forecast_ar1_with_estimated_break(y_train, trim=trim)
            errors["ARMA + Estimated Break (grid)"].append(y_true - f)
        except Exception:
            fails["ARMA + Estimated Break (grid)"] += 1

        try:
            f = mean_forecast_markov_switching(y_train, k_regimes=2)
            errors["Markov Switching (2 regimes)"].append(y_true - f)
        except Exception:
            fails["Markov Switching (2 regimes)"] += 1

    if verbose:
        print(f"--- MC Mean Break END ---\n")

    rows = []
    for method, e in errors.items():
        if len(e) == 0:
            rows.append({"Method": method, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0, "Fails": fails[method]})
        else:
            m = mean_metrics(e)
            rows.append({"Method": method, "RMSE": m["RMSE"], "MAE": m["MAE"], "Bias": m["Bias"], "N": len(e), "Fails": fails[method]})

    results = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return results

def mc_mean_breaks_seasonal(
    n_sim=200,
    T=300,
    Tb=150,
    window=60,
    seed=123,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    s=12,
    A=1.0,
    gap_after_break=20,
    order=(1, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    trim=0.15,
    verbose=True
):
    """
    Monte Carlo simulation for mean breaks with seasonality (SARIMA).
    
    Bakhodir's extension: Tests SARIMA methods on seasonal AR(1) + mean break.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        Tb: Break point
        window: Rolling window size
        seed: Random seed
        mu0, mu1: Pre/post-break means
        phi: AR coefficient
        sigma: Innovation std
        s: Seasonal period (e.g., 12 for monthly)
        A: Seasonal amplitude
        gap_after_break: Forecast origin is Tb + gap_after_break
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal ARIMA order
        trim: Trimming for break detection
        verbose: Print progress
    
    Returns:
        pd.DataFrame: Results with RMSE, MAE, Bias per SARIMA method
    """
    rng = np.random.default_rng(seed)
    t0 = Tb + gap_after_break
    
    if t0 >= T:
        raise ValueError("gap_after_break too large for T.")

    errors = {
        "SARIMA Global": [],
        "SARIMA Rolling": [],
        "SARIMA + Break Dummy (oracle Tb)": [],
        "SARIMA + Estimated Break": [],
    }
    fails = {k: 0 for k in errors}

    if verbose:
        print(f"--- MC Mean Break (Seasonal) START | n_sim={n_sim} ---")

    for i in range(n_sim):
        if verbose and i % 50 == 0:
            print(f"  Iteration {i}/{n_sim}")
            
        y = simulate_mean_break_ar1_seasonal(
            T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, sigma=sigma, 
            s=s, A=A, rng=rng
        )
        y_train = y[:t0]
        y_true = float(y[t0])

        try:
            f = forecast_sarima_global(y_train, order=order, seasonal_order=seasonal_order)
            errors["SARIMA Global"].append(y_true - f)
        except Exception:
            fails["SARIMA Global"] += 1

        try:
            f = forecast_sarima_rolling(y_train, window=window, order=order, seasonal_order=seasonal_order)
            errors["SARIMA Rolling"].append(y_true - f)
        except Exception:
            fails["SARIMA Rolling"] += 1

        try:
            f = forecast_sarima_with_break_dummy(y_train, Tb=Tb, order=order, seasonal_order=seasonal_order)
            errors["SARIMA + Break Dummy (oracle Tb)"].append(y_true - f)
        except Exception:
            fails["SARIMA + Break Dummy (oracle Tb)"] += 1

        try:
            f = forecast_sarima_with_estimated_break(y_train, order=order, seasonal_order=seasonal_order, trim=trim)
            errors["SARIMA + Estimated Break"].append(y_true - f)
        except Exception:
            fails["SARIMA + Estimated Break"] += 1

    if verbose:
        print(f"--- MC Mean Break (Seasonal) END ---\n")

    rows = []
    for method, e in errors.items():
        if len(e) == 0:
            rows.append({"Method": method, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0, "Fails": fails[method]})
        else:
            m = mean_metrics(e)
            rows.append({"Method": method, "RMSE": m["RMSE"], "MAE": m["MAE"], "Bias": m["Bias"], "N": len(e), "Fails": fails[method]})

    results = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return results