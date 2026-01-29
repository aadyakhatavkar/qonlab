"""
Mean Break Monte Carlo Simulations
===================================
MC runners and visualization for mean break analysis.

Extracted from: scripts/legacy_mean_change/meanchange_singlbreak_scenario.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# =============================================================================
# DGP: AR(1) WITH MEAN BREAK
# =============================================================================

def simulate_mean_break_ar1(
    T=300,
    Tb=150,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    y0=0.0,
    rng=None,
    seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the intercept (mean).
    
    Parameters:
        T: Sample size
        Tb: Break point
        mu0: Pre-break mean
        mu1: Post-break mean
        phi: AR coefficient
        sigma: Innovation standard deviation
        y0: Initial value
        rng: Random number generator (optional)
        seed: Random seed (optional, used if rng not provided)
    
    Returns:
        y: Simulated time series
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    y = np.zeros(T, dtype=float)
    y[0] = y0
    
    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi * y[t-1] + rng.normal(0.0, sigma)
    
    return y


# =============================================================================
# FORECASTERS FOR MEAN BREAKS
# =============================================================================

def mean_forecast_global_ar1(y_train):
    """Forecast using global AR(1) model."""
    m = ARIMA(y_train, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def mean_forecast_rolling_ar1(y_train, window=60):
    """Forecast using rolling window AR(1) model."""
    y_sub = y_train[-window:] if len(y_train) > window else y_train
    m = ARIMA(y_sub, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def mean_forecast_ar1_with_break_dummy_oracle(y_train, Tb):
    """
    AR(1) with break dummy (ORACLE - knows true break date).
    Model: y_t = c + φ*y_{t-1} + δ*d_t + u_t
    """
    y = np.asarray(y_train, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    d = (np.arange(1, len(y)) > Tb).astype(float)
    X = np.column_stack([np.ones_like(y_lag), y_lag, d])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    c, phi, delta = beta
    t_next = len(y)
    d_next = 1.0 if t_next > Tb else 0.0
    return float(c + phi * y[-1] + delta * d_next)


def _mean_fit_ar1_ols(y_segment):
    """Helper: Fit AR(1) via OLS and return coefficients + SSE."""
    y = np.asarray(y_segment, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    resid = y_dep - X @ beta
    sse = float(np.sum(resid**2))
    return float(beta[0]), float(beta[1]), sse


def mean_estimate_break_point_gridsearch(y_train, trim=0.15):
    """Estimate mean break point via SSE minimization grid search."""
    y = np.asarray(y_train, dtype=float)
    T = len(y)
    lo = max(int(np.floor(trim * T)), 10)
    hi = min(int(np.ceil((1 - trim) * T)) - 1, T - 11)
    best_Tb, best_sse = None, np.inf
    
    for Tb in range(lo, hi):
        _, _, sse1 = _mean_fit_ar1_ols(y[:Tb+1])
        _, _, sse2 = _mean_fit_ar1_ols(y[Tb+1:])
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    
    return int(best_Tb if best_Tb is not None else T // 2)


def mean_forecast_ar1_with_estimated_break(y_train, trim=0.15):
    """Forecast from post-break regime using estimated break point."""
    y = np.asarray(y_train, dtype=float)
    Tb_hat = mean_estimate_break_point_gridsearch(y, trim=trim)
    regime = y[Tb_hat+1:] if Tb_hat + 1 < len(y) else y
    
    if len(regime) < 20:
        return mean_forecast_global_ar1(y_train)
    
    m = ARIMA(regime, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def mean_forecast_markov_switching(y_train, k_regimes=2):
    """Markov switching model for mean breaks."""
    y = np.asarray(y_train, dtype=float)
    m = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=False).fit(disp=False)
    pred = m.predict(start=len(y), end=len(y))
    return float(np.asarray(pred)[0])


# =============================================================================
# METRICS
# =============================================================================

def mean_metrics(errors):
    """Compute RMSE, MAE, Bias from forecast errors."""
    e = np.asarray(errors, dtype=float)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================

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
        "ARIMA Global (AR1)": [],
        "ARIMA Rolling (AR1)": [],
        "AR1 + Break Dummy (oracle Tb)": [],
        "AR1 + Estimated Break (grid)": [],
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
            f = mean_forecast_global_ar1(y_train)
            errors["ARIMA Global (AR1)"].append(y_true - f)
        except Exception:
            fails["ARIMA Global (AR1)"] += 1

        try:
            f = mean_forecast_rolling_ar1(y_train, window=window)
            errors["ARIMA Rolling (AR1)"].append(y_true - f)
        except Exception:
            fails["ARIMA Rolling (AR1)"] += 1

        try:
            f = mean_forecast_ar1_with_break_dummy_oracle(y_train, Tb=Tb)
            errors["AR1 + Break Dummy (oracle Tb)"].append(y_true - f)
        except Exception:
            fails["AR1 + Break Dummy (oracle Tb)"] += 1

        try:
            f = mean_forecast_ar1_with_estimated_break(y_train, trim=trim)
            errors["AR1 + Estimated Break (grid)"].append(y_true - f)
        except Exception:
            fails["AR1 + Estimated Break (grid)"] += 1

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


# =============================================================================
# VISUALIZATION
# =============================================================================

def mean_plot_dgp_example(T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, seed=42, save_path=None):
    """Plot example realization of mean break DGP."""
    y = simulate_mean_break_ar1(T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, seed=seed)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y, color="black", lw=1.4)
    ax.axvline(Tb, color="red", linestyle="--", linewidth=2, label=f"Break at t={Tb}")
    ax.axhline(mu0/(1-phi), color="blue", linestyle=":", alpha=0.5, label=f"Pre-break mean ≈ {mu0/(1-phi):.2f}")
    ax.axhline(mu1/(1-phi), color="green", linestyle=":", alpha=0.5, label=f"Post-break mean ≈ {mu1/(1-phi):.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(f"DGP: Single Mean Break (μ₀={mu0} → μ₁={mu1})")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def mean_plot_results_bar(df_results, save_path=None):
    """Bar plot of RMSE by method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = df_results["Method"].values
    rmse = df_results["RMSE"].values
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    bars = ax.bar(range(len(methods)), rmse, color=colors)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel("RMSE")
    ax.set_title("Mean Break Forecasting: RMSE by Method")
    
    for bar, val in zip(bars, rmse):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
