"""
Parameter Break Monte Carlo Simulations
========================================
MC runners and visualization for parameter break analysis.

Extracted from: scripts/legacy_parameter_change/parameter_single_break.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# =============================================================================
# DGP: AR(1) WITH PARAMETER BREAK (includes Student-t support)
# =============================================================================

def simulate_parameter_break_ar1(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    innovation="normal",
    df=None,
    rng=None,
    seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the AR coefficient.
    
    Parameters:
        T: Sample size
        Tb: Break point
        phi1: Pre-break AR coefficient
        phi2: Post-break AR coefficient
        sigma: Innovation standard deviation
        innovation: 'normal' or 'student' for distribution type
        df: Degrees of freedom for Student-t (required if innovation='student')
        rng: Random number generator (optional)
        seed: Random seed (optional, used if rng not provided)
    
    Returns:
        y: Simulated time series
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    y = np.zeros(T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2

        if innovation == "normal":
            eps = rng.normal(0.0, sigma)
        elif innovation == "student":
            if df is None or df <= 2:
                raise ValueError("df must be > 2 for Student-t innovations")
            eps = rng.standard_t(df) * sigma / np.sqrt(df / (df - 2))
        else:
            raise ValueError(f"Unknown innovation type: {innovation}")

        y[t] = phi * y[t - 1] + eps

    return y


# =============================================================================
# FORECASTERS FOR PARAMETER BREAKS
# =============================================================================

def param_forecast_global_ar(y):
    """Forecast using global AR(1) model without trend."""
    return float(
        ARIMA(y, order=(1, 0, 0), trend="n")
        .fit()
        .forecast(1)[0]
    )


def param_forecast_rolling_ar(y, window=80):
    """Forecast using rolling window AR(1) model without trend."""
    return float(
        ARIMA(y[-window:], order=(1, 0, 0), trend="n")
        .fit()
        .forecast(1)[0]
    )


def param_forecast_markov_switching_ar(y):
    """
    Forecast using Markov Switching AR model.
    Uses switching_exog to allow AR coefficient to vary by regime.
    """
    y_lag = y[:-1]
    y_curr = y[1:]

    model = MarkovRegression(
        endog=y_curr,
        k_regimes=2,
        trend="n",
        exog=y_lag.reshape(-1, 1),
        switching_exog=True,
        switching_variance=False
    ).fit(disp=False)

    params = dict(zip(model.model.param_names, model.params))
    probs = model.filtered_marginal_probabilities[-1]

    phi0 = params["x1[0]"]
    phi1 = params["x1[1]"]

    return float((probs[0] * phi0 + probs[1] * phi1) * y[-1])


# =============================================================================
# METRICS
# =============================================================================

def param_metrics(errors):
    """Compute RMSE, MAE, Bias from forecast errors."""
    e = np.asarray(errors)
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================

def mc_parameter_breaks_post(
    n_sim=300,
    T=400,
    Tb=200,
    t_post=250,
    window=80,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    innovation="normal",
    df=None,
    seed=123,
    verbose=True
):
    """
    Monte Carlo simulation for parameter break forecasting.
    Evaluates post-break 1-step ahead forecasts.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        Tb: Break point
        t_post: Forecast origin (post-break)
        window: Rolling window size
        phi1, phi2: Pre/post-break AR coefficients
        sigma: Innovation std
        innovation: 'normal' or 'student'
        df: Degrees of freedom for Student-t
        seed: Random seed
        verbose: Print progress
    
    Returns:
        dict: Errors per method
    """
    rng = np.random.default_rng(seed)

    err = {
        "Global AR": [],
        "Rolling AR": [],
        "MS AR": []
    }

    if verbose:
        print(f"--- MC START | innovation={innovation}, df={df} ---")

    for i in range(n_sim):
        if verbose and i % 50 == 0:
            print(f"  MC iteration {i}/{n_sim}")

        y = simulate_parameter_break_ar1(
            T=T,
            Tb=Tb,
            phi1=phi1,
            phi2=phi2,
            sigma=sigma,
            innovation=innovation,
            df=df,
            rng=rng
        )

        y_train = y[:t_post]
        y_true = y[t_post]

        try:
            err["Global AR"].append(y_true - param_forecast_global_ar(y_train))
        except Exception:
            pass
        
        try:
            err["Rolling AR"].append(y_true - param_forecast_rolling_ar(y_train, window))
        except Exception:
            pass
        
        try:
            err["MS AR"].append(y_true - param_forecast_markov_switching_ar(y_train))
        except Exception:
            pass

    if verbose:
        print(f"--- MC END | innovation={innovation} ---\n")

    return err


# =============================================================================
# VISUALIZATION
# =============================================================================

def param_plot_error_distributions(all_err, save_path=None):
    """
    Plot forecast error distributions across innovation types.
    
    Parameters:
        all_err: dict of {label: {model: errors}}
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(len(all_err), 1, figsize=(10, 3*len(all_err)), sharex=True)
    
    if len(all_err) == 1:
        axes = [axes]

    for ax, (label, err) in zip(axes, all_err.items()):
        for model, e in err.items():
            ax.hist(e, bins=40, density=True, alpha=0.4, label=model)

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Forecast Error Distribution — {label}")
        ax.set_ylabel("Density")
        ax.legend()

    axes[-1].set_xlabel("Forecast error")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def param_plot_rmse_by_innovation(df_results, save_path=None):
    """
    Bar plot of RMSE by innovation type and model.
    
    Parameters:
        df_results: DataFrame with columns [Innovation, Model, RMSE]
        save_path: Optional path to save figure
    """
    innovations = df_results["Innovation"].unique()
    models = df_results["Model"].unique()

    x = np.arange(len(innovations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        vals = [
            df_results.loc[
                (df_results["Innovation"] == innov) &
                (df_results["Model"] == model),
                "RMSE"
            ].values[0]
            for innov in innovations
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(innovations)
    ax.set_xlabel("Innovation")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Innovation Distribution and Model")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def param_plot_dgp_example(T=400, Tb=200, phi1=0.2, phi2=0.9, seed=42, save_path=None):
    """
    Plot example realization of parameter break DGP.
    """
    y = simulate_parameter_break_ar1(
        T=T, Tb=Tb, phi1=phi1, phi2=phi2,
        innovation="normal", seed=seed
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y, color="black", lw=1.4)
    ax.axvline(Tb, color="red", linestyle="--", linewidth=2, label=f"Break at t={Tb}")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(f"DGP: Single Parameter Break (φ₁={phi1} → φ₂={phi2})")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
