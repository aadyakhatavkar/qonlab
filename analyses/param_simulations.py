"""
Parameter Break Monte Carlo Simulations
========================================
MC runners for parameter break analysis.
Uses auto-selected ARMA models (Box-Jenkins methodology).
"""
import numpy as np
import pandas as pd

from dgps.parameter import simulate_parameter_break_ar1
from estimators.parameter import (
    param_forecast_global_sarima,
    param_forecast_rolling_sarima,
    param_forecast_markov_switching_ar,
    param_metrics,
)


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
        "Global SARIMA": [],
        "Rolling SARIMA": [],
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
            err["Global SARIMA"].append(y_true - param_forecast_global_sarima(y_train))
        except Exception:
            pass

        try:
            err["Rolling SARIMA"].append(y_true - param_forecast_rolling_sarima(y_train, window))
        except Exception:
            pass
        
        try:
            err["MS AR"].append(y_true - param_forecast_markov_switching_ar(y_train))
        except Exception:
            pass

    if verbose:
        print(f"--- MC END | innovation={innovation} ---\n")

    return err


def mc_parameter_breaks_full(
    n_sim=300,
    T=400,
    Tb=200,
    t_post=250,
    window=80,
    innovations=None,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    seed=123,
    verbose=True
):
    """
    Full MC across multiple innovation types.
    
    Parameters:
        innovations: list of tuples [(label, type, df), ...]
        
    Returns:
        all_err: dict of {label: {model: errors}}
        df_results: DataFrame with RMSE by innovation and model
    """
    if innovations is None:
        innovations = [
            ("Gaussian", "normal", None),
            ("Student-t df=5", "student", 5),
            ("Student-t df=3", "student", 3),
        ]
    
    all_err = {}
    rows = []
    
    for label, innov_type, df in innovations:
        err = mc_parameter_breaks_post(
            n_sim=n_sim, T=T, Tb=Tb, t_post=t_post, window=window,
            phi1=phi1, phi2=phi2, sigma=sigma,
            innovation=innov_type, df=df, seed=seed, verbose=verbose
        )
        all_err[label] = err
        
        for model, e in err.items():
            if len(e) > 0:
                m = param_metrics(e)
                rows.append({
                    "Innovation": label,
                    "Model": model,
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "Bias": m["Bias"],
                    "N": len(e)
                })
    
    df_results = pd.DataFrame(rows)
    return all_err, df_results
