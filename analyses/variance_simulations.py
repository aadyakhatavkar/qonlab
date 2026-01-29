"""
Variance Break Monte Carlo Simulations
=======================================
MC runners for variance break analysis.
Uses ARIMA, GARCH, and post-break detection methods.
"""
import numpy as np
import pandas as pd

from dgps.variance import simulate_variance_break_ar1
from estimators.variance import (
    forecast_variance_dist_arima_global,
    forecast_variance_dist_arima_rolling,
    forecast_garch_variance,
    forecast_variance_arima_post_break,
    variance_rmse_mae_bias,
)


def mc_variance_breaks_post(
    n_sim=200,
    T=400,
    Tb=200,
    t_post=250,
    window=100,
    horizon=20,
    phi=0.6,
    sigma1=1.0,
    sigma2=2.0,
    distribution="normal",
    nu=3,
    seed=123,
    verbose=True
):
    """
    Monte Carlo simulation for variance break forecasting.
    Evaluates multi-step ahead forecasts over horizon.
    
    Parameters:
        n_sim: Number of MC replications
        T: Sample size
        Tb: Break point
        t_post: Forecast origin (post-break)
        window: Rolling window size
        horizon: Forecast horizon
        phi: AR(1) coefficient
        sigma1, sigma2: Pre/post-break variance levels
        distribution: 'normal' or 'student'
        nu: Degrees of freedom for Student-t
        seed: Random seed
        verbose: Print progress
    
    Returns:
        dict: Errors per method
    """
    rng = np.random.default_rng(seed)

    err = {
        "ARIMA Global": [],
        "ARIMA Rolling": [],
        "GARCH": [],
        "ARIMA Post-Break": []
    }

    if verbose:
        print(f"--- MC Variance Break START | distribution={distribution}, nu={nu} ---")

    for i in range(n_sim):
        if verbose and i % 50 == 0:
            print(f"  MC iteration {i}/{n_sim}")

        y = simulate_variance_break_ar1(
            T=T,
            Tb=Tb,
            phi=phi,
            sigma1=sigma1,
            sigma2=sigma2,
            distribution=distribution,
            nu=nu,
            seed=rng.integers(0, 1_000_000_000)
        )

        y_train = y[:t_post]
        y_test = y[t_post:t_post+horizon]
        
        if len(y_test) < horizon:
            continue

        # ARIMA Global
        try:
            mean_g, _ = forecast_variance_dist_arima_global(y_train, horizon=horizon)
            metrics_g = variance_rmse_mae_bias(y_test, mean_g)
            err["ARIMA Global"].append(metrics_g)
        except Exception:
            pass
        
        # ARIMA Rolling
        try:
            mean_r, _ = forecast_variance_dist_arima_rolling(y_train, window=window, horizon=horizon)
            metrics_r = variance_rmse_mae_bias(y_test, mean_r)
            err["ARIMA Rolling"].append(metrics_r)
        except Exception:
            pass
        
        # GARCH
        try:
            mean_garch, _ = forecast_garch_variance(y_train, horizon=horizon)
            metrics_garch = variance_rmse_mae_bias(y_test, mean_garch)
            err["GARCH"].append(metrics_garch)
        except Exception:
            pass
        
        # ARIMA Post-Break
        try:
            mean_pb, _ = forecast_variance_arima_post_break(y_train, horizon=horizon)
            metrics_pb = variance_rmse_mae_bias(y_test, mean_pb)
            err["ARIMA Post-Break"].append(metrics_pb)
        except Exception:
            pass

    if verbose:
        print(f"--- MC Variance Break END ---\n")

    return err


def mc_variance_breaks_full(
    n_sim=200,
    T=400,
    Tb=200,
    t_post=250,
    window=100,
    horizon=20,
    phi=0.6,
    distributions=None,
    seed=123,
    verbose=True
):
    """
    Full MC across multiple distribution types.
    
    Parameters:
        distributions: list of tuples [(label, type, nu), ...]
        
    Returns:
        all_err: dict of {label: {model: errors}}
        df_results: DataFrame with RMSE by distribution and model
    """
    if distributions is None:
        distributions = [
            ("Normal", "normal", None),
            ("Student-t df=5", "student", 5),
            ("Student-t df=3", "student", 3),
        ]
    
    all_err = {}
    rows = []
    
    for label, dist_type, nu in distributions:
        err = mc_variance_breaks_post(
            n_sim=n_sim, T=T, Tb=Tb, t_post=t_post, window=window,
            horizon=horizon, phi=phi, distribution=dist_type, nu=nu,
            seed=seed, verbose=verbose
        )
        all_err[label] = err
        
        for model, e in err.items():
            if len(e) > 0:
                # e is list of tuples (rmse, mae, bias) from variance_rmse_mae_bias
                rmse_vals = [x[0] for x in e]
                mae_vals = [x[1] for x in e]
                bias_vals = [x[2] for x in e]
                rows.append({
                    "Distribution": label,
                    "Model": model,
                    "RMSE": np.mean(rmse_vals),
                    "MAE": np.mean(mae_vals),
                    "Bias": np.mean(bias_vals),
                    "N": len(e),
                })
            else:
                rows.append({
                    "Distribution": label,
                    "Model": model,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "N": 0,
                })
    
    df_results = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return all_err, df_results


if __name__ == "__main__":
    all_err, df_results = mc_variance_breaks_full(
        n_sim=50, T=400, Tb=200, t_post=250, window=100, horizon=20
    )
    print("\n=== VARIANCE BREAK MC RESULTS ===")
    print(df_results.to_string(index=False))
