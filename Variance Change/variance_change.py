import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
try:
    from arch import arch_model
except Exception:
    arch_model = None
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None
try:
    from lstm_forecaster import LSTMForecaster, create_and_train_lstm
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False


def simulate_variance_break(
    T=400, Tb=200, phi=0.6, mu=0.0, sigma1=1.0, sigma2=2.0, seed=None
):
    """
    AR(1) with variance break:
      y_t = mu + phi*(y_{t-1}-mu) + eps_t
      eps_t ~ N(0, sigma1^2) for t < Tb, N(0, sigma2^2) for t >= Tb
    """
    # Validate inputs
    if not (1 <= Tb < T):
        raise ValueError(f"Tb must satisfy 1 <= Tb < T (got Tb={Tb}, T={T})")

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = np.zeros(T)

    eps[:Tb] = rng.normal(0.0, sigma1, size=Tb)
    eps[Tb:] = rng.normal(0.0, sigma2, size=T - Tb)

    for t in range(1, T):
        y[t] = mu + phi * (y[t - 1] - mu) + eps[t]
    return y


def forecast_dist_arima_global(y_train, horizon=1, order=(1, 0, 0)):
    """
    Returns mean forecast and forecast variance using statsmodels get_forecast.
    """
    res = ARIMA(y_train, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_dist_arima_rolling(y_train, window=100, horizon=1, order=(1, 0, 0)):
    """
    Fit on last window only.
    Returns mean forecast and forecast variance for horizon steps.
    """
    y_win = y_train[-window:]
    res = ARIMA(y_win, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_garch_variance(y_train, horizon=1, p=1, q=1):
    """Fit a GARCH(p,q) model (with AR(1) mean) and return mean and variance forecasts.

    If `arch` is not installed, raises ImportError.
    
    NOTE: GARCH is not ideal for structural variance breaks (per Pesaran 2013).
    Use forecast_averaged_window instead for break detection.
    """
    if arch_model is None:
        raise ImportError("arch package is required for GARCH forecasts (pip install arch)")

    # Fit AR(1)-GARCH(p,q) model
    model = arch_model(y_train, mean='AR', lags=1, vol='GARCH', p=p, q=q, rescale=False)
    res = model.fit(disp='off')

    # arch Forecast object: use `res.forecast` to get mean and variance
    fc = res.forecast(horizon=horizon, reindex=False)
    # mean forecasts: (t, h) array, take last row
    mean = fc.mean.values[-1]
    var = fc.variance.values[-1]
    mean = np.asarray(mean)
    var = np.asarray(var)
    return mean, var


def forecast_lstm(y_train, horizon=1, lookback=20, epochs=30):
    """
    LSTM neural network forecast (modern alternative to ARIMA).
    
    Advantages over ARIMA:
    - Better captures non-linear relationships
    - Naturally provides uncertainty via Monte Carlo dropout
    - Better with long sequences (100+ time steps)
    - GPU-accelerated for large datasets
    
    Args:
        y_train: Training data
        horizon: Forecast horizon
        lookback: Context window (default 20 time steps)
        epochs: Training epochs
    
    Returns:
        mean: Point forecast
        var: Forecast variance (from MC dropout)
    
    Raises:
        ImportError if TensorFlow not installed
    """
    if not LSTM_AVAILABLE:
        raise ImportError("LSTM requires TensorFlow. Install: pip install tensorflow")
    
    # Train model
    lstm_model = create_and_train_lstm(y_train, lookback=lookback, epochs=epochs, verbose=0)
    
    # Get probabilistic forecast (uses Monte Carlo dropout for uncertainty)
    mean, var, _ = lstm_model.predict_with_uncertainty(y_train, horizon=horizon, n_samples=50)
    
    return np.asarray(mean), np.asarray(var)


def forecast_averaged_window(y_train, window_sizes=[20, 50, 100], horizon=1, order=(1, 0, 0)):
    """
    Pesaran (2013) Optimal Window Selection: Average forecasts across multiple windows.
    
    This is superior to GARCH for detecting structural variance breaks because:
    1. It doesn't assume mean-reversion (GARCH does)
    2. It explicitly trades off bias (large window) vs. variance (small window)
    3. Recent data is weighted more heavily via smaller windows
    
    Args:
        y_train: Training data
        window_sizes: List of window sizes to average over
        horizon: Forecast horizon
        order: ARIMA order
        
    Returns:
        mean: Average point forecast
        var: Average variance forecast
    """
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    
    means = []
    vars = []
    
    for ws in window_sizes:
        try:
            y_win = y_train[-ws:] if ws < len(y_train) else y_train
            res = ARIMA(y_win, order=order).fit()
            fc = res.get_forecast(steps=horizon)
            m = np.asarray(fc.predicted_mean)
            v = np.asarray(fc.var_pred_mean)
            means.append(m)
            vars.append(v)
        except Exception:
            # If window size fails, skip
            continue
    
    if not means:
        # Fallback to global if all windows fail
        return forecast_dist_arima_global(y_train, horizon=horizon, order=order)
    
    # Equal-weight average (can be modified to bias toward smaller windows)
    mean_avg = np.mean(np.array(means), axis=0)
    var_avg = np.mean(np.array(vars), axis=0)
    
    return mean_avg, var_avg


def rmse_mae_bias(y_true, y_pred):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias


def interval_coverage(y_true, mean, var, level=0.95):
    z = norm.ppf(0.5 + level / 2.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def log_score_normal(y_true, mean, var):
    """Logarithmic score (higher is better). Robust to NaN values."""
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    var = np.asarray(var)
    
    # Handle NaN/Inf
    mask = np.isfinite(mean) & np.isfinite(var) & np.isfinite(y_true)
    if not np.any(mask):
        return np.nan
    
    var = np.maximum(var[mask], 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (y_true[mask] - mean[mask]) ** 2 / var)))


def mc_variance_breaks_grid(
    n_sim=100,
    T=400,
    phi=0.6,
    horizon=20,
    window_sizes=[20, 50, 100, 200],
    break_magnitudes=[1.5, 2.0, 3.0, 5.0],
    seed=42
):
    """
    Pesaran (2013) Optimal Window Selection: Grid-based Monte Carlo.
    
    Tests how optimal window size depends on magnitude of variance break.
    This is the "Loss Surface" as referenced in the course materials.
    
    Args:
        n_sim: Number of MC simulations per (window, break_mag) pair
        T: Sample size
        phi: AR(1) persistence
        horizon: Forecast horizon
        window_sizes: List of rolling window sizes to test
        break_magnitudes: List of variance multipliers (sigma2 = sigma1 * multiplier)
        seed: Random seed
        
    Returns:
        df_loss: DataFrame with columns [Window, BreakMag, RMSE_Avg, Coverage95_Avg, LogScore_Avg]
    """
    rng = np.random.default_rng(seed)
    results = []
    
    Tb = T // 2  # Break at midpoint
    sigma1 = 1.0
    
    for break_mag in break_magnitudes:
        sigma2 = sigma1 * break_mag
        
        for ws in window_sizes:
            rmse_list = []
            cov_list = []
            ls_list = []
            
            # Run simulations
            seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(n_sim)]
            
            for s in seeds:
                y = simulate_variance_break(T=T, Tb=Tb, phi=phi, sigma1=sigma1, sigma2=sigma2, seed=s)
                y_train = y[:-horizon]
                y_test = y[-horizon:]
                
                # Test rolling window
                try:
                    m_roll, v_roll = forecast_dist_arima_rolling(y_train, window=ws, horizon=horizon)
                    rmse, _, _ = rmse_mae_bias(y_test, m_roll)
                    cov95 = interval_coverage(y_test, m_roll, v_roll, level=0.95)
                    ls = log_score_normal(y_test, m_roll, v_roll)
                    
                    rmse_list.append(rmse)
                    cov_list.append(cov95)
                    ls_list.append(ls)
                except Exception:
                    pass
            
            if rmse_list:
                results.append({
                    'Window': ws,
                    'BreakMagnitude': break_mag,
                    'RMSE': np.mean(rmse_list),
                    'Coverage95': np.mean(cov_list),
                    'LogScore': np.mean(ls_list),
                    'N_Sims': len(rmse_list)
                })
    
    return pd.DataFrame(results)


def mc_variance_breaks(
    n_sim=200,
    T=400,
    phi=0.6,
    window=100,
    horizon=20,
    scenarios=None,
    seed=42
):
    """
    Monte Carlo for variance breaks (legacy interface).

    Outputs:
      - Point metrics: RMSE/MAE/Bias
      - Uncertainty metrics: Coverage80/Coverage95/LogScore

    Notes:
      - For advanced Pesaran (2013) analysis, use mc_variance_breaks_grid instead.
      - This function tests a single window size across multiple scenarios.
    """
    rng = np.random.default_rng(seed)

    # Validate and normalize scenarios
    scenarios = _validate_scenarios(scenarios, T)

    point_rows = []
    unc_rows = []

    for sc in scenarios:
        name = sc["name"]
        Tb = sc["Tb"]
        sigma1 = sc["sigma1"]
        sigma2 = sc["sigma2"]

        point_g = []
        point_r = []
        point_garch = []
        point_lstm = []
        unc_g = []
        unc_r = []
        unc_garch = []
        unc_lstm = []

        # prepare seeds for reproducibility
        seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(n_sim)]

        def _run_one(s):
            y = simulate_variance_break(T=T, Tb=Tb, phi=phi, sigma1=sigma1, sigma2=sigma2, seed=s)
            y_train = y[:-horizon]
            y_test = y[-horizon:]

            mg, vg = forecast_dist_arima_global(y_train, horizon=horizon)
            mr, vr = forecast_dist_arima_rolling(y_train, window=window, horizon=horizon)
            try:
                mgarch, vgarch = forecast_garch_variance(y_train, horizon=horizon)
            except Exception:
                mgarch = np.full(horizon, np.nan)
                vgarch = np.full(horizon, np.nan)
            
            # LSTM forecast (if TensorFlow available)
            try:
                mlstm, vlstm = forecast_lstm(y_train, horizon=horizon, lookback=20, epochs=20)
            except Exception:
                mlstm = np.full(horizon, np.nan)
                vlstm = np.full(horizon, np.nan)

            return (
                rmse_mae_bias(y_test, mg),
                rmse_mae_bias(y_test, mr),
                rmse_mae_bias(y_test, mgarch),
                rmse_mae_bias(y_test, mlstm),
                (
                    interval_coverage(y_test, mg, vg, 0.80),
                    interval_coverage(y_test, mg, vg, 0.95),
                    log_score_normal(y_test, mg, vg)
                ),
                (
                    interval_coverage(y_test, mr, vr, 0.80),
                    interval_coverage(y_test, mr, vr, 0.95),
                    log_score_normal(y_test, mr, vr)
                ),
                (
                    interval_coverage(y_test, mgarch, vgarch, 0.80),
                    interval_coverage(y_test, mgarch, vgarch, 0.95),
                    log_score_normal(y_test, mgarch, vgarch)
                ),
                (
                    interval_coverage(y_test, mlstm, vlstm, 0.80),
                    interval_coverage(y_test, mlstm, vlstm, 0.95),
                    log_score_normal(y_test, mlstm, vlstm)
                )
            )

        if Parallel is not None:
            results = Parallel(n_jobs=1)(delayed(_run_one)(s) for s in seeds)
        else:
            results = [_run_one(s) for s in seeds]

        for res in results:
            pg_val, pr_val, pgarch_val, plstm_val, ug_val, ur_val, ugarch_val, ulstm_val = res
            point_g.append(pg_val)
            point_r.append(pr_val)
            point_garch.append(pgarch_val)
            point_lstm.append(plstm_val)
            unc_g.append(ug_val)
            unc_r.append(ur_val)
            unc_garch.append(ugarch_val)
            unc_lstm.append(ulstm_val)

        pg = np.mean(np.array(point_g), axis=0)
        pr = np.mean(np.array(point_r), axis=0)
        pgarch = np.mean(np.array(point_garch), axis=0)
        plstm = np.mean(np.array(point_lstm), axis=0)
        ug = np.mean(np.array(unc_g), axis=0)
        ur = np.mean(np.array(unc_r), axis=0)
        ugarch = np.mean(np.array(unc_garch), axis=0)
        ulstm = np.mean(np.array(unc_lstm), axis=0)

        for metric, idx in [("RMSE", 0), ("MAE", 1), ("Bias", 2)]:
            point_rows.append({
                "Scenario": name,
                "Metric": metric,
                "ARIMA Global": pg[idx],
                "ARIMA Rolling": pr[idx],
                "GARCH": pgarch[idx] if len(point_garch) > 0 else np.nan,
                "LSTM": plstm[idx] if len(point_lstm) > 0 else np.nan,
            })

        for metric, idx in [("Coverage80", 0), ("Coverage95", 1), ("LogScore", 2)]:
            unc_rows.append({
                "Scenario": name,
                "Metric": metric,
                "ARIMA Global": ug[idx],
                "ARIMA Rolling": ur[idx],
                "GARCH": ugarch[idx] if len(unc_garch) > 0 else np.nan,
                "LSTM": ulstm[idx] if len(unc_lstm) > 0 else np.nan,
            })

    return pd.DataFrame(point_rows), pd.DataFrame(unc_rows)


def _validate_scenarios(scenarios, T):
    """Ensure scenarios is a list of dicts with required keys and valid Tb values.

    If scenarios is None, return a default list. If Tb >= T, adjust Tb to T-1.
    """
    if scenarios is None:
        return [{"name": "Single variance break", "Tb": max(1, T // 2), "sigma1": 1.0, "sigma2": 2.0}]

    validated = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            raise ValueError("Each scenario must be a dict")
        for key in ("name", "Tb", "sigma1", "sigma2"):
            if key not in sc:
                raise ValueError(f"Scenario missing required key: {key}")
        Tb = int(sc["Tb"])
        if Tb >= T:
            Tb = T - 1
        if Tb < 1:
            Tb = 1
        validated.append({"name": sc["name"], "Tb": Tb, "sigma1": float(sc["sigma1"]), "sigma2": float(sc["sigma2"])})
    return validated


def main():
    """
    Entry point for variance-break experiments with support for both legacy and Pesaran (2013) analyses.

    Use `--quick` to run a short test (fast, useful while developing).
    Use `--grid` to run the optimal window selection grid analysis (Pesaran framework).
    Other flags allow overriding defaults.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run variance-break Monte Carlo experiments")
    parser.add_argument("--quick", action="store_true", help="run a short, fast simulation")
    parser.add_argument("--grid", action="store_true", help="run grid analysis for optimal window selection (Pesaran 2013)")
    parser.add_argument("--n-sim", type=int, default=200, help="number of Monte Carlo simulations")
    parser.add_argument("--T", type=int, default=400, help="sample size T")
    parser.add_argument("--phi", type=float, default=0.6, help="AR(1) coefficient")
    parser.add_argument("--window", type=int, default=100, help="rolling-window size (legacy mode)")
    parser.add_argument("--horizon", type=int, default=20, help="forecast horizon")
    args = parser.parse_args()

    if args.quick:
        n_sim = min(10, args.n_sim)
        T = min(100, args.T)
        window = min(20, args.window)
        horizon = min(5, args.horizon)
    else:
        n_sim = args.n_sim
        T = args.T
        window = args.window
        horizon = args.horizon

    if args.grid:
        # Pesaran (2013) Grid Analysis: Optimal Window Selection
        print("\n" + "="*70)
        print("VARIANCE BREAK: OPTIMAL WINDOW SELECTION (Pesaran 2013)")
        print("="*70)
        print(f"Testing window sizes vs. break magnitudes (N={n_sim} sims each)")
        print("-"*70)
        
        window_sizes = [20, 50, 100, 200] if not args.quick else [20, 50]
        break_mags = [1.5, 2.0, 3.0, 5.0] if not args.quick else [1.5, 3.0]
        
        df_grid = mc_variance_breaks_grid(
            n_sim=n_sim,
            T=T,
            phi=args.phi,
            horizon=horizon,
            window_sizes=window_sizes,
            break_magnitudes=break_mags,
            seed=42
        )
        
        print("\nLOSS SURFACE: RMSE (lower is better)")
        print(df_grid.pivot(index='Window', columns='BreakMagnitude', values='RMSE').round(4).to_string())
        
        print("\nLOSS SURFACE: Coverage95 (closer to 0.95 is better)")
        print(df_grid.pivot(index='Window', columns='BreakMagnitude', values='Coverage95').round(4).to_string())
        
        print("\nLOSS SURFACE: LogScore (higher is better)")
        print(df_grid.pivot(index='Window', columns='BreakMagnitude', values='LogScore').round(4).to_string())
        
        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("-"*70)
        print("1. As variance break magnitude increases, smaller windows should")
        print("   perform better (lower RMSE, better coverage) because they")
        print("   adapt faster to the structural change.")
        print("2. When break_mag is small (1.5x), larger windows maintain")
        print("   stable variance estimation. As break_mag increases (5.0x),")
        print("   smaller windows dominate.")
        print("3. Log Scores directly measure density forecast quality.")
        print("   Focus on this metric for evaluating uncertainty quantification.")
        print("="*70)
        
    else:
        # Legacy interface: Single window, multiple scenarios
        df_point, df_unc = mc_variance_breaks(
            n_sim=n_sim,
            T=T,
            phi=args.phi,
            window=window,
            horizon=horizon,
        )

        print("\n=== VARIANCE BREAK: POINT METRICS ===")
        print(df_point.round(4).to_string(index=False))

        print("\n=== VARIANCE BREAK: UNCERTAINTY METRICS ===")
        print(df_unc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()