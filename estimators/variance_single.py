"""
Single Variance Break Estimators
=================================
Forecasting methods for AR(1) with a single variance break.
"""
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

try:
    from arch import arch_model
except Exception:
    arch_model = None


def forecast_variance_dist_sarima_global(y_train, horizon=1, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12)):
    """
    Forecast using global SARIMA model on full sample.
    Returns (mean, variance) tuple where variance is the residual variance.
    """
    try:
        res = ARIMA(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n"
        ).fit()
        fc_mean = res.forecast(steps=horizon)
        # Use residual variance as proxy for predictive variance
        residual_var = np.var(res.resid, ddof=1) if len(res.resid) > 1 else np.var(y_train, ddof=1)
        fc_var = np.full(horizon, residual_var)
        return np.asarray(fc_mean), np.asarray(fc_var)
    except Exception:
        return np.full(horizon, np.nan), np.full(horizon, np.nan)


def forecast_variance_dist_sarima_rolling(y_train, window=100, horizon=1, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12)):
    """
    Forecast using rolling window SARIMA model.
    Returns (mean, variance) tuple where variance is the residual variance.
    """
    try:
        y_win = y_train[-window:] if window < len(y_train) else y_train
        res = ARIMA(
            y_win,
            order=order,
            seasonal_order=seasonal_order,
            trend="n"
        ).fit()
        fc_mean = res.forecast(steps=horizon)
        # Use residual variance as proxy for predictive variance
        residual_var = np.var(res.resid, ddof=1) if len(res.resid) > 1 else np.var(y_win, ddof=1)
        fc_var = np.full(horizon, residual_var)
        return np.asarray(fc_mean), np.asarray(fc_var)
    except Exception:
        return np.full(horizon, np.nan), np.full(horizon, np.nan)


def forecast_garch_variance(y_train, horizon=1, p=1, q=1):
    """
    Forecast using GARCH model.
    """
    if arch_model is None:
        raise ImportError("arch package is required for GARCH forecasts (pip install arch)")

    model = arch_model(y_train, mean='AR', lags=1, vol='GARCH', p=p, q=q, rescale=False)
    res = model.fit(disp='off')
    fc = res.forecast(horizon=horizon, reindex=False)
    mean = fc.mean.values[-1]
    var = fc.variance.values[-1]
    mean = np.asarray(mean)
    var = np.asarray(var)
    return mean, var


def forecast_variance_averaged_window(y_train, window_sizes=[20, 50, 100], horizon=1, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12)):
    """
    Forecast by averaging predictions across multiple rolling windows.
    """
    try:
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes]

        forecasts = []

        for ws in window_sizes:
            try:
                fc = forecast_variance_dist_sarima_rolling(
                    y_train, window=ws, horizon=horizon, order=order, seasonal_order=seasonal_order
                )
                forecasts.append(fc)
            except Exception:
                continue

        if not forecasts:
            return forecast_variance_dist_sarima_global(y_train, horizon=horizon, order=order, seasonal_order=seasonal_order)

        return np.mean(np.array(forecasts), axis=0)
    except Exception:
        return np.full(horizon, np.nan)


def variance_rmse_mae_bias(y_true, y_pred):
    """Calculate RMSE, MAE, and Bias."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias


def variance_interval_coverage(y_true, mean, var, level=0.95):
    """Calculate prediction interval coverage probability."""
    z = norm.ppf(0.5 + level / 2.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def variance_log_score_normal(y_true, mean, var):
    """Calculate log-predictive score under normal distribution."""
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    var = np.asarray(var)
    mask = np.isfinite(mean) & np.isfinite(var) & np.isfinite(y_true)
    if not np.any(mask):
        return np.nan
    var = np.maximum(var[mask], 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (y_true[mask] - mean[mask]) ** 2 / var)))
