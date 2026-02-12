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


def _auto_select_arima_order(y_train, max_p=5, max_d=2, max_q=5, method='aic'):
    """
    Auto-select ARIMA order using information criteria.
    
    Uses Box-Jenkins methodology with AIC or BIC to select (p, d, q).
    """
    y = np.asarray(y_train, dtype=float)
    
    # If series is very short, use default
    if len(y) < 20:
        return (1, 0, 0)
    
    best_order = (1, 0, 0)
    best_ic = np.inf
    
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(y, order=(p, d, q))
                    res = model.fit()
                    
                    if method.lower() == 'bic':
                        ic = res.bic
                    else:  # Default to AIC
                        ic = res.aic
                    
                    if ic < best_ic:
                        best_ic = ic
                        best_order = (p, d, q)
                except Exception:
                    continue
    
    return best_order


def forecast_variance_dist_sarima_global(y_train, horizon=1, order=None, seasonal_order=(1, 0, 0, 12), auto_select=True):
    """
    Forecast using global SARIMA model on full sample.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    if order is None and auto_select:
        order = _auto_select_arima_order(y_train)
    elif order is None:
        order = (1, 0, 0)
    
    res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_variance_dist_sarima_rolling(y_train, window=100, horizon=1, order=None, seasonal_order=(1, 0, 0, 12), auto_select=True):
    """
    Forecast using rolling window SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    y_win = y_train[-window:] if window < len(y_train) else y_train
    
    if order is None and auto_select:
        order = _auto_select_arima_order(y_win)
    elif order is None:
        order = (1, 0, 0)
    
    res = SARIMAX(y_win, order=order, seasonal_order=seasonal_order).fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


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


def forecast_variance_sarima_post_break(y_train, horizon=1, order=None, seasonal_order=(1, 0, 0, 12), auto_select=True):
    """
    Detect variance break point and forecast from post-break regime only.
    """
    from dgps.variance_single import estimate_variance_break_point
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    y = np.asarray(y_train, dtype=float)
    v_Tb_hat = estimate_variance_break_point(y, trim=0.15)
    
    y_post = y[v_Tb_hat+1:]
    if len(y_post) < 10:
        return forecast_variance_dist_sarima_global(y_train, horizon=horizon, order=order, seasonal_order=seasonal_order, auto_select=auto_select)
    
    if order is None and auto_select:
        order = _auto_select_arima_order(y_post)
    elif order is None:
        order = (1, 0, 0)
    
    res = SARIMAX(y_post, order=order, seasonal_order=seasonal_order).fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_variance_averaged_window(y_train, window_sizes=[20, 50, 100], horizon=1, order=None, seasonal_order=(1, 0, 0, 12), auto_select=True):
    """
    Forecast by averaging predictions across multiple rolling windows.
    """
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]

    variance_means = []
    variance_vars = []

    for ws in window_sizes:
        try:
            mean, var = forecast_variance_dist_sarima_rolling(
                y_train, window=ws, horizon=horizon, order=order, seasonal_order=seasonal_order, auto_select=auto_select
            )
            variance_means.append(mean)
            variance_vars.append(var)
        except Exception:
            continue

    if not variance_means:
        return forecast_variance_dist_sarima_global(y_train, horizon=horizon, order=order, seasonal_order=seasonal_order, auto_select=auto_select)

    mean_avg = np.mean(np.array(variance_means), axis=0)
    var_avg = np.mean(np.array(variance_vars), axis=0)
    return mean_avg, var_avg


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
