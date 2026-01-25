import numpy as np
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
try:
    from arch import arch_model
except Exception:
    arch_model = None
# LSTM support removed per admin request — keep estimators focused on
# classical methods (ARIMA, GARCH). Neural-model code has been deleted.


def forecast_dist_arima_global(y_train, horizon=1, order=(1, 0, 0)):
    res = ARIMA(y_train, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_dist_arima_rolling(y_train, window=100, horizon=1, order=(1, 0, 0)):
    y_win = y_train[-window:] if window < len(y_train) else y_train
    res = ARIMA(y_win, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_garch_variance(y_train, horizon=1, p=1, q=1):
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


def forecast_lstm(y_train, horizon=1, lookback=20, epochs=30):
    """Deprecated: LSTM support removed.

    The project removed LSTM components per project policy. This
    placeholder raises an informative error so callers are aware.
    """
    raise ImportError(
        "LSTM support removed from this repository. Use classical estimators in `estimators.ols_like`"
    )


def forecast_averaged_window(y_train, window_sizes=[20, 50, 100], horizon=1, order=(1, 0, 0)):
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
            continue

    if not means:
        return forecast_dist_arima_global(y_train, horizon=horizon, order=order)

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
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    var = np.asarray(var)
    mask = np.isfinite(mean) & np.isfinite(var) & np.isfinite(y_true)
    if not np.any(mask):
        return np.nan
    var = np.maximum(var[mask], 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (y_true[mask] - mean[mask]) ** 2 / var)))
