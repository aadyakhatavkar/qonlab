import numpy as np
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
try:
    from arch import arch_model
except Exception:
    arch_model = None
# LSTM support removed per admin request — keep estimators focused on
# classical methods (ARIMA, GARCH). Neural-model code has been deleted.


def _auto_select_arima_order(y_train, max_p=5, max_d=2, max_q=5, method='aic'):
    """
    Auto-select ARIMA order using information criteria or cross-validation.
    
    Uses Box-Jenkins methodology with AIC or BIC to select (p, d, q).
    
    Parameters:
        y_train: Training data
        max_p: Maximum AR order to consider
        max_d: Maximum differencing order to consider
        max_q: Maximum MA order to consider
        method: 'aic' or 'bic' for model selection criterion
    
    Returns:
        Optimal (p, d, q) tuple
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


def forecast_variance_dist_arima_global(y_train, horizon=1, order=None, auto_select=True):
    """
    Forecast using global ARIMA model, providing both mean and variance.
    
    Parameters:
        y_train: Training data
        horizon: Forecast horizon
        order: ARIMA order tuple (p, d, q). If None and auto_select=True, will be auto-selected.
        auto_select: Whether to auto-select order if not provided
    
    Returns:
        mean, variance forecasts
    """
    if order is None and auto_select:
        order = _auto_select_arima_order(y_train)
    elif order is None:
        order = (1, 0, 0)
    
    res = ARIMA(y_train, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_variance_dist_arima_rolling(y_train, window=100, horizon=1, order=None, auto_select=True):
    """
    Forecast using rolling window ARIMA model, providing both mean and variance.
    
    Parameters:
        y_train: Training data
        window: Rolling window size
        horizon: Forecast horizon
        order: ARIMA order tuple (p, d, q). If None and auto_select=True, will be auto-selected.
        auto_select: Whether to auto-select order if not provided
    
    Returns:
        mean, variance forecasts
    """
    y_win = y_train[-window:] if window < len(y_train) else y_train
    
    if order is None and auto_select:
        order = _auto_select_arima_order(y_win)
    elif order is None:
        order = (1, 0, 0)
    
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


def forecast_variance_arima_post_break(y_train, horizon=1, order=None, auto_select=True):
    """
    Detect variance break point and forecast from post-break regime only.
    
    Parameters:
        y_train: Training data
        horizon: Forecast horizon
        order: ARIMA order tuple (p, d, q). If None and auto_select=True, will be auto-selected.
        auto_select: Whether to auto-select order if not provided
    
    Returns:
        mean, variance forecasts
    """
    from dgps.variance import estimate_variance_break_point
    
    y = np.asarray(y_train, dtype=float)
    
    # Estimate break point
    v_Tb_hat = estimate_variance_break_point(y, trim=0.15)
    
    # Fit ARIMA to post-break data only
    y_post = y[v_Tb_hat+1:]
    if len(y_post) < 10:
        # Fall back to global if not enough post-break data
        return forecast_variance_dist_arima_global(y_train, horizon=horizon, order=order, auto_select=auto_select)
    
    if order is None and auto_select:
        order = _auto_select_arima_order(y_post)
    elif order is None:
        order = (1, 0, 0)
    
    res = ARIMA(y_post, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_lstm(y_train, horizon=1, lookback=20, epochs=30):
    """Deprecated: LSTM support removed.

    The project removed LSTM components per project policy. This
    placeholder raises an informative error so callers are aware.
    """
    raise ImportError(
        "LSTM support removed from this repository. Use classical estimators in `estimators.ols_like`"
    )


def forecast_variance_averaged_window(y_train, window_sizes=[20, 50, 100], horizon=1, order=None, auto_select=True):
    """
    Forecast by averaging forecasts across multiple window sizes.
    
    Parameters:
        y_train: Training data
        window_sizes: List of window sizes to average over
        horizon: Forecast horizon
        order: ARIMA order tuple (p, d, q). If None and auto_select=True, will be auto-selected.
        auto_select: Whether to auto-select order if not provided
    
    Returns:
        Averaged mean and variance forecasts
    """
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]

    variance_means = []
    variance_vars = []

    for ws in window_sizes:
        try:
            mean, var = forecast_variance_dist_arima_rolling(
                y_train, window=ws, horizon=horizon, order=order, auto_select=auto_select
            )
            variance_means.append(mean)
            variance_vars.append(var)
        except Exception:
            continue

    if not variance_means:
        return forecast_variance_dist_arima_global(y_train, horizon=horizon, order=order, auto_select=auto_select)

    mean_avg = np.mean(np.array(variance_means), axis=0)
    var_avg = np.mean(np.array(variance_vars), axis=0)
    return mean_avg, var_avg


def forecast_markov_switching(y_train, horizon=1, k_regimes=2, switching_variance=False):
    """
    Forecasting using Markov Switching regression.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        return np.full(horizon, np.nan), np.full(horizon, np.nan)

    y = np.asarray(y_train, dtype=float)
    try:
        model = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=switching_variance).fit(disp=False)
        # Predict next steps
        # MarkovRegression in statsmodels doesn't always support direct h-step var_pred_mean easily
        # but we can get the predicted means.
        pred_mean = model.predict(start=len(y), end=len(y) + horizon - 1)
        # For variance, we use the estimated sigma2 of the current regime or average
        sigma2 = model.sigma2
        return np.asarray(pred_mean), np.full(horizon, sigma2)
    except Exception:
        return np.full(horizon, np.nan), np.full(horizon, np.nan)


def variance_rmse_mae_bias(y_true, y_pred):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias


def variance_interval_coverage(y_true, mean, var, level=0.95):
    z = norm.ppf(0.5 + level / 2.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def variance_log_score_normal(y_true, mean, var):
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    var = np.asarray(var)
    mask = np.isfinite(mean) & np.isfinite(var) & np.isfinite(y_true)
    if not np.any(mask):
        return np.nan
    var = np.maximum(var[mask], 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (y_true[mask] - mean[mask]) ** 2 / var)))
