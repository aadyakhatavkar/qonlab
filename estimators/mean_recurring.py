"""
Recurring (Markov-Switching) Mean Estimators
==============================================
Forecasting methods for AR(1) with Markov-switching mean.
"""
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA


def forecast_ms_ar1_mean(y_train, horizon=1, p00_init=0.9, p11_init=0.9, rng=None):
    """
    Forecast AR(1) mean using Markov-switching model.
    
    Estimates a 2-state Markov-switching AR(1) with different means in each state.
    
    Parameters
    ----------
    y_train : array-like
        Training data
    horizon : int
        Forecast horizon
    p00_init, p11_init : float
        Initial transition probabilities (persistence parameters)
        
    Returns
    -------
    mean : ndarray
        Point forecasts
    var : ndarray
        Variance forecasts
    """
    y = np.asarray(y_train, dtype=float)
    T = len(y)
    rng = rng if rng is not None else np.random.default_rng()
    
    # Detect regime shifts using moving average
    window = max(10, T // 10)
    rolling_mean = np.array([np.mean(y[max(0, t-window):t+1]) for t in range(T)])
    
    # Identify high/low regimes based on quantiles
    q25 = np.percentile(rolling_mean, 25)
    q75 = np.percentile(rolling_mean, 75)
    
    mu0 = np.mean(y[rolling_mean <= q25])
    mu1 = np.mean(y[rolling_mean >= q75])
    
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
    
    # Estimate AR(1) coefficient using full sample
    y_lag = y[:-1]
    y_current = y[1:]
    ar1_coef = float(np.corrcoef(y_current, y_lag)[0, 1]) if len(y) > 2 else 0.5
    ar1_coef = np.clip(ar1_coef, -0.95, 0.95)
    
    # Estimate state-dependent residual variance
    regimes = (rolling_mean > (q25 + q75) / 2).astype(int)
    resid_0 = y[regimes == 0] - mu0
    resid_1 = y[regimes == 1] - mu1
    
    sigma_0 = float(np.std(resid_0)) if len(resid_0) > 0 else 1.0
    sigma_1 = float(np.std(resid_1)) if len(resid_1) > 0 else 1.0
    
    # Estimate transition probabilities
    state_changes = np.sum(np.diff(regimes) != 0)
    p_change = max(0.01, (state_changes + 1.0) / (T - 1))
    p00 = max(1 - p_change, 0.7)
    p11 = max(1 - p_change, 0.7)
    
    # Determine current state
    current_state = 1 if rolling_mean[-1] > np.median(rolling_mean) else 0
    current_val = y[-1]
    
    means = []
    vols = []
    
    for h in range(horizon):
        # Predict next value based on current state
        if current_state == 0:
            next_mean = mu0 + ar1_coef * (current_val - mu0)
            next_vol = sigma_0
            # Transition to next state
            current_state = 1 if rng.random() > p00 else 0
        else:
            next_mean = mu1 + ar1_coef * (current_val - mu1)
            next_vol = sigma_1
            # Transition to next state
            current_state = 0 if rng.random() > p11 else 1
        
        means.append(next_mean)
        vols.append(next_vol ** 2)
        current_val = next_mean
    
    return np.array(means), np.array(vols)


def forecast_mean_arima_global(y_train, horizon=1, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12)):
    """
    Forecast using global SARIMA model on full sample.
    """
    try:
        res = ARIMA(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n"
        ).fit()
        return np.asarray(res.forecast(steps=horizon))
    except Exception:
        return np.full(horizon, np.nan)


def mean_rmse_mae_bias(y_true, y_pred):
    """Calculate RMSE, MAE, and Bias."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias


def mean_interval_coverage(y_true, mean, var, level=0.95):
    """Calculate prediction interval coverage probability."""
    z = norm.ppf(0.5 + level / 2.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def mean_log_score_normal(y_true, mean, var):
    """Calculate log-predictive score under normal distribution."""
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    var = np.asarray(var)
    mask = np.isfinite(mean) & np.isfinite(var) & np.isfinite(y_true)
    if not np.any(mask):
        return np.nan
    var = np.maximum(var[mask], 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (y_true[mask] - mean[mask]) ** 2 / var)))
