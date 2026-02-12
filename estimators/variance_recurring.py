"""
Recurring (Markov-Switching) Variance Estimators
================================================
Forecasting methods for AR(1) with Markov-switching variance.
"""
import numpy as np
from scipy.stats import norm


def forecast_markov_switching(y_train, horizon=1, p00_init=0.9, p11_init=0.9, 
                              ar1_init=0.5, sigma0_init=1.0, sigma1_init=2.0):
    """
    Forecast AR(1) variance using Markov-switching model.
    
    Estimates a 2-state Markov-switching AR(1) with different variance in each state.
    Uses EM-like procedure for parameter estimation.
    
    Parameters
    ----------
    y_train : array-like
        Training data
    horizon : int
        Forecast horizon
    p00_init, p11_init : float
        Initial persistence parameters (transition probabilities)
    ar1_init : float
        Initial AR(1) coefficient
    sigma0_init, sigma1_init : float
        Initial standard deviations in states 0 and 1
        
    Returns
    -------
    mean : ndarray
        Point forecasts
    var : ndarray
        Variance forecasts
    """
    y = np.asarray(y_train, dtype=float)
    T = len(y)
    
    # Very simple estimation: use rolling volatility to detect states
    window = max(10, T // 10)
    rolling_vol = np.array([np.std(y[max(0, t-window):t+1]) for t in range(T)])
    
    # Estimate as mixture of two normals
    vol_mean = np.mean(rolling_vol)
    vol_std = np.std(rolling_vol)
    
    # Use quantiles to define states
    q25 = np.percentile(rolling_vol, 25)
    q75 = np.percentile(rolling_vol, 75)
    
    # Estimate parameters
    sigma0 = max(np.mean(rolling_vol[rolling_vol <= q25]), 0.1)
    sigma1 = max(np.mean(rolling_vol[rolling_vol >= q75]), 0.1)
    
    if sigma0 > sigma1:
        sigma0, sigma1 = sigma1, sigma0
    
    # Estimate AR(1) coefficient
    y_lag = y[:-1]
    y_current = y[1:]
    ar1_coef = float(np.corrcoef(y_current, y_lag)[0, 1]) if len(y) > 2 else ar1_init
    ar1_coef = np.clip(ar1_coef, -0.95, 0.95)
    
    # Estimate transition probabilities from state persistence
    states = (rolling_vol > (q25 + q75) / 2).astype(int)
    state_changes = np.sum(np.diff(states) != 0)
    p_change = (state_changes + 1.0) / (T - 1)
    p00 = max(1 - p_change, 0.7)
    p11 = max(1 - p_change, 0.7)
    
    # Generate forecasts: assume we end in high volatility state
    current_state = 1 if rolling_vol[-1] > np.median(rolling_vol) else 0
    current_val = y[-1]
    
    means = []
    vols = []
    
    for h in range(horizon):
        # Predict next value
        next_mean = ar1_coef * current_val
        
        # State transition
        if current_state == 0:
            next_state = 1 if np.random.rand() > p00 else 0
            current_vol = sigma0
        else:
            next_state = 0 if np.random.rand() > p11 else 1
            current_vol = sigma1
        
        means.append(next_mean)
        vols.append(current_vol ** 2)
        
        current_val = next_mean
        current_state = next_state
    
    return np.array(means), np.array(vols)


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
