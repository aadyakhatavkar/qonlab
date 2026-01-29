"""
Multiple mean break estimators: forecasting models for AR(1) with multiple mean shifts.
"""
import numpy as np
from estimators.mean import forecast_global_ar1, forecast_rolling_ar1


def forecast_ar1_with_multiple_break_dummies_oracle(y_train, breaks):
    """
    Forecast AR(1) with oracle knowledge of multiple break points.
    Uses dummy variables for each break.
    
    Parameters:
        y_train: Training data
        breaks: List of break points [b1, b2, ...]
    
    Returns:
        Forecast value (float)
    """
    y = np.asarray(y_train, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    t_idx = np.arange(1, len(y))
    
    # Create dummy variables for each break
    dummies = [(t_idx > b).astype(float) for b in breaks]
    
    # Build design matrix: [const, lag, d1, d2, ...]
    X = np.column_stack([np.ones_like(y_lag), y_lag] + dummies)
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    
    c = beta[0]
    phi = beta[1]
    alphas = beta[2:]
    
    # Make forecast for next period
    t_next = len(y)
    d_next = np.array([1.0 if t_next > b else 0.0 for b in breaks])
    
    return float(c + phi * y[-1] + np.sum(alphas * d_next))


def forecast_ar1_single_break_dummy_oracle(y_train, Tb):
    """
    Forecast AR(1) with oracle knowledge of single break point.
    Uses dummy variable for post-break regime.
    """
    y = np.asarray(y_train, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    d = (np.arange(1, len(y)) > Tb).astype(float)
    X = np.column_stack([np.ones_like(y_lag), y_lag, d])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    c, phi, delta = beta
    t_next = len(y)
    d_next = 1.0 if t_next > Tb else 0.0
    return float(c + phi * y[-1] + delta * d_next)
