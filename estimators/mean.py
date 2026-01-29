"""
Mean break estimators: forecasting models for AR(1) with mean shifts.
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def forecast_global_ar1(y_train):
    """Forecast using global AR(1) model."""
    m = ARIMA(y_train, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def forecast_rolling_ar1(y_train, window=60):
    """Forecast using rolling window AR(1) model."""
    y_sub = y_train[-window:] if len(y_train) > window else y_train
    m = ARIMA(y_sub, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def forecast_ar1_with_break_dummy_oracle(y_train, Tb):
    """
    Forecast AR(1) with oracle knowledge of break point.
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


def estimate_break_point_grid_search(y_train, trim=0.15):
    """Estimate mean break point via grid search over SSE."""
    y = np.asarray(y_train, dtype=float)
    T = len(y)
    lo = max(int(np.floor(trim * T)), 10)
    hi = min(int(np.ceil((1 - trim) * T)) - 1, T - 11)
    best_Tb, best_sse = None, np.inf
    
    for Tb in range(lo, hi):
        c1, p1, sse1 = _fit_ar1_ols(y[:Tb+1])
        c2, p2, sse2 = _fit_ar1_ols(y[Tb+1:])
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    return int(best_Tb if best_Tb is not None else T // 2)


def _fit_ar1_ols(y_segment):
    """Fit AR(1) via OLS on a segment."""
    y = np.asarray(y_segment, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    resid = y_dep - X @ beta
    sse = float(np.sum(resid**2))
    return float(beta[0]), float(beta[1]), sse


def forecast_ar1_with_estimated_break(y_train, trim=0.15):
    """Forecast from post-break regime using estimated break point."""
    y = np.asarray(y_train, dtype=float)
    Tb_hat = estimate_break_point_grid_search(y, trim=trim)
    regime = y[Tb_hat+1:] if Tb_hat + 1 < len(y) else y
    if len(regime) < 20:
        return forecast_global_ar1(y_train)
    m = ARIMA(regime, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def forecast_markov_switching(y_train, k_regimes=2):
    """Forecast using Markov Switching regression."""
    y = np.asarray(y_train, dtype=float)
    m = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=False).fit(disp=False)
    pred = m.predict(start=len(y), end=len(y))
    return float(np.asarray(pred)[0])
