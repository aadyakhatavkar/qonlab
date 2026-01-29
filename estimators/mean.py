"""
Mean Break Forecasters
======================
Forecasting methods for ARMA models with mean (intercept) breaks.
Uses auto-selected ARMA orders via AIC/BIC (Box-Jenkins methodology).
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _auto_select_arma_order(y, max_p=3, max_q=3, criterion='aic'):
    """
    Auto-select ARMA(p, q) order using information criteria.
    
    Uses Box-Jenkins methodology with AIC or BIC to select optimal (p, q).
    This is the "golden standard" for ARMA model selection.
    
    Parameters:
        y: Time series data
        max_p: Maximum AR order to consider
        max_q: Maximum MA order to consider
        criterion: 'aic' or 'bic' for model selection
    
    Returns:
        Optimal (p, q) tuple
    """
    y = np.asarray(y, dtype=float)
    
    # If series too short, use conservative order
    if len(y) < 30:
        return (1, 1)
    
    best_order = (1, 1)
    best_ic = np.inf
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            # Skip (0, 0) - need at least one component
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y, order=(p, 0, q))
                res = model.fit()
                
                ic = res.bic if criterion.lower() == 'bic' else res.aic
                
                if ic < best_ic:
                    best_ic = ic
                    best_order = (p, q)
            except Exception:
                continue
    
    return best_order


def mean_forecast_global_arma(y_train, auto_select=True, order=None):
    """
    Forecast using global auto-selected ARMA model.
    
    Parameters:
        y_train: Training data
        auto_select: Whether to auto-select order via AIC
        order: Manual (p, q) if auto_select=False
    
    Returns:
        One-step ahead forecast
    """
    if auto_select:
        p, q = _auto_select_arma_order(y_train)
    else:
        p, q = order if order else (1, 1)
    
    m = ARIMA(y_train, order=(p, 0, q)).fit()
    return float(m.forecast(1)[0])


def mean_forecast_rolling_arma(y_train, window=60, auto_select=True, order=None):
    """
    Forecast using rolling window auto-selected ARMA model.
    
    Parameters:
        y_train: Training data
        window: Rolling window size
        auto_select: Whether to auto-select order via AIC
        order: Manual (p, q) if auto_select=False
    
    Returns:
        One-step ahead forecast
    """
    y_sub = y_train[-window:] if len(y_train) > window else y_train
    
    if auto_select:
        p, q = _auto_select_arma_order(y_sub)
    else:
        p, q = order if order else (1, 1)
    
    m = ARIMA(y_sub, order=(p, 0, q)).fit()
    return float(m.forecast(1)[0])


# Backward compatibility aliases
def mean_forecast_global_ar1(y_train):
    """Deprecated: Use mean_forecast_global_arma instead."""
    return mean_forecast_global_arma(y_train, auto_select=True)


def mean_forecast_rolling_ar1(y_train, window=60):
    """Deprecated: Use mean_forecast_rolling_arma instead."""
    return mean_forecast_rolling_arma(y_train, window=window, auto_select=True)


def mean_forecast_ar1_with_break_dummy_oracle(y_train, Tb):
    """
    AR(1) with break dummy (ORACLE - knows true break date).
    Model: y_t = c + φ*y_{t-1} + δ*d_t + u_t
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


def _mean_fit_ar1_ols(y_segment):
    """Helper: Fit AR(1) via OLS and return coefficients + SSE."""
    y = np.asarray(y_segment, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    resid = y_dep - X @ beta
    sse = float(np.sum(resid**2))
    return float(beta[0]), float(beta[1]), sse


def mean_estimate_break_point_gridsearch(y_train, trim=0.15):
    """Estimate mean break point via SSE minimization grid search."""
    y = np.asarray(y_train, dtype=float)
    T = len(y)
    lo = max(int(np.floor(trim * T)), 10)
    hi = min(int(np.ceil((1 - trim) * T)) - 1, T - 11)
    best_Tb, best_sse = None, np.inf
    
    for Tb in range(lo, hi):
        _, _, sse1 = _mean_fit_ar1_ols(y[:Tb+1])
        _, _, sse2 = _mean_fit_ar1_ols(y[Tb+1:])
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    
    return int(best_Tb if best_Tb is not None else T // 2)


def mean_forecast_ar1_with_estimated_break(y_train, trim=0.15):
    """Forecast from post-break regime using estimated break point."""
    y = np.asarray(y_train, dtype=float)
    Tb_hat = mean_estimate_break_point_gridsearch(y, trim=trim)
    regime = y[Tb_hat+1:] if Tb_hat + 1 < len(y) else y
    
    if len(regime) < 20:
        return mean_forecast_global_ar1(y_train)
    
    m = ARIMA(regime, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])


def mean_forecast_markov_switching(y_train, k_regimes=2):
    """Markov switching model for mean breaks."""
    y = np.asarray(y_train, dtype=float)
    m = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=False).fit(disp=False)
    pred = m.predict(start=len(y), end=len(y))
    return float(np.asarray(pred)[0])


def mean_metrics(errors):
    """Compute RMSE, MAE, Bias from forecast errors."""
    e = np.asarray(errors, dtype=float)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }

# =========================================================
# SARIMA Methods (Seasonal ARIMA for mean breaks)
# =========================================================

def forecast_sarima_global(y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)):
    """
    SARIMA(p,d,q)(P,D,Q)_s forecasting (1-step ahead).
    
    Parameters:
        y_train: Training data
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal order with period s
    
    Returns:
        1-step ahead forecast
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        m = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        return float(m.forecast(1)[0])
    except Exception:
        # Fallback: simple mean
        return float(np.mean(y_train[-min(10, len(y_train)):]))


def forecast_sarima_rolling(y_train, window=60, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)):
    """
    SARIMA forecasting with rolling window.
    
    Parameters:
        y_train: Training data
        window: Rolling window size
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal order
    
    Returns:
        1-step ahead forecast
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        sub = y_train[-window:] if len(y_train) > window else y_train
        m = SARIMAX(
            sub,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        return float(m.forecast(1)[0])
    except Exception:
        return float(np.mean(sub[-10:]))


def forecast_sarima_with_break_dummy(y_train, Tb, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)):
    """
    SARIMA with exogenous break dummy (oracle Tb).
    
    Parameters:
        y_train: Training data
        Tb: Known break point
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal order
    
    Returns:
        1-step ahead forecast
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        y = np.asarray(y_train, dtype=float)
        t_idx = np.arange(len(y))
        d = (t_idx > Tb).astype(float).reshape(-1, 1)
        
        m = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            exog=d,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        d_next = np.array([[1.0 if len(y) > Tb else 0.0]])
        return float(m.forecast(1, exog=d_next)[0])
    except Exception:
        return float(np.mean(y_train[-10:]))


def forecast_sarima_with_estimated_break(y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), trim=0.15):
    """
    SARIMA with estimated break point.
    Estimates Tb via grid search on mean-only model, then uses dummy.
    
    Parameters:
        y_train: Training data
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal order
        trim: Trimming fraction for break detection
    
    Returns:
        1-step ahead forecast
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        y = np.asarray(y_train, dtype=float)
        Tn = len(y)
        lo = max(int(trim * Tn), 10)
        hi = min(int((1 - trim) * Tn) - 1, Tn - 10)
        
        # Estimate break point via SSE minimization
        best_Tb, best_sse = None, np.inf
        for Tb in range(lo, hi):
            m1 = np.mean(y[:Tb+1])
            m2 = np.mean(y[Tb+1:])
            sse = np.sum((y[:Tb+1] - m1)**2) + np.sum((y[Tb+1:] - m2)**2)
            if sse < best_sse:
                best_sse, best_Tb = sse, Tb
        
        Tb_hat = best_Tb if best_Tb is not None else Tn // 2
        
        # Fit SARIMA with break dummy
        t_idx = np.arange(len(y))
        d = (t_idx > Tb_hat).astype(float).reshape(-1, 1)
        
        m = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            exog=d,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        d_next = np.array([[1.0 if len(y) > Tb_hat else 0.0]])
        return float(m.forecast(1, exog=d_next)[0])
    except Exception:
        return float(np.mean(y_train[-10:]))