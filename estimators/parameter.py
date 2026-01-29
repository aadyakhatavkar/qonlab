"""
Parameter Break Forecasters
===========================
Forecasting methods for ARMA models with parameter (AR coefficient) breaks.
Uses auto-selected ARMA orders via AIC/BIC (Box-Jenkins methodology).
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _auto_select_arma_order(y, max_p=3, max_q=3, criterion='aic'):
    """
    Auto-select ARMA(p, q) order using information criteria.
    
    Uses Box-Jenkins methodology with AIC or BIC to select optimal (p, q).
    
    Parameters:
        y: Time series data
        max_p: Maximum AR order to consider
        max_q: Maximum MA order to consider
        criterion: 'aic' or 'bic' for model selection
    
    Returns:
        Optimal (p, q) tuple
    """
    y = np.asarray(y, dtype=float)
    
    if len(y) < 30:
        return (1, 1)
    
    best_order = (1, 1)
    best_ic = np.inf
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y, order=(p, 0, q), trend="n")
                res = model.fit()
                
                ic = res.bic if criterion.lower() == 'bic' else res.aic
                
                if ic < best_ic:
                    best_ic = ic
                    best_order = (p, q)
            except Exception:
                continue
    
    return best_order


def param_forecast_global_arma(y, auto_select=True, order=None):
    """
    Forecast using global auto-selected ARMA model without trend.
    
    Parameters:
        y: Training data
        auto_select: Whether to auto-select order via AIC
        order: Manual (p, q) if auto_select=False
    
    Returns:
        One-step ahead forecast
    """
    if auto_select:
        p, q = _auto_select_arma_order(y)
    else:
        p, q = order if order else (1, 1)
    
    return float(
        ARIMA(y, order=(p, 0, q), trend="n")
        .fit()
        .forecast(1)[0]
    )


def param_forecast_rolling_arma(y, window=80, auto_select=True, order=None):
    """
    Forecast using rolling window auto-selected ARMA model without trend.
    
    Parameters:
        y: Training data
        window: Rolling window size
        auto_select: Whether to auto-select order via AIC
        order: Manual (p, q) if auto_select=False
    
    Returns:
        One-step ahead forecast
    """
    y_sub = y[-window:] if len(y) > window else y
    
    if auto_select:
        p, q = _auto_select_arma_order(y_sub)
    else:
        p, q = order if order else (1, 1)
    
    return float(
        ARIMA(y_sub, order=(p, 0, q), trend="n")
        .fit()
        .forecast(1)[0]
    )


# Backward compatibility aliases
def param_forecast_global_ar(y):
    """Deprecated: Use param_forecast_global_arma instead."""
    return param_forecast_global_arma(y, auto_select=True)


def param_forecast_rolling_ar(y, window=80):
    """Deprecated: Use param_forecast_rolling_arma instead."""
    return param_forecast_rolling_arma(y, window=window, auto_select=True)


def param_forecast_markov_switching_ar(y):
    """
    Forecast using Markov Switching AR model.
    Uses switching_exog to allow AR coefficient to vary by regime.
    """
    y_lag = y[:-1]
    y_curr = y[1:]

    model = MarkovRegression(
        endog=y_curr,
        k_regimes=2,
        trend="n",
        exog=y_lag.reshape(-1, 1),
        switching_exog=True,
        switching_variance=False
    ).fit(disp=False)

    params = dict(zip(model.model.param_names, model.params))
    probs = model.filtered_marginal_probabilities[-1]

    phi0 = params["x1[0]"]
    phi1 = params["x1[1]"]

    return float((probs[0] * phi0 + probs[1] * phi1) * y[-1])


def param_metrics(errors):
    """Compute RMSE, MAE, Bias from forecast errors."""
    e = np.asarray(errors)
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }
