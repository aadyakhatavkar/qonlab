import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# =========================================================
# 2) SARIMA forecasting methods (1-step ahead)
# =========================================================
def forecast_sarima_global(y_train, order=(1,0,0), seasonal_order=(1,0,0,12)):
    """
    SARIMA(p,d,q)(P,D,Q)_s implemented as SARIMAX.
    """
    m = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    return float(m.forecast(1)[0])

def forecast_sarima_rolling(y_train, window=60, order=(1,0,0), seasonal_order=(1,0,0,12)):
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

def forecast_sarima_break_dummy_oracle(y_train, Tb, order=(1,0,0), seasonal_order=(1,0,0,12)):
    """
    SARIMA with exogenous break dummy (oracle Tb).
    Model includes dummy in exog to shift the mean after Tb.
    """
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

    # next-step dummy value
    d_next = np.array([[1.0 if len(y) > Tb else 0.0]])
    return float(m.forecast(1, exog=d_next)[0])

def estimate_break_grid_sse_mean_only(y_train, trim=0.15):
    """
    Simple break estimator:
    choose Tb_hat that minimizes SSE of two means (level shift only).
    (This is robust and fast for teaching purposes.)
    """
    y = np.asarray(y_train, dtype=float)
    Tn = len(y)
    lo = max(int(trim*Tn), 10)
    hi = min(int((1-trim)*Tn)-1, Tn-10)

    best_Tb, best_sse = None, np.inf
    for Tb in range(lo, hi):
        m1 = np.mean(y[:Tb+1])
        m2 = np.mean(y[Tb+1:])
        sse = np.sum((y[:Tb+1]-m1)**2) + np.sum((y[Tb+1:]-m2)**2)
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    return int(best_Tb if best_Tb is not None else Tn//2)

def forecast_sarima_estimated_break(y_train, order=(1,0,0), seasonal_order=(1,0,0,12), trim=0.15):
    """
    1) Estimate Tb_hat from y_train
    2) Fit SARIMA using break dummy based on Tb_hat
    3) Forecast 1-step ahead
    """
    y = np.asarray(y_train, dtype=float)
    Tb_hat = estimate_break_grid_sse_mean_only(y, trim=trim)

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

def forecast_ses(y_train):
    m = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(optimized=True)
    return float(m.forecast(1)[0])

