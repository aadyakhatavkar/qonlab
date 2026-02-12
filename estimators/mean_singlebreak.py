import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# =========================================================
# 2) SARIMA forecasting methods (1-step ahead)
# =========================================================
def forecast_sarima_global(y_train, order=(1,0,1), seasonal_order=(1,0,0,12)):
    """
    SARIMA(p,d,q)(P,D,Q)_s global model on full sample.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        res = ARIMA(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n"
        ).fit()
        return float(res.forecast(1)[0])
    except Exception:
        return np.nan

def forecast_sarima_rolling(y_train, window=60, order=(1,0,1), seasonal_order=(1,0,0,12)):
    """
    SARIMA(p,d,q)(P,D,Q)_s rolling window model.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        sub = y_train[-window:] if len(y_train) > window else y_train
        res = ARIMA(
            sub,
            order=order,
            seasonal_order=seasonal_order,
            trend="n"
        ).fit()
        return float(res.forecast(1)[0])
    except Exception:
        return np.nan

def forecast_sarima_break_dummy_oracle(y_train, Tb, order=(1,0,1), seasonal_order=(1,0,0,12)):
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
    ).fit()

    # next-step dummy value
    d_next = np.array([[1.0 if len(y) > Tb else 0.0]])
    return float(m.forecast(1, exog=d_next)[0])

def forecast_ses(y_train):
    m = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(optimized=True)
    return float(m.forecast(1)[0])

