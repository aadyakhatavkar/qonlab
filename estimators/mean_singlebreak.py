import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def _fit_arima_safely(y, order, seasonal_order, trend="n"):
    """Fit ARIMA while suppressing expected optimizer/start-value warnings."""
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*Non-stationary starting.*")
        warnings.filterwarnings("ignore", message=".*Non-invertible starting.*")
        return ARIMA(
            y,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend
        ).fit(method_kwargs={"maxiter": 200})

# =========================================================
# 2) SARIMA forecasting methods (1-step ahead)
# =========================================================
def forecast_sarima_global(y_train, order=(1,0,1), seasonal_order=(1,0,0,12)):
    """
    SARIMA(p,d,q)(P,D,Q)_s global model on full sample.
    """
    try:
        res = _fit_arima_safely(y_train, order=order, seasonal_order=seasonal_order, trend="n")
        return float(res.forecast(1)[0])
    except Exception:
        return np.nan

def forecast_sarima_rolling(y_train, window=60, order=(1,0,1), seasonal_order=(1,0,0,12)):
    """
    SARIMA(p,d,q)(P,D,Q)_s rolling window model.
    """
    try:
        sub = y_train[-window:] if len(y_train) > window else y_train
        res = _fit_arima_safely(sub, order=order, seasonal_order=seasonal_order, trend="n")
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

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message=".*Non-stationary starting.*")
            warnings.filterwarnings("ignore", message=".*Non-invertible starting.*")
            m = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                exog=d,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False, maxiter=250)
    except Exception:
        return np.nan

    # next-step dummy value
    d_next = np.array([[1.0 if len(y) > Tb else 0.0]])
    return float(m.forecast(1, exog=d_next)[0])

def forecast_ses(y_train):
    """Simple Exponential Smoothing (SES)."""
    try:
        m = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(optimized=True)
        return float(m.forecast(1)[0])
    except Exception:
        return np.nan

def forecast_holt_winters(y_train):
    """
    Holt-Winters Exponential Smoothing (additive).
    Useful for data with trend and seasonality.
    """
    try:
        y = np.asarray(y_train, dtype=float)
        if len(y) < 13:
            # If too short, use SES instead
            return forecast_ses(y_train)
        
        # Try additive Holt-Winters with seasonal_periods=12
        try:
            m = ExponentialSmoothing(
                y,
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                initialization_method="estimated"
            ).fit(optimized=True)
            return float(m.forecast(1)[0])
        except:
            # Fall back to additive without seasonal component
            m = ExponentialSmoothing(
                y,
                trend='add',
                seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            return float(m.forecast(1)[0])
    except Exception:
        return np.nan
