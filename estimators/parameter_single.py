import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def _fit_arima_safely(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12), trend="n"):
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

# =====================================================
# 2) Forecasting models (1-step ahead)
# =====================================================

# ---- Global SARIMA ----
def forecast_global_sarima(y):
    try:
        res = _fit_arima_safely(y)
        return float(res.forecast(1)[0])
    except Exception:
        return np.nan

# ---- Rolling SARIMA ----
def forecast_rolling_sarima(y, window=80):
    try:
        res = _fit_arima_safely(y[-window:])
        return float(res.forecast(1)[0])
    except Exception:
        return np.nan

# ---- MS-AR (NaN-safe) ----
def forecast_markov_switching_ar(y):
    try:
        y_lag = y[:-1]
        y_curr = y[1:]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = MarkovRegression(
                endog=y_curr,
                k_regimes=2,
                trend="n",
                exog=y_lag.reshape(-1, 1),
                switching_exog=True,
                switching_variance=False
            ).fit(disp=False, maxiter=250)

        params = dict(zip(model.model.param_names, model.params))
        probs = model.filtered_marginal_probabilities[-1]

        phi0 = params["x1[0]"]
        phi1 = params["x1[1]"]

        return float((probs[0] * phi0 + probs[1] * phi1) * y[-1])

    except Exception:
        return np.nan


# =====================================================
# 3) Metrics (NaN-filtered)
# =====================================================
def metrics(e):
    e = np.asarray(e)
    e = e[~np.isnan(e)]
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }
