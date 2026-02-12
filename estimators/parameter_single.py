import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# =====================================================
# 2) Forecasting models (1-step ahead)
# =====================================================

# ---- Global SARIMA ----
def forecast_global_sarima(y):
    try:
        return float(
            ARIMA(
                y,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 12),
                trend="n"
            )
            .fit()
            .forecast(1)[0]
        )
    except Exception:
        return np.nan

# ---- Rolling SARIMA ----
def forecast_rolling_sarima(y, window=80):
    try:
        return float(
            ARIMA(
                y[-window:],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 12),
                trend="n"
            )
            .fit()
            .forecast(1)[0]
        )
    except Exception:
        return np.nan

# ---- MS-AR (NaN-safe) ----
def forecast_markov_switching_ar(y):
    try:
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

