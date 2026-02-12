import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# =====================================================
# 2) Forecasting models (Recurring) (1-step ahead)
# =====================================================

# ---- Global SARIMA (replaces Global AR) ----
def forecast_global_sarima(y):
    try:
        return float(
            ARIMA(
                y,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 12),
                trend="n"
            ).fit().forecast(1)[0]
        )
    except Exception:
        return np.nan

# ---- Rolling SARIMA (replaces Rolling AR) ----
def forecast_rolling_sarima(y, window=60):
    try:
        return float(
            ARIMA(
                y[-window:],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 12),
                trend="n"
            ).fit().forecast(1)[0]
        )
    except Exception:
        return np.nan

# ---- MS-AR (unchanged) ----
def forecast_markov_switching_ar(y):
    try:
        y_dep = y[1:]
        x = y[:-1].reshape(-1, 1)

        model = MarkovRegression(
            y_dep,
            k_regimes=2,
            trend="n",
            exog=x,
            switching_exog=True,
            switching_variance=False
        ).fit(disp=False)

        probs = model.filtered_marginal_probabilities[-1]
        params = dict(zip(model.model.param_names, model.params))

        phi0 = params["x1[0]"]
        phi1 = params["x1[1]"]

        return float((probs[0] * phi0 + probs[1] * phi1) * y[-1])

    except Exception:
        return np.nan

# =====================================================
# 3) Metrics (NaN-safe)
# =====================================================
def metrics(e):
    e = np.asarray(e)
    e = e[~np.isnan(e)]
    return {
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "MAE": np.mean(np.abs(e)),
        "Bias": np.mean(e)
    }
