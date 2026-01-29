"""
Parameter break estimators: forecasting models for AR(1) with parameter shifts.
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def forecast_global_ar(y):
    """Forecast using global AR(1) model without trend."""
    return float(
        ARIMA(y, order=(1, 0, 0), trend="n")
        .fit()
        .forecast(1)[0]
    )


def forecast_rolling_ar(y, window=80):
    """Forecast using rolling window AR(1) model without trend."""
    return float(
        ARIMA(y[-window:], order=(1, 0, 0), trend="n")
        .fit()
        .forecast(1)[0]
    )


def forecast_markov_switching_ar(y):
    """
    Forecast using Markov Switching AR model.
    Estimates two regimes with switching AR coefficients.
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
