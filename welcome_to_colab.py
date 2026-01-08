import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

np.random.seed(42)

# =================================================
# 1) SINGLE STRUCTURAL BREAK DGP
# =================================================
def simulate_single_break_ar1(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0
):
    y = np.zeros(T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2
        eps = np.random.normal(0, sigma)
        y[t] = phi * y[t-1] + eps

    return y


# =================================================
# 2) FORECASTING MODELS
# =================================================
def forecast_global_ar(y_train):
    res = ARIMA(y_train, order=(1, 0, 0)).fit()
    phi = res.params[1]
    return phi * y_train[-1]


def forecast_rolling_ar(y_train, window=120):
    y_win = y_train[-window:]
    res = ARIMA(y_win, order=(1, 0, 0)).fit()
    phi = res.params[1]
    return phi * y_train[-1]


def forecast_markov_switching_ar(y_train):
    model = MarkovRegression(
        y_train,
        k_regimes=2,
        trend="n",
        switching_variance=True
    ).fit(disp=False)

    params = model.params
    phi0, phi1 = params[0], params[1]

    probs = model.filtered_marginal_probabilities[-1]
    phi_hat = probs[0] * phi0 + probs[1] * phi1

    return phi_hat * y_train[-1]


# =================================================
# 3) METRICS
# =================================================
def compute_metrics(errors):
    errors = np.asarray(errors)
    return {
        "RMSE": np.sqrt(np.mean(errors**2)),
        "MAE": np.mean(np.abs(errors)),
        "Bias": np.mean(errors)
    }


# =================================================
# 4) MONTE CARLO EXPERIMENT
# =================================================
def monte_carlo_single_break(
    n_sim=300,
    T=400,
    Tb=200,
    window=120
):
    err_g, err_r, err_m = [], [], []

    for i in range(n_sim):
        y = simulate_single_break_ar1(T=T, Tb=Tb)
        y_train = y[:-1]
        y_true = y[-1]

        fg = forecast_global_ar(y_train)
        fr = forecast_rolling_ar(y_train, window)
        fm = forecast_markov_switching_ar(y_train)

        err_g.append(y_true - fg)
        err_r.append(y_true - fr)
        err_m.append(y_true - fm)

    return (
        compute_metrics(err_g),
        compute_metrics(err_r),
        compute_metrics(err_m)
    )


# =================================================
# 5) RUN
# =================================================
global_ar, rolling_ar, markov_ar = monte_carlo_single_break()

print("\nGLOBAL ARIMA")
print(global_ar)

print("\nROLLING ARIMA")
print(rolling_ar)

print("\nMARKOV-SWITCHING ARIMA")
print(markov_ar)