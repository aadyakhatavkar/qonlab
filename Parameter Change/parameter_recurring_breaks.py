import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings

# DGP: Markov-switching AR(1), ONLY phi changes

def simulate_ms_ar1_phi_only(
    T=300,
    p00=0.97, p11=0.97,
    phi0=0.2, phi1=0.9,
    sigma=1.0,
    y0=0.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)
    s = np.zeros(T, dtype=int)

    y[0] = y0
    s[0] = 0 if rng.random() < 0.5 else 1

    for t in range(1, T):
        # regime transition
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0

        phi = phi0 if s[t] == 0 else phi1
        y[t] = phi * y[t-1] + rng.normal(0.0, sigma)

    return y, s



# 2) Forecasting models (1-step ahead)

def forecast_global_ar1(y_train):
    res = ARIMA(y_train, order=(1, 0, 0), trend="n").fit()
    return float(res.forecast(1)[0])


def forecast_rolling_ar1(y_train, window=80):
    y_win = y_train[-window:]
    res = ARIMA(y_win, order=(1, 0, 0), trend="n").fit()
    return float(res.forecast(1)[0])


def forecast_markov_switching_ar1(y_train):
    y = np.asarray(y_train)

    y_dep = y[1:]
    x = y[:-1].reshape(-1, 1)

    mod = MarkovRegression(
        y_dep,
        k_regimes=2,
        trend="n",              # no mean
        exog=x,
        switching_exog=True,    # phi switches
        switching_variance=False
    )

    res = mod.fit(disp=False)

    probs = res.filtered_marginal_probabilities[-1]

    names = res.model.param_names
    vals = res.params
    par = {n: vals[i] for i, n in enumerate(names)}

    phi0 = par["x1[0]"]
    phi1 = par["x1[1]"]

    y_last = y[-1]
    f0 = phi0 * y_last
    f1 = phi1 * y_last

    return float(probs[0] * f0 + probs[1] * f1)



# 3) Monte Carlo experiment

def monte_carlo_phi_only(
    n_sim=200,
    T=300,
    t0=250,
    window=80,
    seed=123
):
    rng = np.random.default_rng(seed)
    warnings.filterwarnings("ignore")

    err_g, err_r, err_m = [], [], []

    for _ in range(n_sim):
        y, _ = simulate_ms_ar1_phi_only(T=T, rng=rng)

        y_train = y[:t0]
        y_true = y[t0]

        try:
            fg = forecast_global_ar1(y_train)
            fr = forecast_rolling_ar1(y_train, window)
            fm = forecast_markov_switching_ar1(y_train)
        except Exception:
            continue

        err_g.append(y_true - fg)
        err_r.append(y_true - fr)
        err_m.append(y_true - fm)

    def metrics(e):
        e = np.asarray(e)
        return {
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE":  float(np.mean(np.abs(e))),
            "Bias": float(np.mean(e))
        }

    return metrics(err_g), metrics(err_r), metrics(err_m)



# 4) RUN

if __name__ == "__main__":
    global_ar, rolling_ar, markov_ar = monte_carlo_phi_only(
        n_sim=200,
        T=300,
        t0=250,
        window=80,
        seed=123
    )

    print("\nGLOBAL AR(1)")
    print(global_ar)

    print("\nROLLING AR(1)")
    print(rolling_ar)

    print("\nMARKOV-SWITCHING AR(1)")
    print(markov_ar)
