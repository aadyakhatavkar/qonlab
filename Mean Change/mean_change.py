import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1) DGP: AR(1) with ONE deterministic mean break
# =========================================================
def simulate_single_mean_break(
    T=300,
    Tb=150,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    y0=0.0,
    rng=None
):
    """
    y_t = mu_t + phi * y_{t-1} + eps_t
    mu_t = mu0 for t <= Tb, mu1 for t > Tb
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)
    y[0] = y0

    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi * y[t-1] + rng.normal(0.0, sigma)

    return y


# =========================================================
# 2) Forecasting models
# =========================================================
def forecast_global_ar1(y_train):
    model = ARIMA(y_train, order=(1, 0, 0)).fit()
    return float(model.forecast(1)[0])


def forecast_rolling_ar1(y_train, window=60):
    model = ARIMA(y_train[-window:], order=(1, 0, 0)).fit()
    return float(model.forecast(1)[0])


def forecast_ar1_with_break_dummy(y_train, Tb):
    """
    AR(1) with known break dummy:
        y_t = alpha + delta * 1(t > Tb) + phi y_{t-1} + eps_t
    """
    y = np.asarray(y_train)

    y_dep = y[1:]
    y_lag = y[:-1]

    # Break dummy aligned with y_dep
    d = (np.arange(1, len(y)) > Tb).astype(int)

    X = np.column_stack([y_lag, d])

    model = ARIMA(
        y_dep,
        order=(0, 0, 0),
        exog=X
    ).fit()

    alpha = model.params[0]
    phi = model.params[1]
    delta = model.params[2]

    d_next = 1 if len(y_train) > Tb else 0
    y_last = y[-1]

    return float(alpha + phi * y_last + delta * d_next)


# =========================================================
# 3) Monte Carlo evaluation (post-break forecasting)
# =========================================================
def monte_carlo_mean_break(
    n_sim=300,
    T=300,
    Tb=150,
    window=60,
    seed=123
):
    rng = np.random.default_rng(seed)

    eg, er, ed = [], [], []

    # Forecast strictly after break
    t0 = Tb + 20

    for _ in range(n_sim):
        y = simulate_single_mean_break(T=T, Tb=Tb, rng=rng)

        y_train = y[:t0]
        y_true = y[t0]

        try:
            fg = forecast_global_ar1(y_train)
            fr = forecast_rolling_ar1(y_train, window)
            fd = forecast_ar1_with_break_dummy(y_train, Tb)
        except Exception:
            continue

        eg.append(y_true - fg)
        er.append(y_true - fr)
        ed.append(y_true - fd)

    def metrics(errs):
        e = np.asarray(errs)
        return {
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE":  float(np.mean(np.abs(e))),
            "Bias": float(np.mean(e))
        }

    return metrics(eg), metrics(er), metrics(ed)


# =========================================================
# 4) RUN
# =========================================================
if __name__ == "__main__":
    mg, mr, md = monte_carlo_mean_break(
        n_sim=300,
        T=300,
        Tb=150,
        window=60,
        seed=123
    )

    print("\nGLOBAL AR(1)")
    print(mg)

    print("\nROLLING AR(1)")
    print(mr)

    print("\nAR(1) + BREAK DUMMY")
    print(md)
