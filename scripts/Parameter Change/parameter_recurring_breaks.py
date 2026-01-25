import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

# =====================================================
# 1) DGP: Markov-switching AR(1), phi only
# =====================================================
def simulate_ms_ar1_phi_only(
    T=400,
    p00=0.97, p11=0.97,
    phi0=0.2, phi1=0.9,
    sigma=1.0,
    y0=0.0,
    innovation="normal",
    df=None,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)
    s = np.zeros(T, dtype=int)

    y[0] = y0
    s[0] = rng.integers(0, 2)

    for t in range(1, T):
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0

        if innovation == "normal":
            eps = rng.normal(0.0, sigma)
        elif innovation == "student":
            eps = (
                rng.standard_t(df)
                * sigma
                / np.sqrt(df / (df - 2))
            )
        else:
            raise ValueError("Unknown innovation")

        phi = phi0 if s[t] == 0 else phi1
        y[t] = phi * y[t-1] + eps

    return y, s


# =====================================================
# 2) Forecasting models
# =====================================================
def forecast_global_ar1(y):
    res = ARIMA(y, order=(1, 0, 0), trend="n").fit()
    return float(res.forecast(1)[0])


def forecast_rolling_ar1(y, window=80):
    res = ARIMA(y[-window:], order=(1, 0, 0), trend="n").fit()
    return float(res.forecast(1)[0])


def forecast_markov_switching_ar1(y):
    y_dep = y[1:]
    x = y[:-1].reshape(-1, 1)

    mod = MarkovRegression(
        y_dep,
        k_regimes=2,
        trend="n",
        exog=x,
        switching_exog=True,
        switching_variance=False
    )

    res = mod.fit(disp=False)
    probs = res.filtered_marginal_probabilities[-1]
    params = dict(zip(res.model.param_names, res.params))

    phi0 = params["x1[0]"]
    phi1 = params["x1[1]"]

    y_last = y[-1]
    return float(probs[0] * phi0 * y_last + probs[1] * phi1 * y_last)


# =====================================================
# 3) Monte Carlo experiment
# =====================================================
def monte_carlo_experiment(
    innovation="normal",
    df=None,
    n_sim=200,
    T=400,
    t0=300,
    window=80,
    seed=123
):
    rng = np.random.default_rng(seed)

    err_g, err_r, err_m = [], [], []

    for _ in range(n_sim):
        y, _ = simulate_ms_ar1_phi_only(
            T=T,
            innovation=innovation,
            df=df,
            rng=rng
        )

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
            "MAE": float(np.mean(np.abs(e))),
            "Bias": float(np.mean(e))
        }

    return metrics(err_g), metrics(err_r), metrics(err_m)


# =====================================================
# 4) DGP visualization
# =====================================================
def plot_ms_dgp(
    T=400,
    innovation="normal",
    df=None,
    seed=42
):
    rng = np.random.default_rng(seed)
    y, s = simulate_ms_ar1_phi_only(
        T=T,
        innovation=innovation,
        df=df,
        rng=rng
    )

    plt.figure(figsize=(10, 4))
    plt.plot(y, color="black", lw=1.5, label="y_t")

    for t in range(1, T):
        if s[t] == 1:
            plt.axvspan(t-1, t, color="red", alpha=0.15)

    title = "DGP: Markov-switching AR(1)"
    if innovation == "student":
        title += f" with Student-t Innovations (df={df})"
    else:
        title += " with Gaussian Innovations"

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# 5) RUN — RESULTS FIRST, GRAPHS AFTER
# =====================================================
if __name__ == "__main__":

    # ---------- NUMERICAL RESULTS ----------
    cases = [
        ("Gaussian", "normal", None),
        ("Student-t (df=10)", "student", 10),
        ("Student-t (df=5)", "student", 5),
        ("Student-t (df=3)", "student", 3),
    ]

    results = {}

    for label, innov, df in cases:
        g, r, m = monte_carlo_experiment(
            innovation=innov,
            df=df
        )
        results[label] = (g, r, m)

        print(f"\n===== {label} Innovations =====")
        print("Global AR:", g)
        print("Rolling AR:", r)
        print("MS AR:", m)

    # ---------- GRAPHS ----------
    plot_ms_dgp(innovation="normal")
    plot_ms_dgp(innovation="student", df=10)
    plot_ms_dgp(innovation="student", df=5)
    plot_ms_dgp(innovation="student", df=3)
