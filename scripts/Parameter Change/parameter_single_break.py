import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

# =====================================================
# 1) DGP: AR(1) with SINGLE deterministic break in phi
# =====================================================
def simulate_single_break_ar1(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    sigma=1.0,
    innovation="normal",
    df=None,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2

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

        y[t] = phi * y[t-1] + eps

    return y


# =====================================================
# 2) Forecasting models (1-step ahead)
# =====================================================
def forecast_global_ar(y):
    res = ARIMA(y, order=(1, 0, 0), trend="n").fit()
    return float(res.forecast(1)[0])


def forecast_rolling_ar(y, window=120):
    res = ARIMA(y[-window:], order=(1, 0, 0), trend="n").fit()
    return float(res.forecast(1)[0])


def forecast_markov_switching_ar(y):
    model = MarkovRegression(
        y,
        k_regimes=2,
        trend="n",
        switching_variance=False
    ).fit(disp=False)

    phi0, phi1 = model.params[:2]
    probs = model.filtered_marginal_probabilities[-1]
    phi_hat = probs[0] * phi0 + probs[1] * phi1

    return float(phi_hat * y[-1])


# =====================================================
# 3) Metrics
# =====================================================
def metrics(e):
    e = np.asarray(e)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE":  float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }


# =====================================================
# 4) Monte Carlo — PRE & POST break (FAST, comparable)
# =====================================================
def monte_carlo_single_break(
    n_sim=300,
    T=400,
    Tb=200,
    t_pre=150,
    t_post=300,
    window=120,
    innovation="normal",
    df=None,
    seed=123
):
    rng = np.random.default_rng(seed)

    err_pre = {"Global AR": [], "Rolling AR": [], "MS AR": []}
    err_post = {"Global AR": [], "Rolling AR": [], "MS AR": []}

    for _ in range(n_sim):
        y = simulate_single_break_ar1(
            T=T,
            Tb=Tb,
            innovation=innovation,
            df=df,
            rng=rng
        )

        # --- pre-break forecast ---
        y_train = y[:t_pre]
        y_true = y[t_pre]

        err_pre["Global AR"].append(y_true - forecast_global_ar(y_train))
        err_pre["Rolling AR"].append(y_true - forecast_rolling_ar(y_train, window))
        err_pre["MS AR"].append(y_true - forecast_markov_switching_ar(y_train))

        # --- post-break forecast ---
        y_train = y[:t_post]
        y_true = y[t_post]

        err_post["Global AR"].append(y_true - forecast_global_ar(y_train))
        err_post["Rolling AR"].append(y_true - forecast_rolling_ar(y_train, window))
        err_post["MS AR"].append(y_true - forecast_markov_switching_ar(y_train))

    return {
        "Pre-break": {k: metrics(v) for k, v in err_pre.items()},
        "Post-break": {k: metrics(v) for k, v in err_post.items()}
    }


# =====================================================
# 5) Convert results to Power BI–ready table
# =====================================================
def results_to_rows(results, scenario, innovation):
    rows = []
    for period, models in results.items():
        for model, stats in models.items():
            rows.append({
                "Scenario": scenario,
                "Innovation": innovation,
                "Period": period,
                "Model": model,
                "RMSE": stats["RMSE"],
                "MAE": stats["MAE"],
                "Bias": stats["Bias"]
            })
    return rows


# =====================================================
# 6) Visualization — DGP only
# =====================================================
def plot_single_break_dgp(T=400, Tb=200, innovation="normal", df=None, seed=42):
    rng = np.random.default_rng(seed)
    y = simulate_single_break_ar1(
        T=T, Tb=Tb, innovation=innovation, df=df, rng=rng
    )

    plt.figure(figsize=(10, 4))
    plt.plot(y, color="black", lw=1.5, label="y_t")
    plt.axvline(Tb, color="red", linestyle="--", label="True break")

    plt.title("DGP: AR(1) with Single Structural Break in Persistence")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# 7) Visualization — DGP + forecasts
# =====================================================
def plot_single_break_forecasts(
    T=400,
    Tb=200,
    t_start=50,
    window=120,
    innovation="normal",
    df=None,
    seed=42
):
    rng = np.random.default_rng(seed)
    y = simulate_single_break_ar1(
        T=T, Tb=Tb, innovation=innovation, df=df, rng=rng
    )

    fg, fr, fm = [], [], []

    for t in range(T - 1):
        if t < t_start:
            fg.append(np.nan)
            fr.append(np.nan)
            fm.append(np.nan)
            continue

        y_train = y[:t + 1]

        fg.append(forecast_global_ar(y_train))
        fr.append(forecast_rolling_ar(y_train, window))
        fm.append(forecast_markov_switching_ar(y_train))

    plt.figure(figsize=(10, 4))
    plt.plot(y, color="black", lw=2, label="True y_t")
    plt.plot(fg, "--", label="Global AR")
    plt.plot(fr, "--", label="Rolling AR")
    plt.plot(fm, "--", label="Markov-switching AR")
    plt.axvline(Tb, color="red", linestyle=":", label="True break")

    plt.title("One-step-ahead Forecasts Around a Structural Break")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# 8) RUN — RESULTS FIRST, GRAPHS AFTER
# =====================================================
if __name__ == "__main__":

    all_rows = []

    cases = [
        ("Single break", "Gaussian", "normal", None),
        ("Single break", "Student-t df=10", "student", 10),
        ("Single break", "Student-t df=5", "student", 5),
        ("Single break", "Student-t df=3", "student", 3),
    ]

    # --------- NUMERICAL RESULTS ---------
    for scenario, label, innov, df in cases:
        res = monte_carlo_single_break(
            innovation=innov,
            df=df
        )

        all_rows.extend(
            results_to_rows(res, scenario, label)
        )

    df_results = pd.DataFrame(all_rows)
    print("\n=== FINAL RESULTS TABLE ===\n")
    print(df_results.to_string(index=False))

    # Export for Power BI
    df_results.to_csv("single_break_results.csv", index=False)

    # --------- VISUALIZATIONS ---------
    plot_single_break_dgp(innovation="normal")
    plot_single_break_forecasts(innovation="normal")

    plot_single_break_dgp(innovation="student", df=5)
    plot_single_break_forecasts(innovation="student", df=5)
