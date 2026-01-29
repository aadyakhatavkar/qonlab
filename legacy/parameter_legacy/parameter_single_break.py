import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

# =====================================================
# VERSION / EXECUTION CHECK
# =====================================================
print("RUNNING parameter_single_break.py (SARIMA + MS-AR, NaN-safe)")
print("FILE:", __file__)
print("=" * 60)

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
            if df <= 2:
                raise ValueError("df must be > 2 for finite variance")
            eps = rng.standard_t(df) * sigma / np.sqrt(df / (df - 2))
        else:
            raise ValueError("Unknown innovation")

        y[t] = phi * y[t - 1] + eps

    return y

# =====================================================
# 2) Forecasting models (1-step ahead)
# =====================================================

# ---- Global SARIMA ----
def forecast_global_sarima(y):
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

# ---- Rolling SARIMA ----
def forecast_rolling_sarima(y, window=80):
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
    e = e[~np.isnan(e)]   # CRITICAL FIX
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e))
    }

# =====================================================
# 4) Monte Carlo — POST-BREAK ONLY
# =====================================================
def monte_carlo_single_break_post(
    n_sim=300,
    T=400,
    Tb=200,
    t_post=250,
    window=80,
    innovation="normal",
    df=None,
    seed=123
):
    rng = np.random.default_rng(seed)

    err = {
        "Global SARIMA": [],
        "Rolling SARIMA": [],
        "MS AR": []
    }

    print(f"--- Monte Carlo START | innovation={innovation}, df={df} ---")

    for i in range(n_sim):
        if i % 50 == 0:
            print(f"  MC iteration {i}/{n_sim}")

        y = simulate_single_break_ar1(
            T=T,
            Tb=Tb,
            innovation=innovation,
            df=df,
            rng=rng
        )

        y_train = y[:t_post]
        y_true = y[t_post]

        err["Global SARIMA"].append(
            y_true - forecast_global_sarima(y_train)
        )
        err["Rolling SARIMA"].append(
            y_true - forecast_rolling_sarima(y_train, window)
        )
        err["MS AR"].append(
            y_true - forecast_markov_switching_ar(y_train)
        )

    print(
        f"--- Monte Carlo END | innovation={innovation} "
        f"| MS-AR NaNs: {np.isnan(err['MS AR']).sum()} ---\n"
    )

    return err

# =====================================================
# 5) Plots
# =====================================================
def plot_combined_distributions(all_err):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, (label, err) in zip(axes, all_err.items()):
        for model, e in err.items():
            e = np.asarray(e)
            e = e[~np.isnan(e)]
            ax.hist(e, bins=40, density=True, alpha=0.4, label=model)

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Forecast Error Distribution — {label}")
        ax.set_ylabel("Density")
        ax.legend()

    axes[-1].set_xlabel("Forecast error")
    plt.tight_layout()
    plt.show()

def plot_rmse_by_innovation(df_results):
    innovations = df_results["Innovation"].unique()
    models = ["Global SARIMA", "Rolling SARIMA", "MS AR"]

    x = np.arange(len(innovations))
    width = 0.25

    plt.figure(figsize=(8, 5))

    for i, model in enumerate(models):
        vals = [
            df_results.loc[
                (df_results["Innovation"] == innov) &
                (df_results["Model"] == model),
                "RMSE"
            ].values[0]
            for innov in innovations
        ]
        plt.bar(x + (i - 1) * width, vals, width, label=model)

    plt.xticks(x, innovations)
    plt.xlabel("Innovation")
    plt.ylabel("RMSE")
    plt.title("RMSE (Innovations Standardized to Unit Variance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_single_break_dgp(
    T=400,
    Tb=200,
    phi1=0.2,
    phi2=0.9,
    seed=42
):
    rng = np.random.default_rng(seed)

    y = simulate_single_break_ar1(
        T=T,
        Tb=Tb,
        phi1=phi1,
        phi2=phi2,
        innovation="normal",
        rng=rng
    )

    plt.figure(figsize=(10, 4))
    plt.plot(y, color="black", lw=1.4)
    plt.axvline(Tb, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(r"$y_t$")
    plt.title("DGP: Single Deterministic Parameter Break")
    plt.tight_layout()
    plt.show()

# =====================================================
# 6) RUN
# =====================================================
if __name__ == "__main__":

    cases = [
        ("Gaussian", "normal", None),
        ("Student-t df=5", "student", 5),
        ("Student-t df=3", "student", 3),
    ]

    all_err = {}
    rows = []

    print("\n=== SINGLE BREAK: POST-BREAK FORECAST EXPERIMENT ===\n")

    for label, innov, df in cases:
        start = time.time()

        err = monte_carlo_single_break_post(
            innovation=innov,
            df=df
        )

        elapsed = time.time() - start
        print(f"Finished {label} in {elapsed:.2f} seconds\n")

        all_err[label] = err

        for model, e in err.items():
            m = metrics(e)
            rows.append({
                "Scenario": "Single break",
                "Innovation": label,
                "Model": model,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"]
            })

    df_results = pd.DataFrame(rows)

    print("\nPOST-BREAK FORECAST RESULTS (SINGLE BREAK)\n")
    print(df_results.to_string(index=False))

    plot_combined_distributions(all_err)
    plot_rmse_by_innovation(df_results)
    plot_single_break_dgp()
