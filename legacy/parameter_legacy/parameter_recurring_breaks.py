import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

# =====================================================
# 1) DGP: Markov-switching AR(1), Gaussian
# =====================================================
def simulate_ms_ar1_phi_only(
    T=400,
    p00=0.97,
    p11=0.97,
    phi0=0.2,
    phi1=0.9,
    sigma=1.0,
    y0=0.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)
    s = np.zeros(T, dtype=int)

    y[0] = y0
    s[0] = rng.integers(0, 2)

    for t in range(1, T):
        if s[t - 1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0

        eps = rng.normal(0.0, sigma)
        phi = phi0 if s[t] == 0 else phi1
        y[t] = phi * y[t - 1] + eps

    return y, s

# =====================================================
# 2) Forecasting models (1-step ahead)
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

# =====================================================
# 4) Monte Carlo experiment (recurring breaks)
# =====================================================
def monte_carlo_recurring(
    p,
    n_sim=300,
    T=400,
    t0=300,
    window=60,
    seed=123
):
    rng = np.random.default_rng(seed)

    err = {
        "Global SARIMA": [],
        "Rolling SARIMA": [],
        "MS AR": []
    }

    for _ in range(n_sim):
        y, _ = simulate_ms_ar1_phi_only(
            T=T,
            p00=p,
            p11=p,
            rng=rng
        )

        y_train = y[:t0]
        y_true = y[t0]

        err["Global SARIMA"].append(
            y_true - forecast_global_sarima(y_train)
        )
        err["Rolling SARIMA"].append(
            y_true - forecast_rolling_sarima(y_train, window)
        )
        err["MS AR"].append(
            y_true - forecast_markov_switching_ar(y_train)
        )

    return err

# =====================================================
# 5) Combined error distribution figure
# =====================================================
def plot_error_distributions_all(err_by_p, persistence_levels):
    fig, axes = plt.subplots(
        len(persistence_levels),
        1,
        figsize=(9, 3 * len(persistence_levels)),
        sharex=True,
        sharey=True
    )

    for ax, p in zip(axes, persistence_levels):
        for model, e in err_by_p[p].items():
            e = np.asarray(e)
            e = e[~np.isnan(e)]
            ax.hist(e, bins=40, density=True, alpha=0.4, label=model)

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Persistence p = {p}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Forecast error")
    axes[0].set_ylabel("Density")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

# =====================================================
# 6) Bar charts for RMSE / MAE / Bias
# =====================================================
def plot_metric_bars(df, metric):
    persistences = df["Persistence"].unique()
    models = ["Global SARIMA", "Rolling SARIMA", "MS AR"]

    x = np.arange(len(persistences))
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, model in enumerate(models):
        vals = (
            df[df["Model"] == model]
            .set_index("Persistence")
            .loc[persistences][metric]
            .values
        )
        plt.bar(x + (i - 1) * width, vals, width, label=model)

    plt.xticks(x, persistences)
    plt.xlabel("Regime Persistence")
    plt.ylabel(metric)
    plt.title(f"{metric} across Persistence Levels (Gaussian)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================================
# 7) DGP plots
# =====================================================
def plot_dgp_by_persistence(
    persistence_levels,
    T=400,
    phi0=0.2,
    phi1=0.9,
    seed=42
):
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(
        len(persistence_levels),
        1,
        figsize=(10, 3 * len(persistence_levels)),
        sharex=True
    )

    for ax, p in zip(axes, persistence_levels):
        y, s = simulate_ms_ar1_phi_only(
            T=T,
            p00=p,
            p11=p,
            phi0=phi0,
            phi1=phi1,
            rng=rng
        )

        ax.plot(y, color="black", lw=1.2)

        for t in range(1, T):
            if s[t] == 1:
                ax.axvspan(t - 1, t, color="pink", alpha=0.6)

        ax.set_title(f"DGP: Markov-Switching AR(1), persistence p = {p}")
        ax.set_ylabel(r"$y_t$")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

# =====================================================
# 8) RUN
# =====================================================
if __name__ == "__main__":

    persistence_levels = [0.90, 0.95, 0.97, 0.995]
    rows = []
    err_by_p = {}

    for p in persistence_levels:
        print(f"Running persistence p = {p}")
        err = monte_carlo_recurring(p=p)
        err_by_p[p] = err

        for model, e in err.items():
            m = metrics(e)
            rows.append({
                "Persistence": p,
                "Model": model,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"]
            })

    df_results = pd.DataFrame(rows)

    print("\nForecast Performance — Recurring Instability (Gaussian)\n")
    print(df_results)

    plot_error_distributions_all(err_by_p, persistence_levels)
    plot_metric_bars(df_results, "RMSE")
    plot_metric_bars(df_results, "MAE")
    plot_metric_bars(df_results, "Bias")
    plot_dgp_by_persistence(persistence_levels)
