"""
Parameter Break Plots
=====================
Visualization for parameter break analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

from dgps.parameter import simulate_parameter_break_ar1

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