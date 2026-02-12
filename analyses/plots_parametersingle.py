import numpy as np
import matplotlib.pyplot as plt
from dgps.parameter_single import simulate_single_break_ar1
from dgps.parameter_recurring import simulate_ms_ar1_phi_only

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
            if not df_results.loc[(df_results["Innovation"] == innov) & (df_results["Model"] == model)].empty
            else np.nan
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
