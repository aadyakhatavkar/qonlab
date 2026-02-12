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

"""

def param_plot_error_distributions(all_err, save_path=None):
    """
    Plot forecast error distributions across innovation types.
    
    Parameters:
        all_err: dict of {label: {model: errors}}
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(len(all_err), 1, figsize=(10, 3*len(all_err)), sharex=True)
    
    if len(all_err) == 1:
        axes = [axes]

    for ax, (label, err) in zip(axes, all_err.items()):
        for model, e in err.items():
            ax.hist(e, bins=40, density=True, alpha=0.4, label=model)

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Forecast Error Distribution — {label}")
        ax.set_ylabel("Density")
        ax.legend()

    axes[-1].set_xlabel("Forecast error")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def param_plot_rmse_by_innovation(df_results, save_path=None):
    """
    Bar plot of RMSE by innovation type and model.
    
    Parameters:
        df_results: DataFrame with columns [Innovation, Model, RMSE]
        save_path: Optional path to save figure
    """
    innovations = df_results["Innovation"].unique()
    models = df_results["Model"].unique()

    x = np.arange(len(innovations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        vals = [
            df_results.loc[
                (df_results["Innovation"] == innov) &
                (df_results["Model"] == model),
                "RMSE"
            ].values[0]
            for innov in innovations
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(innovations)
    ax.set_xlabel("Innovation")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Innovation Distribution and Model")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def param_plot_dgp_example(T=400, Tb=200, phi1=0.2, phi2=0.9, seed=42, save_path=None):
    """Plot example realization of parameter break DGP."""
    y = simulate_parameter_break_ar1(
        T=T, Tb=Tb, phi1=phi1, phi2=phi2,
        innovation="normal", seed=seed
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y, color="black", lw=1.4)
    ax.axvline(Tb, color="red", linestyle="--", linewidth=2, label=f"Break at t={Tb}")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(f"DGP: Single Parameter Break (φ₁={phi1} → φ₂={phi2})")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

"""