"""
Parameter Break Plots
=====================
Visualization for parameter break analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

from dgps.parameter import simulate_parameter_break_ar1


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
