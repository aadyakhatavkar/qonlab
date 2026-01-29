"""
Mean Break Plots
================
Visualization for mean break analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

from dgps.mean import simulate_mean_break_ar1


def mean_plot_dgp_example(T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, seed=42, save_path=None):
    """Plot example realization of mean break DGP."""
    y = simulate_mean_break_ar1(T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, seed=seed)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y, color="black", lw=1.4)
    ax.axvline(Tb, color="red", linestyle="--", linewidth=2, label=f"Break at t={Tb}")
    ax.axhline(mu0/(1-phi), color="blue", linestyle=":", alpha=0.5, label=f"Pre-break mean ≈ {mu0/(1-phi):.2f}")
    ax.axhline(mu1/(1-phi), color="green", linestyle=":", alpha=0.5, label=f"Post-break mean ≈ {mu1/(1-phi):.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(f"DGP: Single Mean Break (μ₀={mu0} → μ₁={mu1})")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def mean_plot_results_bar(df_results, save_path=None):
    """Bar plot of RMSE by method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = df_results["Method"].values
    rmse = df_results["RMSE"].values
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    bars = ax.bar(range(len(methods)), rmse, color=colors)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel("RMSE")
    ax.set_title("Mean Break Forecasting: RMSE by Method")
    
    for bar, val in zip(bars, rmse):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
