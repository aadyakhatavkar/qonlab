"""
Mean Break Plots
================
Visualization for mean break analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

from dgps.mean_singlebreaks import simulate_mean_break_ar1

# =========================================================
# 5) GRAPHS: RMSE/MAE comparison across scenarios
# =========================================================
def bar_compare(metric="RMSE"):
    pivot = all_results.pivot_table(index="Method", columns="Scenario", values=metric, aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(12,5))
    ax.set_title(f"{metric} Comparison: Single vs Multiple Breaks (SARIMA setting)")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

bar_compare("RMSE")
bar_compare("MAE")


# =========================================================
# 6) Example series plots (single vs multiple) with breaks
# =========================================================
rng_demo = np.random.default_rng(999)
y1 = simulate_single_break_with_seasonality(rng=rng_demo)
y2 = simulate_multiple_breaks_with_seasonality(rng=rng_demo)

plt.figure(figsize=(12,4))
plt.plot(y1, label="Single break series")
plt.axvline(150, linestyle="--", linewidth=2, label="Tb=150")
plt.title("Example: Single break + seasonality")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(y2, label="Multiple breaks series")
plt.axvline(100, linestyle="--", linewidth=2, label="b1=100")
plt.axvline(200, linestyle="--", linewidth=2, label="b2=200")
plt.title("Example: Multiple breaks + seasonality")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

"""

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
"""