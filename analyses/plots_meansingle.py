import numpy as np
import matplotlib.pyplot as plt
from dgps.mean_singlebreaks import simulate_single_break_ar1

# =========================================================
# 5) VISUAL GRAPHS
# =========================================================
def plot_mean_single_break_results(results):
    plot_df = results.copy().sort_values("RMSE", na_position="last")

    plt.figure(figsize=(12,4))
    plt.bar(plot_df["Method"], plot_df["RMSE"])
    plt.title("Single-break: RMSE by method (SARIMA setting)")
    plt.ylabel("RMSE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,4))
    plt.bar(plot_df["Method"], plot_df["MAE"])
    plt.title("Single-break: MAE by method (SARIMA setting)")
    plt.ylabel("MAE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,4))
    plt.bar(plot_df["Method"], plot_df["Bias"])
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title("Single-break: Bias by method (SARIMA setting)")
    plt.ylabel("Bias")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

def plot_mean_single_break_example(T=300, Tb=150):
    # Example series with break
    y_demo = simulate_single_break_ar1(
        T=T, Tb=Tb, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, rng=np.random.default_rng(999)
    )

    plt.figure(figsize=(12,4))
    plt.plot(y_demo, label="Simulated series (AR(1))")
    plt.axvline(Tb, linestyle="--", linewidth=2, label=f"Break Tb={Tb}")
    plt.title("Example series: single mean break (AR(1))")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()