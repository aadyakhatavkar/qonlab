# Aligned simulation runners
from scripts.runner import mc_unified, mc_mean_breaks, mc_variance_breaks

# Keep backward compatibility aliases if needed, mapping to unified runner
mc_mean_breaks_multi = mc_mean_breaks
mc_parameter_breaks_post = mc_unified
monte_carlo_single_break_post = mc_unified
monte_carlo_recurring = mc_unified

# Plots
from .plots_variance import plot_loss_surfaces, plot_logscore_comparison, plot_time_series_example
from .plots_meansingle import plot_mean_single_break_results, plot_mean_single_break_example
from .plots_parametersingle import (
    plot_combined_distributions as param_plot_combined_distributions,
    plot_rmse_by_innovation as param_plot_rmse_by_innovation,
    plot_single_break_dgp as param_plot_single_break_dgp,
)

__all__ = [
    "mc_unified",
    "mc_mean_breaks",
    "mc_variance_breaks",
    "mc_mean_breaks_multi",
    "mc_parameter_breaks_post",
    "monte_carlo_single_break_post",
    "monte_carlo_recurring",
    # Plot exports
    "plot_loss_surfaces",
    "plot_logscore_comparison",
    "plot_time_series_example",
    "plot_mean_single_break_results",
    "plot_mean_single_break_example",
    "param_plot_combined_distributions",
    "param_plot_rmse_by_innovation",
    "param_plot_single_break_dgp",
]
