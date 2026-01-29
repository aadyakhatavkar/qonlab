# Variance break simulations (main)
from .simulations import mc_variance_breaks, mc_variance_breaks_grid

# Mean break simulations
from .mean_simulations import (
    mc_mean_breaks,
    simulate_mean_break_ar1,
    mean_forecast_global_ar1,
    mean_forecast_rolling_ar1,
    mean_forecast_ar1_with_break_dummy_oracle,
    mean_forecast_ar1_with_estimated_break,
    mean_forecast_markov_switching,
    mean_metrics,
    mean_plot_dgp_example,
    mean_plot_results_bar,
)

# Parameter break simulations
from .param_simulations import (
    mc_parameter_breaks_post,
    simulate_parameter_break_ar1,
    param_forecast_global_ar,
    param_forecast_rolling_ar,
    param_forecast_markov_switching_ar,
    param_metrics,
    param_plot_error_distributions,
    param_plot_rmse_by_innovation,
    param_plot_dgp_example,
)

__all__ = [
    # Variance
    "mc_variance_breaks",
    "mc_variance_breaks_grid",
    # Mean
    "mc_mean_breaks",
    "simulate_mean_break_ar1",
    "mean_forecast_global_ar1",
    "mean_forecast_rolling_ar1",
    "mean_forecast_ar1_with_break_dummy_oracle",
    "mean_forecast_ar1_with_estimated_break",
    "mean_forecast_markov_switching",
    "mean_metrics",
    "mean_plot_dgp_example",
    "mean_plot_results_bar",
    # Parameter
    "mc_parameter_breaks_post",
    "simulate_parameter_break_ar1",
    "param_forecast_global_ar",
    "param_forecast_rolling_ar",
    "param_forecast_markov_switching_ar",
    "param_metrics",
    "param_plot_error_distributions",
    "param_plot_rmse_by_innovation",
    "param_plot_dgp_example",
]
