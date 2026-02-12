"""
Unified Estimator Aggregator
============================
Aggregates all forecasting methods for structural breaks.
Maintains legacy aliases for backward compatibility.
"""
import numpy as np
from .variance import (
    forecast_variance_dist_sarima_global,
    forecast_variance_dist_sarima_rolling,
    forecast_garch_variance,
    forecast_variance_sarima_post_break,
    variance_rmse_mae_bias,
    variance_interval_coverage,
    variance_log_score_normal,
)
from .mean_multiplebreaks import (
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_sarima_break_dummy_oracle_single,
    forecast_sarima_break_dummy_oracle_multiple,
    forecast_sarima_estimated_break_single,
    forecast_sarima_estimated_breaks_multiple,
    forecast_ses,
    forecast_holt_winters_seasonal
)
from .parameter_single import (
    forecast_global_sarima as param_forecast_global_sarima,
    forecast_rolling_sarima as param_forecast_rolling_sarima,
    forecast_markov_switching_ar as param_forecast_markov_switching_ar,
    metrics as param_metrics
)

# Legacy aliases for Mean (previously imported from mean_multiplebreaks)
forecast_mean_ar1_global = forecast_sarima_global
forecast_mean_ar1_rolling = forecast_sarima_rolling
forecast_mean_oracle_single = forecast_sarima_break_dummy_oracle_single
forecast_mean_markov = param_forecast_markov_switching_ar # MS-AR is common across mean/param

def mean_metrics(e):
    # Wrapper around calculation in protocols or local implementation
    from protocols import calculate_metrics
    return calculate_metrics(e)

__all__ = [
    # Variance
    "forecast_variance_dist_sarima_global",
    "forecast_variance_dist_sarima_rolling",
    "forecast_garch_variance",
    "forecast_variance_sarima_post_break",
    "variance_rmse_mae_bias",
    # Mean
    "forecast_sarima_global",
    "forecast_sarima_rolling",
    "forecast_mean_ar1_global",
    "forecast_mean_ar1_rolling",
    "forecast_mean_oracle_single",
    "forecast_mean_markov",
    "mean_metrics",
    # Parameter
    "param_forecast_global_sarima",
    "param_forecast_rolling_sarima",
    "param_forecast_markov_switching_ar",
    "param_metrics",
]
