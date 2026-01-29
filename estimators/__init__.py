# Variance forecasters
from .variance import (
    forecast_variance_dist_arima_global,
    forecast_variance_dist_arima_rolling,
    forecast_garch_variance,
    forecast_variance_arima_post_break,
    forecast_variance_averaged_window,
    forecast_markov_switching,
    _auto_select_arima_order,
    variance_rmse_mae_bias,
    variance_interval_coverage,
    variance_log_score_normal,
)

# Mean forecasters (auto-selected ARMA)
from .mean import (
    mean_forecast_global_arma,
    mean_forecast_rolling_arma,
    mean_forecast_global_ar1,  # Backward compatibility
    mean_forecast_rolling_ar1,  # Backward compatibility
    mean_forecast_ar1_with_break_dummy_oracle,
    mean_forecast_ar1_with_estimated_break,
    mean_forecast_markov_switching,
    mean_estimate_break_point_gridsearch,
    mean_metrics,
    _auto_select_arma_order,
)

# Mean multiple breaks
from .mean_multiplebreaks import (
    forecast_ar1_single_break_dummy_oracle,
    forecast_ar1_with_multiple_break_dummies_oracle,
)

# Parameter forecasters (auto-selected ARMA)
from .parameter import (
    param_forecast_global_arma,
    param_forecast_rolling_arma,
    param_forecast_global_ar,  # Backward compatibility
    param_forecast_rolling_ar,  # Backward compatibility
    param_forecast_markov_switching_ar,
    param_metrics,
    _auto_select_arma_order as param_auto_select_arma_order,
)

__all__ = [
    # Variance
    "forecast_variance_dist_arima_global",
    "forecast_variance_dist_arima_rolling",
    "forecast_garch_variance",
    "forecast_variance_arima_post_break",
    "forecast_variance_averaged_window",
    "forecast_markov_switching",
    "_auto_select_arima_order",
    "variance_rmse_mae_bias",
    "variance_interval_coverage",
    "variance_log_score_normal",
    # Mean (auto-selected ARMA)
    "mean_forecast_global_arma",
    "mean_forecast_rolling_arma",
    "mean_forecast_global_ar1",
    "mean_forecast_rolling_ar1",
    "mean_forecast_ar1_with_break_dummy_oracle",
    "mean_forecast_ar1_with_estimated_break",
    "mean_forecast_markov_switching",
    "mean_estimate_break_point_gridsearch",
    "mean_metrics",
    "_auto_select_arma_order",
    # Mean (multiple breaks)
    "forecast_ar1_single_break_dummy_oracle",
    "forecast_ar1_with_multiple_break_dummies_oracle",
    # Parameter (auto-selected ARMA)
    "param_forecast_global_arma",
    "param_forecast_rolling_arma",
    "param_forecast_global_ar",
    "param_forecast_rolling_ar",
    "param_forecast_markov_switching_ar",
    "param_metrics",
    "param_auto_select_arma_order",
]
