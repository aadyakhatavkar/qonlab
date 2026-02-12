"""
Unified Estimator Aggregator
============================
Aggregates all forecasting methods for structural breaks.
"""
import numpy as np
from .variance_single import (
    forecast_variance_dist_sarima_global,
    forecast_variance_dist_sarima_rolling,
    forecast_garch_variance,
)
from .variance_recurring import (
    forecast_markov_switching,
)
from .mean_singlebreak import (
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_ses,
)
from .mean_recurring import (
    forecast_ms_ar1_mean,
)
from .parameter_single import (
    forecast_global_sarima as param_forecast_global_sarima,
    forecast_rolling_sarima as param_forecast_rolling_sarima,
    forecast_markov_switching_ar as param_forecast_markov_switching_ar,
)
from .parameter_recurring import (
    forecast_markov_switching_ar as param_forecast_markov_switching_ar_recurring,
)

from protocols import calculate_metrics

__all__ = [
    # Variance - Single
    "forecast_variance_dist_sarima_global",
    "forecast_variance_dist_sarima_rolling",
    "forecast_garch_variance",
    # Variance - Recurring
    "forecast_markov_switching",
    # Mean - Single
    "forecast_sarima_global",
    "forecast_sarima_rolling",
    "forecast_ses",
    # Mean - Recurring
    "forecast_ms_ar1_mean",
    # Parameter - Single
    "param_forecast_global_sarima",
    "param_forecast_rolling_sarima",
    "param_forecast_markov_switching_ar",
    # Parameter - Recurring (same methods as single, different DGP)
    "param_forecast_markov_switching_ar_recurring",
    # Utilities
    "calculate_metrics",
]
