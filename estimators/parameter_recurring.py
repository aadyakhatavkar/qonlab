"""
Parameter Break Estimators (Recurring/Markov-Switching)
========================================================
Forecasting methods for AR(1) with recurring (Markov-switching) parameter breaks.

Note: These are re-exported from parameter_single as the forecasting methods
are identical for both single and recurring breaks.
"""

from .parameter_single import (
    forecast_global_sarima,
    forecast_rolling_sarima,
    forecast_markov_switching_ar,
)

__all__ = [
    "forecast_global_sarima",
    "forecast_rolling_sarima",
    "forecast_markov_switching_ar",
]

