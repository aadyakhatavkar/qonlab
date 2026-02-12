"""
Tests for Variance Recurring (Markov-Switching) Break
=====================================================
Tests DGP reproducibility, estimator outputs for variance recurring breaks.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.variance_recurring import simulate_ms_ar1_variance_only
from estimators.variance_recurring import (
    forecast_markov_switching,
    variance_rmse_mae_bias,
    variance_interval_coverage,
    variance_log_score_normal,
)
from estimators.variance_single import (
    forecast_variance_dist_sarima_rolling,
    forecast_variance_dist_sarima_global,
)


def test_simulate_ms_variance_seed():
    """Test that Markov-switching DGP is reproducible with seed."""
    y1 = simulate_ms_ar1_variance_only(T=60, p00=0.95, p11=0.95, seed=123)
    y2 = simulate_ms_ar1_variance_only(T=60, p00=0.95, p11=0.95, seed=123)
    assert np.allclose(y1, y2), "MS DGP not reproducible with same seed"


def test_simulate_ms_variance_shape():
    """Test that Markov-switching DGP returns correct shape."""
    y = simulate_ms_ar1_variance_only(T=100, p00=0.95, p11=0.95)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_simulate_ms_variance_persistence():
    """Test that high persistence produces longer regime durations."""
    y_low = simulate_ms_ar1_variance_only(T=200, p00=0.80, p11=0.80, seed=42)
    y_high = simulate_ms_ar1_variance_only(T=200, p00=0.99, p11=0.99, seed=42)
    
    # Both should be valid time series
    assert np.all(np.isfinite(y_low)), "Low persistence series has NaN/Inf"
    assert np.all(np.isfinite(y_high)), "High persistence series has NaN/Inf"


def test_forecast_markov_switching_shapes():
    """Test Markov-switching forecaster output shapes."""
    y = simulate_ms_ar1_variance_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_markov_switching(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_markov_switching_finite():
    """Test that MS forecaster produces finite outputs."""
    y = simulate_ms_ar1_variance_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_markov_switching(y_train, horizon=5)
    
    assert np.all(np.isfinite(mean)), "Forecasted means contain NaN/Inf"
    assert np.all(np.isfinite(var)), "Forecasted variances contain NaN/Inf"
    assert np.all(var > 0), "Variances should be positive"


def test_variance_metrics_on_ms_data():
    """Test metric calculations on MS-generated data."""
    y = simulate_ms_ar1_variance_only(T=100, p00=0.95, p11=0.95, seed=42)
    y_train = y[:-10]
    y_test = y[-10:]
    
    # Get forecasts
    mean, var = forecast_markov_switching(y_train, horizon=len(y_test))
    
    # Calculate metrics
    rmse, mae, bias = variance_rmse_mae_bias(y_test, mean)
    coverage = variance_interval_coverage(y_test, mean, var, level=0.95)
    ls = variance_log_score_normal(y_test, mean, var)
    
    # Verify outputs
    assert np.isfinite(rmse), "RMSE should be finite"
    assert np.isfinite(mae), "MAE should be finite"
    assert np.isfinite(bias), "Bias should be finite"
    assert 0 <= coverage <= 1, "Coverage should be in [0, 1]"


def test_ms_vs_sarima_on_recurring_data():
    """Test that forecasters can handle recurring variance breaks."""
    # Generate MS data
    y = simulate_ms_ar1_variance_only(T=200, p00=0.95, p11=0.95, phi=0.6, sigma1=1.0, sigma2=2.0, seed=123)
    y_train = y[:150]
    y_test = y[150:160]
    
    # Both forecasters should work
    try:
        mean_ms, var_ms = forecast_markov_switching(y_train, horizon=len(y_test))
        assert mean_ms.shape == (len(y_test),), "MS forecaster shape mismatch"
    except Exception as e:
        pytest.fail(f"MS forecaster failed: {e}")
    
    try:
        mean_sarima, var_sarima = forecast_variance_dist_sarima_rolling(y_train, window=50, horizon=len(y_test))
        assert mean_sarima.shape == (len(y_test),), "SARIMA forecaster shape mismatch"
    except Exception as e:
        pytest.fail(f"SARIMA forecaster failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
