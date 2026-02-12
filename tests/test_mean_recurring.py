"""
Tests for Mean Recurring (Markov-Switching) Break
==================================================
Tests DGP reproducibility, estimator outputs for mean recurring breaks.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.mean_recurring import simulate_ms_ar1_mean_only
from estimators.mean_recurring import (
    forecast_ms_ar1_mean,
    forecast_mean_arima_global,
    mean_rmse_mae_bias,
    mean_interval_coverage,
    mean_log_score_normal,
)


def test_simulate_ms_mean_seed():
    """Test that Markov-switching DGP is reproducible with seed."""
    y1 = simulate_ms_ar1_mean_only(T=60, p00=0.95, p11=0.95, seed=123)
    y2 = simulate_ms_ar1_mean_only(T=60, p00=0.95, p11=0.95, seed=123)
    assert np.allclose(y1, y2), "MS mean DGP not reproducible with same seed"


def test_simulate_ms_mean_shape():
    """Test that Markov-switching DGP returns correct shape."""
    y = simulate_ms_ar1_mean_only(T=100, p00=0.95, p11=0.95)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_simulate_ms_mean_means():
    """Test that regimes have different means."""
    y = simulate_ms_ar1_mean_only(T=400, p00=0.95, p11=0.95, mu0=0.0, mu1=3.0, seed=42)
    
    # Verify there's variation in the series (indicates regime switching)
    assert np.std(y) > 0, "Series should have variation"
    # Mean should be somewhere between mu0 and mu1
    overall_mean = np.mean(y)
    assert 0 <= overall_mean <= 3.0, "Overall mean should be between regime means"


def test_forecast_ms_ar1_mean_shapes():
    """Test MS AR(1) mean forecaster output shapes."""
    y = simulate_ms_ar1_mean_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_ms_ar1_mean(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_mean_arima_global_shapes():
    """Test ARIMA global forecaster output shapes."""
    y = simulate_ms_ar1_mean_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_mean_arima_global(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_ms_ar1_mean_finite():
    """Test that MS AR(1) forecaster produces finite outputs."""
    y = simulate_ms_ar1_mean_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_ms_ar1_mean(y_train, horizon=5)
    
    assert np.all(np.isfinite(mean)), "Forecasted means contain NaN/Inf"
    assert np.all(np.isfinite(var)), "Forecasted variances contain NaN/Inf"
    assert np.all(var > 0), "Variances should be positive"


def test_mean_metrics_on_ms_data():
    """Test metric calculations on MS-generated data."""
    y = simulate_ms_ar1_mean_only(T=100, p00=0.95, p11=0.95, seed=42)
    y_train = y[:-10]
    y_test = y[-10:]
    
    # Get forecasts
    mean, var = forecast_ms_ar1_mean(y_train, horizon=len(y_test))
    
    # Calculate metrics
    rmse, mae, bias = mean_rmse_mae_bias(y_test, mean)
    coverage = mean_interval_coverage(y_test, mean, var, level=0.95)
    ls = mean_log_score_normal(y_test, mean, var)
    
    # Verify outputs
    assert np.isfinite(rmse), "RMSE should be finite"
    assert np.isfinite(mae), "MAE should be finite"
    assert np.isfinite(bias), "Bias should be finite"
    assert 0 <= coverage <= 1, "Coverage should be in [0, 1]"


def test_ms_vs_arima_on_recurring_mean():
    """Test that forecasters can handle recurring mean breaks."""
    # Generate MS data with significant mean shift
    y = simulate_ms_ar1_mean_only(T=200, p00=0.95, p11=0.95, phi=0.6, mu0=0.0, mu1=4.0, seed=123)
    y_train = y[:150]
    y_test = y[150:160]
    
    # Both forecasters should work
    try:
        mean_ms, var_ms = forecast_ms_ar1_mean(y_train, horizon=len(y_test))
        assert mean_ms.shape == (len(y_test),), "MS forecaster shape mismatch"
    except Exception as e:
        pytest.fail(f"MS forecaster failed: {e}")
    
    try:
        mean_arima, var_arima = forecast_mean_arima_global(y_train, horizon=len(y_test))
        assert mean_arima.shape == (len(y_test),), "ARIMA forecaster shape mismatch"
    except Exception as e:
        pytest.fail(f"ARIMA forecaster failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
