"""
Tests for Mean Single Break
============================
Tests DGP reproducibility, estimator outputs for mean single breaks.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.mean_singlebreaks import simulate_mean_break_ar1
from estimators.mean_singlebreak import (
    forecast_mean_sarima_global,
    forecast_mean_sarima_rolling,
    mean_rmse_mae_bias,
    mean_interval_coverage,
    mean_log_score_normal,
)


def test_simulate_mean_break_seed():
    """Test that DGP is reproducible with seed."""
    y1 = simulate_mean_break_ar1(T=60, Tb=30, mu0=0.0, mu1=2.0, seed=123)
    y2 = simulate_mean_break_ar1(T=60, Tb=30, mu0=0.0, mu1=2.0, seed=123)
    assert np.allclose(y1, y2), "Mean DGP not reproducible with same seed"


def test_simulate_mean_break_shape():
    """Test that DGP returns correct shape."""
    y = simulate_mean_break_ar1(T=100, Tb=50, mu0=0.0, mu1=1.5)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_simulate_mean_break_means():
    """Test that means are approximately correct before/after break."""
    y = simulate_mean_break_ar1(T=400, Tb=200, mu0=0.0, mu1=3.0, phi=0.1, sigma=0.5, seed=42)
    
    mean_before = np.mean(y[:200])
    mean_after = np.mean(y[200:])
    
    # Should be roughly separated by the break magnitude
    assert mean_after > mean_before, "Mean should increase after break"
    assert abs((mean_after - mean_before) - 3.0) < 2.0, "Mean difference should be roughly 3.0"


def test_forecast_mean_sarima_global_shapes():
    """Test SARIMA global forecaster output shapes."""
    y = simulate_mean_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_mean_sarima_global(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_mean_sarima_rolling_shapes():
    """Test SARIMA rolling forecaster output shapes."""
    y = simulate_mean_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_mean_sarima_rolling(y_train, window=50, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_mean_metrics():
    """Test metric calculations."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    rmse, mae, bias = mean_rmse_mae_bias(y_true, y_pred)
    
    assert isinstance(rmse, float), "RMSE should be float"
    assert isinstance(mae, float), "MAE should be float"
    assert isinstance(bias, float), "Bias should be float"
    assert rmse > 0, "RMSE should be positive"


def test_mean_interval_coverage():
    """Test interval coverage calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    var = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    coverage = mean_interval_coverage(y_true, mean, var, level=0.95)
    
    assert isinstance(coverage, float), "Coverage should be float"
    assert 0 <= coverage <= 1, "Coverage should be between 0 and 1"


def test_mean_log_score_normal():
    """Test log score calculation."""
    y_true = np.array([1.0, 2.0, 3.0])
    mean = np.array([1.0, 2.0, 3.0])
    var = np.array([0.5, 0.5, 0.5])
    
    ls = mean_log_score_normal(y_true, mean, var)
    
    assert isinstance(ls, float), "Log score should be float"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
