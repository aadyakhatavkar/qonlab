"""
Tests for Variance Single Break
================================
Tests DGP reproducibility, estimator outputs for variance single breaks.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.variance_single import simulate_variance_break_ar1, estimate_variance_break_point
from estimators.variance_single import (
    forecast_variance_dist_sarima_global,
    forecast_variance_dist_sarima_rolling,
    forecast_garch_variance,
    forecast_variance_sarima_post_break,
    forecast_variance_averaged_window,
    variance_rmse_mae_bias,
    variance_interval_coverage,
    variance_log_score_normal,
)


def test_simulate_variance_break_seed():
    """Test that DGP is reproducible with seed."""
    y1 = simulate_variance_break_ar1(T=60, Tb=30, seed=123)
    y2 = simulate_variance_break_ar1(T=60, Tb=30, seed=123)
    assert np.allclose(y1, y2), "DGP not reproducible with same seed"


def test_simulate_variance_break_shape():
    """Test that DGP returns correct shape."""
    y = simulate_variance_break_ar1(T=100, Tb=50)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_estimate_variance_break_point():
    """Test break point estimation."""
    y = simulate_variance_break_ar1(T=200, Tb=100, sigma1=1.0, sigma2=3.0, seed=42)
    Tb_est = estimate_variance_break_point(y, trim=0.15)
    # Should be reasonably close to true break point
    assert 80 < Tb_est < 120, f"Estimated break point {Tb_est} far from true {100}"


def test_forecast_variance_sarima_global_shapes():
    """Test SARIMA global forecaster output shapes."""
    y = simulate_variance_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_variance_dist_sarima_global(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_variance_sarima_rolling_shapes():
    """Test SARIMA rolling forecaster output shapes."""
    y = simulate_variance_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_variance_dist_sarima_rolling(y_train, window=50, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_garch_variance_shapes():
    """Test GARCH forecaster output shapes."""
    import estimators.variance_single as var_mod
    if getattr(var_mod, 'arch_model', None) is None:
        pytest.skip('arch package not installed')
    
    y = simulate_variance_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_garch_variance(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_variance_averaged_window_shapes():
    """Test averaged window forecaster output shapes."""
    y = simulate_variance_break_ar1(T=150, Tb=75, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_variance_averaged_window(y_train, window_sizes=[30, 50, 70], horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_variance_rmse_mae_bias():
    """Test metric calculations."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    rmse, mae, bias = variance_rmse_mae_bias(y_true, y_pred)
    
    assert isinstance(rmse, float), "RMSE should be float"
    assert isinstance(mae, float), "MAE should be float"
    assert isinstance(bias, float), "Bias should be float"
    assert rmse > 0, "RMSE should be positive"
    assert mae > 0, "MAE should be positive"


def test_variance_interval_coverage():
    """Test interval coverage calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    var = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    coverage = variance_interval_coverage(y_true, mean, var, level=0.95)
    
    assert isinstance(coverage, float), "Coverage should be float"
    assert 0 <= coverage <= 1, "Coverage should be between 0 and 1"


def test_variance_log_score_normal():
    """Test log score calculation."""
    y_true = np.array([1.0, 2.0, 3.0])
    mean = np.array([1.0, 2.0, 3.0])
    var = np.array([0.5, 0.5, 0.5])
    
    ls = variance_log_score_normal(y_true, mean, var)
    
    assert isinstance(ls, float), "Log score should be float"
    # Log score should be negative for normal distribution
    assert ls < 0 or np.isnan(ls), "Log score should be negative or NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
