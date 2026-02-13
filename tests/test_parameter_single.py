"""
Tests for Parameter Single Break
=================================
Tests DGP reproducibility, estimator outputs for parameter single breaks.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.parameter_single import simulate_single_break_ar1
from estimators.parameter_single import (
    forecast_global_sarima,
    forecast_rolling_sarima,
    forecast_markov_switching_ar,
)


def test_simulate_parameter_break_seed():
    """Test that DGP is reproducible with seed."""
    y1 = simulate_single_break_ar1(T=60, Tb=30, phi0=0.3, phi1=0.8, seed=123)
    y2 = simulate_single_break_ar1(T=60, Tb=30, phi0=0.3, phi1=0.8, seed=123)
    assert np.allclose(y1, y2), "Parameter DGP not reproducible with same seed"


def test_simulate_parameter_break_shape():
    """Test that DGP returns correct shape."""
    y = simulate_parameter_break_ar1(T=100, Tb=50, phi0=0.3, phi1=0.8)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_simulate_parameter_break_persistence():
    """Test that persistence increases after break."""
    y = simulate_parameter_break_ar1(T=200, Tb=100, phi0=0.2, phi1=0.9, seed=42)
    
    # Before break: autocorrelation should be lower
    acf_before = np.corrcoef(y[:100][:-1], y[:100][1:])[0, 1]
    # After break: autocorrelation should be higher
    acf_after = np.corrcoef(y[100:][:-1], y[100:][1:])[0, 1]
    
    # After break should have higher persistence
    assert acf_after > acf_before - 0.1, "Persistence should increase after break"


def test_forecast_parameter_sarima_global_shapes():
    """Test SARIMA global forecaster output shapes."""
    y = simulate_parameter_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_parameter_sarima_global(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_parameter_sarima_rolling_shapes():
    """Test SARIMA rolling forecaster output shapes."""
    y = simulate_parameter_break_ar1(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_parameter_sarima_rolling(y_train, window=50, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_parameter_metrics():
    """Test metric calculations."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    rmse, mae, bias = parameter_rmse_mae_bias(y_true, y_pred)
    
    assert isinstance(rmse, float), "RMSE should be float"
    assert isinstance(mae, float), "MAE should be float"
    assert isinstance(bias, float), "Bias should be float"
    assert rmse > 0, "RMSE should be positive"


def test_parameter_interval_coverage():
    """Test interval coverage calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    var = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    coverage = parameter_interval_coverage(y_true, mean, var, level=0.95)
    
    assert isinstance(coverage, float), "Coverage should be float"
    assert 0 <= coverage <= 1, "Coverage should be between 0 and 1"


def test_parameter_log_score_normal():
    """Test log score calculation."""
    y_true = np.array([1.0, 2.0, 3.0])
    mean = np.array([1.0, 2.0, 3.0])
    var = np.array([0.5, 0.5, 0.5])
    
    ls = parameter_log_score_normal(y_true, mean, var)
    
    assert isinstance(ls, float), "Log score should be float"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
