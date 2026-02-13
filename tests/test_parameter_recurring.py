"""
Tests for Parameter Recurring (Markov-Switching) Break
=======================================================
Tests DGP reproducibility, estimator outputs for parameter recurring breaks.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.parameter_recurring import simulate_ms_ar1_phi_only
from estimators.parameter_recurring import (
    forecast_markov_switching_ar,
)
from estimators.parameter_single import forecast_global_sarima


def test_simulate_ms_parameter_seed():
    """Test that Markov-switching DGP is reproducible with seed."""
    y1, _ = simulate_ms_ar1_phi_only(T=60, p00=0.95, p11=0.95, seed=123)
    y2, _ = simulate_ms_ar1_phi_only(T=60, p00=0.95, p11=0.95, seed=123)
    assert np.allclose(y1, y2), "MS parameter DGP not reproducible with same seed"


def test_simulate_ms_parameter_shape():
    """Test that Markov-switching DGP returns correct shapes."""
    y, states = simulate_ms_ar1_phi_only(T=100, p00=0.95, p11=0.95)
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    assert states.shape == (100,), f"Expected states shape (100,), got {states.shape}"


def test_simulate_ms_parameter_states():
    """Test that state sequence is binary (0 or 1)."""
    y, states = simulate_ms_ar1_phi_only(T=200, p00=0.95, p11=0.95, seed=42)
    
    unique_states = np.unique(states)
    assert set(unique_states).issubset({0, 1}), "States should be 0 or 1"
    assert len(unique_states) == 2, "Should have both states 0 and 1"


def test_simulate_ms_parameter_persistence():
    """Test that high persistence shows persistent states."""
    y_low, states_low = simulate_ms_ar1_phi_only(T=200, p00=0.70, p11=0.70, seed=42)
    y_high, states_high = simulate_ms_ar1_phi_only(T=200, p00=0.99, p11=0.99, seed=42)
    
    # Count state changes
    changes_low = np.sum(np.diff(states_low) != 0)
    changes_high = np.sum(np.diff(states_high) != 0)
    
    # Higher persistence should have fewer changes
    assert changes_low > changes_high, "Higher persistence should have fewer state changes"


def test_forecast_markov_switching_ar_shapes():
    """Test MS AR forecaster output shapes."""
    y, _ = simulate_ms_ar1_phi_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_markov_switching_ar(y_train, horizon=3)
    assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
    assert var.shape == (3,), f"Expected var shape (3,), got {var.shape}"


def test_forecast_markov_switching_ar_finite():
    """Test that MS AR forecaster produces finite outputs."""
    y, _ = simulate_ms_ar1_phi_only(T=150, p00=0.95, p11=0.95, seed=1)
    y_train = y[:-5]
    
    mean, var = forecast_markov_switching_ar(y_train, horizon=5)
    
    assert np.all(np.isfinite(mean)), "Forecasted means contain NaN/Inf"
    assert np.all(np.isfinite(var)), "Forecasted variances contain NaN/Inf"


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


def test_ms_vs_sarima_on_recurring_parameter():
    """Test that forecasters can handle recurring parameter breaks."""
    # Generate MS data
    y, _ = simulate_ms_ar1_phi_only(T=200, p00=0.95, p11=0.95, phi0=0.2, phi1=0.9, seed=123)
    y_train = y[:150]
    y_test = y[150:160]
    
    # Both forecasters should work
    try:
        mean_ms, var_ms = forecast_markov_switching_ar(y_train, horizon=len(y_test))
        assert mean_ms.shape == (len(y_test),), "MS forecaster shape mismatch"
    except Exception as e:
        pytest.fail(f"MS forecaster failed: {e}")
    
    try:
        mean_sarima, var_sarima = forecast_parameter_sarima_global(y_train, horizon=len(y_test))
        assert mean_sarima.shape == (len(y_test),), "SARIMA forecaster shape mismatch"
    except Exception as e:
        pytest.fail(f"SARIMA forecaster failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
