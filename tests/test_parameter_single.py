"""
Tests for Parameter Single Break
================================
Basic tests for parameter single break DGP and estimators.
"""

import numpy as np
import pytest

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.parameter_single import simulate_single_break_ar1
from estimators.parameter_single import (
    forecast_global_sarima,
    forecast_rolling_sarima,
)


def test_simulate_param_break_reproducible():
    """Test that DGP is reproducible with same rng."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    y1 = simulate_single_break_ar1(T=60, Tb=30, phi1=0.3, phi2=0.9, rng=rng1)
    y2 = simulate_single_break_ar1(T=60, Tb=30, phi1=0.3, phi2=0.9, rng=rng2)
    assert np.allclose(y1, y2), "Parameter DGP not reproducible with same rng"


def test_simulate_param_break_shape():
    """Test that DGP returns correct shape."""
    y = simulate_single_break_ar1(T=100, Tb=50, phi1=0.3, phi2=0.8)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_forecast_global_sarima_returns_values():
    """Test SARIMA global forecaster returns a numeric value."""
    rng = np.random.default_rng(1)
    y = simulate_single_break_ar1(T=120, Tb=60, phi1=0.3, phi2=0.8, rng=rng)
    y_train = y[:-5]
    
    result = forecast_global_sarima(y_train)
    assert isinstance(result, (int, float, np.floating)) or np.isnan(result)


def test_forecast_rolling_sarima_returns_values():
    """Test SARIMA rolling forecaster returns a numeric value."""
    rng = np.random.default_rng(1)
    y = simulate_single_break_ar1(T=120, Tb=60, phi1=0.3, phi2=0.8, rng=rng)
    y_train = y[:-5]
    
    result = forecast_rolling_sarima(y_train, window=50)
    assert isinstance(result, (int, float, np.floating)) or np.isnan(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
