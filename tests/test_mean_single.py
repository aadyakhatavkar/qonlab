"""
Tests for Mean Single Break
============================
Basic tests for mean single break DGP and estimators.
"""

import numpy as np
import pytest

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.mean_singlebreaks import simulate_single_break_ar1
from estimators.mean_singlebreak import (
    forecast_sarima_global,
    forecast_sarima_rolling,
)


def test_simulate_mean_break_reproducible():
    """Test that DGP is reproducible with same rng."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    y1 = simulate_single_break_ar1(T=60, Tb=30, mu0=0.0, mu1=2.0, rng=rng1)
    y2 = simulate_single_break_ar1(T=60, Tb=30, mu0=0.0, mu1=2.0, rng=rng2)
    assert np.allclose(y1, y2), "Mean DGP not reproducible with same rng"


def test_simulate_mean_break_shape():
    """Test that DGP returns correct shape."""
    y = simulate_single_break_ar1(T=100, Tb=50, mu0=0.0, mu1=1.5)
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"


def test_forecast_sarima_global_returns_values():
    """Test SARIMA global forecaster returns a numeric value."""
    rng = np.random.default_rng(1)
    y = simulate_single_break_ar1(T=120, Tb=60, rng=rng)
    y_train = y[:-5]
    
    result = forecast_sarima_global(y_train)
    assert isinstance(result, (int, float, np.floating)) or np.isnan(result)


def test_forecast_sarima_rolling_returns_values():
    """Test SARIMA rolling forecaster returns a numeric value."""
    rng = np.random.default_rng(1)
    y = simulate_single_break_ar1(T=120, Tb=60, rng=rng)
    y_train = y[:-5]
    
    result = forecast_sarima_rolling(y_train, window=50)
    assert isinstance(result, (int, float, np.floating)) or np.isnan(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
