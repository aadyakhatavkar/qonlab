"""
Tests for Mean Recurring Break
==============================
Basic tests for mean recurring (Markov-switching) DGP and estimators.
"""

import numpy as np
import pytest

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.mean_recurring import simulate_ms_ar1_mean_only
from estimators.mean_recurring import forecast_ms_ar1_mean


def test_simulate_ms_mean_reproducible():
    """Test that DGP is reproducible with same seed."""
    y1, s1 = simulate_ms_ar1_mean_only(T=100, p00=0.95, p11=0.95, seed=123)
    y2, s2 = simulate_ms_ar1_mean_only(T=100, p00=0.95, p11=0.95, seed=123)
    assert np.allclose(y1, y2), "Mean MS-DGP not reproducible with same seed"


def test_simulate_ms_mean_shape():
    """Test that DGP returns correct shapes."""
    y, s = simulate_ms_ar1_mean_only(T=100, p00=0.95, p11=0.95)
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    assert s.shape == (100,), f"Expected s shape (100,), got {s.shape}"


def test_simulate_ms_mean_states_valid():
    """Test that states are only 0 or 1."""
    y, s = simulate_ms_ar1_mean_only(T=200, p00=0.95, p11=0.95)
    assert set(np.unique(s)).issubset({0, 1}), "States should only be 0 or 1"


def test_forecast_ms_ar1_mean_returns_values():
    """Test MS AR1 mean forecaster returns a numeric value or array."""
    y, _ = simulate_ms_ar1_mean_only(T=150, p00=0.95, p11=0.95, seed=42)
    y_train = y[:-10]
    
    result = forecast_ms_ar1_mean(y_train, horizon=5)
    assert result is not None, "Forecast should return a result"
    if np.isscalar(result):
        assert isinstance(result, (int, float, np.floating)) or np.isnan(result)
    else:
        assert hasattr(result, '__len__'), "Result should be iterable or scalar"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
