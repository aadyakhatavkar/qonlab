"""
Tests for Parameter Recurring Break
===================================
Basic tests for parameter recurring (Markov-switching) DGP and estimators.
"""

import numpy as np
import pytest

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.parameter_recurring import simulate_ms_ar1_phi_only
from estimators.parameter_recurring import forecast_markov_switching_ar


def test_simulate_ms_param_reproducible():
    """Test that DGP is reproducible with same rng."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    y1, s1 = simulate_ms_ar1_phi_only(T=100, persistence=0.95, rng=rng1)
    y2, s2 = simulate_ms_ar1_phi_only(T=100, persistence=0.95, rng=rng2)
    assert np.allclose(y1, y2), "Parameter MS-DGP not reproducible with same rng"


def test_simulate_ms_param_shape():
    """Test that DGP returns correct shapes."""
    y, s = simulate_ms_ar1_phi_only(T=100, persistence=0.95)
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    assert s.shape == (100,), f"Expected s shape (100,), got {s.shape}"


def test_simulate_ms_param_states_valid():
    """Test that states are only 0 or 1."""
    y, s = simulate_ms_ar1_phi_only(T=200, persistence=0.95)
    assert set(np.unique(s)).issubset({0, 1}), "States should only be 0 or 1"


def test_forecast_ms_ar_returns_values():
    """Test MS AR forecaster returns a numeric value."""
    y, _ = simulate_ms_ar1_phi_only(T=150, persistence=0.95, rng=np.random.default_rng(42))
    y_train = y[:-10]
    
    result = forecast_markov_switching_ar(y_train)
    assert isinstance(result, (int, float, np.floating)) or np.isnan(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
