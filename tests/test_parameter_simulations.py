"""
Tests for parameter break simulations
=====================================
Tests DGP reproducibility, estimator outputs, and MC runner execution
for parameter break forecasting methods.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.parameter import simulate_parameter_break_ar1
from estimators.parameter import (
    param_forecast_global_arma,
    param_forecast_rolling_arma,
    param_forecast_markov_switching_ar,
    param_metrics,
)
from analyses.param_simulations import mc_parameter_breaks_post, mc_parameter_breaks_full


def test_simulate_parameter_break_seed():
    """Test that DGP is reproducible with seed."""
    y1 = simulate_parameter_break_ar1(T=100, Tb=50, seed=123)
    y2 = simulate_parameter_break_ar1(T=100, Tb=50, seed=123)
    assert np.allclose(y1, y2), "DGP not reproducible with same seed"


def test_param_forecast_shapes():
    """Test that parameter forecasters produce scalar outputs."""
    y_train = simulate_parameter_break_ar1(T=80, Tb=40, seed=42)
    
    # Test global ARMA
    forecast_global = param_forecast_global_arma(y_train)
    assert isinstance(forecast_global, (float, np.floating)), f"Expected scalar, got {type(forecast_global)}"
    
    # Test rolling ARMA
    forecast_rolling = param_forecast_rolling_arma(y_train, window=40)
    assert isinstance(forecast_rolling, (float, np.floating)), f"Expected scalar, got {type(forecast_rolling)}"
    
    # Test Markov Switching
    forecast_ms = param_forecast_markov_switching_ar(y_train)
    assert isinstance(forecast_ms, (float, np.floating)), f"Expected scalar, got {type(forecast_ms)}"


def test_mc_parameter_breaks_post_runs():
    """Test that post-break MC runner executes without errors on small sample."""
    np.random.seed(123)
    # Run minimal MC (2 simulations) to ensure function works
    results = mc_parameter_breaks_post(
        n_sim=2,
        T=50,
        Tb=25,
        t_post=40,
        window=20,
        seed=123,
        verbose=False
    )
    assert isinstance(results, dict), "Should return dict of results"
    assert "Global ARMA (auto)" in results, "Should have Global ARMA results"
    assert len(results) > 0, "Should have results for at least one method"


def test_mc_parameter_breaks_full_runs():
    """Test that full distribution MC runner executes without errors."""
    np.random.seed(123)
    # Run minimal MC (1 simulation) on small sample
    result = mc_parameter_breaks_full(
        n_sim=1,
        T=50,
        Tb=25,
        t_post=40,
        window=20,
        innovations=[("Normal", "normal", None)],
        seed=123,
        verbose=False
    )
    assert isinstance(result, tuple), "Should return tuple of (dict, dataframe)"
    assert len(result) == 2, "Should have 2 elements: dict and dataframe"
    assert isinstance(result[0], dict), "First element should be dict"
    assert "Normal" in result[0], "Should have Normal innovation results"


def test_parameter_student_t_distribution():
    """Test that parameter DGP handles Student-t innovations."""
    y = simulate_parameter_break_ar1(T=100, Tb=50, innovation="student", df=5, seed=42)
    assert len(y) == 100, "Should produce time series of correct length"
    assert not np.any(np.isnan(y)), "Should not produce NaN values"
    # Student-t should have heavier tails but still be finite
    assert np.isfinite(y).all(), "All values should be finite"
