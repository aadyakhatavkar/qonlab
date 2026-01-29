"""
Tests for mean break simulations
================================
Tests DGP reproducibility, estimator outputs, and MC runner execution
for mean break forecasting methods.
"""

import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgps.mean import simulate_mean_break_ar1, simulate_mean_break_ar1_seasonal
from estimators.mean import (
    mean_forecast_global_arma,
    mean_forecast_rolling_arma,
    mean_forecast_ar1_with_break_dummy_oracle,
    mean_forecast_ar1_with_estimated_break,
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_sarima_with_break_dummy,
    forecast_sarima_with_estimated_break,
)
from analyses.mean_simulations import mc_mean_breaks, mc_mean_breaks_seasonal


def test_simulate_mean_break_seed():
    """Test that DGP is reproducible with seed."""
    y1 = simulate_mean_break_ar1(T=100, Tb=50, seed=123)
    y2 = simulate_mean_break_ar1(T=100, Tb=50, seed=123)
    assert np.allclose(y1, y2), "DGP not reproducible with same seed"


def test_simulate_mean_break_seasonal_seed():
    """Test that seasonal DGP is reproducible with seed."""
    y1 = simulate_mean_break_ar1_seasonal(T=120, Tb=60, seed=456, s=12)
    y2 = simulate_mean_break_ar1_seasonal(T=120, Tb=60, seed=456, s=12)
    assert np.allclose(y1, y2), "Seasonal DGP not reproducible with same seed"


def test_mean_forecast_shapes():
    """Test that mean forecasters produce scalar outputs."""
    y_train = simulate_mean_break_ar1(T=80, Tb=40, seed=42)
    
    # Test global ARMA
    forecast_global = mean_forecast_global_arma(y_train)
    assert isinstance(forecast_global, (float, np.floating)), f"Expected scalar, got {type(forecast_global)}"
    
    # Test rolling ARMA
    forecast_rolling = mean_forecast_rolling_arma(y_train, window=40)
    assert isinstance(forecast_rolling, (float, np.floating)), f"Expected scalar, got {type(forecast_rolling)}"
    
    # Test break dummy
    forecast_dummy = mean_forecast_ar1_with_break_dummy_oracle(y_train, Tb=40)
    assert isinstance(forecast_dummy, (float, np.floating)), f"Expected scalar, got {type(forecast_dummy)}"
    
    # Test estimated break
    forecast_est = mean_forecast_ar1_with_estimated_break(y_train, trim=0.15)
    assert isinstance(forecast_est, (float, np.floating)), f"Expected scalar, got {type(forecast_est)}"


def test_mean_forecast_sarima_shapes():
    """Test that SARIMA forecasters produce scalar outputs."""
    y_train = simulate_mean_break_ar1_seasonal(T=100, Tb=50, seed=99, s=12)
    
    # Test SARIMA global
    f_global = forecast_sarima_global(y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    assert isinstance(f_global, (float, np.floating)), "SARIMA global should be scalar"
    
    # Test SARIMA rolling
    f_rolling = forecast_sarima_rolling(y_train, window=40, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    assert isinstance(f_rolling, (float, np.floating)), "SARIMA rolling should be scalar"
    
    # Test SARIMA with break dummy
    f_dummy = forecast_sarima_with_break_dummy(y_train, Tb=50, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    assert isinstance(f_dummy, (float, np.floating)), "SARIMA break dummy should be scalar"
    
    # Test SARIMA with estimated break
    f_est = forecast_sarima_with_estimated_break(y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), trim=0.15)
    assert isinstance(f_est, (float, np.floating)), "SARIMA estimated break should be scalar"


def test_mc_mean_breaks_runs_quick():
    """Test MC simulation runs without errors on minimal sample."""
    df = mc_mean_breaks(
        n_sim=2,
        T=50,
        Tb=25,
        window=15,
        seed=123,
        verbose=False
    )
    assert not df.empty, "MC results DataFrame is empty"
    assert "RMSE" in df.columns, "RMSE column missing"
    assert "MAE" in df.columns, "MAE column missing"
    assert len(df) > 0, "No methods in results"


def test_mc_mean_breaks_seasonal_runs_quick():
    """Test seasonal MC simulation runs without errors on minimal sample."""
    df = mc_mean_breaks_seasonal(
        n_sim=2,
        T=60,
        Tb=30,
        window=15,
        seed=456,
        verbose=False
    )
    assert not df.empty, "Seasonal MC results DataFrame is empty"
    assert "RMSE" in df.columns, "RMSE column missing"
    assert len(df) > 0, "No SARIMA methods in results"
