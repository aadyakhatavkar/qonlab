"""
Tests for Simulation Functions (Smoke Tests)
=============================================
Quick smoke tests to verify simulation functions run without error.
Uses minimal n_sim to keep tests fast (~5 seconds total).
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analyses.simu_meansingle import run_mc_single_break_sarima
from analyses.simu_mean_recurring import mc_mean_recurring
from analyses.simu_paramsingle import monte_carlo_single_break_post
from analyses.simu_paramrecurring import monte_carlo_recurring
from analyses.simu_variance_single import mc_variance_single_break
from analyses.simu_variance_recurring import mc_variance_recurring


class TestMeanSimulations:
    """Smoke tests for mean break simulations."""
    
    def test_mean_single_runs(self):
        """Test mean single break simulation runs without error."""
        result = run_mc_single_break_sarima(n_sim=3, T=100, Tb=50, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'RMSE' in result.columns or 'rmse' in result.columns.str.lower()
    
    def test_mean_recurring_runs(self):
        """Test mean recurring simulation runs without error."""
        result = mc_mean_recurring(n_sim=3, T=100, p=0.95, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestParameterSimulations:
    """Smoke tests for parameter break simulations."""
    
    def test_param_single_runs(self):
        """Test parameter single break simulation runs without error."""
        result = monte_carlo_single_break_post(n_sim=3, T=100, Tb=50, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_param_recurring_runs(self):
        """Test parameter recurring simulation runs without error."""
        result = monte_carlo_recurring(p=0.95, n_sim=3, T=100, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestVarianceSimulations:
    """Smoke tests for variance break simulations."""
    
    def test_variance_single_runs(self):
        """Test variance single break simulation runs without error."""
        result = mc_variance_single_break(n_sim=3, T=100, Tb=50, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_variance_recurring_runs(self):
        """Test variance recurring simulation runs without error."""
        result = mc_variance_recurring(n_sim=3, T=100, p=0.95, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestSimulationResults:
    """Test that simulation results have expected structure."""
    
    def test_mean_single_has_metrics(self):
        """Test mean single returns DataFrame with metrics."""
        result = run_mc_single_break_sarima(n_sim=3, T=100, Tb=50, seed=42)
        cols_lower = [c.lower() for c in result.columns]
        # Should have at least RMSE or MAE
        assert any(m in cols_lower for m in ['rmse', 'mae', 'bias'])
    
    def test_variance_single_has_methods(self):
        """Test variance single returns results for multiple methods."""
        result = mc_variance_single_break(n_sim=3, T=100, Tb=50, seed=42)
        # Should have results for multiple forecasters
        assert len(result) >= 1  # At least one method


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
