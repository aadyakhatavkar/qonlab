import numpy as np
import pytest

import sys
import os
# ensure package import from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from dgps.static import simulate_variance_break
from estimators.ols_like import forecast_garch_variance, rmse_mae_bias
import estimators.ols_like as ols_mod


def test_simulate_variance_break_seed():
    y1 = simulate_variance_break(T=60, Tb=30, seed=123)
    y2 = simulate_variance_break(T=60, Tb=30, seed=123)
    assert np.allclose(y1, y2)


def test_forecast_garch_shapes():
    if getattr(ols_mod, 'arch_model', None) is None:
        pytest.skip('arch not installed')
    y = simulate_variance_break(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    mean, var = forecast_garch_variance(y_train, horizon=3)
    assert mean.shape == (3,)
    assert var.shape == (3,)


def test_mc_variance_breaks_runs_quick():
    scenarios = [{'name':'test','Tb':40,'sigma1':1.0,'sigma2':1.5}]
    from analyses.mc import mc_variance_breaks
    pg, pu = mc_variance_breaks(n_sim=2, T=80, phi=0.3, window=20, horizon=5, scenarios=scenarios)
    assert not pg.empty
    assert not pu.empty
