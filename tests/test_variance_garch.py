import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Variance Change'))
import variance_change as vc


def test_simulate_variance_break_seed():
    y1 = vc.simulate_variance_break(T=60, Tb=30, seed=123)
    y2 = vc.simulate_variance_break(T=60, Tb=30, seed=123)
    assert np.allclose(y1, y2)


@pytest.mark.skipif(vc.arch_model is None, reason="arch not installed")
def test_forecast_garch_shapes():
    y = vc.simulate_variance_break(T=120, Tb=60, seed=1)
    y_train = y[:-5]
    mean, var = vc.forecast_garch_variance(y_train, horizon=3)
    assert mean.shape == (3,)
    assert var.shape == (3,)


def test_mc_variance_breaks_runs_quick():
    scenarios = [{'name':'test','Tb':40,'sigma1':1.0,'sigma2':1.5}]
    pg, pu = vc.mc_variance_breaks(n_sim=2, T=80, phi=0.3, window=20, horizon=5, scenarios=scenarios)
    assert not pg.empty
    assert not pu.empty
