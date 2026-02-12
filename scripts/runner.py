#!/usr/bin/env python3
"""
Unified Structural Break Experiment Runner (Aligned Version)
============================================================
Handles Mean, Parameter, and Variance breaks with standardized DGPs and SARIMA forecasting.
Standard Configuration: T=400, Tb=200, n_sim=300.
Metrics: RMSE, MAE, Bias, Variance.

Features:
- Random forecast origin after the break.
- Support for Gaussian and Student-t (df=3, 5) innovations.
- Support for Single and Multiple (Recurring/MS) breaks across all categories.
- Persistence levels (0.90, 0.95, 0.99) for recurring scenarios.
- SARIMA-based estimators.
"""
import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

from protocols import validate_scenarios, calculate_metrics

# =========================================================
# 1) Standardized DGPs with Multi-Innovation Support
# =========================================================

def get_innovation(rng, size, dist="normal", nu=5, sigma=1.0):
    """Generate standardized innovations."""
    if dist.lower() == "normal":
        return rng.normal(0.0, sigma, size=size)
    elif dist.lower() == "student" or dist.lower() == "t":
        # Standardize Student-t such that its variance is sigma^2
        # Var(t_nu) = nu / (nu - 2).
        if nu <= 2:
            raise ValueError("Degrees of freedom nu must be > 2 for finite variance.")
        scale = sigma / np.sqrt(nu / (nu - 2))
        return rng.standard_t(nu, size=size) * scale
    else:
        return rng.normal(0.0, sigma, size=size)


def simulate_break_single(T=400, Tb=200, task="mean", val0=0.0, val1=1.0, phi=0.6, mu=0.0, sigma=1.0, dist="normal", nu=5, rng=None):
    """
    Unified Single Break DGP.
    - task='mean': intercept switches from val0 to val1
    - task='parameter': AR(1) phi switches from val0 to val1
    - task='variance': innovation sigma switches from val0 to val1
    """
    if rng is None: rng = np.random.default_rng()
    y = np.zeros(T)
    # We pre-generate eps to handle variance breaks correctly
    eps = np.zeros(T)
    
    # Base params
    curr_mu = mu
    curr_phi = phi
    curr_sigma = sigma

    for t in range(1, T):
        # Update param based on break
        if task == "mean":
            curr_mu = val0 if t <= Tb else val1
        elif task == "parameter":
            curr_phi = val0 if t <= Tb else val1
        elif task == "variance":
            curr_sigma = val0 if t <= Tb else val1
        
        err = get_innovation(rng, 1, dist=dist, nu=nu, sigma=curr_sigma)[0]
        y[t] = curr_mu + curr_phi * (y[t-1] - curr_mu) + err
    return y


def simulate_break_recurring(T=400, task="mean", val0=0.0, val1=1.0, p00=0.95, p11=0.95, phi=0.6, mu=0.0, sigma=1.0, dist="normal", nu=5, rng=None):
    """
    Unified Recurring (Markov-Switching) DGP.
    - Persistence levels p01, p11 define the regime switching.
    """
    if rng is None: rng = np.random.default_rng()
    y = np.zeros(T)
    s = np.zeros(T, dtype=int)
    s[0] = rng.integers(0, 2)
    
    # Base params
    curr_mu = mu
    curr_phi = phi
    curr_sigma = sigma

    for t in range(1, T):
        # State transition
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0
            
        # Param mapping
        if task == "mean":
            curr_mu = val0 if s[t] == 0 else val1
        elif task == "parameter":
            curr_phi = val0 if s[t] == 0 else val1
        elif task == "variance":
            curr_sigma = val0 if s[t] == 0 else val1

        err = get_innovation(rng, 1, dist=dist, nu=nu, sigma=curr_sigma)[0]
        y[t] = curr_mu + curr_phi * (y[t-1] - curr_mu) + err
    return y, s

# =========================================================
# 2) Unified MC Engine
# =========================================================

def mc_unified(n_sim=300, T=400, window=100, horizon=1, task="mean", scenarios=None, seed=42):
    """
    Aligned Monte Carlo Engine.
    """
    rng = np.random.default_rng(seed)
    scenarios = validate_scenarios(scenarios, T)
    all_results = []
    
    # Local imports to avoid circular deps and isolate SARIMA logic
    from estimators.variance import (
        forecast_variance_dist_sarima_global, 
        forecast_variance_dist_sarima_rolling,
        forecast_garch_variance
    )
    from estimators.parameter_single import (
        forecast_markov_switching_ar
    )

    for sc in scenarios:
        name = sc["name"]
        task_type = sc.get("task", task)
        is_recurring = "p" in sc
        dist = sc.get("distribution", "normal")
        nu = sc.get("nu", 5)
        
        # Method selection
        # Default methods: SARIMA Global, SARIMA Rolling, plus specific ones
        methods = {
            "SARIMA Global": lambda ytr: forecast_variance_dist_sarima_global(ytr, horizon=horizon)[0],
            "SARIMA Rolling": lambda ytr: forecast_variance_dist_sarima_rolling(ytr, window=window, horizon=horizon)[0],
        }
        
        if task_type == "parameter":
            methods["MS-AR"] = lambda ytr: forecast_markov_switching_ar(ytr)
        elif task_type == "variance":
            try:
                methods["GARCH"] = lambda ytr: forecast_garch_variance(ytr, horizon=horizon)[0]
            except Exception:
                pass
        elif task_type == "mean":
            methods["MS-AR"] = lambda ytr: forecast_markov_switching_ar(ytr)

        errors = {m: [] for m in methods}
        
        for _ in range(n_sim):
            # DGP selection
            if is_recurring:
                y, _ = simulate_break_recurring(
                    T=T, task=task_type, 
                    val0=sc.get("val0", 0.0), val1=sc.get("val1", 2.0),
                    p00=sc["p"], p11=sc["p"],
                    phi=sc.get("phi", 0.6), mu=sc.get("mu", 0.0), sigma=sc.get("sigma", 1.0),
                    dist=dist, nu=nu, rng=rng
                )
                Tb_ref = T // 4 # heuristic origin minimum
            else:
                Tb = sc.get("Tb", sc.get("variance_Tb", T//2))
                y = simulate_break_single(
                    T=T, Tb=Tb, task=task_type,
                    val0=sc.get("val0", sc.get("mu0", sc.get("phi1", sc.get("variance_sigma1", 0.0)))),
                    val1=sc.get("val1", sc.get("mu1", sc.get("phi2", sc.get("variance_sigma2", 1.0)))),
                    phi=sc.get("phi", 0.6), mu=sc.get("mu", 0.0), sigma=sc.get("sigma", 1.0),
                    dist=dist, nu=nu, rng=rng
                )
                Tb_ref = Tb

            # Random origin randomization (MUST BE AFTER BREAK Tb)
            # Ensure at least 'horizon' points remaining
            lower = int(Tb_ref + 10)
            upper = int(T - horizon - 2)
            if lower >= upper:
                t_orig = T - horizon - 2
            else:
                t_orig = int(rng.integers(lower, upper))
                
            y_train = y[:t_orig]
            y_true = float(y[t_orig]) if horizon == 1 else y[t_orig:t_orig+horizon]

            for mname, mfunc in methods.items():
                try:
                    pred = mfunc(y_train)
                    e = y_true - pred
                    errors[mname].append(e[0] if isinstance(e, (np.ndarray, list)) else e)
                except Exception:
                    errors[mname].append(np.nan)

        for mname, errs in errors.items():
            m = calculate_metrics(errs)
            all_results.append({
                "Scenario": name,
                "Task": task_type,
                "Break": "Recurring" if is_recurring else "Single",
                "Dist": dist,
                "Nu": nu if dist == "student" else "-",
                "Method": mname,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"],
                "Variance": m["Variance"]
            })

    return pd.DataFrame(all_results)


# =========================================================
# 3) Backward Compatibility Wrappers
# =========================================================

def mc_variance_breaks(n_sim=300, T=400, phi=0.6, window=100, horizon=20, scenarios=None, seed=42):
    df = mc_unified(n_sim=n_sim, T=T, window=window, horizon=horizon, task="variance", scenarios=scenarios, seed=seed)
    return df, df

def mc_mean_breaks(n_sim=300, T=400, Tb=200, window=60, seed=123, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, horizon=1, **kwargs):
    sc = [{"name": "Mean Break", "task": "mean", "Tb": Tb, "val0": mu0, "val1": mu1, "phi": phi, "sigma": sigma}]
    return mc_unified(n_sim=n_sim, T=T, window=window, horizon=horizon, task="mean", scenarios=sc, seed=seed)

# =========================================================
# 4) Main Execution
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='Aligned Structural Break Experiment Runner')
    parser.add_argument('--quick', action='store_true', help='Short run for verification')
    parser.add_argument('--n-sim', type=int, default=300)
    args = parser.parse_args()
    
    n_sim = 10 if args.quick else args.n_sim
    T = 150 if args.quick else 400
    
    tasks = ["mean", "parameter", "variance"]
    dists = [("normal", 5), ("student", 3), ("student", 5)]
    ps = [0.90, 0.95, 0.99]

    final_dfs = []
    
    for task in tasks:
        scenarios = []
        # Single breaks with diff innovations
        for d, nu in dists:
            if task == "mean":
                sc = {"name": f"Mean Single ({d}{nu if d=='student' else ''})", "task": "mean", "Tb": T//2, "val0": 0.0, "val1": 2.0, "distribution": d, "nu": nu}
            elif task == "parameter":
                sc = {"name": f"Param Single ({d}{nu if d=='student' else ''})", "task": "parameter", "Tb": T//2, "val0": 0.2, "val1": 0.9, "distribution": d, "nu": nu}
            else: # variance
                sc = {"name": f"Var Single ({d}{nu if d=='student' else ''})", "task": "variance", "Tb": T//2, "val0": 1.0, "val1": 2.0, "distribution": d, "nu": nu}
            scenarios.append(sc)
            
        # Recurring breaks with diff persistence
        for p in ps:
            if task == "mean":
                sc = {"name": f"Mean Recurring (p={p})", "task": "mean", "p": p, "val0": 0.0, "val1": 2.0}
            elif task == "parameter":
                sc = {"name": f"Param Recurring (p={p})", "task": "parameter", "p": p, "val0": 0.2, "val1": 0.9}
            else:
                sc = {"name": f"Var Recurring (p={p})", "task": "variance", "p": p, "val0": 1.0, "val1": 2.0}
            scenarios.append(sc)

        print(f"--- Running {task.upper()} experiments ---")
        df = mc_unified(n_sim=n_sim, T=T, task=task, scenarios=scenarios)
        final_dfs.append(df)
        print(df.round(4).to_string(index=False))
        print("\n")

    full_df = pd.concat(final_dfs)
    os.makedirs("results", exist_ok=True)
    full_df.to_csv(f"results/aligned_results_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    print(f"All results saved to results/aligned_results_{datetime.now().strftime('%Y%m%d')}.csv")

if __name__ == '__main__':
    main()
