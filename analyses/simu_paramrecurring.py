import numpy as np
import pandas as pd
from pathlib import Path
from dgps.parameter_recurring import simulate_ms_ar1_phi_only
from estimators.parameter_recurring import (
    forecast_global_sarima,
    forecast_rolling_sarima,
    forecast_markov_switching_ar,
)
from analyses.metrics import rmse, mae, bias, var_error, coverage_from_errors, logscore_from_errors

# =====================================================
# 4) Monte Carlo experiment (recurring breaks)
# =====================================================
def monte_carlo_recurring(
    p,
    n_sim=300,
    T=400,
    window=70,
    seed=123
):
    """
    Monte Carlo evaluation for Markov-switching parameter breaks with random forecast origin.
    
    Parameters:
        p: Persistence level (probability of staying in same regime)
           - p=0.90: Frequent regime switches
           - p=0.95: Moderate persistence
           - p=0.99: High persistence (stable regimes)
        n_sim: Number of Monte Carlo replications
        T: Total time series length
        window: Rolling window size
        seed: Random seed
    
    Returns:
        DataFrame with RMSE, MAE, Bias, Variance for each method
    """
    rng = np.random.default_rng(seed)

    err = {
        "Global SARIMA": [],
        "Rolling SARIMA": [],
        "MS AR": []
    }

    for _ in range(n_sim):
        y, _ = simulate_ms_ar1_phi_only(
            T=T,
            persistence=p,
            rng=rng
        )

        # Choose random forecast origin (allow sufficient data for training)
        t0_random = rng.integers(max(T // 4, 50), T - 1)
        y_train = y[:t0_random]
        y_true = y[t0_random]

        err["Global SARIMA"].append(
            y_true - forecast_global_sarima(y_train)
        )
        err["Rolling SARIMA"].append(
            y_true - forecast_rolling_sarima(y_train, window)
        )
        err["MS AR"].append(
            y_true - forecast_markov_switching_ar(y_train)
        )

    # Convert to results DataFrame
    rows = []
    for method_name in err:
        e = np.asarray(err[method_name], dtype=float)
        e = e[~np.isnan(e)]
        n_success = len(e)
        
        rows.append({
            "Method": method_name,
            "RMSE": rmse(e),
            "MAE": mae(e),
            "Bias": bias(e),
            "Var(error)": var_error(e),
            "Successes": n_success,
            "Failures": n_sim - n_success,
            "N": n_sim
        })
    
    df = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    
    # Save to outputs/
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs_dir / f"parameter_recurring_p{p}_results.csv", index=False)
    
    return df