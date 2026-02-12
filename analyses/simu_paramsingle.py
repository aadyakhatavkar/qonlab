import numpy as np
import pandas as pd
from dgps.parameter_single import simulate_single_break_ar1
from estimators.parameter_single import (
    forecast_global_sarima,
    forecast_rolling_sarima,
    forecast_markov_switching_ar,
)
from protocols import calculate_metrics

# =====================================================
# 4) Monte Carlo — POST-BREAK ONLY
# =====================================================
def monte_carlo_single_break_post(
    n_sim=300,
    T=400,
    Tb=200,
    t_post=250,
    window=80,
    innovation_type='gaussian',
    dof=None,
    seed=123
):
    """
    Monte Carlo evaluation for single parameter break at forecast origin t_post (after Tb).
    
    Parameters:
        n_sim: Number of Monte Carlo replications
        T: Total time series length
        Tb: Structural break point
        t_post: Forecast origin (1-step ahead forecast target is at t_post)
        window: Rolling window size
        innovation_type: 'gaussian' or 'student' (Student-t innovations)
        dof: Degrees of freedom for Student-t (required if innovation_type='student')
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

    for i in range(n_sim):
        y = simulate_single_break_ar1(
            T=T,
            Tb=Tb,
            innovation_type=innovation_type,
            dof=dof,
            rng=rng
        )

        y_train = y[:t_post]
        y_true = y[t_post]

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
        
        if len(e) == 0:
            metrics = {"RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "Variance": np.nan}
        else:
            from protocols import calculate_metrics
            metrics = calculate_metrics(e)
        
        rows.append({
            "Method": method_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Bias": metrics["Bias"],
            "Variance": metrics["Variance"],
        })
    
    import pandas as pd
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
