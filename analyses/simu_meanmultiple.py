import numpy as np
import pandas as pd
from dgps import (
    simulate_single_break_with_seasonality,
    simulate_multiple_breaks_with_seasonality,
)
from estimators import (
    forecast_sarima_global,
    forecast_sarima_rolling,
    forecast_sarima_break_dummy_oracle_single,
    forecast_sarima_estimated_break_single,
    forecast_sarima_break_dummy_oracle_multiple,
    forecast_sarima_estimated_breaks_multiple,
    forecast_ses,
    forecast_holt_winters_seasonal,
)

# =========================================================
# 3) Monte Carlo runners: single vs multiple
# =========================================================
def mc_single_sarima(
    n_sim=200, T=300, Tb=150, window=60, seed=123,
    mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, s=12, A=1.0,
    gap_after_break=20,
    order=(1,0,0), seasonal_order=(1,0,0,12),
    trim=0.15
):
    rng = np.random.default_rng(seed)
    t0 = Tb + gap_after_break

    methods = [
        ("SARIMA Global", lambda ytr: forecast_sarima_global(ytr, order, seasonal_order)),
        ("SARIMA Rolling", lambda ytr: forecast_sarima_rolling(ytr, window, order, seasonal_order)),
        ("SARIMA + Break Dummy (oracle Tb)", lambda ytr: forecast_sarima_break_dummy_oracle_single(ytr, Tb, order, seasonal_order)),
        ("SARIMA + Estimated Break (grid)", lambda ytr: forecast_sarima_estimated_break_single(ytr, order, seasonal_order, trim=trim)),
        ("SES (level smoothing)", lambda ytr: forecast_ses(ytr)),
    ]

    errors = {m: [] for m,_ in methods}
    fails  = {m: 0  for m,_ in methods}

    for _ in range(n_sim):
        y = simulate_single_break_with_seasonality(T, Tb, mu0, mu1, phi, sigma, s, A, rng=rng)
        y_train, y_true = y[:t0], float(y[t0])
        for name, func in methods:
            try:
                f = func(y_train)
                errors[name].append(y_true - f)
            except Exception:
                fails[name] += 1

    rows = []
    for name in errors:
        e = np.asarray(errors[name], float)
        if len(e)==0:
            rows.append({"Method":name,"RMSE":np.nan,"MAE":np.nan,"Bias":np.nan,"N":0,"Fails":fails[name]})
        else:
            rows.append({
                "Method": name,
                "RMSE": float(np.sqrt(np.mean(e**2))),
                "MAE":  float(np.mean(np.abs(e))),
                "Bias": float(np.mean(e)),
                "N": int(len(e)),
                "Fails": fails[name],
            })
    res = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    res["Scenario"] = "Single break"
    return res

def mc_multiple_sarima(
    n_sim=200, T=300, b1=100, b2=200, window=60, seed=456,
    mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, s=12, A=1.0,
    gap_after_last_break=20,
    order=(1,0,0), seasonal_order=(1,0,0,12),
    trim=0.15, min_seg=25
):
    rng = np.random.default_rng(seed)
    t0 = b2 + gap_after_last_break

    methods = [
        ("SARIMA Global", lambda ytr: forecast_sarima_global(ytr, order, seasonal_order)),
        ("SARIMA Rolling", lambda ytr: forecast_sarima_rolling(ytr, window, order, seasonal_order)),
        ("SARIMA + 2 Break Dummies (oracle)", lambda ytr: forecast_sarima_break_dummy_oracle_multiple(ytr, b1, b2, order, seasonal_order)),
        ("SARIMA + 2 Breaks Estimated (grid)", lambda ytr: forecast_sarima_estimated_breaks_multiple(ytr, order, seasonal_order, trim=trim, min_seg=min_seg)),
        ("Holt-Winters Seasonal Smoothing", lambda ytr: forecast_holt_winters_seasonal(ytr, s=s)),
    ]

    errors = {m: [] for m,_ in methods}
    fails  = {m: 0  for m,_ in methods}

    for _ in range(n_sim):
        y = simulate_multiple_breaks_with_seasonality(T, b1, b2, mu0, mu1, mu2, phi, sigma, s, A, rng=rng)
        y_train, y_true = y[:t0], float(y[t0])
        for name, func in methods:
            try:
                f = func(y_train)
                errors[name].append(y_true - f)
            except Exception:
                fails[name] += 1

    rows = []
    for name in errors:
        e = np.asarray(errors[name], float)
        if len(e)==0:
            rows.append({"Method":name,"RMSE":np.nan,"MAE":np.nan,"Bias":np.nan,"N":0,"Fails":fails[name]})
        else:
            rows.append({
                "Method": name,
                "RMSE": float(np.sqrt(np.mean(e**2))),
                "MAE":  float(np.mean(np.abs(e))),
                "Bias": float(np.mean(e)),
                "N": int(len(e)),
                "Fails": fails[name],
            })
    res = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    res["Scenario"] = "Multiple breaks"
    return res

