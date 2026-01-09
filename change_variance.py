import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

def simulate_variance_break(
    T=400, Tb=200, phi=0.6, mu=0.0, sigma1=1.0, sigma2=2.0, seed=None
):
    """
    AR(1) with variance break:
      y_t = mu + phi*(y_{t-1}-mu) + eps_t
      eps_t ~ N(0, sigma1^2) for t < Tb, N(0, sigma2^2) for t >= Tb
    """
    # Validate inputs
    if not (1 <= Tb < T):
        raise ValueError(f"Tb must satisfy 1 <= Tb < T (got Tb={Tb}, T={T})")

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = np.zeros(T)

    eps[:Tb] = rng.normal(0.0, sigma1, size=Tb)
    eps[Tb:] = rng.normal(0.0, sigma2, size=T - Tb)

    for t in range(1, T):
        y[t] = mu + phi * (y[t - 1] - mu) + eps[t]
    return y


def forecast_dist_arima_global(y_train, horizon=1, order=(1, 0, 0)):
    """
    Returns mean forecast and forecast variance using statsmodels get_forecast.
    """
    res = ARIMA(y_train, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def forecast_dist_arima_rolling(y_train, window=100, horizon=1, order=(1, 0, 0)):
    """
    Fit on last window only (in the same spirit as Mahir's rolling).
    Returns mean forecast and forecast variance for horizon steps.
    """
    y_win = y_train[-window:]
    res = ARIMA(y_win, order=order).fit()
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    return mean, var


def rmse_mae_bias(y_true, y_pred):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias


def interval_coverage(y_true, mean, var, level=0.95):
    z = norm.ppf(0.5 + level / 2.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def log_score_normal(y_true, mean, var):
    var = np.maximum(var, 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)))


def mc_variance_breaks(
    n_sim=200,
    T=400,
    phi=0.6,
    window=100,
    horizon=20,
    scenarios=None,
    seed=42
):
    """
    Monte Carlo for variance breaks.
    Outputs:
      - Point metrics: RMSE/MAE/Bias
      - Uncertainty metrics: Coverage80/Coverage95/LogScore
    """
    rng = np.random.default_rng(seed)

    if scenarios is None:
        scenarios = [
            {"name": "Single variance break", "Tb": 200, "sigma1": 1.0, "sigma2": 2.0},
            {"name": "Multiple variance breaks (piecewise)", "Tb": 200, "sigma1": 1.0, "sigma2": 2.0},  # placeholder label
        ]

    point_rows = []
    unc_rows = []

    for sc in scenarios:
        name = sc["name"]
        Tb = sc["Tb"]
        sigma1 = sc["sigma1"]
        sigma2 = sc["sigma2"]

        point_g = []
        point_r = []
        unc_g = []
        unc_r = []

        for _ in range(n_sim):
            s = int(rng.integers(0, 1_000_000_000))
            y = simulate_variance_break(T=T, Tb=Tb, phi=phi, sigma1=sigma1, sigma2=sigma2, seed=s)

            # simple end-of-sample evaluation block
            y_train = y[:-horizon]
            y_test = y[-horizon:]

            mg, vg = forecast_dist_arima_global(y_train, horizon=horizon)
            mr, vr = forecast_dist_arima_rolling(y_train, window=window, horizon=horizon)

            point_g.append(rmse_mae_bias(y_test, mg))
            point_r.append(rmse_mae_bias(y_test, mr))

            unc_g.append((
                interval_coverage(y_test, mg, vg, 0.80),
                interval_coverage(y_test, mg, vg, 0.95),
                log_score_normal(y_test, mg, vg)
            ))
            unc_r.append((
                interval_coverage(y_test, mr, vr, 0.80),
                interval_coverage(y_test, mr, vr, 0.95),
                log_score_normal(y_test, mr, vr)
            ))

        pg = np.mean(np.array(point_g), axis=0)
        pr = np.mean(np.array(point_r), axis=0)
        ug = np.mean(np.array(unc_g), axis=0)
        ur = np.mean(np.array(unc_r), axis=0)

        for metric, idx in [("RMSE", 0), ("MAE", 1), ("Bias", 2)]:
            point_rows.append({
                "Scenario": name,
                "Metric": metric,
                "ARIMA Global": pg[idx],
                "ARIMA Rolling": pr[idx],
            })

        for metric, idx in [("Coverage80", 0), ("Coverage95", 1), ("LogScore", 2)]:
            unc_rows.append({
                "Scenario": name,
                "Metric": metric,
                "ARIMA Global": ug[idx],
                "ARIMA Rolling": ur[idx],
            })

    return pd.DataFrame(point_rows), pd.DataFrame(unc_rows)


def main():
    print("\n============================")
    print("MAHIR: PARAMETER BREAK")
    print("============================")
    rmse_g, rmse_r = mc_simulate()
    print("Global ARIMA RMSE:", rmse_g)
    print("Rolling ARIMA RMSE:", rmse_r)

    print("\n============================")
    print("AADYA: VARIANCE BREAK")
    print("============================")
    df_point, df_unc = mc_variance_breaks(
        n_sim=200,
        T=400,
        phi=0.6,
        window=100,
        horizon=20
    )

    print("\n=== VARIANCE BREAK: POINT METRICS ===")
    print(df_point.round(4).to_string(index=False))

    print("\n=== VARIANCE BREAK: UNCERTAINTY METRICS ===")
    print(df_unc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()