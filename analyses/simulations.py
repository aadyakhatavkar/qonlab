import numpy as np
import pandas as pd
from scipy.stats import norm
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

from dgps.variance import simulate_variance_break_ar1
from dgps.mean import simulate_mean_break_ar1
from dgps.parameter import simulate_parameter_break_ar1
from dgps.utils import validate_scenarios
from estimators.forecasters import (
    forecast_variance_dist_arima_global,
    forecast_variance_dist_arima_rolling,
    forecast_garch_variance,
    forecast_variance_arima_post_break,
    variance_rmse_mae_bias,
    variance_interval_coverage,
    variance_log_score_normal,
)


def mc_variance_breaks_grid(
    n_sim=100,
    T=400,
    phi=0.6,
    horizon=20,
    window_sizes=[20, 50, 100, 200],
    break_magnitudes=[1.5, 2.0, 3.0, 5.0],
    seed=42
):
    rng = np.random.default_rng(seed)
    results = []
    variance_Tb = T // 2
    variance_sigma1 = 1.0

    for break_mag in break_magnitudes:
        variance_sigma2 = variance_sigma1 * break_mag

        for ws in window_sizes:
            rmse_list = []
            cov_list = []
            ls_list = []

            seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(n_sim)]

            for s in seeds:
                y = simulate_variance_break(T=T, variance_Tb=variance_Tb, phi=phi, variance_sigma1=variance_sigma1, variance_sigma2=variance_sigma2, seed=s)
                y_train = y[:-horizon]
                y_test = y[-horizon:]

                try:
                    m_roll, v_roll = forecast_variance_dist_arima_rolling(y_train, window=ws, horizon=horizon)
                    rmse, _, _ = variance_rmse_mae_bias(y_test, m_roll)
                    cov95 = variance_interval_coverage(y_test, m_roll, v_roll, level=0.95)
                    ls = variance_log_score_normal(y_test, m_roll, v_roll)

                    rmse_list.append(rmse)
                    cov_list.append(cov95)
                    ls_list.append(ls)
                except Exception:
                    pass

            if rmse_list:
                results.append({
                    'Window': ws,
                    'BreakMagnitude': break_mag,
                    'RMSE': np.mean(rmse_list),
                    'Coverage95': np.mean(cov_list),
                    'LogScore': np.mean(ls_list),
                    'N_Sims': len(rmse_list)
                })

    return pd.DataFrame(results)


def mc_variance_breaks(
    n_sim=200,
    T=400,
    phi=0.6,
    window=100,
    horizon=20,
    scenarios=None,
    seed=42
):
    rng = np.random.default_rng(seed)
    scenarios = validate_scenarios(scenarios, T)

    variance_point_rows = []
    variance_unc_rows = []

    for sc in scenarios:
        name = sc["name"]
        task = sc.get("task", "variance")

        v_point_g = []
        v_point_r = []
        v_point_garch = []
        v_point_pb = []
        v_unc_g = []
        v_unc_r = []
        v_unc_garch = []
        v_unc_pb = []

        seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(n_sim)]

        def _run_one(s):
            if task == "variance":
                y = simulate_variance_break_ar1(
                    T=T, Tb=sc["variance_Tb"], phi=phi, 
                    sigma1=sc["variance_sigma1"], sigma2=sc["variance_sigma2"], 
                    distribution=sc.get("distribution", "normal"), nu=sc.get("nu", 3), seed=s
                )
            elif task == "mean":
                y = simulate_mean_break_ar1(
                    T=T, Tb=sc["Tb"], mu0=sc["mu0"], mu1=sc["mu1"], 
                    phi=sc.get("phi", phi), sigma=sc.get("sigma", 1.0), seed=s
                )
            elif task == "parameter":
                y = simulate_parameter_break_ar1(
                    T=T, Tb=sc["Tb"], phi1=sc["phi1"], phi2=sc["phi2"], 
                    sigma=sc.get("sigma", 1.0), seed=s
                )
            else:
                raise ValueError(f"Unknown task: {task}")

            y_train = y[:-horizon]
            y_test = y[-horizon:]

            mg, vg = forecast_variance_dist_arima_global(y_train, horizon=horizon)
            mr, vr = forecast_variance_dist_arima_rolling(y_train, window=window, horizon=horizon)
            try:
                mgarch, vgarch = forecast_garch_variance(y_train, horizon=horizon)
            except Exception:
                mgarch = np.full(horizon, np.nan)
                vgarch = np.full(horizon, np.nan)
            
            try:
                mpb, vpb = forecast_variance_arima_post_break(y_train, horizon=horizon)
            except Exception:
                mpb = np.full(horizon, np.nan)
                vpb = np.full(horizon, np.nan)

            return (
                variance_rmse_mae_bias(y_test, mg),
                variance_rmse_mae_bias(y_test, mr),
                variance_rmse_mae_bias(y_test, mgarch),
                variance_rmse_mae_bias(y_test, mpb),
                (
                    variance_interval_coverage(y_test, mg, vg, 0.80),
                    variance_interval_coverage(y_test, mg, vg, 0.95),
                    variance_log_score_normal(y_test, mg, vg)
                ),
                (
                    variance_interval_coverage(y_test, mr, vr, 0.80),
                    variance_interval_coverage(y_test, mr, vr, 0.95),
                    variance_log_score_normal(y_test, mr, vr)
                ),
                (
                    variance_interval_coverage(y_test, mgarch, vgarch, 0.80),
                    variance_interval_coverage(y_test, mgarch, vgarch, 0.95),
                    variance_log_score_normal(y_test, mgarch, vgarch)
                ),
                (
                    variance_interval_coverage(y_test, mpb, vpb, 0.80),
                    variance_interval_coverage(y_test, mpb, vpb, 0.95),
                    variance_log_score_normal(y_test, mpb, vpb)
                ),
            )

        if Parallel is not None:
            results = Parallel(n_jobs=1)(delayed(_run_one)(s) for s in seeds)
        else:
            results = [_run_one(s) for s in seeds]

        for res in results:
            pg_val, pr_val, pgarch_val, ppb_val, ug_val, ur_val, ugarch_val, upb_val = res
            v_point_g.append(pg_val)
            v_point_r.append(pr_val)
            v_point_garch.append(pgarch_val)
            v_point_pb.append(ppb_val)
            v_unc_g.append(ug_val)
            v_unc_r.append(ur_val)
            v_unc_garch.append(ugarch_val)
            v_unc_pb.append(upb_val)

        v_pg = np.mean(np.array(v_point_g), axis=0) if v_point_g else np.zeros(3)
        v_pr = np.mean(np.array(v_point_r), axis=0) if v_point_r else np.zeros(3)
        v_pgarch = np.mean(np.array(v_point_garch), axis=0) if v_point_garch else np.zeros(3)
        v_ppb = np.mean(np.array(v_point_pb), axis=0) if v_point_pb else np.zeros(3)
        v_ug = np.mean(np.array(v_unc_g), axis=0) if v_unc_g else np.zeros(3)
        v_ur = np.mean(np.array(v_unc_r), axis=0) if v_unc_r else np.zeros(3)
        v_ugarch = np.mean(np.array(v_unc_garch), axis=0) if v_unc_garch else np.zeros(3)
        v_upb = np.mean(np.array(v_unc_pb), axis=0) if v_unc_pb else np.zeros(3)

        for metric, idx in [("RMSE", 0), ("MAE", 1), ("Bias", 2)]:
            variance_point_rows.append({
                "Scenario": name,
                "Metric": metric,
                "ARIMA Global": v_pg[idx],
                "ARIMA Rolling": v_pr[idx],
                "GARCH": v_pgarch[idx] if len(v_point_garch) > 0 else np.nan,
                "ARIMA PostBreak": v_ppb[idx] if len(v_point_pb) > 0 else np.nan,
            })

        for metric, idx in [("Coverage80", 0), ("Coverage95", 1), ("LogScore", 2)]:
            variance_unc_rows.append({
                "Scenario": name,
                "Metric": metric,
                "ARIMA Global": v_ug[idx],
                "ARIMA Rolling": v_ur[idx],
                "GARCH": v_ugarch[idx] if len(v_unc_garch) > 0 else np.nan,
                "ARIMA PostBreak": v_upb[idx] if len(v_unc_pb) > 0 else np.nan,
            })

    return pd.DataFrame(variance_point_rows), pd.DataFrame(variance_unc_rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run variance-break Monte Carlo experiments")
    parser.add_argument("--quick", action="store_true", help="run a short, fast simulation")
    parser.add_argument("--grid", action="store_true", help="run grid analysis for optimal window selection (Pesaran 2013)")
    parser.add_argument("--n-sim", type=int, default=200, help="number of Monte Carlo simulations")
    parser.add_argument("--T", type=int, default=400, help="sample size T")
    parser.add_argument("--phi", type=float, default=0.6, help="AR(1) coefficient")
    parser.add_argument("--window", type=int, default=100, help="rolling-window size (legacy mode)")
    parser.add_argument("--horizon", type=int, default=20, help="forecast horizon")
    args = parser.parse_args()

    if args.quick:
        n_sim = min(10, args.n_sim)
        T = min(100, args.T)
        window = min(20, args.window)
        horizon = min(5, args.horizon)
    else:
        n_sim = args.n_sim
        T = args.T
        window = args.window
        horizon = args.horizon

    if args.grid:
        window_sizes = [20, 50, 100, 200] if not args.quick else [20, 50]
        break_mags = [1.5, 2.0, 3.0, 5.0] if not args.quick else [1.5, 3.0]

        df_grid = mc_variance_breaks_grid(
            n_sim=n_sim,
            T=T,
            phi=args.phi,
            horizon=horizon,
            window_sizes=window_sizes,
            break_magnitudes=break_mags,
            seed=42
        )

        print("\nLOSS SURFACE: RMSE (lower is better)")
        print(df_grid.pivot(index='Window', columns='BreakMagnitude', values='RMSE').round(4).to_string())
        print("\nLOSS SURFACE: Coverage95 (closer to 0.95 is better)")
        print(df_grid.pivot(index='Window', columns='BreakMagnitude', values='Coverage95').round(4).to_string())
        print("\nLOSS SURFACE: LogScore (higher is better)")
        print(df_grid.pivot(index='Window', columns='BreakMagnitude', values='LogScore').round(4).to_string())

    else:
        df_point, df_unc = mc_variance_breaks(
            n_sim=n_sim,
            T=T,
            phi=args.phi,
            window=window,
            horizon=horizon,
        )

        print("\n=== VARIANCE BREAK: POINT METRICS ===")
        print(df_point.round(4).to_string(index=False))

        print("\n=== VARIANCE BREAK: UNCERTAINTY METRICS ===")
        print(df_unc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
