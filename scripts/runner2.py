# =====================================================
# VERSION / EXECUTION CHECK (Parameter Single Break)
# =====================================================
print("RUNNING parameter_single_break.py (SARIMA + MS-AR, NaN-safe)")
print("FILE:", __file__)
print("=" * 60)

# =====================================================
# 6) RUN (Parameter Single Break)
# =====================================================
if __name__ == "__main__":

    cases = [
        ("Gaussian", "normal", None),
        ("Student-t df=5", "student", 5),
        ("Student-t df=3", "student", 3),
    ]

    all_err = {}
    rows = []

    print("\n=== SINGLE BREAK: POST-BREAK FORECAST EXPERIMENT ===\n")

    for label, innov, df in cases:
        start = time.time()

        err = monte_carlo_single_break_post(
            innovation=innov,
            df=df
        )

        elapsed = time.time() - start
        print(f"Finished {label} in {elapsed:.2f} seconds\n")

        all_err[label] = err

        for model, e in err.items():
            m = metrics(e)
            rows.append({
                "Scenario": "Single break",
                "Innovation": label,
                "Model": model,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"]
            })

    df_results = pd.DataFrame(rows)

    print("\nPOST-BREAK FORECAST RESULTS (SINGLE BREAK)\n")
    print(df_results.to_string(index=False))

    plot_combined_distributions(all_err)
    plot_rmse_by_innovation(df_results)
    plot_single_break_dgp()

# =====================================================
# 8) RUN (Parameter Recurring Breaks)
# =====================================================
if __name__ == "__main__":

    persistence_levels = [0.90, 0.95, 0.97, 0.995]
    rows = []
    err_by_p = {}

    for p in persistence_levels:
        print(f"Running persistence p = {p}")
        err = monte_carlo_recurring(p=p)
        err_by_p[p] = err

        for model, e in err.items():
            m = metrics(e)
            rows.append({
                "Persistence": p,
                "Model": model,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"]
            })

    df_results = pd.DataFrame(rows)

    print("\nForecast Performance — Recurring Instability (Gaussian)\n")
    print(df_results)

    plot_error_distributions_all(err_by_p, persistence_levels)
    plot_metric_bars(df_results, "RMSE")
    plot_metric_bars(df_results, "MAE")
    plot_metric_bars(df_results, "Bias")
    plot_dgp_by_persistence(persistence_levels)

# =========================================================
# 4) RUN BOTH + COMBINE + PRINT (Mean Break Multiple)
# =========================================================
single = mc_single_sarima()
multi  = mc_multiple_sarima()
all_results = pd.concat([single, multi], ignore_index=True)

print("\n=== COMPARISON TABLE (SARIMA single vs SARIMA multiple) ===")
print(all_results.sort_values(["Scenario","RMSE"], na_position="last").to_string(index=False))

for scen in ["Single break", "Multiple breaks"]:
    sub = all_results[(all_results["Scenario"]==scen) & (~all_results["RMSE"].isna())].sort_values("RMSE")
    if len(sub)>0:
        best = sub.iloc[0]
        print(f"\nBest method for {scen}: {best['Method']} (RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f})")

# =========================================================
# 4) RUN + RESULTS (Mean Break Single)
# =========================================================
results = run_mc_single_break_sarima(
    n_sim=200,
    T=300,
    Tb=150,
    window=60,
    seed=123,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    s=12,
    A=1.0,
    gap_after_break=20,
    order=(1,0,0),
    seasonal_order=(1,0,0,12),
    trim=0.15
)

print("\nSingle-break SARIMA-based results (lower RMSE = better):")
print(results.to_string(index=False))

best = results.dropna(subset=["RMSE"]).head(1)
if len(best) > 0:
    print(f"\nConclusion: Best method (lowest RMSE) = {best.iloc[0]['Method']}")
else:
    print("\nConclusion: All methods failed.")
