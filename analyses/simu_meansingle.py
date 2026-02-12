# =========================================================
# 3) Monte Carlo evaluation
# =========================================================
def run_mc_single_break_sarima(
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
):
    rng = np.random.default_rng(seed)
    t0 = Tb + gap_after_break
    if t0 >= T:
        raise ValueError("gap_after_break too large for T.")

    methods = [
        ("SARIMA Global", lambda ytr: forecast_sarima_global(ytr, order=order, seasonal_order=seasonal_order)),
        ("SARIMA Rolling", lambda ytr: forecast_sarima_rolling(ytr, window=window, order=order, seasonal_order=seasonal_order)),
        ("SARIMA + Break Dummy (oracle Tb)", lambda ytr: forecast_sarima_break_dummy_oracle(ytr, Tb=Tb, order=order, seasonal_order=seasonal_order)),
        ("SARIMA + Estimated Break (grid)", lambda ytr: forecast_sarima_estimated_break(ytr, order=order, seasonal_order=seasonal_order, trim=trim)),
        ("Simple Exp. Smoothing (SES)", lambda ytr: forecast_ses(ytr)),
    ]

    errors = {name: [] for name, _ in methods}
    fails  = {name: 0 for name, _ in methods}

    for _ in range(n_sim):
        y = simulate_single_break_with_seasonality(
            T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, sigma=sigma, s=s, A=A, rng=rng
        )
        y_train = y[:t0]
        y_true  = float(y[t0])

        for name, func in methods:
            try:
                f = func(y_train)
                errors[name].append(y_true - f)
            except Exception:
                fails[name] += 1

    rows = []
    for name in errors:
        e = np.asarray(errors[name], dtype=float)
        if len(e) == 0:
            rows.append({"Method": name, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0, "Fails": fails[name]})
        else:
            rows.append({
                "Method": name,
                "RMSE": float(np.sqrt(np.mean(e**2))),
                "MAE":  float(np.mean(np.abs(e))),
                "Bias": float(np.mean(e)),
                "N": int(len(e)),
                "Fails": fails[name]
            })

    return pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)

