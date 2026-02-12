# =====================================================
# 4) Monte Carlo experiment (recurring breaks)
# =====================================================
def monte_carlo_recurring(
    p,
    n_sim=300,
    T=400,
    t0=300,
    window=60,
    seed=123
):
    rng = np.random.default_rng(seed)

    err = {
        "Global SARIMA": [],
        "Rolling SARIMA": [],
        "MS AR": []
    }

    for _ in range(n_sim):
        y, _ = simulate_ms_ar1_phi_only(
            T=T,
            p00=p,
            p11=p,
            rng=rng
        )

        y_train = y[:t0]
        y_true = y[t0]

        err["Global SARIMA"].append(
            y_true - forecast_global_sarima(y_train)
        )
        err["Rolling SARIMA"].append(
            y_true - forecast_rolling_sarima(y_train, window)
        )
        err["MS AR"].append(
            y_true - forecast_markov_switching_ar(y_train)
        )

    return err