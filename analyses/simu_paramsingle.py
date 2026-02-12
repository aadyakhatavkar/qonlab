# =====================================================
# 4) Monte Carlo — POST-BREAK ONLY
# =====================================================
def monte_carlo_single_break_post(
    n_sim=300,
    T=400,
    Tb=200,
    t_post=250,
    window=80,
    innovation="normal",
    df=None,
    seed=123
):
    rng = np.random.default_rng(seed)

    err = {
        "Global SARIMA": [],
        "Rolling SARIMA": [],
        "MS AR": []
    }

    print(f"--- Monte Carlo START | innovation={innovation}, df={df} ---")

    for i in range(n_sim):
        if i % 50 == 0:
            print(f"  MC iteration {i}/{n_sim}")

        y = simulate_single_break_ar1(
            T=T,
            Tb=Tb,
            innovation=innovation,
            df=df,
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

    print(
        f"--- Monte Carlo END | innovation={innovation} "
        f"| MS-AR NaNs: {np.isnan(err['MS AR']).sum()} ---\n"
    )

    return err
