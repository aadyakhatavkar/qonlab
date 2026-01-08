import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings

# -----------------------------
# 1) DGP: 2-regime Markov-switching AR(1)
# -----------------------------
def simulate_ms_ar1(
    T=300,
    p00=0.97, p11=0.97,
    c0=-1.0, phi0=0.15, sigma0=0.35,
    c1=+1.0, phi1=0.95, sigma1=1.20,
    y0=0.0,
    rng=None
):
    """
    Regime s_t follows a 2-state Markov chain.
    y_t = c_s + phi_s * y_{t-1} + eps_t,  eps_t ~ N(0, sigma_s^2)
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T)
    s = np.zeros(T, dtype=int)

    y[0] = y0
    s[0] = 0 if rng.random() < 0.5 else 1

    for t in range(1, T):
        # transition
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0

        # regime parameters
        if s[t] == 0:
            c, phi, sig = c0, phi0, sigma0
        else:
            c, phi, sig = c1, phi1, sigma1

        y[t] = c + phi * y[t-1] + rng.normal(0.0, sig)

    return y, s


# -----------------------------
# 2) Forecasting models
# -----------------------------
def forecast_global_ar1(y_train):
    """Fit single-regime AR(1) on full train, 1-step forecast."""
    model = ARIMA(y_train, order=(1, 0, 0)).fit()
    return float(model.forecast(steps=1)[0])

def forecast_rolling_ar1(y_train, window=80):
    """Fit single-regime AR(1) on last 'window' obs, 1-step forecast."""
    y_win = y_train[-window:]
    model = ARIMA(y_win, order=(1, 0, 0)).fit()
    return float(model.forecast(steps=1)[0])

def forecast_markov_switching_ar1(y_train):
    """
    Fit 2-regime Markov-switching regression:
        y_t = const(regime) + phi(regime)*y_{t-1} + eps_t
    We compute 1-step forecast manually using filtered regime probs at last time.
    """
    # Build lagged regressor: x_t = y_{t-1}
    y = np.asarray(y_train)
    y_dep = y[1:]                  # y_1..y_{T-1}
    x = y[:-1].reshape(-1, 1)      # y_0..y_{T-2}

    # Fit MarkovRegression with switching intercept + switching slope + switching variance
    mod = MarkovRegression(
        y_dep,
        k_regimes=2,
        trend="c",
        exog=x,
        order=0,
        switching_trend=True,
        switching_exog=True,
        switching_variance=True
    )
    res = mod.fit(disp=False)

    # Filtered regime probabilities at last in-sample time (for y_dep)
    probs = res.filtered_marginal_probabilities[-1]  # shape (2,)

    # Map parameters by name -> value
    names = res.model.param_names
    vals = res.params
    par = {n: vals[i] for i, n in enumerate(names)}

    # Extract regime-specific intercepts and AR(1) slopes (x1)
    c0 = par["const[0]"]; c1 = par["const[1]"]
    phi0 = par["x1[0]"];  phi1 = par["x1[1]"]

    # 1-step forecast uses last observed y_T
    y_last = y[-1]
    f0 = c0 + phi0 * y_last
    f1 = c1 + phi1 * y_last
    f = probs[0] * f0 + probs[1] * f1
    return float(f)


# -----------------------------
# 3) Monte Carlo evaluation (light)
# -----------------------------
def monte_carlo(
    n_sim=200,
    T=300,
    t0=250,        # forecast origin (train uses y[:t0], forecast y[t0])
    window=80,
    seed=123
):
    rng = np.random.default_rng(seed)

    errors_g = []
    errors_r = []
    errors_m = []

    # Silence convergence spam; we handle failures with try/except.
    warnings.filterwarnings("ignore")

    for _ in range(n_sim):
        y, s = simulate_ms_ar1(T=T, rng=rng)

        # Train up to t0-1, forecast y[t0]
        y_train = y[:t0]
        y_true = y[t0]

        # Global AR(1)
        try:
            fg = forecast_global_ar1(y_train)
        except Exception:
            # if ARIMA fails, skip this replication
            continue

        # Rolling AR(1)
        try:
            fr = forecast_rolling_ar1(y_train, window=window)
        except Exception:
            continue

        # Markov-switching AR(1)
        try:
            fm = forecast_markov_switching_ar1(y_train)
        except Exception:
            # If MS fit fails in this replication, skip it
            continue

        errors_g.append(y_true - fg)
        errors_r.append(y_true - fr)
        errors_m.append(y_true - fm)

    def metrics(errs):
        errs = np.asarray(errs)
        return {
            "RMSE": float(np.sqrt(np.mean(errs**2))),
            "MAE":  float(np.mean(np.abs(errs))),
            "Bias": float(np.mean(errs)),
            "N":    int(len(errs))
        }

    return metrics(errors_g), metrics(errors_r), metrics(errors_m)


# -----------------------------
# 4) RUN
# -----------------------------
if __name__ == "__main__":
    mg, mr, mm = monte_carlo(
        n_sim=200, T=300, t0=250, window=80, seed=123
    )

    print("\nGLOBAL AR(1)\n", mg)
    print("\nROLLING AR(1)\n", mr)
    print("\nMARKOV-SWITCHING AR(1)\n", mm)
