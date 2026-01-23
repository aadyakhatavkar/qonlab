import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# =========================================================
# 1) DGP: AR(1) with ONE deterministic mean break
# =========================================================
def simulate_single_mean_break(
    T=300,
    Tb=150,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    y0=0.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(T, dtype=float)
    y[0] = y0
    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi * y[t-1] + rng.normal(0.0, sigma)
    return y


# =========================================================
# 2) Forecasting models (1-step ahead)
# =========================================================
def forecast_global_ar1(y_train):
    m = ARIMA(y_train, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])

def forecast_rolling_ar1(y_train, window=60):
    y_sub = y_train[-window:] if len(y_train) > window else y_train
    m = ARIMA(y_sub, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])

def forecast_ar1_with_break_dummy_oracleTb(y_train, Tb):
    y = np.asarray(y_train, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    d = (np.arange(1, len(y)) > Tb).astype(float)
    X = np.column_stack([np.ones_like(y_lag), y_lag, d])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]  # [c, phi, delta]
    c, phi, delta = beta
    t_next = len(y)
    d_next = 1.0 if t_next > Tb else 0.0
    return float(c + phi * y[-1] + delta * d_next)

def _fit_ar1_ols(y_segment):
    y = np.asarray(y_segment, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    resid = y_dep - X @ beta
    sse = float(np.sum(resid**2))
    return float(beta[0]), float(beta[1]), sse

def estimate_break_Tb_gridsearch(y_train, trim=0.15):
    y = np.asarray(y_train, dtype=float)
    T = len(y)
    lo = max(int(np.floor(trim * T)), 10)
    hi = min(int(np.ceil((1 - trim) * T)) - 1, T - 11)
    best_Tb, best_sse = None, np.inf
    for Tb in range(lo, hi):
        c1, p1, sse1 = _fit_ar1_ols(y[:Tb+1])
        c2, p2, sse2 = _fit_ar1_ols(y[Tb+1:])
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    return int(best_Tb if best_Tb is not None else T // 2)

def forecast_ar1_with_estimated_break(y_train, trim=0.15):
    y = np.asarray(y_train, dtype=float)
    Tb_hat = estimate_break_Tb_gridsearch(y, trim=trim)
    regime = y[Tb_hat+1:] if Tb_hat + 1 < len(y) else y
    if len(regime) < 20:
        return forecast_global_ar1(y_train)
    m = ARIMA(regime, order=(1, 0, 0)).fit()
    return float(m.forecast(1)[0])

def forecast_markov_switching(y_train, k_regimes=2):
    """
    More stable settings:
    - switching_variance=False reduces convergence failures
    """
    y = np.asarray(y_train, dtype=float)
    m = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=False).fit(disp=False)
    # predict next step mean
    pred = m.predict(start=len(y), end=len(y))
    return float(np.asarray(pred)[0])


# =========================================================
# 3) Monte Carlo evaluation (post-break 1-step forecast)
# =========================================================
def run_monte_carlo_single_break(
    n_sim=200,
    T=300,
    Tb=150,
    window=60,
    seed=123,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    gap_after_break=20,
    trim=0.15
):
    rng = np.random.default_rng(seed)
    t0 = Tb + gap_after_break
    if t0 >= T:
        raise ValueError("gap_after_break too large for T.")

    # store forecast errors per method
    errors = {
        "ARIMA Global (AR1)": [],
        "ARIMA Rolling (AR1)": [],
        "AR1 + Break Dummy (oracle Tb)": [],
        "AR1 + Estimated Break (grid)": [],
        "Markov Switching (2 regimes)": [],
    }

    # track failures (helps debugging)
    fails = {k: 0 for k in errors}

    for _ in range(n_sim):
        y = simulate_single_mean_break(T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, sigma=sigma, rng=rng)
        y_train = y[:t0]
        y_true = float(y[t0])

        # Each method gets its own try/except (NO more skipping everything)
        try:
            f = forecast_global_ar1(y_train)
            errors["ARIMA Global (AR1)"].append(y_true - f)
        except Exception:
            fails["ARIMA Global (AR1)"] += 1

        try:
            f = forecast_rolling_ar1(y_train, window=window)
            errors["ARIMA Rolling (AR1)"].append(y_true - f)
        except Exception:
            fails["ARIMA Rolling (AR1)"] += 1

        try:
            f = forecast_ar1_with_break_dummy_oracleTb(y_train, Tb=Tb)
            errors["AR1 + Break Dummy (oracle Tb)"].append(y_true - f)
        except Exception:
            fails["AR1 + Break Dummy (oracle Tb)"] += 1

        try:
            f = forecast_ar1_with_estimated_break(y_train, trim=trim)
            errors["AR1 + Estimated Break (grid)"].append(y_true - f)
        except Exception:
            fails["AR1 + Estimated Break (grid)"] += 1

        try:
            f = forecast_markov_switching(y_train, k_regimes=2)
            errors["Markov Switching (2 regimes)"].append(y_true - f)
        except Exception:
            fails["Markov Switching (2 regimes)"] += 1

    def metrics(e):
        e = np.asarray(e, dtype=float)
        return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))), float(np.mean(e))

    rows = []
    for method, e in errors.items():
        if len(e) == 0:
            rows.append({"Method": method, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0, "Fails": fails[method]})
        else:
            rmse, mae, bias = metrics(e)
            rows.append({"Method": method, "RMSE": rmse, "MAE": mae, "Bias": bias, "N": len(e), "Fails": fails[method]})

    results = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return results


# =========================================================
# 4) RUN
# =========================================================
results = run_monte_carlo_single_break(
    n_sim=200,
    T=300,
    Tb=150,
    window=60,
    seed=123,
    mu0=0.0,
    mu1=2.0,
    phi=0.6,
    sigma=1.0,
    gap_after_break=20,
    trim=0.15
)

print("\nSingle-break results (lower RMSE = better):")
print(results.to_string(index=False))

best_row = results.dropna(subset=["RMSE"]).head(1)
if len(best_row) > 0:
    print(f"\nConclusion: Best method (lowest RMSE) = {best_row.iloc[0]['Method']}")
else:
    print("\nConclusion: All methods failed (check statsmodels installation / Markov settings).")
