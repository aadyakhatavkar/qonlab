import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# =========================================================
# 1) DATA GENERATING PROCESSES
# =========================================================
def simulate_single_mean_break(T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, y0=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    y = np.zeros(T)
    y[0] = y0
    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi*y[t-1] + rng.normal(0, sigma)
    return y

def simulate_multiple_mean_breaks(T=300, b1=100, b2=200, mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, y0=0.0, rng=None):
    """
    3 regimes:
      t <= b1: mu0
      b1 < t <= b2: mu1
      t > b2: mu2
    """
    if rng is None:
        rng = np.random.default_rng()
    y = np.zeros(T)
    y[0] = y0
    for t in range(1, T):
        if t <= b1:
            mu = mu0
        elif t <= b2:
            mu = mu1
        else:
            mu = mu2
        y[t] = mu + phi*y[t-1] + rng.normal(0, sigma)
    return y


# =========================================================
# 2) FORECAST MODELS (1-step ahead)
# =========================================================
def forecast_global_ar1(y_train):
    m = ARIMA(y_train, order=(1,0,0)).fit()
    return float(m.forecast(1)[0])

def forecast_rolling_ar1(y_train, window=60):
    sub = y_train[-window:] if len(y_train) > window else y_train
    m = ARIMA(sub, order=(1,0,0)).fit()
    return float(m.forecast(1)[0])

def forecast_ar1_break_dummy_oracle_single(y_train, Tb):
    # y_t = c + phi*y_{t-1} + delta*1(t>Tb) + e_t
    y = np.asarray(y_train, float)
    y_dep = y[1:]
    y_lag = y[:-1]
    d = (np.arange(1, len(y)) > Tb).astype(float)
    X = np.column_stack([np.ones_like(y_lag), y_lag, d])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    c, phi, delta = beta
    t_next = len(y)
    d_next = 1.0 if t_next > Tb else 0.0
    return float(c + phi*y[-1] + delta*d_next)

def forecast_ar1_break_dummy_oracle_multiple(y_train, b1, b2):
    """
    2 dummies:
      d1 = 1(t > b1)
      d2 = 1(t > b2)
    So mean can step at b1 and b2.
    """
    y = np.asarray(y_train, float)
    y_dep = y[1:]
    y_lag = y[:-1]
    t_idx = np.arange(1, len(y))
    d1 = (t_idx > b1).astype(float)
    d2 = (t_idx > b2).astype(float)
    X = np.column_stack([np.ones_like(y_lag), y_lag, d1, d2])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    c, phi, a1, a2 = beta
    t_next = len(y)
    d1_next = 1.0 if t_next > b1 else 0.0
    d2_next = 1.0 if t_next > b2 else 0.0
    return float(c + phi*y[-1] + a1*d1_next + a2*d2_next)

def _fit_ar1_ols(y_seg):
    y = np.asarray(y_seg, float)
    y_dep = y[1:]
    y_lag = y[:-1]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    resid = y_dep - X @ beta
    sse = float(np.sum(resid**2))
    return beta, sse

def estimate_single_break_grid(y_train, trim=0.15):
    y = np.asarray(y_train, float)
    T = len(y)
    lo = max(int(trim*T), 10)
    hi = min(int((1-trim)*T)-1, T-11)
    best_Tb, best_sse = None, np.inf
    for Tb in range(lo, hi):
        _, sse1 = _fit_ar1_ols(y[:Tb+1])
        _, sse2 = _fit_ar1_ols(y[Tb+1:])
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    return int(best_Tb if best_Tb is not None else T//2)

def forecast_ar1_estimated_break_single(y_train, trim=0.15):
    y = np.asarray(y_train, float)
    Tb_hat = estimate_single_break_grid(y, trim=trim)
    regime = y[Tb_hat+1:] if Tb_hat+1 < len(y) else y
    if len(regime) < 20:
        return forecast_global_ar1(y_train)
    m = ARIMA(regime, order=(1,0,0)).fit()
    return float(m.forecast(1)[0])

def forecast_markov_switching(y_train, k_regimes=2):
    # Markov switching is fragile; we keep it but track fails.
    y = np.asarray(y_train, float)
    m = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=False).fit(disp=False)
    pred = m.predict(start=len(y), end=len(y))
    return float(np.asarray(pred)[0])


# =========================================================
# 3) MONTE CARLO RUNNER
# =========================================================
def evaluate_methods(errors_dict):
    rows = []
    for method, errs in errors_dict.items():
        errs = np.asarray(errs, float)
        if len(errs) == 0:
            rows.append({"Method": method, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0})
        else:
            rows.append({
                "Method": method,
                "RMSE": float(np.sqrt(np.mean(errs**2))),
                "MAE":  float(np.mean(np.abs(errs))),
                "Bias": float(np.mean(errs)),
                "N": int(len(errs)),
            })
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)

def run_mc_single(n_sim=200, T=300, Tb=150, window=60, seed=123, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, gap_after_break=20, trim=0.15):
    rng = np.random.default_rng(seed)
    t0 = Tb + gap_after_break

    errors = {
        "ARIMA Global (AR1)": [],
        "ARIMA Rolling (AR1)": [],
        "AR1 + Break Dummy (oracle)": [],
        "AR1 + Est. Break (grid)": [],
        "Markov Switching (2)": [],
    }
    fails = {k: 0 for k in errors}

    for _ in range(n_sim):
        y = simulate_single_mean_break(T=T, Tb=Tb, mu0=mu0, mu1=mu1, phi=phi, sigma=sigma, rng=rng)
        y_train, y_true = y[:t0], float(y[t0])

        # each method independent
        for name, func in [
            ("ARIMA Global (AR1)", lambda: forecast_global_ar1(y_train)),
            ("ARIMA Rolling (AR1)", lambda: forecast_rolling_ar1(y_train, window=window)),
            ("AR1 + Break Dummy (oracle)", lambda: forecast_ar1_break_dummy_oracle_single(y_train, Tb=Tb)),
            ("AR1 + Est. Break (grid)", lambda: forecast_ar1_estimated_break_single(y_train, trim=trim)),
            ("Markov Switching (2)", lambda: forecast_markov_switching(y_train, k_regimes=2)),
        ]:
            try:
                f = func()
                errors[name].append(y_true - f)
            except Exception:
                fails[name] += 1

    res = evaluate_methods(errors)
    res["Scenario"] = "Single break"
    res["Fails"] = res["Method"].map(fails)
    return res

def run_mc_multiple(n_sim=200, T=300, b1=100, b2=200, window=60, seed=456, mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, gap_after_last_break=20):
    rng = np.random.default_rng(seed)
    t0 = b2 + gap_after_last_break

    errors = {
        "ARIMA Global (AR1)": [],
        "ARIMA Rolling (AR1)": [],
        "AR1 + Break Dummy (oracle)": [],
        "Markov Switching (2)": [],
    }
    fails = {k: 0 for k in errors}

    for _ in range(n_sim):
        y = simulate_multiple_mean_breaks(T=T, b1=b1, b2=b2, mu0=mu0, mu1=mu1, mu2=mu2, phi=phi, sigma=sigma, rng=rng)
        y_train, y_true = y[:t0], float(y[t0])

        for name, func in [
            ("ARIMA Global (AR1)", lambda: forecast_global_ar1(y_train)),
            ("ARIMA Rolling (AR1)", lambda: forecast_rolling_ar1(y_train, window=window)),
            ("AR1 + Break Dummy (oracle)", lambda: forecast_ar1_break_dummy_oracle_multiple(y_train, b1=b1, b2=b2)),
            ("Markov Switching (2)", lambda: forecast_markov_switching(y_train, k_regimes=2)),
        ]:
            try:
                f = func()
                errors[name].append(y_true - f)
            except Exception:
                fails[name] += 1

    res = evaluate_methods(errors)
    res["Scenario"] = "Multiple breaks"
    res["Fails"] = res["Method"].map(fails)
    return res


# =========================================================
# 4) RUN BOTH + COMBINE + PRINT
# =========================================================
single = run_mc_single(n_sim=200, T=300, Tb=150, window=60, seed=123, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, gap_after_break=20, trim=0.15)
multi  = run_mc_multiple(n_sim=200, T=300, b1=100, b2=200, window=60, seed=456, mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, gap_after_last_break=20)

all_results = pd.concat([single, multi], ignore_index=True)

print("\n=== COMPARISON TABLE (Single vs Multiple) ===")
print(all_results.sort_values(["Scenario","RMSE"], na_position="last").to_string(index=False))

# Best method per scenario (by RMSE)
for scen in ["Single break", "Multiple breaks"]:
    sub = all_results[(all_results["Scenario"] == scen) & (~all_results["RMSE"].isna())]
    if len(sub) > 0:
        best = sub.sort_values("RMSE").iloc[0]
        print(f"\nBest method for {scen}: {best['Method']} (RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f})")
    else:
        print(f"\nBest method for {scen}: No successful methods.")


# =========================================================
# 5) GRAPHS: RMSE and MAE comparison
# =========================================================
def bar_compare(metric="RMSE"):
    pivot = all_results.pivot_table(index="Method", columns="Scenario", values=metric, aggfunc="first")
    pivot = pivot.sort_values(by="Single break", na_position="last")  # sort by single break
    ax = pivot.plot(kind="bar", figsize=(11,5))
    ax.set_title(f"{metric} Comparison: Single vs Multiple Breaks (lower is better)")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

bar_compare("RMSE")
bar_compare("MAE")


# =========================================================
# 6) EXAMPLE PLOT: one single-break vs one multiple-break path
# =========================================================
rng_demo = np.random.default_rng(999)
y_single = simulate_single_mean_break(T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, rng=rng_demo)
y_multi  = simulate_multiple_mean_breaks(T=300, b1=100, b2=200, mu0=0.0, mu1=2.0, mu2=-2.0, phi=0.6, sigma=1.0, rng=rng_demo)

plt.figure(figsize=(11,4))
plt.plot(y_single, label="Single break (Tb=150)")
plt.plot(y_multi, label="Multiple breaks (b1=100, b2=200)")
plt.axvline(150, linestyle="--")
plt.axvline(100, linestyle="--")
plt.axvline(200, linestyle="--")
plt.title("Example simulated series: single vs multiple mean breaks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
