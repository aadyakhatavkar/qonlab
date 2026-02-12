"""
Mean Break Forecasters
======================
Forecasting methods for ARMA models with mean (intercept) breaks.
Uses auto-selected ARMA orders via AIC/BIC (Box-Jenkins methodology).
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# =========================================================
# 2) SARIMA forecasting helpers
# =========================================================
def forecast_sarima(y_train, order=(1,0,0), seasonal_order=(1,0,0,12), exog=None, exog_next=None):
        m = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        if exog is None:
            return float(m.forecast(1)[0])
        return float(m.forecast(1, exog=exog_next)[0])


def forecast_sarima_global(y_train, order, seasonal_order):
    return forecast_sarima(y_train, order=order, seasonal_order=seasonal_order)

def forecast_sarima_rolling(y_train, window, order, seasonal_order):
    sub = y_train[-window:] if len(y_train) > window else y_train
    return forecast_sarima(sub, order=order, seasonal_order=seasonal_order)

def forecast_sarima_break_dummy_oracle_single(y_train, Tb, order, seasonal_order):
    y = np.asarray(y_train, dtype=float)
    t_idx = np.arange(len(y))
    d = (t_idx > Tb).astype(float).reshape(-1,1)
    d_next = np.array([[1.0 if len(y) > Tb else 0.0]])
    return forecast_sarima(y, order, seasonal_order, exog=d, exog_next=d_next)

def forecast_sarima_break_dummy_oracle_multiple(y_train, b1, b2, order, seasonal_order):
    y = np.asarray(y_train, dtype=float)
    t_idx = np.arange(len(y))
    d1 = (t_idx > b1).astype(float)
    d2 = (t_idx > b2).astype(float)
    exog = np.column_stack([d1, d2])
    exog_next = np.array([[1.0 if len(y) > b1 else 0.0, 1.0 if len(y) > b2 else 0.0]])
    return forecast_sarima(y, order, seasonal_order, exog=exog, exog_next=exog_next)


# --- simple break estimation (mean-only SSE) for report-friendly realism
def estimate_single_break_mean_only(y_train, trim=0.15):
    y = np.asarray(y_train, dtype=float)
    Tn = len(y)
    lo = max(int(trim*Tn), 10)
    hi = min(int((1-trim)*Tn)-1, Tn-10)
    best_Tb, best_sse = None, np.inf
    for Tb in range(lo, hi):
        m1 = np.mean(y[:Tb+1]); m2 = np.mean(y[Tb+1:])
        sse = np.sum((y[:Tb+1]-m1)**2) + np.sum((y[Tb+1:]-m2)**2)
        if sse < best_sse:
            best_sse, best_Tb = sse, Tb
    return int(best_Tb if best_Tb is not None else Tn//2)

def estimate_two_breaks_mean_only(y_train, trim=0.15, min_seg=25):
    y = np.asarray(y_train, dtype=float)
    Tn = len(y)
    lo = max(int(trim*Tn), min_seg)
    hi = min(int((1-trim)*Tn), Tn-min_seg)
    best = (Tn//3, 2*Tn//3)
    best_sse = np.inf
    for b1 in range(lo, hi-min_seg):
        for b2 in range(b1+min_seg, hi):
            seg1 = y[:b1+1]; seg2 = y[b1+1:b2+1]; seg3 = y[b2+1:]
            if len(seg1)<min_seg or len(seg2)<min_seg or len(seg3)<min_seg:
                continue
            m1,m2,m3 = np.mean(seg1), np.mean(seg2), np.mean(seg3)
            sse = np.sum((seg1-m1)**2)+np.sum((seg2-m2)**2)+np.sum((seg3-m3)**2)
            if sse < best_sse:
                best_sse = sse
                best = (b1,b2)
    return int(best[0]), int(best[1])

def forecast_sarima_estimated_break_single(y_train, order, seasonal_order, trim=0.15):
    Tb_hat = estimate_single_break_mean_only(y_train, trim=trim)
    return forecast_sarima_break_dummy_oracle_single(y_train, Tb=Tb_hat, order=order, seasonal_order=seasonal_order)

def forecast_sarima_estimated_breaks_multiple(y_train, order, seasonal_order, trim=0.15, min_seg=25):
    b1_hat, b2_hat = estimate_two_breaks_mean_only(y_train, trim=trim, min_seg=min_seg)
    return forecast_sarima_break_dummy_oracle_multiple(y_train, b1=b1_hat, b2=b2_hat, order=order, seasonal_order=seasonal_order)


# Smoothing methods
def forecast_ses(y_train):
    m = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(optimized=True)
    return float(m.forecast(1)[0])

def forecast_holt_winters_seasonal(y_train, s=12):
    m = ExponentialSmoothing(y_train, trend=None, seasonal="add", seasonal_periods=s).fit(optimized=True)
    return float(m.forecast(1)[0])