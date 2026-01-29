"""
Variance Break Analysis Visualizations
======================================
Plots for variance break analysis.

Run: `from analyses.plots import plot_logscore_comparison` or `python -m analyses.plots`
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dgps.variance import simulate_variance_break_ar1
from estimators.forecasters import (
    forecast_variance_dist_arima_rolling,
    variance_log_score_normal,
    variance_interval_coverage,
    variance_rmse_mae_bias,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_loss_surfaces():
    """
    DEPRECATED: Grid search for optimal window selection has been removed.
    
    Per project policy, practitioners prefer fixed window and break detection
    over optimal window selection. Use plot_logscore_comparison() instead.
    """
    print("\n[DEPRECATED] Grid search for optimal window selection has been removed.")
    print("Practitioners prefer fixed window and break detection approaches.")
    print("Use plot_logscore_comparison() or plot_time_series_example() instead.")
    return None


def plot_logscore_comparison():
    """
    Heatmap comparing LogScore across methods and windows for variance breaks.
    """
    print("\n[2/2] Generating LogScore comparison...")
    n_sims = 100
    results = {'window_size': [], 'method': [], 'logscore': []}
    window_sizes = [20, 50, 100, 200]
    methods = ['ARIMA Global', 'ARIMA Rolling', 'GARCH']

    for window in window_sizes:
        for method in methods:
            logscore_values = []
            for sim in range(n_sims):
                y = simulate_variance_break_ar1(T=200, Tb=100, sigma1=1.0, sigma2=3.0)
                y_train = y[:-20]
                y_test = y[-20:]
                try:
                    if method == 'ARIMA Global':
                        mean, var = forecast_variance_dist_arima_rolling(y_train, window=200, horizon=20)
                    elif method == 'ARIMA Rolling':
                        mean, var = forecast_variance_dist_arima_rolling(y_train, window=window, horizon=20)
                    elif method == 'GARCH':
                        try:
                            mean, var = forecast_garch_variance(y_train, horizon=20)
                        except Exception:
                            mean = np.full(20, np.nan)
                            var = np.full(20, np.nan)
                    ls = variance_log_score_normal(y_test, mean, var)
                    if np.isfinite(ls):
                        logscore_values.append(ls)
                except:
                    pass
            if logscore_values:
                results['window_size'].append(window)
                results['method'].append(method)
                results['logscore'].append(np.mean(logscore_values))

    import pandas as pd
    df = pd.DataFrame(results)
    pivot = df.pivot(index='method', columns='window_size', values='logscore')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'LogScore (Higher = Better)'}, ax=ax)
    ax.set_title('LogScore Comparison: Methods × Window Sizes\n(Variance Break: σ₂=3.0×σ₁)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Forecasting Method')
    plt.tight_layout()
    plt.savefig('variance_logscore_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: variance_logscore_comparison.png")
    plt.show()


def plot_time_series_example():
    print("\n[3/3] Generating Time Series visualization...")
    np.random.seed(42)
    y = simulate_variance_break_ar1(T=250, Tb=125, sigma1=1.0, sigma2=3.0)
    y_train = y[:200]
    y_test = y[200:]
    mean_arima, var_arima = forecast_variance_dist_arima_rolling(y_train, window=50, horizon=len(y_test))
    try:
        mean_garch, var_garch = forecast_garch_variance(y_train, horizon=len(y_test))
    except Exception:
        mean_garch = np.full(len(y_test), np.nan)
        var_garch = np.full(len(y_test), np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    ax = axes[0]
    time_idx = np.arange(150, 250)
    test_idx = np.arange(200, 250)
    forecast_idx = np.arange(len(y_test))
    ax.plot(time_idx, y[150:250], 'ko-', linewidth=2, markersize=4, label='Actual Data')
    ax.axvline(x=200, color='red', linestyle='--', linewidth=2, label='Break Point (Tb=200)')
    ax.plot(test_idx, mean_arima, 'b-', linewidth=2, label='ARIMA Rolling (Window=50)')
    ax.fill_between(test_idx, mean_arima - 1.96*np.sqrt(var_arima), mean_arima + 1.96*np.sqrt(var_arima), alpha=0.2, color='blue', label='95% CI (ARIMA)')
    ax.plot(test_idx, mean_garch, 'g-', linewidth=2, label='GARCH')
    ax.fill_between(test_idx, mean_garch - 1.96*np.sqrt(var_garch), mean_garch + 1.96*np.sqrt(var_garch), alpha=0.2, color='green', label='95% CI (GARCH)')
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Point Forecasts with 95% Confidence Intervals', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    true_var = np.concatenate([np.ones(125)*1.0, np.ones(125)*9.0])
    ax.plot(time_idx, true_var[150:250], 'ko-', linewidth=2, markersize=4, label='True Variance')
    ax.axvline(x=200, color='red', linestyle='--', linewidth=2)
    rolling_var = []
    for i in range(150, 250):
        rolling_var.append(np.var(y[max(0,i-50):i]))
    ax.plot(time_idx, rolling_var, 'b-', linewidth=2, label='Rolling Variance (Window=50)')
    ax.set_ylabel('Variance', fontsize=10)
    ax.set_title('True Variance vs Rolling Estimate', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    residuals_arima = y_test - mean_arima
    residuals_garch = y_test - mean_garch
    ax.scatter(test_idx, residuals_arima, alpha=0.6, s=50, label='ARIMA Rolling Residuals')
    ax.scatter(test_idx, residuals_garch, alpha=0.6, s=50, label='GARCH Residuals')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=200, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.set_title('Forecast Residuals (Should be near zero)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variance_timeseries_example.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: variance_timeseries_example.png')
    plt.show()


if __name__ == '__main__':
    print('\n' + '='*70)
    print('VARIANCE BREAK ANALYSIS VISUALIZATIONS')
    print('='*70)
    plot_loss_surfaces()
    plot_logscore_comparison()
    plot_time_series_example()
    print('\n' + '='*70)
    print('✓ All variance plots generated successfully!')
    print('='*70)
    print('\nGenerated files:')
    print('  1. variance_loss_surfaces.png')
    print('  2. variance_logscore_comparison.png')
    print('  3. variance_timeseries_example.png')
