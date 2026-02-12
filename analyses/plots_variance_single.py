"""
Variance Single Break Analysis Visualizations
==============================================
Plots for single variance break analysis.

Run: `from analyses.plots_variance_single import plot_logscore_comparison`
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dgps.variance_single import simulate_variance_break_ar1
from estimators.variance_single import (
    forecast_variance_dist_sarima_rolling,
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
    Heatmap comparing LogScore across methods and windows for single variance breaks.
    """
    print("\n[1/2] Generating LogScore comparison for single variance break...")
    n_sims = 100
    results = {'window_size': [], 'method': [], 'logscore': []}
    window_sizes = [20, 50, 100, 200]
    methods = ['SARIMA Global', 'SARIMA Rolling', 'GARCH']

    for window in window_sizes:
        for method in methods:
            print(f"  {method} (window={window})...", end="", flush=True)
            logscores = []
            for _ in range(n_sims):
                y = simulate_variance_break_ar1(T=400, Tb=200, phi=0.6, sigma1=1.0, sigma2=2.0)
                y_train = y[:300]
                y_test = y[300:310]
                
                try:
                    if method == 'SARIMA Global':
                        mean, var = forecast_variance_dist_sarima_rolling(y_train, window=window, horizon=len(y_test))
                    elif method == 'SARIMA Rolling':
                        mean, var = forecast_variance_dist_sarima_rolling(y_train, window=window, horizon=len(y_test))
                    elif method == 'GARCH':
                        from estimators.variance_single import forecast_garch_variance
                        mean, var = forecast_garch_variance(y_train, horizon=len(y_test))
                    
                    ls = variance_log_score_normal(y_test, mean, var)
                    if not np.isnan(ls):
                        logscores.append(ls)
                except Exception:
                    pass
            
            if logscores:
                avg_logscore = float(np.mean(logscores))
                results['window_size'].append(window)
                results['method'].append(method)
                results['logscore'].append(avg_logscore)
                print(f" ✓ {avg_logscore:.3f}")
            else:
                print(" ✗ Failed")

    if results['method']:
        import pandas as pd
        df_results = pd.DataFrame(results)
        pivot_table = df_results.pivot_table(index='method', columns='window_size', values='logscore')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'LogScore'})
        ax.set_title('LogScore Comparison: Single Variance Break', fontsize=12, fontweight='bold')
        ax.set_xlabel('Window Size', fontsize=11)
        ax.set_ylabel('Method', fontsize=11)
        plt.tight_layout()
        plt.savefig('variance_single_logscore_comparison.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: variance_single_logscore_comparison.png')
        plt.show()


def plot_time_series_example():
    """
    Example time series with forecasts and confidence intervals for single variance break.
    """
    print("\n[2/2] Generating time series example for single variance break...")
    np.random.seed(42)
    
    # Generate data
    y = simulate_variance_break_ar1(T=400, Tb=200, phi=0.6, sigma1=1.0, sigma2=2.0)
    y_train = y[:200]
    y_test = y[200:250]
    
    # Forecasts
    try:
        mean_sarima, var_sarima = forecast_variance_dist_sarima_rolling(y_train, window=50, horizon=len(y_test))
    except Exception:
        mean_sarima = np.full(len(y_test), np.nan)
        var_sarima = np.full(len(y_test), np.nan)
    
    try:
        from estimators.variance_single import forecast_garch_variance
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
    ax.plot(test_idx, mean_sarima, 'b-', linewidth=2, label='SARIMA Rolling (Window=50)')
    ax.fill_between(test_idx, mean_sarima - 1.96*np.sqrt(var_sarima), mean_sarima + 1.96*np.sqrt(var_sarima), alpha=0.2, color='blue', label='95% CI (SARIMA)')
    ax.plot(test_idx, mean_garch, 'g-', linewidth=2, label='GARCH')
    ax.fill_between(test_idx, mean_garch - 1.96*np.sqrt(var_garch), mean_garch + 1.96*np.sqrt(var_garch), alpha=0.2, color='green', label='95% CI (GARCH)')
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Point Forecasts with 95% Confidence Intervals (Single Variance Break)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    true_var = np.concatenate([np.ones(125)*1.0, np.ones(125)*4.0])
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
    residuals_sarima = y_test - mean_sarima
    residuals_garch = y_test - mean_garch
    ax.scatter(test_idx, residuals_sarima, alpha=0.6, s=50, label='SARIMA Rolling Residuals')
    ax.scatter(test_idx, residuals_garch, alpha=0.6, s=50, label='GARCH Residuals')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=200, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.set_title('Forecast Residuals (Should be near zero)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variance_single_timeseries_example.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: variance_single_timeseries_example.png')
    plt.show()


if __name__ == '__main__':
    print('\n' + '='*70)
    print('VARIANCE SINGLE BREAK ANALYSIS VISUALIZATIONS')
    print('='*70)
    plot_loss_surfaces()
    plot_logscore_comparison()
    plot_time_series_example()
    print('\n' + '='*70)
    print('✓ All variance single break plots generated successfully!')
    print('='*70)
