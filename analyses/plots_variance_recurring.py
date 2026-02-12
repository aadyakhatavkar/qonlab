"""
Variance Recurring (Markov-Switching) Break Analysis Visualizations
===================================================================
Plots for recurring (Markov-switching) variance break analysis.

Run: `from analyses.plots_variance_recurring import plot_logscore_comparison`
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dgps.variance_recurring import simulate_ms_ar1_variance_only
from estimators.variance_single import (
    forecast_variance_dist_sarima_rolling,
    variance_log_score_normal,
)
from estimators.variance_recurring import forecast_markov_switching

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_logscore_comparison():
    """
    Heatmap comparing LogScore across methods for recurring (Markov-switching) variance breaks.
    """
    print("\n[1/2] Generating LogScore comparison for recurring variance break...")
    n_sims = 100
    results = {'persistence': [], 'method': [], 'logscore': []}
    persistence_levels = [0.90, 0.95, 0.99]
    methods = ['SARIMA Global', 'SARIMA Rolling', 'MS AR(1)']

    for p in persistence_levels:
        for method in methods:
            print(f"  {method} (p={p})...", end="", flush=True)
            logscores = []
            for _ in range(n_sims):
                y = simulate_ms_ar1_variance_only(T=400, p00=p, p11=p, phi=0.6, sigma1=1.0, sigma2=2.0)
                t_orig = min(400 - 10, max(400 // 2, 100))
                y_train = y[:t_orig]
                y_test = y[t_orig:t_orig+10]
                
                try:
                    if method == 'SARIMA Global':
                        from estimators.variance_single import forecast_variance_dist_sarima_global
                        mean, var = forecast_variance_dist_sarima_global(y_train, horizon=len(y_test))
                    elif method == 'SARIMA Rolling':
                        mean, var = forecast_variance_dist_sarima_rolling(y_train, window=100, horizon=len(y_test))
                    elif method == 'MS AR(1)':
                        mean, var = forecast_markov_switching(y_train, horizon=len(y_test))
                    
                    ls = variance_log_score_normal(y_test, mean, var)
                    if not np.isnan(ls):
                        logscores.append(ls)
                except Exception:
                    pass
            
            if logscores:
                avg_logscore = float(np.mean(logscores))
                results['persistence'].append(p)
                results['method'].append(method)
                results['logscore'].append(avg_logscore)
                print(f" ✓ {avg_logscore:.3f}")
            else:
                print(" ✗ Failed")

    if results['method']:
        import pandas as pd
        df_results = pd.DataFrame(results)
        pivot_table = df_results.pivot_table(index='method', columns='persistence', values='logscore')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'LogScore'})
        ax.set_title('LogScore Comparison: Recurring (Markov-Switching) Variance Break', fontsize=12, fontweight='bold')
        ax.set_xlabel('Persistence (p00, p11)', fontsize=11)
        ax.set_ylabel('Method', fontsize=11)
        plt.tight_layout()
        plt.savefig('variance_recurring_logscore_comparison.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: variance_recurring_logscore_comparison.png')
        plt.show()


def plot_time_series_example():
    """
    Example time series with forecasts for recurring (Markov-switching) variance break.
    """
    print("\n[2/2] Generating time series example for recurring variance break...")
    np.random.seed(42)
    
    # Generate data
    y = simulate_ms_ar1_variance_only(T=400, p00=0.95, p11=0.95, phi=0.6, sigma1=1.0, sigma2=2.0)
    t_orig = 300
    y_train = y[:t_orig]
    y_test = y[t_orig:t_orig+50]
    
    # Forecasts
    try:
        from estimators.variance_single import forecast_variance_dist_sarima_global
        mean_sarima, var_sarima = forecast_variance_dist_sarima_global(y_train, horizon=len(y_test))
    except Exception:
        mean_sarima = np.full(len(y_test), np.nan)
        var_sarima = np.full(len(y_test), np.nan)
    
    try:
        mean_ms, var_ms = forecast_markov_switching(y_train, horizon=len(y_test))
    except Exception:
        mean_ms = np.full(len(y_test), np.nan)
        var_ms = np.full(len(y_test), np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    ax = axes[0]
    time_idx = np.arange(250, 350)
    test_idx = np.arange(t_orig, t_orig + len(y_test))
    ax.plot(time_idx, y[250:350], 'ko-', linewidth=2, markersize=4, label='Actual Data')
    ax.plot(test_idx, mean_sarima, 'b-', linewidth=2, label='SARIMA Global')
    ax.fill_between(test_idx, mean_sarima - 1.96*np.sqrt(var_sarima), mean_sarima + 1.96*np.sqrt(var_sarima), 
                     alpha=0.2, color='blue', label='95% CI (SARIMA)')
    ax.plot(test_idx, mean_ms, 'r-', linewidth=2, label='MS AR(1)')
    ax.fill_between(test_idx, mean_ms - 1.96*np.sqrt(var_ms), mean_ms + 1.96*np.sqrt(var_ms), 
                     alpha=0.2, color='red', label='95% CI (MS)')
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Forecasts with 95% Confidence Intervals (Recurring Variance Break, p=0.95)', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    rolling_var = []
    for i in range(250, 350):
        rolling_var.append(np.var(y[max(0,i-50):i]))
    ax.plot(time_idx, rolling_var, 'g-', linewidth=2, label='Rolling Variance (Window=50)')
    ax.set_ylabel('Variance', fontsize=10)
    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_title('Rolling Variance Estimate', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variance_recurring_timeseries_example.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: variance_recurring_timeseries_example.png')
    plt.show()


if __name__ == '__main__':
    print('\n' + '='*70)
    print('VARIANCE RECURRING BREAK ANALYSIS VISUALIZATIONS')
    print('='*70)
    plot_logscore_comparison()
    plot_time_series_example()
    print('\n' + '='*70)
    print('✓ All variance recurring break plots generated successfully!')
    print('='*70)
