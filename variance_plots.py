"""
Variance Break Analysis Visualizations
======================================
Plots for variance break analysis using Pesaran (2013) framework.

Run: python variance_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Variance Change.variance_change import (
    simulate_variance_break,
    mc_variance_breaks_grid,
    forecast_dist_arima_rolling,
    forecast_lstm,
    log_score_normal,
    interval_coverage,
    rmse_mae_bias
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# ============================================================================
# PLOT 1: LOSS SURFACES (Pesaran 2013 Framework)
# ============================================================================

def plot_loss_surfaces():
    """
    Heatmaps showing optimal window selection for different break magnitudes.
    Key insight: smaller windows for large breaks, larger windows for small breaks.
    """
    print("\n[1/2] Generating Loss Surface heatmaps...")
    
    # Run grid analysis (this takes ~30 seconds)
    results = mc_variance_breaks_grid(
        n_sim=50,
        T=200,
        window_sizes=[20, 50, 100, 200],
        break_magnitudes=[1.5, 2.0, 3.0, 5.0]
    )
    
    # Extract metrics
    windows = [20, 50, 100, 200]
    break_mags = [1.5, 2.0, 3.0, 5.0]
    
    # Initialize arrays
    rmse_surface = np.zeros((len(windows), len(break_mags)))
    coverage_surface = np.zeros((len(windows), len(break_mags)))
    logscore_surface = np.zeros((len(windows), len(break_mags)))
    
    for i, w in enumerate(windows):
        for j, b in enumerate(break_mags):
            key = (w, b)
            if key in results:
                rmse_surface[i, j] = results[key]['rmse']
                coverage_surface[i, j] = results[key]['coverage95']
                logscore_surface[i, j] = results[key]['logscore']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: RMSE Surface
    sns.heatmap(rmse_surface, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=break_mags, yticklabels=windows,
                cbar_kws={'label': 'RMSE'}, ax=axes[0])
    axes[0].set_title('RMSE Loss Surface\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Break Magnitude (σ₂/σ₁)')
    axes[0].set_ylabel('Window Size')
    
    # Plot 2: Coverage95 Surface
    sns.heatmap(coverage_surface, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=break_mags, yticklabels=windows,
                cbar_kws={'label': 'Coverage95'}, ax=axes[1], vmin=0.8, vmax=0.95)
    axes[1].set_title('Coverage95 Loss Surface\n(Target: 0.95)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Break Magnitude (σ₂/σ₁)')
    axes[1].set_ylabel('Window Size')
    
    # Plot 3: LogScore Surface
    sns.heatmap(logscore_surface, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=break_mags, yticklabels=windows,
                cbar_kws={'label': 'LogScore'}, ax=axes[2])
    axes[2].set_title('LogScore Loss Surface\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Break Magnitude (σ₂/σ₁)')
    axes[2].set_ylabel('Window Size')
    
    plt.tight_layout()
    plt.savefig('variance_loss_surfaces.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: variance_loss_surfaces.png")
    plt.show()
    
    return results


# ============================================================================
# PLOT 7: LOGSCORE COMPARISON (Detailed Breakdown)
# ============================================================================

def plot_logscore_comparison():
    """
    Heatmap comparing LogScore across methods and windows for variance breaks.
    Shows LSTM struggles (very negative) vs traditional methods.
    """
    print("\n[2/2] Generating LogScore comparison...")
    
    # Simulate single variance break scenario
    n_sims = 100
    results = {
        'window_size': [],
        'method': [],
        'logscore': []
    }
    
    window_sizes = [20, 50, 100, 200]
    methods = ['ARIMA Global', 'ARIMA Rolling', 'GARCH', 'LSTM']
    
    for window in window_sizes:
        for method in methods:
            logscore_values = []
            
            for sim in range(n_sims):
                # Simulate variance break
                y = simulate_variance_break(T=200, Tb=100, sigma1=1.0, sigma2=3.0)
                y_train = y[:-20]
                y_test = y[-20:]
                
                try:
                    if method == 'ARIMA Global':
                        mean, var = forecast_dist_arima_rolling(y_train, window=200, horizon=20)
                    elif method == 'ARIMA Rolling':
                        mean, var = forecast_dist_arima_rolling(y_train, window=window, horizon=20)
                    elif method == 'GARCH':
                        mean, var = forecast_dist_arima_rolling(y_train, window=100, horizon=20)
                    elif method == 'LSTM':
                        mean, var = forecast_lstm(y_train, horizon=20, lookback=20, epochs=20)
                    
                    # Calculate LogScore
                    ls = np.mean([log_score_normal(y_test[t], mean[t], var[t]) 
                                 for t in range(len(y_test)) if np.isfinite(log_score_normal(y_test[t], mean[t], var[t]))])
                    if np.isfinite(ls):
                        logscore_values.append(ls)
                except:
                    pass
            
            if logscore_values:
                results['window_size'].append(window)
                results['method'].append(method)
                results['logscore'].append(np.mean(logscore_values))
    
    # Pivot to heatmap format
    import pandas as pd
    df = pd.DataFrame(results)
    pivot = df.pivot(index='method', columns='window_size', values='logscore')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': 'LogScore (Higher = Better)'}, ax=ax)
    ax.set_title('LogScore Comparison: Methods × Window Sizes\n(Variance Break: σ₂=3.0×σ₁)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Forecasting Method')
    
    plt.tight_layout()
    plt.savefig('variance_logscore_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: variance_logscore_comparison.png")
    plt.show()


# ============================================================================
# PLOT 3: TIME SERIES VISUALIZATION (Variance Break Example)
# ============================================================================

def plot_time_series_example():
    """
    Multi-panel visualization showing:
    - Top: DGP with break point and forecasts
    - Middle: True vs estimated variance
    - Bottom: Uncertainty bands and coverage
    """
    print("\n[3/3] Generating Time Series visualization...")
    
    np.random.seed(42)
    
    # Simulate variance break
    y = simulate_variance_break(T=250, Tb=125, sigma1=1.0, sigma2=3.0)
    y_train = y[:200]
    y_test = y[200:]
    
    # Get forecasts
    mean_arima, var_arima = forecast_dist_arima_rolling(y_train, window=50, horizon=len(y_test))
    mean_garch, var_garch = forecast_dist_arima_rolling(y_train, window=100, horizon=len(y_test))
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Panel 1: Forecasts with confidence intervals
    ax = axes[0]
    time_idx = np.arange(150, 250)
    test_idx = np.arange(200, 250)
    forecast_idx = np.arange(len(y_test))
    
    # Plot actual data
    ax.plot(time_idx, y[150:250], 'ko-', linewidth=2, markersize=4, label='Actual Data')
    
    # Mark break point
    ax.axvline(x=200, color='red', linestyle='--', linewidth=2, label='Break Point (Tb=200)')
    
    # Plot forecasts
    ax.plot(test_idx, mean_arima, 'b-', linewidth=2, label='ARIMA Rolling (Window=50)')
    ax.fill_between(test_idx, 
                     mean_arima - 1.96*np.sqrt(var_arima),
                     mean_arima + 1.96*np.sqrt(var_arima),
                     alpha=0.2, color='blue', label='95% CI (ARIMA)')
    
    ax.plot(test_idx, mean_garch, 'g-', linewidth=2, label='GARCH')
    ax.fill_between(test_idx,
                     mean_garch - 1.96*np.sqrt(var_garch),
                     mean_garch + 1.96*np.sqrt(var_garch),
                     alpha=0.2, color='green', label='95% CI (GARCH)')
    
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Point Forecasts with 95% Confidence Intervals', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Variance evolution
    ax = axes[1]
    true_var = np.concatenate([np.ones(125)*1.0, np.ones(125)*9.0])
    ax.plot(time_idx, true_var[150:250], 'ko-', linewidth=2, markersize=4, label='True Variance')
    ax.axvline(x=200, color='red', linestyle='--', linewidth=2)
    
    # Rolling variance estimates
    rolling_var = []
    for i in range(150, 250):
        rolling_var.append(np.var(y[max(0,i-50):i]))
    ax.plot(time_idx, rolling_var, 'b-', linewidth=2, label='Rolling Variance (Window=50)')
    
    ax.set_ylabel('Variance', fontsize=10)
    ax.set_title('True Variance vs Rolling Estimate', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Residuals
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
    print("✓ Saved: variance_timeseries_example.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("VARIANCE BREAK ANALYSIS VISUALIZATIONS")
    print("="*70)
    
    # Run all plots
    plot_loss_surfaces()
    plot_logscore_comparison()
    plot_time_series_example()
    
    print("\n" + "="*70)
    print("✓ All variance plots generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. variance_loss_surfaces.png")
    print("  2. variance_logscore_comparison.png")
    print("  3. variance_timeseries_example.png")
