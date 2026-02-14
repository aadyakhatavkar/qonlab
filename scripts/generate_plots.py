#!/usr/bin/env python3
"""
Generate Analysis Figures
=========================

Creates figures using REAL simulation data from outputs/tables/.
No hardcoded illustrative data - all values from Monte Carlo simulations.

Outputs: outputs/figures/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("📊 Generating Figures from Real Simulation Data\n" + "="*60 + "\n")

# =========================================================================
# HELPER: Load simulation results
# =========================================================================

def load_results(pattern):
    """Load latest CSV matching pattern."""
    files = sorted(TABLES_DIR.glob(pattern))
    if not files:
        return None
    return pd.read_csv(files[-1])

# =========================================================================
# DGP VISUALIZATION FIGURES
# =========================================================================

print("="*60)
print("📈 DGP Visualization Figures")
print("="*60 + "\n")

# DGP Figure 1: Variance Break
print("[DGP 1/6] Variance Break DGP...")

np.random.seed(42)
T, Tb = 300, 150
phi, sigma1, sigma2 = 0.6, 1.0, 2.5
y_var_dgp = np.zeros(T)
y_var_dgp[0] = np.random.normal(0, sigma1)
for t in range(1, T):
    sigma = sigma1 if t < Tb else sigma2
    y_var_dgp[t] = phi * y_var_dgp[t-1] + np.random.normal(0, sigma)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Time series
ax = axes[0]
ax.plot(range(Tb), y_var_dgp[:Tb], color='steelblue', lw=1.5, label='Regime 1 (σ=1.0)')
ax.plot(range(Tb, T), y_var_dgp[Tb:], color='darkred', lw=1.5, label='Regime 2 (σ=2.5)')
ax.axvline(x=Tb, color='black', ls='--', lw=2, label=f'Break Point (t={Tb})')
ax.set_ylabel('y(t)', fontsize=11, fontweight='bold')
ax.set_title('Variance Break: AR(1) with σ Change', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom: Rolling variance
window = 30
roll_var = [np.var(y_var_dgp[max(0, t-window):t]) if t > 0 else 0 for t in range(T)]
ax = axes[1]
ax.plot(range(T), roll_var, color='purple', lw=2, label=f'Rolling Variance (w={window})')
ax.axhline(y=sigma1**2, color='steelblue', ls=':', lw=2, label=f'True σ₁² = {sigma1**2}')
ax.axhline(y=sigma2**2, color='darkred', ls=':', lw=2, label=f'True σ₂² = {sigma2**2}')
ax.axvline(x=Tb, color='black', ls='--', lw=2)
ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
ax.set_ylabel('Variance', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dgp_variance_break.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved")

# DGP Figure 2: Mean Break
print("[DGP 2/6] Mean Break DGP...")

np.random.seed(42)
T, Tb = 300, 150
mu1, mu2, phi, sigma = 0.0, 2.0, 0.6, 1.0
y_mean_dgp = np.zeros(T)
y_mean_dgp[0] = mu1 + np.random.normal(0, sigma)
for t in range(1, T):
    mu = mu1 if t < Tb else mu2
    mu_prev = mu1 if t-1 < Tb else mu2
    y_mean_dgp[t] = mu + phi * (y_mean_dgp[t-1] - mu_prev) + np.random.normal(0, sigma)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Time series
ax = axes[0]
ax.plot(range(Tb), y_mean_dgp[:Tb], color='steelblue', lw=1.5, label='Regime 1 (μ=0.0)')
ax.plot(range(Tb, T), y_mean_dgp[Tb:], color='darkred', lw=1.5, label='Regime 2 (μ=2.0)')
ax.axvline(x=Tb, color='black', ls='--', lw=2, label=f'Break Point (t={Tb})')
ax.axhline(y=mu1, color='steelblue', ls=':', lw=1.5, alpha=0.7)
ax.axhline(y=mu2, color='darkred', ls=':', lw=1.5, alpha=0.7)
ax.set_ylabel('y(t)', fontsize=11, fontweight='bold')
ax.set_title('Mean Break: AR(1) with μ Change', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom: Rolling mean
window = 30
roll_mean = [np.mean(y_mean_dgp[max(0, t-window):t]) if t > 0 else mu1 for t in range(T)]
ax = axes[1]
ax.plot(range(T), roll_mean, color='purple', lw=2, label=f'Rolling Mean (w={window})')
ax.axhline(y=mu1, color='steelblue', ls=':', lw=2, label=f'True μ₁ = {mu1}')
ax.axhline(y=mu2, color='darkred', ls=':', lw=2, label=f'True μ₂ = {mu2}')
ax.axvline(x=Tb, color='black', ls='--', lw=2)
ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dgp_mean_break.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved")

# DGP Figure 3: Parameter Break
print("[DGP 3/6] Parameter Break DGP...")

np.random.seed(42)
T, Tb = 300, 150
phi1, phi2, sigma = 0.3, 0.8, 1.0
y_param_dgp = np.zeros(T)
y_param_dgp[0] = np.random.normal(0, sigma)
for t in range(1, T):
    phi = phi1 if t < Tb else phi2
    y_param_dgp[t] = phi * y_param_dgp[t-1] + np.random.normal(0, sigma)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Time series
ax = axes[0]
ax.plot(range(Tb), y_param_dgp[:Tb], color='steelblue', lw=1.5, label='Regime 1 (φ=0.3, low persistence)')
ax.plot(range(Tb, T), y_param_dgp[Tb:], color='darkred', lw=1.5, label='Regime 2 (φ=0.8, high persistence)')
ax.axvline(x=Tb, color='black', ls='--', lw=2, label=f'Break Point (t={Tb})')
ax.set_ylabel('y(t)', fontsize=11, fontweight='bold')
ax.set_title('Parameter Break: AR(1) with φ Change', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom: Rolling autocorrelation estimate
window = 40
def estimate_phi(y_seg):
    if len(y_seg) < 5:
        return 0.0
    return np.corrcoef(y_seg[:-1], y_seg[1:])[0, 1] if np.var(y_seg) > 1e-10 else 0.0

roll_phi = [estimate_phi(y_param_dgp[max(0, t-window):t]) for t in range(T)]
ax = axes[1]
ax.plot(range(T), roll_phi, color='purple', lw=2, label=f'Rolling φ Estimate (w={window})')
ax.axhline(y=phi1, color='steelblue', ls=':', lw=2, label=f'True φ₁ = {phi1}')
ax.axhline(y=phi2, color='darkred', ls=':', lw=2, label=f'True φ₂ = {phi2}')
ax.axvline(x=Tb, color='black', ls='--', lw=2)
ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
ax.set_ylabel('AR(1) Coefficient (φ)', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dgp_parameter_break.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved")

# DGP Figure 4: Parameter Persistence Panel (p=0.90, 0.95, 0.99)
print("[DGP 4/6] Parameter Persistence Panel...")

persistence_levels = [0.90, 0.95, 0.99]
T = 400
phi0, phi1, sigma = 0.3, 0.8, 1.0

fig, axes = plt.subplots(len(persistence_levels), 1, figsize=(14, 3*len(persistence_levels)), sharex=True)

for ax, p_stay in zip(axes, persistence_levels):
    np.random.seed(42)  # Same seed for comparison
    
    # Simulate regime sequence
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        if np.random.random() < p_stay:
            regime[t] = regime[t-1]
        else:
            regime[t] = 1 - regime[t-1]
    
    # Simulate time series with parameter switching
    y = np.zeros(T)
    y[0] = np.random.normal(0, sigma)
    for t in range(1, T):
        phi = phi0 if regime[t] == 0 else phi1
        y[t] = phi * y[t-1] + np.random.normal(0, sigma)
    
    # Plot time series
    ax.plot(y, color='black', lw=1.2)
    
    # Highlight regime 1 periods
    for t in range(1, T):
        if regime[t] == 1:
            ax.axvspan(t-1, t, color='lightcoral', alpha=0.4)
    
    ax.set_title(f'Markov-Switching AR(1) Parameter Break: Persistence p = {p_stay}', fontsize=11, fontweight='bold')
    ax.set_ylabel('y(t)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add legend only to top plot
    if ax == axes[0]:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label=f'Regime 0 (φ={phi0})'),
            Patch(facecolor='lightcoral', alpha=0.6, label=f'Regime 1 (φ={phi1})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

axes[-1].set_xlabel('Time (t)', fontsize=11, fontweight='bold')
plt.suptitle('Recurring Parameter Break DGP Across Persistence Levels', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'recurring_parameter_dgp_persistence_panel.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved")

# DGP Figure 5: Recurring Mean DGP (Top Only)
print("[DGP 5/6] Recurring Mean DGP (Top Only)...")

np.random.seed(42)
T = 400
p_stay = 0.95
mu0, mu1, phi, sigma = 0.0, 2.0, 0.6, 1.0

# Simulate regime sequence
regime = np.zeros(T, dtype=int)
for t in range(1, T):
    if np.random.random() < p_stay:
        regime[t] = regime[t-1]
    else:
        regime[t] = 1 - regime[t-1]

# Simulate time series with mean switching
y = np.zeros(T)
y[0] = mu0 + np.random.normal(0, sigma)
for t in range(1, T):
    mu_curr = mu0 if regime[t] == 0 else mu1
    mu_prev = mu0 if regime[t-1] == 0 else mu1
    y[t] = mu_curr + phi * (y[t-1] - mu_prev) + np.random.normal(0, sigma)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y, color='black', lw=1.2)

# Highlight regime 1 periods
for t in range(1, T):
    if regime[t] == 1:
        ax.axvspan(t-1, t, color='lightcoral', alpha=0.4)

# Add mean reference lines
ax.axhline(y=mu0, color='steelblue', ls=':', lw=1.5, alpha=0.7)
ax.axhline(y=mu1, color='darkred', ls=':', lw=1.5, alpha=0.7)

ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
ax.set_ylabel('y(t)', fontsize=10)
ax.set_title(f'Recurring Mean Break DGP (persistence p = {p_stay})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label=f'Regime 0 (μ={mu0})'),
    Patch(facecolor='lightcoral', alpha=0.6, label=f'Regime 1 (μ={mu1})')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'recurring_mean_dgp_toponly.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved")

# DGP Figure 9: Recurring Variance DGP (Top Only)
print("[DGP 6/6] Recurring Variance DGP (Top Only)...")

np.random.seed(42)
T = 400
p_stay = 0.95
phi, sigma0, sigma1 = 0.6, 1.0, 2.5

# Simulate regime sequence
regime = np.zeros(T, dtype=int)
for t in range(1, T):
    if np.random.random() < p_stay:
        regime[t] = regime[t-1]
    else:
        regime[t] = 1 - regime[t-1]

# Simulate time series with variance switching
y = np.zeros(T)
y[0] = np.random.normal(0, sigma0)
for t in range(1, T):
    sigma = sigma0 if regime[t] == 0 else sigma1
    y[t] = phi * y[t-1] + np.random.normal(0, sigma)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y, color='black', lw=1.2)

# Highlight regime 1 periods
for t in range(1, T):
    if regime[t] == 1:
        ax.axvspan(t-1, t, color='lightcoral', alpha=0.4)

ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
ax.set_ylabel('y(t)', fontsize=10)
ax.set_title(f'Recurring Variance Break DGP (persistence p = {p_stay})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label=f'Regime 0 (σ={sigma0})'),
    Patch(facecolor='lightcoral', alpha=0.6, label=f'Regime 1 (σ={sigma1})')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'recurring_variance_dgp_toponly.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved")

print("\n" + "="*60)
print("📊 Analysis Figures (Real Simulation Data)")
print("="*60 + "\n")

# =========================================================================
# FIGURE 1: Variance Break - Method Comparison (REAL DATA)
# =========================================================================

print("[1/7] Variance Break Method Comparison (Real Data)...")

df_var_single = load_results('variance_single_results.csv')
df_var_rec = load_results('variance_recurring_p0.95_results.csv')

if df_var_single is not None and df_var_rec is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Single break
    ax = axes[0]
    methods = df_var_single['Method'].tolist()
    rmse = df_var_single['RMSE'].tolist()
    colors = ['steelblue', 'coral', 'gold', 'darkgreen'][:len(methods)]
    bars = ax.barh(range(len(methods)), rmse, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Single Variance Break', fontsize=12, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, rmse)):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim([min(rmse)*0.95, max(rmse)*1.08])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Recurring break
    ax = axes[1]
    methods = df_var_rec['Method'].tolist()
    rmse = df_var_rec['RMSE'].tolist()
    colors = ['darkred', 'darkgreen', 'seagreen', 'orange'][:len(methods)]
    bars = ax.barh(range(len(methods)), rmse, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Recurring Variance Break (p=0.95)', fontsize=12, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, rmse)):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim([min(rmse)*0.99, max(rmse)*1.03])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Variance Break: Method Performance Comparison (300 MC simulations)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'variance_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
else:
    print("  ⚠ Missing variance data files")

# =========================================================================
# FIGURE 1b: Variance Deep Dive - Coverage Calibration & Log Score
# =========================================================================

print("[1b/7] Variance Deep Dive (Coverage & Probabilistic Accuracy)...")

df_var = load_results('variance_single_results.csv')

if df_var is not None and 'Coverage80' in df_var.columns and 'LogScore' in df_var.columns:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    methods = df_var['Method'].tolist()
    cov80_actual = df_var['Coverage80'].tolist()
    cov95_actual = df_var['Coverage95'].tolist()
    logscore = df_var['LogScore'].tolist()
    
    # Colors for each method
    colors = {'GARCH': 'darkred', 'SARIMA Rolling': 'darkblue', 
              'SARIMA Avg-Window': 'purple', 'SARIMA Global': 'darkorange'}
    method_colors = [colors.get(m, 'gray') for m in methods]
    
    # Panel 1: Coverage Calibration Plot
    ax = axes[0]
    ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', lw=2, alpha=0.5, label='Perfect Calibration')
    
    for i, m in enumerate(methods):
        # Plot 80% coverage point
        ax.scatter(0.80, cov80_actual[i], s=150, c=method_colors[i], marker='o', 
                   edgecolors='black', linewidth=1.5, zorder=3)
        # Plot 95% coverage point
        ax.scatter(0.95, cov95_actual[i], s=150, c=method_colors[i], marker='s',
                   edgecolors='black', linewidth=1.5, zorder=3, label=m)
    
    ax.set_xlabel('Nominal Coverage', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Coverage', fontsize=11, fontweight='bold')
    ax.set_title('Coverage Calibration\n(closer to diagonal = better)', fontsize=11, fontweight='bold')
    ax.set_xlim([0.75, 1.0])
    ax.set_ylim([0.6, 1.0])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for under/over coverage
    ax.fill_between([0.75, 1.0], [0.75, 1.0], [0.6, 0.6], alpha=0.1, color='red')
    ax.fill_between([0.75, 1.0], [0.75, 1.0], [1.0, 1.0], alpha=0.1, color='green')
    ax.text(0.77, 0.62, 'Under-coverage\n(intervals too narrow)', fontsize=8, color='darkred')
    ax.text(0.77, 0.97, 'Over-coverage\n(intervals too wide)', fontsize=8, color='darkgreen')
    
    # Panel 2: Log Score Comparison
    ax = axes[1]
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, logscore, color=method_colors, alpha=0.85, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('Log Score (less negative = better)', fontsize=11, fontweight='bold')
    ax.set_title('Probabilistic Forecast Quality\n(penalizes mis-calibrated variance)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, logscore):
        ax.text(val - 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='right', fontsize=9, fontweight='bold', color='white')
    
    # Mark best
    best_idx = np.argmax(logscore)
    ax.annotate('BEST', xy=(logscore[best_idx], best_idx), xytext=(logscore[best_idx] + 0.05, best_idx),
               fontsize=9, fontweight='bold', color='darkgreen',
               arrowprops=dict(arrowstyle='->', color='darkgreen'))
    
    # Panel 3: Variance Dynamics Illustration
    ax = axes[2]
    np.random.seed(42)
    T, Tb = 200, 100
    sigma1, sigma2, phi = 1.0, 2.5, 0.6
    
    # Simulate
    y = np.zeros(T)
    for t in range(1, T):
        sigma = sigma1 if t < Tb else sigma2
        y[t] = phi * y[t-1] + np.random.normal(0, sigma)
    
    # Rolling variance estimates (different windows)
    windows = {'w=20': 20, 'w=40': 40, 'w=80': 80}
    
    ax.plot(y, color='black', lw=0.8, alpha=0.5, label='Time Series')
    ax.axvline(x=Tb, color='red', ls='--', lw=2, label='Variance Break')
    
    # Plot ±2σ bands for true variance
    true_sigma = np.array([sigma1 if t < Tb else sigma2 for t in range(T)])
    ax.fill_between(range(T), -2*true_sigma, 2*true_sigma, alpha=0.15, color='gray', label='True ±2σ')
    
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('y(t)', fontsize=11, fontweight='bold')
    ax.set_title('Variance Break Dynamics\n(spread increases after break)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Variance Break Deep Dive: Calibration, Probabilistic Accuracy, and Dynamics', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'variance_deep_dive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
else:
    print("  ⚠ Missing variance data or required columns")

# =========================================================================
# FIGURE 2: Mean Break - Method Comparison (REAL DATA)
# =========================================================================

print("[2/7] Mean Break Method Comparison (Real Data)...")

df_mean_single = load_results('mean_single_results.csv')
df_mean_rec = load_results('mean_recurring_p0.95_results.csv')

if df_mean_single is not None and df_mean_rec is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Single break
    ax = axes[0]
    methods = df_mean_single['Method'].tolist()
    rmse = df_mean_single['RMSE'].tolist()
    colors = ['steelblue', 'coral', 'gold', 'darkgreen', 'purple'][:len(methods)]
    bars = ax.barh(range(len(methods)), rmse, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Single Mean Break', fontsize=12, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, rmse)):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim([min(rmse)*0.95, max(rmse)*1.06])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Recurring break
    ax = axes[1]
    methods = df_mean_rec['Method'].tolist()
    rmse = df_mean_rec['RMSE'].tolist()
    colors = ['darkred', 'darkgreen', 'seagreen', 'orange'][:len(methods)]
    bars = ax.barh(range(len(methods)), rmse, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Recurring Mean Break (p=0.95)', fontsize=12, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, rmse)):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim([min(rmse)*0.99, max(rmse)*1.03])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Mean Break: Method Performance Comparison (300 MC simulations)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mean_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
else:
    print("  ⚠ Missing mean data files")

# =========================================================================
# FIGURE 3: Parameter Break - Method Comparison (REAL DATA)
# =========================================================================

print("[3/7] Parameter Break Method Comparison (Real Data)...")

df_param_single = load_results('parameter_single_results.csv')
df_param_rec = load_results('parameter_recurring_p0.95_results.csv')

if df_param_single is not None and df_param_rec is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Single break
    ax = axes[0]
    methods = df_param_single['Method'].tolist()
    rmse = df_param_single['RMSE'].tolist()
    colors = ['steelblue', 'coral', 'gold', 'darkgreen'][:len(methods)]
    bars = ax.barh(range(len(methods)), rmse, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Single Parameter Break', fontsize=12, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, rmse)):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim([min(rmse)*0.95, max(rmse)*1.08])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Recurring break
    ax = axes[1]
    methods = df_param_rec['Method'].tolist()
    rmse = df_param_rec['RMSE'].tolist()
    colors = ['darkred', 'darkgreen', 'orange'][:len(methods)]
    bars = ax.barh(range(len(methods)), rmse, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Recurring Parameter Break (p=0.95)', fontsize=12, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, rmse)):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim([min(rmse)*0.99, max(rmse)*1.05])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Parameter Break: Method Performance Comparison (300 MC simulations)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'parameter_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
else:
    print("  ⚠ Missing parameter data files")

# =========================================================================
# FIGURE 4: Bias Comparison Across Break Types (REAL DATA)
# =========================================================================

print("[4/7] Bias Comparison (Real Data)...")

df_var = load_results('variance_recurring_p0.95_results.csv')
df_mean = load_results('mean_recurring_p0.95_results.csv')
df_param = load_results('parameter_recurring_p0.95_results.csv')

if all([df_var is not None, df_mean is not None, df_param is not None]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, df, title in zip(axes, [df_var, df_mean, df_param], 
                               ['Variance', 'Mean', 'Parameter']):
        methods = df['Method'].tolist()
        bias = df['Bias'].tolist()
        
        colors = ['darkred' if b < 0 else 'darkgreen' for b in bias]
        bars = ax.barh(range(len(methods)), bias, color=colors, alpha=0.8, edgecolor='black')
        ax.axvline(x=0, color='black', ls='-', lw=1.5)
        
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel('Bias', fontsize=10, fontweight='bold')
        ax.set_title(f'{title} Recurring (p=0.95)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, v) in enumerate(zip(bars, bias)):
            offset = 0.01 if v >= 0 else -0.01
            ha = 'left' if v >= 0 else 'right'
            ax.text(v + offset, i, f'{v:.3f}', va='center', ha=ha, fontsize=8)
    
    plt.suptitle('Forecast Bias by Method (300 MC simulations)\nNegative = underprediction, Positive = overprediction', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bias_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
else:
    print("  ⚠ Missing data files for bias comparison")

# =========================================================================
# FIGURE 5: Persistence Comparison Across p (REAL DATA)
# =========================================================================

print("[5/5] Persistence Analysis (Real Data)...")

# Load all persistence levels
df_p09 = load_results('parameter_recurring_p0.9_results.csv')
df_p095 = load_results('parameter_recurring_p0.95_results.csv')
df_p099 = load_results('parameter_recurring_p0.99_results.csv')

if all([df_p09 is not None, df_p095 is not None, df_p099 is not None]):
    persistence_levels = [0.90, 0.95, 0.99]
    
    # Extract RMSE for each method across persistence levels
    ms_rmse = []
    rolling_rmse = []
    global_rmse = []
    
    for df in [df_p09, df_p095, df_p099]:
        for _, row in df.iterrows():
            method = row['Method']
            if 'MS' in method:
                ms_rmse.append(row['RMSE'])
            elif 'Rolling' in method:
                rolling_rmse.append(row['RMSE'])
            elif 'Global' in method:
                global_rmse.append(row['RMSE'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(persistence_levels, ms_rmse, marker='s', linewidth=2.5, markersize=10,
            label='Markov-Switching AR', color='darkred', alpha=0.85, zorder=3)
    ax.plot(persistence_levels, rolling_rmse, marker='o', linewidth=2.5, markersize=10,
            label='Rolling SARIMA', color='darkgreen', alpha=0.85, zorder=3)
    ax.plot(persistence_levels, global_rmse, marker='^', linewidth=2.5, markersize=10,
            label='Global SARIMA', color='orange', alpha=0.85, zorder=3)
    
    ax.fill_between(persistence_levels, ms_rmse, rolling_rmse, alpha=0.15, color='darkred')
    
    ax.set_xlabel('Regime Persistence (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE (1-step ahead)', fontsize=12, fontweight='bold')
    ax.set_title('Real Simulation Data: MS AR vs SARIMA Across Persistence Levels\n(300 MC simulations per setting)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.90, 0.95, 0.99])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'persistence_comparison_real.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
else:
    print("  ⚠ Missing persistence data files")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'='*60}")
print("✅ Figure generation complete!")

print(f"\n📁 Figures saved to {FIGURES_DIR}:")
for i, f in enumerate(sorted(FIGURES_DIR.glob('*.png')), 1):
    print(f"   {i}. {f.name}")

print(f"\n📊 NEXT STEP: Build PDF with figures:")
print("   pixi run python scripts/build_pdfs.py --figures")
print("")
print("   Or build combined PDF (tables + figures):")
print("   pixi run python scripts/build_pdfs.py --all")
print("="*60)
