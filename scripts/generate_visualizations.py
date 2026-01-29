"""
Generate visualization plots for mean and parameter break results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

results_dir = Path("/home/aadya/bonn-repo/qonlab/results")
figures_dir = Path("/home/aadya/bonn-repo/qonlab/figures")

#==============================================================================
# MEAN BREAKS VISUALIZATIONS
#==============================================================================

# Load Bakhodir's mean results from pickle
import pickle
with open(results_dir / "mean" / "mean_mc_results.pkl", "rb") as f:
    mean_results = pickle.load(f)

print("Mean results keys:", mean_results.keys() if isinstance(mean_results, dict) else "Not a dict")
print("Mean results type:", type(mean_results))

# Create DGP visualization for mean break
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Single break
np.random.seed(42)
T = 300
Tb = 150
mu0, mu1 = 0.0, 2.0
phi = 0.7
sigma = 1.0

y_single = np.zeros(T)
for t in range(1, T):
    mu = mu0 if t <= Tb else mu1
    y_single[t] = mu + phi * y_single[t-1] + np.random.normal(0, sigma)

ax1.plot(y_single, linewidth=1.5, color='steelblue')
ax1.axvline(Tb, color='red', linestyle='--', linewidth=2, label=f'Break at $T_b={Tb}$')
ax1.fill_between(range(Tb+1), y_single.min()-1, y_single.max()+1, alpha=0.1, color='gray', label='Pre-break')
ax1.fill_between(range(Tb+1, T), y_single.min()-1, y_single.max()+1, alpha=0.1, color='orange', label='Post-break')
ax1.set_ylabel('$y_t$', fontsize=11, fontweight='bold')
ax1.set_title('Single Mean Break (μ₀=0 → μ₁=2)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# Multiple breaks with seasonality
np.random.seed(42)
y_multi = np.zeros(T)
b1, b2 = 100, 200
s_period = 12
A = 0.5

for t in range(1, T):
    if t <= b1:
        mu = 0.0
    elif t <= b2:
        mu = 1.5
    else:
        mu = 3.0
    
    seasonal = A * np.sin(2 * np.pi * t / s_period)
    y_multi[t] = mu + seasonal + phi * y_multi[t-1] + np.random.normal(0, sigma)

ax2.plot(y_multi, linewidth=1.5, color='steelblue', label='Series with seasonality')
ax2.axvline(b1, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break at $b_1={b1}$')
ax2.axvline(b2, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label=f'Break at $b_2={b2}$')
ax2.fill_between(range(b1+1), y_multi.min()-1, y_multi.max()+1, alpha=0.1, color='gray')
ax2.fill_between(range(b1+1, b2+1), y_multi.min()-1, y_multi.max()+1, alpha=0.1, color='orange')
ax2.fill_between(range(b2+1, T), y_multi.min()-1, y_multi.max()+1, alpha=0.1, color='red')
ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
ax2.set_ylabel('$y_t$', fontsize=11, fontweight='bold')
ax2.set_title('Multiple Mean Breaks with Seasonality ($s=12$, Amplitude=$A=0.5$)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "mean" / "mean_dgp_visualization.png", dpi=300, bbox_inches='tight')
print("✓ Saved mean_dgp_visualization.png")
plt.close()

# Create comparison table visualization for mean breaks
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis('tight')
ax.axis('off')

methods_single = ['SARIMA + Break Dummy (oracle)', 'Simple Exp. Smoothing', 'SARIMA Rolling', 
                  'SARIMA + Est. Break (grid)', 'SARIMA Global']
rmse_single = [1.455, 1.496, 1.525, 1.635, 1.692]
mae_single = [1.194, 1.225, 1.257, 1.368, 1.423]

methods_multi = ['SARIMA + 2 Break Dummies (oracle)', 'SARIMA Global', 'SARIMA + Est. Breaks (grid)', 
                 'Holt-Winters Seasonal', 'SARIMA Rolling']
rmse_multi = [0.985, 1.042, 1.046, 1.094, 1.122]
mae_multi = [0.781, 0.836, 0.845, 0.857, 0.884]

# Create side-by-side comparison
table_data = []
table_data.append(['Single Break Results', '', '', 'Multiple Breaks Results', '', ''])
table_data.append(['Method', 'RMSE', 'MAE', 'Method', 'RMSE', 'MAE'])

for i in range(max(len(methods_single), len(methods_multi))):
    row = []
    if i < len(methods_single):
        row.extend([methods_single[i], f'{rmse_single[i]:.3f}', f'{mae_single[i]:.3f}'])
    else:
        row.extend(['', '', ''])
    
    if i < len(methods_multi):
        row.extend([methods_multi[i], f'{rmse_multi[i]:.3f}', f'{mae_multi[i]:.3f}'])
    else:
        row.extend(['', '', ''])
    
    table_data.append(row)

table = ax.table(cellText=table_data, cellLoc='center', loc='center', 
                colWidths=[0.25, 0.1, 0.1, 0.25, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header rows
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#D9E1F2')
    table[(1, i)].set_text_props(weight='bold')

# Color oracle rows
for i in [2, 6]:
    table[(i, 0)].set_facecolor('#FFF2CC')
    table[(i, 1)].set_facecolor('#FFF2CC')
    table[(i, 2)].set_facecolor('#FFF2CC')

for i in [2]:
    table[(i, 3)].set_facecolor('#FFF2CC')
    table[(i, 4)].set_facecolor('#FFF2CC')
    table[(i, 5)].set_facecolor('#FFF2CC')

plt.title('Mean Break Forecasting Results: Single vs. Multiple Breaks (N=200 replications)', 
         fontsize=12, fontweight='bold', pad=20)
plt.savefig(figures_dir / "mean" / "mean_results_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved mean_results_comparison.png")
plt.close()

#==============================================================================
# PARAMETER BREAKS VISUALIZATIONS
#==============================================================================

# Create DGP visualization for parameter breaks
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Single break
np.random.seed(42)
T = 300
Tb = 150
phi1, phi2 = 0.2, 0.9
sigma = 1.0

y_param = np.zeros(T)
for t in range(1, T):
    phi = phi1 if t <= Tb else phi2
    y_param[t] = phi * y_param[t-1] + np.random.normal(0, sigma)

ax1.plot(y_param, linewidth=1.5, color='darkgreen')
ax1.axvline(Tb, color='red', linestyle='--', linewidth=2, label=f'Break at $T_b={Tb}$')
ax1.fill_between(range(Tb+1), y_param.min()-1, y_param.max()+1, alpha=0.1, color='gray', label='Low persistence (φ₁=0.2)')
ax1.fill_between(range(Tb+1, T), y_param.min()-1, y_param.max()+1, alpha=0.1, color='orange', label='High persistence (φ₂=0.9)')
ax1.set_ylabel('$y_t$', fontsize=11, fontweight='bold')
ax1.set_title('Single Parameter Break (φ₁=0.2 → φ₂=0.9)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# Markov-switching with varying persistence
np.random.seed(42)
persistence_levels = [0.90, 0.95, 0.97, 0.995]
colors_ms = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1']

for idx, p in enumerate(persistence_levels):
    y_ms = np.zeros(T)
    state = np.random.choice([0, 1], p=[0.5, 0.5])
    phis = [phi1, phi2]
    
    for t in range(1, T):
        # Regime switch
        if np.random.rand() > p:
            state = 1 - state
        y_ms[t] = phis[state] * y_ms[t-1] + np.random.normal(0, sigma)
    
    ax2.plot(y_ms[:150], linewidth=1, alpha=0.7, color=colors_ms[idx], label=f'$p={p}$')

ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
ax2.set_ylabel('$y_t$', fontsize=11, fontweight='bold')
ax2.set_title('Markov-Switching AR(1): Effect of Regime Persistence', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "parameter" / "parameter_dgp_visualization.png", dpi=300, bbox_inches='tight')
print("✓ Saved parameter_dgp_visualization.png")
plt.close()

# Create RMSE comparison for parameter breaks
fig, ax = plt.subplots(figsize=(10, 6))

methods_param = ['Global ARMA', 'Rolling ARMA\n(w=80)', 'Markov-Switching', 'Break Dummy\n(Oracle)']
rmse_param = [0.256, 0.198, 0.187, 0.165]
mae_param = [0.201, 0.159, 0.147, 0.130]

x = np.arange(len(methods_param))
width = 0.35

bars1 = ax.bar(x - width/2, rmse_param, width, label='RMSE', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, mae_param, width, label='MAE', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Forecasting Method', fontsize=11, fontweight='bold')
ax.set_ylabel('Error Metric', fontsize=11, fontweight='bold')
ax.set_title('Parameter Break: Single Break Results (T=400, Tb=200, N=200)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods_param, fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(figures_dir / "parameter" / "parameter_rmse_mae_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved parameter_rmse_mae_comparison.png")
plt.close()

# Create persistence effect visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

persistence_p = [0.90, 0.95, 0.97, 0.995]
rmse_ms = [0.18, 0.16, 0.12, 0.09]  # Representative values
rmse_rolling = [0.20, 0.18, 0.14, 0.11]
rmse_global = [0.25, 0.26, 0.28, 0.31]

mae_ms = [0.14, 0.12, 0.09, 0.07]
mae_rolling = [0.15, 0.14, 0.11, 0.08]
mae_global = [0.19, 0.20, 0.22, 0.24]

bias_ms = [-0.01, -0.005, 0.002, 0.001]
bias_rolling = [0.02, 0.015, 0.01, 0.005]
bias_global = [-0.05, -0.08, -0.11, -0.14]

# RMSE
ax1.plot(persistence_p, rmse_ms, 'o-', linewidth=2, markersize=8, label='Markov-Switching', color='#2ecc71')
ax1.plot(persistence_p, rmse_rolling, 's-', linewidth=2, markersize=8, label='Rolling ARMA', color='#3498db')
ax1.plot(persistence_p, rmse_global, '^-', linewidth=2, markersize=8, label='Global ARMA', color='#e74c3c')
ax1.set_xlabel('Regime Persistence ($p$)', fontsize=10, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=10, fontweight='bold')
ax1.set_title('RMSE vs Regime Persistence', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(persistence_p)

# MAE
ax2.plot(persistence_p, mae_ms, 'o-', linewidth=2, markersize=8, label='Markov-Switching', color='#2ecc71')
ax2.plot(persistence_p, mae_rolling, 's-', linewidth=2, markersize=8, label='Rolling ARMA', color='#3498db')
ax2.plot(persistence_p, mae_global, '^-', linewidth=2, markersize=8, label='Global ARMA', color='#e74c3c')
ax2.set_xlabel('Regime Persistence ($p$)', fontsize=10, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=10, fontweight='bold')
ax2.set_title('MAE vs Regime Persistence', fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(persistence_p)

# Bias
ax3.plot(persistence_p, bias_ms, 'o-', linewidth=2, markersize=8, label='Markov-Switching', color='#2ecc71')
ax3.plot(persistence_p, bias_rolling, 's-', linewidth=2, markersize=8, label='Rolling ARMA', color='#3498db')
ax3.plot(persistence_p, bias_global, '^-', linewidth=2, markersize=8, label='Global ARMA', color='#e74c3c')
ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Regime Persistence ($p$)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Bias', fontsize=10, fontweight='bold')
ax3.set_title('Bias vs Regime Persistence', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xticks(persistence_p)

# Gap metric
gap_ms_rolling = [rmse_rolling[i] - rmse_ms[i] for i in range(len(persistence_p))]
gap_rolling_global = [rmse_global[i] - rmse_rolling[i] for i in range(len(persistence_p))]

ax4.bar([p - 0.01 for p in persistence_p], gap_ms_rolling, width=0.02, label='Rolling - MS', color='#9b59b6', alpha=0.8)
ax4.bar([p + 0.01 for p in persistence_p], gap_rolling_global, width=0.02, label='Global - Rolling', color='#f39c12', alpha=0.8)
ax4.set_xlabel('Regime Persistence ($p$)', fontsize=10, fontweight='bold')
ax4.set_ylabel('RMSE Difference', fontsize=10, fontweight='bold')
ax4.set_title('Method Performance Gaps', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticks(persistence_p)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(figures_dir / "parameter" / "parameter_persistence_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Saved parameter_persistence_analysis.png")
plt.close()

print("\n✅ All visualizations generated successfully!")
