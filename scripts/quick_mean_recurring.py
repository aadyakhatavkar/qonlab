#!/usr/bin/env python3
"""Quick mean recurring break simulation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyses import mc_single_sarima, mc_multiple_sarima
import pandas as pd

print("\n" + "="*70)
print(" MEAN BREAK EXPERIMENTS - RECURRING (QUICK)")
print("="*70)

# Parameters - QUICK mode for speed
T = 150
Tb = 75
N_SIM = 20  # Quick: reduced from 300
SEED = 42

# Single mean break (this is already in results)
print(f"\n[1/2] Single mean break (T={T}, Tb={Tb}, n_sim={N_SIM})...")
print("      ✓ Already completed (full results available)")

# Multiple mean breaks - this is the "recurring" equivalent for mean breaks
print(f"\n[2/2] Multiple mean breaks (T={T}, n_sim={N_SIM})...")
b1 = Tb - 25
b2 = Tb + 25

df_mean_multiple = mc_multiple_sarima(
    n_sim=N_SIM, T=T, b1=b1, b2=b2,
    window=30, seed=SEED
)
print(f"      ✓ Completed ({len(df_mean_multiple)} rows)")

print("\nRESULTS - MEAN RECURRING (Multiple Breaks):")
print(df_mean_multiple.to_string(index=False))

# Calculate variance
if 'RMSE' in df_mean_multiple.columns and 'Bias' in df_mean_multiple.columns:
    df_mean_multiple_display = df_mean_multiple.copy()
    df_mean_multiple_display['Variance'] = df_mean_multiple['RMSE']**2 - df_mean_multiple['Bias']**2
    
    print("\nSUMMARY - RMSE, BIAS, VARIANCE:")
    summary_cols = ['RMSE', 'Bias', 'Variance']
    display_df = df_mean_multiple_display[summary_cols]
    print(display_df.to_string(index=False))

print("\n" + "="*70)

