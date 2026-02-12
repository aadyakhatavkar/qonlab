#!/usr/bin/env python3
"""
Structural Break Experiments Runner
====================================
Orchestrates MC simulations for all 3 break types with standardized parameters.
Uses existing MC functions from analyses/ folder.

STANDARDIZED PARAMETERS:
  T = 400       (time series length)
  Tb = 200      (single break point)
  n_sim = 300   (Monte Carlo simulations)

BREAK TYPES:
  1. Variance break   → mc_variance_single_break(), mc_variance_recurring()
  2. Mean break       → mc_single_sarima(), mc_multiple_sarima()
  3. Parameter break  → monte_carlo_single_break_post(), monte_carlo_recurring()

Calls existing MC functions from analyses/ with aligned parameters.
Features:
  - Multiple innovation types (Gaussian, Student-t)
  - Multiple persistence levels for recurring breaks
  - Summary statistics and best method reporting
  - Results aggregation and comparison
"""
import argparse
import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path so imports work from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyses import (
    mc_variance_single_break, mc_variance_recurring,
    mc_single_sarima, mc_multiple_sarima,
    monte_carlo_single_break_post, monte_carlo_recurring
)

# =========================================================
# STANDARDIZED PARAMETERS (ALL EXPERIMENTS)
# =========================================================
T = 400                 # Time series length
Tb = 200                # Break point (single breaks)
N_SIM = 300             # Monte Carlo simulations
WINDOW = 100            # SARIMA rolling window
SEED = 42               # Random seed

# Experiment variants
INNOVATION_TYPES = [("normal", None), ("student", 3), ("student", 5)]
PERSISTENCE_LEVELS = [0.90, 0.95, 0.99]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_best_methods(df, group_col="Scenario", metric="RMSE"):
    """Print best methods grouped by scenario."""
    if len(df) == 0 or metric not in df.columns:
        return
    
    print(f"\n--- Best Methods (by {metric}) ---")
    for group in df[group_col].unique():
        if pd.isna(group):
            continue
        sub = df[(df[group_col] == group) & (~df[metric].isna())].sort_values(metric)
        if len(sub) > 0:
            best = sub.iloc[0]
            print(f"  {group:30s}: {best.get('Method', 'N/A'):15s} ({metric}={best[metric]:.4f})")


def print_summary_stats(df):
    """Print summary statistics of results."""
    print(f"\n--- Summary Statistics ---")
    numeric_cols = ['RMSE', 'MAE', 'Bias', 'Variance']
    for col in numeric_cols:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                print(f"  {col:10s}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                      f"min={valid.min():.4f}, max={valid.max():.4f}")


# =========================================================
# MAIN EXECUTION - Run All 3 Break Types
# =========================================================

def run_variance_breaks():
    """Run variance break experiments (single + recurring with multiple persistence levels)."""
    print("\n" + "="*70)
    print(" VARIANCE BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    
    # Single variance break with multiple innovation types
    print(f"\n[1/2] Single variance break (T={T}, Tb={Tb}, n_sim={N_SIM})...")
    for innov_type, dof in INNOVATION_TYPES:
        innov_label = f"t(df={dof})" if dof else "Gaussian"
        print(f"      └─ {innov_label}...", end=" ", flush=True)
        t0 = time.time()
        df_var_single = mc_variance_single_break(
            n_sim=N_SIM, T=T, Tb=Tb, 
            sigma1=1.0, sigma2=2.0,
            window=WINDOW, seed=SEED,
            innovation_type=innov_type, dof=dof
        )
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df_var_single)} rows)")
        all_dfs.append(df_var_single)
    
    # Recurring variance break (no persistence levels needed for variance)
    print(f"\n[2/2] Recurring variance breaks (T={T}, n_sim={N_SIM})...")
    t0 = time.time()
    df_var_recurring = mc_variance_recurring(
        n_sim=N_SIM, T=T, p=0.95,
        sigma1=1.0, sigma2=2.0,
        window=WINDOW, seed=SEED
    )
    elapsed = time.time() - t0
    print(f"      ✓ Completed in {elapsed:.1f}s ({len(df_var_recurring)} rows)")
    all_dfs.append(df_var_recurring)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "Scenario", "RMSE")
    print_summary_stats(df_combined)
    return df_combined


def run_mean_breaks():
    """Run mean break experiments (single + multiple)."""
    print("\n" + "="*70)
    print(" MEAN BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    
    # Single mean break with multiple innovation types
    print(f"\n[1/2] Single mean break (T={T}, Tb={Tb}, n_sim={N_SIM})...")
    for innov_type, dof in INNOVATION_TYPES:
        innov_label = f"t(df={dof})" if dof else "Gaussian"
        print(f"      └─ {innov_label}...", end=" ", flush=True)
        t0 = time.time()
        df_mean_single = mc_single_sarima(
            n_sim=N_SIM, T=T, Tb=Tb,
            window=60, seed=SEED,
            innovation_type=innov_type, dof=dof
        )
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df_mean_single)} rows)")
        all_dfs.append(df_mean_single)
    
    # Multiple mean breaks
    print(f"\n[2/2] Multiple mean breaks (T={T}, n_sim={N_SIM})...")
    t0 = time.time()
    b1 = Tb - 50 if Tb >= 50 else max(50, Tb // 2)
    b2 = Tb + 50 if Tb + 50 < T else min(T - 50, Tb + (T - Tb) // 2)
    df_mean_multiple = mc_multiple_sarima(
        n_sim=N_SIM, T=T, b1=b1, b2=b2,
        window=60, seed=SEED
    )
    elapsed = time.time() - t0
    print(f"      ✓ Completed in {elapsed:.1f}s ({len(df_mean_multiple)} rows)")
    all_dfs.append(df_mean_multiple)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "Scenario", "RMSE")
    print_summary_stats(df_combined)
    return df_combined


def run_parameter_breaks():
    """Run parameter break experiments (single + recurring with multiple persistence levels)."""
    print("\n" + "="*70)
    print(" PARAMETER BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    
    # Single parameter break with multiple innovation types
    print(f"\n[1/2] Single parameter break (T={T}, Tb={Tb}, n_sim={N_SIM})...")
    for innov_type, dof in INNOVATION_TYPES:
        innov_label = f"t(df={dof})" if dof else "Gaussian"
        print(f"      └─ {innov_label}...", end=" ", flush=True)
        t0 = time.time()
        df_param_single = monte_carlo_single_break_post(
            n_sim=N_SIM, T=T, Tb=Tb,
            seed=SEED,
            innovation_type=innov_type, dof=dof
        )
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df_param_single)} rows)")
        all_dfs.append(df_param_single)
    
    # Recurring parameter break with multiple persistence levels
    print(f"\n[2/2] Recurring parameter breaks (T={T}, n_sim={N_SIM})...")
    for p in PERSISTENCE_LEVELS:
        print(f"      └─ Persistence p={p}...", end=" ", flush=True)
        t0 = time.time()
        df_param_recurring = monte_carlo_recurring(
            p=p, n_sim=N_SIM, T=T,
            seed=SEED
        )
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df_param_recurring)} rows)")
        all_dfs.append(df_param_recurring)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "Scenario", "RMSE")
    print_summary_stats(df_combined)
    return df_combined


def main():
    parser = argparse.ArgumentParser(
        description='Aligned Structural Break Experiments (All 3 Break Types)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Standardized Parameters:
  T = 400      (time series length)
  Tb = 200     (single break point)
  n_sim = 300  (Monte Carlo simulations)
  
Break Types:
  1. Variance:   single break + recurring (multiple persistence levels)
  2. Mean:       single break + multiple breaks
  3. Parameter:  single break + recurring (multiple persistence levels)
        """
    )
    parser.add_argument('--quick', action='store_true', help='Quick run (n_sim=10, T=150)')
    parser.add_argument('--variance', action='store_true', help='Run variance breaks only')
    parser.add_argument('--mean', action='store_true', help='Run mean breaks only')
    parser.add_argument('--parameter', action='store_true', help='Run parameter breaks only')
    args = parser.parse_args()
    
    # Override params for quick mode
    global T, Tb, N_SIM
    if args.quick:
        T = 150
        Tb = T // 2
        N_SIM = 10
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║    ALIGNED STRUCTURAL BREAK EXPERIMENTS                            ║
║                                                                    ║
║    Parameters: T={T}, Tb={Tb}, n_sim={N_SIM}                      ║
║    Persistence levels: {PERSISTENCE_LEVELS}                         ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    all_results = []
    total_time = time.time()
    
    # Determine which experiments to run
    run_all = not (args.variance or args.mean or args.parameter)
    
    if run_all or args.variance:
        all_results.append(run_variance_breaks())
    
    if run_all or args.mean:
        all_results.append(run_mean_breaks())
    
    if run_all or args.parameter:
        all_results.append(run_parameter_breaks())
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    total_time = time.time() - total_time
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/aligned_breaks_{timestamp}.csv"
    df_all.to_csv(filename, index=False)
    
    # Display final summary
    print(f"""
{'='*70}
OVERALL SUMMARY
{'='*70}
Total experiments: {len(df_all)} results
Time elapsed: {total_time:.1f} seconds

Results saved to: {filename}

Break Type Counts:
""")
    
    if 'Task' in df_all.columns:
        for task in ['variance', 'mean', 'parameter']:
            count = (df_all['Task'] == task).sum()
            if count > 0:
                print(f"  ✓ {task.capitalize():12s}: {count:3d} results")
    
    if 'Break' in df_all.columns:
        print("\nBreak Type Counts:")
        for btype in ['Single', 'Recurring']:
            count = (df_all['Break'] == btype).sum()
            if count > 0:
                print(f"  ✓ {btype:12s}: {count:3d} results")
    
    print(f"{'='*70}\n")
    
    # Show top methods overall
    if 'RMSE' in df_all.columns:
        print("TOP 5 BEST METHODS (by RMSE):")
        top_methods = df_all.nsmallest(5, 'RMSE')[['Scenario', 'Method', 'RMSE', 'MAE', 'Bias']]
        print(top_methods.to_string(index=False))
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
