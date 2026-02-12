#!/usr/bin/env python3
"""
Aligned Structural Break Experiments Runner
============================================
Authors: Aadya Khatavkar, Mahir Baylarov, Bakhodir Izzatulloev
University of Bonn | Winter Semester 2025/26

Unified runner for all 3 break types with standardized parameters.

STANDARDIZED PARAMETERS:
  T = 400       (time series length)
  Tb = 200      (single break point)
  n_sim = 300   (Monte Carlo simulations)

BREAK TYPES:
  1. Variance break   → single (3 innovations) + recurring (no persistence)
  2. Mean break       → single (3 innovations) + recurring (no persistence)
  3. Parameter break  → single (3 innovations) + recurring (3 persistence levels)

Features:
  - Innovation types for single breaks: Gaussian, Student-t(df=3), Student-t(df=5)
  - Persistence levels for parameter recurring: 0.90, 0.95, 0.99
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyses import (
    mc_variance_single_break, mc_variance_recurring,
    mc_single_sarima, mc_mean_recurring,
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
INNOVATION_TYPES = [
    ("gaussian", None),
    ("student", 5),
    ("student", 3),
]
PERSISTENCE_LEVELS = [0.90, 0.95, 0.99]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_best_methods(df, metric="RMSE"):
    """Print best methods grouped by innovation/persistence."""
    if len(df) == 0 or metric not in df.columns:
        return
    
    print(f"\n--- Best Methods (by {metric}) ---")
    
    # Check for Innovation column (single breaks)
    if 'Innovation' in df.columns:
        for inn in df['Innovation'].unique():
            if pd.isna(inn):
                continue
            sub = df[(df['Innovation'] == inn) & (~df[metric].isna())].sort_values(metric)
            if len(sub) > 0:
                best = sub.iloc[0]
                print(f"  {inn:20s}: {best.get('Method', 'N/A'):20s} ({metric}={best[metric]:.4f})")
    
    # Check for Persistence column (recurring breaks)
    if 'Persistence' in df.columns:
        for p in sorted(df['Persistence'].unique()):
            if pd.isna(p):
                continue
            sub = df[(df['Persistence'] == p) & (~df[metric].isna())].sort_values(metric)
            if len(sub) > 0:
                best = sub.iloc[0]
                print(f"  p={p:.2f}             : {best.get('Method', 'N/A'):20s} ({metric}={best[metric]:.4f})")


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
# VARIANCE BREAKS
# =========================================================

def run_variance_breaks():
    """Run variance break experiments (single + recurring)."""
    print("\n" + "="*70)
    print(" VARIANCE BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    
    # Single variance breaks - test across innovations
    print(f"\n[1/2] Single variance breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    variance_single_results = []
    
    for inn_type, dof in INNOVATION_TYPES:
        inn_name = f"Student-t(df={dof})" if inn_type == "student" else "Gaussian"
        print(f"      └─ {inn_name}...", end=" ", flush=True)
        t0 = time.time()
        
        df = mc_variance_single_break(
            n_sim=N_SIM, T=T, Tb=Tb,
            innovation_type=inn_type, dof=dof,
            seed=SEED
        )
        df['Innovation'] = inn_name
        variance_single_results.append(df)
        
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
    
    variance_single = pd.concat(variance_single_results, ignore_index=True)
    all_dfs.append(variance_single)
    
    # Recurring variance breaks
    print(f"\n[2/2] Recurring variance breaks (T={T}, n_sim={N_SIM})")
    print(f"      └─ Markov-switching...", end=" ", flush=True)
    t0 = time.time()
    
    variance_recurring = mc_variance_recurring(
        n_sim=N_SIM, T=T,
        seed=SEED
    )
    
    elapsed = time.time() - t0
    print(f"✓ ({elapsed:.1f}s, {len(variance_recurring)} rows)")
    all_dfs.append(variance_recurring)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "RMSE")
    print_summary_stats(df_combined)
    return df_combined


# =========================================================
# MEAN BREAKS
# =========================================================

def run_mean_breaks():
    """Run mean break experiments (single + recurring)."""
    print("\n" + "="*70)
    print(" MEAN BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    
    # Single mean breaks - test across innovations
    print(f"\n[1/2] Single mean breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    mean_single_results = []
    
    for inn_type, dof in INNOVATION_TYPES:
        inn_name = f"Student-t(df={dof})" if inn_type == "student" else "Gaussian"
        print(f"      └─ {inn_name}...", end=" ", flush=True)
        t0 = time.time()
        
        df = mc_single_sarima(
            n_sim=N_SIM, T=T, Tb=Tb,
            innovation_type=inn_type, dof=dof,
            seed=SEED
        )
        df['Innovation'] = inn_name
        mean_single_results.append(df)
        
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
    
    mean_single = pd.concat(mean_single_results, ignore_index=True)
    all_dfs.append(mean_single)
    
    # Recurring mean breaks
    print(f"\n[2/2] Recurring mean breaks (T={T}, n_sim={N_SIM})")
    print(f"      └─ Markov-switching...", end=" ", flush=True)
    t0 = time.time()
    
    mean_recurring = mc_mean_recurring(
        n_sim=N_SIM, T=T,
        seed=SEED
    )
    
    elapsed = time.time() - t0
    print(f"✓ ({elapsed:.1f}s, {len(mean_recurring)} rows)")
    all_dfs.append(mean_recurring)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "RMSE")
    print_summary_stats(df_combined)
    return df_combined


# =========================================================
# PARAMETER BREAKS
# =========================================================

def run_parameter_breaks():
    """Run parameter break experiments (single + recurring)."""
    print("\n" + "="*70)
    print(" PARAMETER BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    
    # Single parameter breaks - test across innovations
    print(f"\n[1/2] Single parameter breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    parameter_single_results = []
    
    # Set forecast origin to be Tb + gap
    t_post = min(Tb + 30, T - 5)
    
    for inn_type, dof in INNOVATION_TYPES:
        inn_name = f"Student-t(df={dof})" if inn_type == "student" else "Gaussian"
        print(f"      └─ {inn_name}...", end=" ", flush=True)
        t0 = time.time()
        
        df = monte_carlo_single_break_post(
            n_sim=N_SIM, T=T, Tb=Tb, t_post=t_post,
            innovation_type=inn_type, dof=dof,
            seed=SEED
        )
        df['Innovation'] = inn_name
        parameter_single_results.append(df)
        
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
    
    parameter_single = pd.concat(parameter_single_results, ignore_index=True)
    all_dfs.append(parameter_single)
    
    # Recurring parameter breaks - test across persistence levels
    print(f"\n[2/2] Recurring parameter breaks (T={T}, n_sim={N_SIM})")
    parameter_recurring_results = []
    
    # Set forecast origin for recurring
    t0_recurring = min(T - 50, max(T // 2, 100))
    
    for p in PERSISTENCE_LEVELS:
        print(f"      └─ p={p}...", end=" ", flush=True)
        t0 = time.time()
        
        df = monte_carlo_recurring(
            p=p, n_sim=N_SIM, T=T, t0=t0_recurring,
            seed=SEED
        )
        df['Persistence'] = p
        parameter_recurring_results.append(df)
        
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
    
    parameter_recurring = pd.concat(parameter_recurring_results, ignore_index=True)
    all_dfs.append(parameter_recurring)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "RMSE")
    print_summary_stats(df_combined)
    return df_combined


# =========================================================
# MAIN
# =========================================================

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
  1. Variance:   single (3 innovations) + recurring (Markov-switching)
  2. Mean:       single (3 innovations) + recurring (Markov-switching)
  3. Parameter:  single (3 innovations) + recurring (3 persistence levels)

Innovations: Gaussian, Student-t(df=3), Student-t(df=5)
Persistence: 0.90, 0.95, 0.99 (parameter recurring only)
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
║    Innovations: 3 types (Gaussian, t(df=5), t(df=3))               ║
║    Persistence: 3 levels (0.90, 0.95, 0.99) [parameter only]       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    all_results = []
    total_start = time.time()
    
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
    total_elapsed = time.time() - total_start
    
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
Total results: {len(df_all)} rows
Time elapsed: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)

Results saved to: {filename}

Metrics available: RMSE, MAE, Bias, Variance
Methods compared: ~5 per experiment
""")
    
    if 'RMSE' in df_all.columns:
        print("\nTOP 5 BEST METHODS (by RMSE):")
        top_methods = df_all.nsmallest(5, 'RMSE')[['Method', 'RMSE', 'MAE', 'Bias']]
        print(top_methods.to_string(index=False))
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
