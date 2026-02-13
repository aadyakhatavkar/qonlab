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

QUICK MODE:
  --quick       Same as above but n_sim = 30

BREAK TYPES:
  1. Variance break   → single (3 innovations) + recurring (no persistence)
  2. Mean break       → single (3 innovations) + recurring (no persistence)
  3. Parameter break  → single (3 innovations) + recurring (3 persistence levels)

Features:
  - Innovation types for single breaks: Gaussian, Student-t(df=3), Student-t(df=5)
  - Persistence levels for parameter recurring: 0.90, 0.95, 0.99
  - Summary statistics and best method reporting
  - Results saved to outputs/ directory with separate tables by innovation type
  - Results aggregation and comparison
"""
import warnings

# Suppress expected optimizer/start-value warnings from repeated SARIMA/GARCH fits.
warnings.filterwarnings("ignore", message=".*Non-stationary starting autoregressive.*")
warnings.filterwarnings("ignore", message=".*Non-stationary starting seasonal autoregressive.*")
warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", message=".*optimizer returned code.*")

import argparse
import os
import sys
import time
import traceback
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# LOGGING SETUP
# =========================================================
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = f"{LOG_DIR}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*70)
logger.info("STRUCTURAL BREAK EXPERIMENTS RUNNER STARTED")
logger.info("="*70)

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
N_SIM = 300           # Monte Carlo simulations
WINDOW = 100            # SARIMA rolling window
SEED = 42               # Random seed

# Experiment variants
INNOVATION_TYPES = [
    ("gaussian", None),
    ("student", 5),
    ("student", 3),
]
PERSISTENCE_LEVELS = [0.90, 0.95, 0.99]

# Ensure results directory exists
RESULTS_DIR = "outputs"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def save_results(df, break_type, variant_name=None):
    """Save results to CSV with timestamp and variant info."""
    try:
        # Ensure tables directory exists
        os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Drop Successes/Failures columns before saving
        df_save = df.drop(columns=['Successes', 'Failures'], errors='ignore')
        
        if variant_name:
            filename = f"{RESULTS_DIR}/tables/{break_type}_{timestamp}_{variant_name}.csv"
        else:
            filename = f"{RESULTS_DIR}/tables/{break_type}_{timestamp}.csv"
        
        df_save.to_csv(filename, index=False)
        logger.info(f"✓ CSV saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"✗ Failed to save CSV: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def save_latex_table(df, break_type, variant_name=None, timestamp=None):
    """Save results as LaTeX table."""
    try:
        # Ensure tex directory exists
        os.makedirs(f"{RESULTS_DIR}/tex", exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if variant_name:
            filename = f"{RESULTS_DIR}/tex/{break_type}_{timestamp}_{variant_name}.tex"
            label = f"tab:{break_type}_{variant_name.lower().replace(' ', '_')}"
            caption = f"{break_type.replace('_', ' ').title()} Results - {variant_name}"
        else:
            filename = f"{RESULTS_DIR}/tex/{break_type}_{timestamp}.tex"
            label = f"tab:{break_type}"
            caption = f"{break_type.replace('_', ' ').title()} Results"
        
        # Select columns for LaTeX table
        columns_to_show = [col for col in df.columns if col in ['Method', 'RMSE', 'MAE', 'Bias', 'Variance', 'Coverage80', 'Coverage95', 'LogScore', 'Successes']]
        df_latex = df[columns_to_show].copy()
        
        # Format numeric columns to 4 decimal places
        for col in ['RMSE', 'MAE', 'Bias', 'Variance', 'Coverage80', 'Coverage95', 'LogScore']:
            if col in df_latex.columns:
                df_latex[col] = df_latex[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        
        # Generate LaTeX - use simple method without Styler to avoid Jinja2 dependency
        latex_str = df_latex.to_latex(
            index=False,
            float_format="%.4f"
        )
        
        # Wrap with table environment and add caption/label
        latex_wrapped = f"""\\begin{{table}}[h!]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex_str}
\\end{{table}}"""
        
        with open(filename, 'w') as f:
            f.write(latex_wrapped)
        
        logger.info(f"✓ LaTeX table saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"✗ Failed to save LaTeX table: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


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
    logger.info("="*70)
    logger.info("VARIANCE BREAK EXPERIMENTS START")
    logger.info("="*70)
    print("\n" + "="*70)
    print(" VARIANCE BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    failed_variants = []
    
    # Single variance breaks - test across innovations
    logger.info(f"[1/2] Single variance breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    print(f"\n[1/2] Single variance breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    variance_single_results = []
    
    for inn_type, dof in INNOVATION_TYPES:
        inn_name = f"Student-t(df={dof})" if inn_type == "student" else "Gaussian"
        print(f"      └─ {inn_name}...", end=" ", flush=True)
        logger.info(f"Starting variance_single: {inn_name}")
        t0 = time.time()
        
        try:
            df = mc_variance_single_break(
                n_sim=N_SIM, T=T, Tb=Tb,
                innovation_type=inn_type, dof=dof,
                seed=SEED
            )
            
            if df is None or len(df) == 0:
                raise ValueError(f"Returned empty dataframe")
            
            df['Innovation'] = inn_name
            variance_single_results.append(df)
            
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
            logger.info(f"✓ variance_single {inn_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ FAILED ({elapsed:.1f}s)")
            logger.error(f"✗ variance_single {inn_name} FAILED: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_variants.append(f"variance_single: {inn_name}")
            continue
    
    if len(variance_single_results) == 0:
        logger.error("✗ All variance_single innovations failed!")
        print("✗ All variance_single innovations failed!")
        return []
    
    variance_single = pd.concat(variance_single_results, ignore_index=True)
    
    # Print and save single break results by innovation type
    print(f"\n  --- Single Break Results by Innovation Type ---")
    logger.info("Saving variance_single results by innovation type...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for inn_name in sorted(variance_single['Innovation'].unique()):
        try:
            sub = variance_single[variance_single['Innovation'] == inn_name].sort_values('RMSE')
            print(f"\n  {inn_name}:")
            print(f"    Best method: {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
            logger.info(f"  {inn_name}: best method = {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
            # Save this innovation type's results
            inn_safe = inn_name.replace("(", "").replace(")", "").replace("=", "").replace(",", "")
            save_results(sub, "variance_single", inn_safe)
            save_latex_table(sub, "variance_single", inn_safe, timestamp)
        except Exception as e:
            logger.error(f"✗ Failed to save variance_single {inn_name}: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
    
    all_dfs.append(variance_single)
    
    # Recurring variance breaks
    logger.info(f"[2/2] Recurring variance breaks (T={T}, n_sim={N_SIM})")
    print(f"\n[2/2] Recurring variance breaks (T={T}, n_sim={N_SIM})")
    print(f"      └─ Markov-switching...", end=" ", flush=True)
    logger.info("Starting variance_recurring (Markov-switching)")
    t0 = time.time()
    
    try:
        variance_recurring = mc_variance_recurring(
            n_sim=N_SIM, T=T,
            seed=SEED
        )
        
        if variance_recurring is None or len(variance_recurring) == 0:
            raise ValueError("Returned empty dataframe")
        
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(variance_recurring)} rows)")
        logger.info(f"✓ variance_recurring completed in {elapsed:.1f}s")
        
        # Save recurring results
        print(f"\n  --- Recurring Break Results ---")
        logger.info("Saving variance_recurring results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            save_results(variance_recurring, "variance_recurring", "MarkovSwitching")
            save_latex_table(variance_recurring, "variance_recurring", "MarkovSwitching", timestamp)
            print(f"  ✓ Saved variance_recurring")
            logger.info(f"  ✓ Saved variance_recurring")
        except Exception as save_e:
            logger.error(f"✗ Failed to save variance_recurring: {type(save_e).__name__}: {str(save_e)}")
            logger.error(traceback.format_exc())
        
        all_dfs.append(variance_recurring)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"✗ FAILED ({elapsed:.1f}s)")
        logger.error(f"✗ variance_recurring FAILED: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        failed_variants.append("variance_recurring")
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "RMSE")
    print_summary_stats(df_combined)
    
    # Log summary
    logger.info("="*70)
    logger.info("VARIANCE BREAK EXPERIMENTS COMPLETED")
    if failed_variants:
        logger.warning(f"Failed variants: {', '.join(failed_variants)}")
    logger.info("="*70)
    
    return df_combined


# =========================================================
# MEAN BREAKS
# =========================================================

def run_mean_breaks():
    """Run mean break experiments (single + recurring)."""
    logger.info("="*70)
    logger.info("MEAN BREAK EXPERIMENTS START")
    logger.info("="*70)
    print("\n" + "="*70)
    print(" MEAN BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    failed_variants = []
    
    # Single mean breaks - test across innovations
    logger.info(f"[1/2] Single mean breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    print(f"\n[1/2] Single mean breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    mean_single_results = []
    
    for inn_type, dof in INNOVATION_TYPES:
        inn_name = f"Student-t(df={dof})" if inn_type == "student" else "Gaussian"
        print(f"      └─ {inn_name}...", end=" ", flush=True)
        logger.info(f"Starting mean_single: {inn_name}")
        t0 = time.time()
        
        try:
            df = mc_single_sarima(
                n_sim=N_SIM, T=T, Tb=Tb,
                innovation_type=inn_type, dof=dof,
                seed=SEED
            )
            
            if df is None or len(df) == 0:
                raise ValueError("Returned empty dataframe")
            
            df['Innovation'] = inn_name
            mean_single_results.append(df)
            
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
            logger.info(f"✓ mean_single {inn_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ FAILED ({elapsed:.1f}s)")
            logger.error(f"✗ mean_single {inn_name} FAILED: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_variants.append(f"mean_single: {inn_name}")
            continue
    
    if len(mean_single_results) == 0:
        logger.error("✗ All mean_single innovations failed!")
        print("✗ All mean_single innovations failed!")
        return []
    
    mean_single = pd.concat(mean_single_results, ignore_index=True)
    
    # Print and save single break results by innovation type
    print(f"\n  --- Single Break Results by Innovation Type ---")
    logger.info("Saving mean_single results by innovation type...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for inn_name in sorted(mean_single['Innovation'].unique()):
        try:
            sub = mean_single[mean_single['Innovation'] == inn_name].sort_values('RMSE')
            print(f"\n  {inn_name}:")
            print(f"    Best method: {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
            logger.info(f"  {inn_name}: best method = {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
            # Save this innovation type's results
            inn_safe = inn_name.replace("(", "").replace(")", "").replace("=", "").replace(",", "")
            save_results(sub, "mean_single", inn_safe)
            save_latex_table(sub, "mean_single", inn_safe, timestamp)
        except Exception as e:
            logger.error(f"✗ Failed to save mean_single {inn_name}: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
    
    all_dfs.append(mean_single)
    
    # Recurring mean breaks
    logger.info(f"[2/2] Recurring mean breaks (T={T}, n_sim={N_SIM})")
    print(f"\n[2/2] Recurring mean breaks (T={T}, n_sim={N_SIM})")
    print(f"      └─ Markov-switching...", end=" ", flush=True)
    logger.info("Starting mean_recurring (Markov-switching)")
    t0 = time.time()
    
    try:
        mean_recurring = mc_mean_recurring(
            n_sim=N_SIM, T=T,
            seed=SEED
        )
        
        if mean_recurring is None or len(mean_recurring) == 0:
            raise ValueError("Returned empty dataframe")
        
        elapsed = time.time() - t0
        print(f"✓ ({elapsed:.1f}s, {len(mean_recurring)} rows)")
        logger.info(f"✓ mean_recurring completed in {elapsed:.1f}s")
        
        # Save recurring results
        print(f"\n  --- Recurring Break Results ---")
        logger.info("Saving mean_recurring results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            save_results(mean_recurring, "mean_recurring", "MarkovSwitching")
            save_latex_table(mean_recurring, "mean_recurring", "MarkovSwitching", timestamp)
            print(f"  ✓ Saved mean_recurring")
            logger.info(f"  ✓ Saved mean_recurring")
        except Exception as save_e:
            logger.error(f"✗ Failed to save mean_recurring: {type(save_e).__name__}: {str(save_e)}")
            logger.error(traceback.format_exc())
        
        all_dfs.append(mean_recurring)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"✗ FAILED ({elapsed:.1f}s)")
        logger.error(f"✗ mean_recurring FAILED: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        failed_variants.append("mean_recurring")
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "RMSE")
    print_summary_stats(df_combined)
    
    # Log summary
    logger.info("="*70)
    logger.info("MEAN BREAK EXPERIMENTS COMPLETED")
    if failed_variants:
        logger.warning(f"Failed variants: {', '.join(failed_variants)}")
    logger.info("="*70)
    
    return df_combined


# =========================================================
# PARAMETER BREAKS
# =========================================================

def run_parameter_breaks():
    """Run parameter break experiments (single + recurring)."""
    logger.info("="*70)
    logger.info("PARAMETER BREAK EXPERIMENTS START")
    logger.info("="*70)
    print("\n" + "="*70)
    print(" PARAMETER BREAK EXPERIMENTS")
    print("="*70)
    
    all_dfs = []
    failed_variants = []
    
    # Single parameter breaks - test across innovations
    logger.info(f"[1/2] Single parameter breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    print(f"\n[1/2] Single parameter breaks (T={T}, Tb={Tb}, n_sim={N_SIM})")
    parameter_single_results = []
    
    for inn_type, dof in INNOVATION_TYPES:
        inn_name = f"Student-t(df={dof})" if inn_type == "student" else "Gaussian"
        print(f"      └─ {inn_name}...", end=" ", flush=True)
        logger.info(f"Starting parameter_single: {inn_name}")
        t0 = time.time()
        
        try:
            df = monte_carlo_single_break_post(
                n_sim=N_SIM, T=T, Tb=Tb,
                innovation_type=inn_type, dof=dof,
                seed=SEED
            )
            
            if df is None or len(df) == 0:
                raise ValueError("Returned empty dataframe")
            
            df['Innovation'] = inn_name
            parameter_single_results.append(df)
            
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
            logger.info(f"✓ parameter_single {inn_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ FAILED ({elapsed:.1f}s)")
            logger.error(f"✗ parameter_single {inn_name} FAILED: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_variants.append(f"parameter_single: {inn_name}")
            continue
    
    if len(parameter_single_results) == 0:
        logger.error("✗ All parameter_single innovations failed!")
        print("✗ All parameter_single innovations failed!")
        return []
    
    parameter_single = pd.concat(parameter_single_results, ignore_index=True)
    
    # Print and save single break results by innovation type
    print(f"\n  --- Single Break Results by Innovation Type ---")
    logger.info("Saving parameter_single results by innovation type...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for inn_name in sorted(parameter_single['Innovation'].unique()):
        try:
            sub = parameter_single[parameter_single['Innovation'] == inn_name].sort_values('RMSE')
            print(f"\n  {inn_name}:")
            print(f"    Best method: {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
            logger.info(f"  {inn_name}: best method = {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
            # Save this innovation type's results
            inn_safe = inn_name.replace("(", "").replace(")", "").replace("=", "").replace(",", "")
            save_results(sub, "parameter_single", inn_safe)
            save_latex_table(sub, "parameter_single", inn_safe, timestamp)
        except Exception as e:
            logger.error(f"✗ Failed to save parameter_single {inn_name}: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
    
    all_dfs.append(parameter_single)
    
    # Recurring parameter breaks - test across persistence levels
    logger.info(f"[2/2] Recurring parameter breaks (T={T}, n_sim={N_SIM})")
    print(f"\n[2/2] Recurring parameter breaks (T={T}, n_sim={N_SIM})")
    parameter_recurring_results = []
    
    for p in PERSISTENCE_LEVELS:
        print(f"      └─ p={p}...", end=" ", flush=True)
        logger.info(f"Starting parameter_recurring: p={p}")
        t0 = time.time()
        
        try:
            df = monte_carlo_recurring(
                p=p, n_sim=N_SIM, T=T,
                seed=SEED
            )
            
            if df is None or len(df) == 0:
                raise ValueError("Returned empty dataframe")
            
            df['Persistence'] = p
            parameter_recurring_results.append(df)
            
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s, {len(df)} rows)")
            logger.info(f"✓ parameter_recurring p={p} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ FAILED ({elapsed:.1f}s)")
            logger.error(f"✗ parameter_recurring p={p} FAILED: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_variants.append(f"parameter_recurring: p={p}")
            continue
    
    if len(parameter_recurring_results) > 0:
        parameter_recurring = pd.concat(parameter_recurring_results, ignore_index=True)
        
        # Save recurring results by persistence level
        print(f"\n  --- Recurring Break Results ---")
        logger.info("Saving parameter_recurring results by persistence level...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for p in sorted(parameter_recurring['Persistence'].unique()):
            try:
                sub = parameter_recurring[parameter_recurring['Persistence'] == p].sort_values('RMSE')
                print(f"  p={p}: best method = {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
                logger.info(f"  p={p}: best method = {sub.iloc[0]['Method']} (RMSE={sub.iloc[0]['RMSE']:.4f})")
                p_safe = f"p{str(p).replace('.', '')}"
                save_results(sub, "parameter_recurring", p_safe)
                save_latex_table(sub, "parameter_recurring", p_safe, timestamp)
            except Exception as e:
                logger.error(f"✗ Failed to save parameter_recurring p={p}: {type(e).__name__}: {str(e)}")
                logger.error(traceback.format_exc())
        
        all_dfs.append(parameter_recurring)
    else:
        logger.warning("No parameter_recurring results collected")
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print_best_methods(df_combined, "RMSE")
    print_summary_stats(df_combined)
    
    # Log summary
    logger.info("="*70)
    logger.info("PARAMETER BREAK EXPERIMENTS COMPLETED")
    if failed_variants:
        logger.warning(f"Failed variants: {', '.join(failed_variants)}")
    logger.info("="*70)
    
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
    parser.add_argument('--quick', action='store_true', help='Quick run (n_sim=30, same T and Tb)')
    parser.add_argument('--variance', action='store_true', help='Run variance breaks only')
    parser.add_argument('--mean', action='store_true', help='Run mean breaks only')
    parser.add_argument('--parameter', action='store_true', help='Run parameter breaks only')
    parser.add_argument('--generate-plots', action='store_true', help='Generate plots from results after run')
    parser.add_argument('--plots-only', action='store_true', help='Only generate plots (skip experiments)')
    args = parser.parse_args()
    
    # Override params for quick mode
    global T, Tb, N_SIM
    if args.quick:
        N_SIM = 30
    
    # Handle plots-only mode
    if args.plots_only:
        print("Generating plots from latest results...")
        import subprocess
        result = subprocess.run(
            ['python', 'scripts/generate_plots.py', '--latest'],
            capture_output=False
        )
        return
    
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
    
    logger.info(f"Parameters: T={T}, Tb={Tb}, n_sim={N_SIM}")
    logger.info(f"Quick mode: {args.quick}")
    
    # Determine which experiments to run
    run_all = not (args.variance or args.mean or args.parameter)
    
    try:
        if run_all or args.variance:
            logger.info("Starting variance break experiments...")
            variance_results = run_variance_breaks()
            if len(variance_results) > 0:
                all_results.append(variance_results)
            else:
                logger.warning("Variance break experiments returned no results")
    except Exception as e:
        logger.error(f"✗ Variance breaks FAILED: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        if run_all or args.mean:
            logger.info("Starting mean break experiments...")
            mean_results = run_mean_breaks()
            if len(mean_results) > 0:
                all_results.append(mean_results)
            else:
                logger.warning("Mean break experiments returned no results")
    except Exception as e:
        logger.error(f"✗ Mean breaks FAILED: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        if run_all or args.parameter:
            logger.info("Starting parameter break experiments...")
            parameter_results = run_parameter_breaks()
            if len(parameter_results) > 0:
                all_results.append(parameter_results)
            else:
                logger.warning("Parameter break experiments returned no results")
    except Exception as e:
        logger.error(f"✗ Parameter breaks FAILED: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Combine all results
    if len(all_results) == 0:
        logger.error("✗ All experiments failed - no results to save")
        print("✗ All experiments failed - no results to save")
        return
    
    try:
        df_all = pd.concat(all_results, ignore_index=True)
        total_elapsed = time.time() - total_start
        
        # Save results
        os.makedirs("outputs/tables", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Remove metadata columns from combined results before saving
        df_combined = df_all.drop(columns=['Break Type', 'Innovation', 'Persistence', 'Successes', 'Failures', 'N'], errors='ignore')
        
        # Filter out rows where all metric columns are NaN (failed forecasts)
        metric_cols = ['RMSE', 'MAE', 'Bias', 'Variance', 'LogScore']
        metric_cols = [c for c in metric_cols if c in df_combined.columns]
        df_combined = df_combined.dropna(subset=metric_cols, how='all')
        
        filename = f"outputs/tables/aligned_breaks_{timestamp}.csv"
        df_combined.to_csv(filename, index=False)
        logger.info(f"✓ Combined results saved: {filename}")
        
        # Replace NaN in Coverage and LogScore columns with "NA"
        for col in ['Coverage80', 'Coverage95', 'LogScore']:
            if col in df_combined.columns:
                df_combined[col] = df_combined[col].where(df_combined[col].notna(), 'NA')
        
        # Save combined results as LaTeX
        os.makedirs("outputs/tex", exist_ok=True)
        filename_latex = f"outputs/tex/aligned_breaks_{timestamp}.tex"
        latex_str = df_combined.to_latex(
            index=False,
            caption="Complete Structural Break Forecasting Results",
            label="tab:all_results",
            float_format="%.4f",
            escape=False
        )
        with open(filename_latex, 'w') as f:
            f.write(latex_str)
        logger.info(f"✓ LaTeX table saved: {filename_latex}")
        
        # Display final summary
        summary = f"""
{'='*70}
 OVERALL SUMMARY
{'='*70}
Total results: {len(df_all)} rows
Time elapsed: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)

Results saved to: 
  CSV: {filename}
  LaTeX: {filename_latex}
  Log: {log_file}

Metrics available: RMSE, MAE, Bias, Variance
"""
        print(summary)
        logger.info(summary)
        
        if 'RMSE' in df_all.columns:
            print("\nTOP 5 BEST METHODS (by RMSE):")
            top_methods = df_all.nsmallest(5, 'RMSE')[['Method', 'RMSE', 'MAE', 'Bias']]
            print(top_methods.to_string(index=False))
            logger.info(f"Top methods:\n{top_methods.to_string(index=False)}")
        
        logger.info("="*70)
        logger.info("RUNNER COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        print(f"\n{'='*70}\n")
        
        # Generate plots if requested
        if args.generate_plots:
            print("\nGenerating plots from results...")
            import subprocess
            result = subprocess.run(
                ['python', 'scripts/generate_plots.py', '--results', filename, '--output-dir', 'figures'],
                capture_output=False
            )
            if result.returncode == 0:
                logger.info("✓ Plots generated successfully")
            else:
                logger.warning("✗ Plot generation encountered issues")
    
    except Exception as e:
        logger.error(f"✗ Failed to save/summarize results: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
