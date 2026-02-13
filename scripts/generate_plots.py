#!/usr/bin/env python3
"""
Generate Publication-Quality Plots from Results
================================================
Reads CSV results from results/ folder and generates coherent matplotlib visualizations.
Uses unified styling across all break types for professional appearance.
Saves to figures/ folder organized by break type.

Usage:
    python scripts/generate_plots.py [--break-type {variance,mean,parameter,all}]
    python scripts/generate_plots.py --latest  # Use most recent results
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================================================
# UNIFIED STYLING
# =========================================================

# Professional color scheme (consistent across all plots)
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'accent': '#A23B72',       # Accent purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
}

# Coverage target colors
COVERAGE_COLORS = {
    'Coverage80': '#FF6B6B',   # Red
    'Coverage95': '#4ECDC4',   # Teal
}

def apply_unified_style():
    """Apply consistent styling to all matplotlib figures."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'patch.edgecolor': 'black',
        'patch.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =========================================================
# COHERENT PLOTTING FUNCTIONS
# =========================================================

def plot_method_comparison_metrics(df, break_type, save_path=None):
    """
    Unified metric comparison across methods.
    Shows RMSE, MAE, Bias, Variance in one coherent figure.
    """
    apply_unified_style()
    
    # Select available metrics
    metrics = ['RMSE', 'MAE', 'Bias', 'Variance']
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) if n_metrics == 4 else plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics < 4:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot each metric consistently
    for idx, metric in enumerate(available_metrics):
        ax = axes.flatten()[idx]
        data = df.sort_values(metric)
        
        bars = ax.bar(range(len(data)), data[metric], color=COLORS['primary'], alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Forecasting Method', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['Method'], rotation=45, ha='right', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Remove empty subplots
    for idx in range(n_metrics, len(axes.flatten())):
        fig.delaxes(axes.flatten()[idx])
    
    fig.suptitle(f'{break_type.title()} Break - Method Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_uncertainty_metrics(df, break_type, save_path=None):
    """
    Unified uncertainty quantification plot (Coverage80, Coverage95, LogScore).
    Only for variance scenarios.
    """
    if 'Coverage80' not in df.columns or 'Coverage95' not in df.columns:
        return
    
    apply_unified_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = df.sort_values('Coverage95')
    x = range(len(data))
    
    # Coverage 80%
    bars1 = axes[0].bar(x, data['Coverage80'], color=COVERAGE_COLORS['Coverage80'], alpha=0.85, edgecolor='black', linewidth=0.5)
    axes[0].axhline(0.80, color='darkred', linestyle='--', linewidth=2.5, label='Target (80%)', alpha=0.7)
    axes[0].set_ylabel('Coverage Probability', fontsize=11)
    axes[0].set_title('80% Prediction Interval Coverage', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(data['Method'], rotation=45, ha='right')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, axis='y', alpha=0.3, linestyle='--')
    axes[0].legend(loc='lower right')
    axes[0].set_axisbelow(True)
    
    # Coverage 95%
    bars2 = axes[1].bar(x, data['Coverage95'], color=COVERAGE_COLORS['Coverage95'], alpha=0.85, edgecolor='black', linewidth=0.5)
    axes[1].axhline(0.95, color='darkred', linestyle='--', linewidth=2.5, label='Target (95%)', alpha=0.7)
    axes[1].set_ylabel('Coverage Probability', fontsize=11)
    axes[1].set_title('95% Prediction Interval Coverage', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(data['Method'], rotation=45, ha='right')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, axis='y', alpha=0.3, linestyle='--')
    axes[1].legend(loc='lower right')
    axes[1].set_axisbelow(True)
    
    # Log Score
    if 'LogScore' in df.columns:
        # Higher is better for log score, so sort descending
        data_ls = data.sort_values('LogScore', ascending=False)
        bars3 = axes[2].bar(range(len(data_ls)), data_ls['LogScore'], color=COLORS['success'], alpha=0.85, edgecolor='black', linewidth=0.5)
        axes[2].set_ylabel('Log-Predictive Score', fontsize=11)
        axes[2].set_title('Log-Predictive Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[2].set_xticks(range(len(data_ls)))
        axes[2].set_xticklabels(data_ls['Method'], rotation=45, ha='right')
        axes[2].grid(True, axis='y', alpha=0.3, linestyle='--')
        axes[2].set_axisbelow(True)
    
    fig.suptitle(f'{break_type.title()} Break - Uncertainty Quantification', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_rmse_comparison_all_innovations(df_all_innovations, break_type, save_path=None):
    """
    Compare RMSE across all innovation types for a given break type.
    Shows methods on x-axis, different colors for each innovation type.
    """
    if 'Innovation' not in df_all_innovations.columns:
        return
    
    apply_unified_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    innovations = sorted(df_all_innovations['Innovation'].unique())
    methods = sorted(df_all_innovations['Method'].unique())
    
    x = np.arange(len(methods))
    width = 0.25
    
    color_palette = [COLORS['primary'], COLORS['accent'], COLORS['warning']]
    
    for idx, inn in enumerate(innovations):
        df_inn = df_all_innovations[df_all_innovations['Innovation'] == inn].set_index('Method')
        rmse_values = [df_inn.loc[m, 'RMSE'] if m in df_inn.index else np.nan for m in methods]
        
        ax.bar(x + idx*width, rmse_values, width, label=inn, 
               color=color_palette[idx % len(color_palette)], alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Forecasting Method', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'{break_type.title()} Break - RMSE Across Innovation Types', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(title='Innovation Type', fontsize=10, title_fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def find_latest_results(results_dir='results'):
    """Find most recent results file."""
    pattern = 'aligned_breaks_*.csv'
    files = sorted(Path(results_dir).glob(pattern))
    return files[-1] if files else None


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate coherent, publication-quality plots from Monte Carlo results'
    )
    parser.add_argument(
        '--results',
        default=None,
        help='Path to results CSV file (default: most recent)'
    )
    parser.add_argument(
        '--break-type',
        choices=['variance', 'mean', 'parameter', 'all'],
        default='all',
        help='Break type to plot'
    )
    parser.add_argument(
        '--output-dir',
        default='figures',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use most recent results file'
    )
    
    args = parser.parse_args()
    
    # Find results file
    if args.latest:
        results_file = find_latest_results()
        if not results_file:
            print("✗ No results files found in results/")
            return
    elif args.results:
        results_file = args.results
    else:
        results_file = find_latest_results()
    
    if not results_file or not os.path.exists(results_file):
        print(f"✗ Results file not found: {results_file}")
        return
    
    # Load results
    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    
    # Determine which break types to plot
    if args.break_type == 'all':
        break_types = sorted(df['Task'].unique()) if 'Task' in df.columns else ['variance', 'mean', 'parameter']
    else:
        break_types = [args.break_type]
    
    # Generate coherent plots
    print(f"\nGenerating coherent plots for: {', '.join(break_types)}\n")
    
    for break_type in break_types:
        print(f"{break_type.upper()}:")
        
        if 'Task' in df.columns:
            df_break = df[df['Task'] == break_type].copy()
        else:
            df_break = df[df.get('Scenario', '').str.contains(break_type, case=False, na=False)].copy()
        
        if len(df_break) == 0:
            print(f"  ✗ No data for {break_type}")
            continue
        
        # 1. Method Comparison (Core metrics)
        save_path = os.path.join(args.output_dir, break_type, f'{break_type}_method_comparison.png')
        plot_method_comparison_metrics(df_break, break_type, save_path)
        
        # 2. Uncertainty Quantification (for variance only)
        if break_type == 'variance':
            save_path = os.path.join(args.output_dir, break_type, f'{break_type}_uncertainty.png')
            plot_uncertainty_metrics(df_break, break_type, save_path)
        
        # 3. Innovation Type Comparison (if available)
        if 'Innovation' in df_break.columns and len(df_break['Innovation'].unique()) > 1:
            save_path = os.path.join(args.output_dir, break_type, f'{break_type}_innovation_comparison.png')
            plot_rmse_comparison_all_innovations(df_break, break_type, save_path)
    
    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
