#!/usr/bin/env python3
"""
Generate Publication-Quality Plots (Tier 1 & 2 per PLOT_RECOMMENDATIONS.md)
===========================================================================

This script orchestrates plot generation from the analyses/ modules, which contain
the actual plotting functions.

Tier 1 Plots: Core metric comparisons (method comparison, coverage, logscore)
Tier 2 Plots: DGP visualizations (time series examples, regime switches)

Usage:
    # List all available plots
    python scripts/generate_plots.py --list
    
    # Generate all plots
    python scripts/generate_plots.py --all
    
    # Generate plots for specific break types
    python scripts/generate_plots.py --variance --single
    python scripts/generate_plots.py --mean --recurring
    python scripts/generate_plots.py --parameter --all
"""
import argparse
import os
import sys
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================================================
# PLOT REGISTRY - Maps to functions in analyses/ modules
# =========================================================

PLOT_REGISTRY = {
    'variance': {
        'single': {
            'tier1': [
                ('plots_variance_single', 'plot_logscore_comparison', 'LogScore heatmap (window × method)'),
            ],
            'tier2': [
                ('plots_variance_single', 'plot_time_series_example', 'Example time series with variance break'),
            ]
        },
        'recurring': {
            'tier1': [
                ('plots_variance_recurring', 'plot_logscore_comparison', 'LogScore heatmap (window × method)'),
            ],
            'tier2': [
                ('plots_variance_recurring', 'plot_time_series_example', 'Example time series with recurring variance changes'),
            ]
        }
    },
    'mean': {
        'single': {
            'tier1': [
                ('plots_meansingle', 'plot_mean_single_break_results', 'Method comparison (RMSE, MAE, Bias, Variance)'),
            ],
            'tier2': [
                ('plots_meansingle', 'plot_mean_single_break_example', 'Example time series with mean break'),
            ]
        },
        'recurring': {
            'tier1': [
                ('plots_mean_recurring', 'plot_mean_recurring_results', 'Method comparison (RMSE, MAE, Bias, Variance)'),
            ],
            'tier2': [
                ('plots_mean_recurring', 'plot_mean_recurring_example', 'Example time series with Markov-switching'),
            ]
        }
    },
    'parameter_extra': {
        'single': {
            'tier1': [
                ('plots_parametersingle', 'plot_combined_distributions', 'Error distributions (Gaussian, t-df3, t-df5)'),
                ('plots_parametersingle', 'plot_rmse_by_innovation', 'RMSE by innovation type'),
            ],
            'tier2': [
                ('plots_parametersingle', 'plot_single_break_dgp', 'Example time series with parameter break'),
            ]
        },
        'recurring': {
            'tier1': [
                ('plots_parameterrecurring', 'plot_metric_bars', 'Method comparison (RMSE, MAE, Bias) by persistence'),
                ('plots_parameterrecurring', 'plot_error_distributions_all', 'Error distributions across persistence levels'),
            ],
            'tier2': [
                ('plots_parameterrecurring', 'plot_dgp_by_persistence', 'DGP visualization (p=0.90, 0.95, 0.99)'),
            ]
        }
    },
    'parameter': {
        'single': {
            'tier1': [
                ('plots_parametersingle', 'plot_rmse_by_innovation', 'RMSE by innovation type'),
            ],
            'tier2': [
                ('plots_parametersingle', 'plot_single_break_dgp', 'Example time series with parameter break'),
            ]
        },
        'recurring': {
            'tier1': [
                ('plots_parameterrecurring', 'plot_metric_bars', 'Method comparison (RMSE, MAE, Bias) by persistence'),
            ],
            'tier2': [
                ('plots_parameterrecurring', 'plot_dgp_by_persistence', 'DGP visualization (p=0.90, 0.95, 0.99)'),
            ]
        }
    }
}

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def list_plots():
    """Print all available plots."""
    print("\n" + "="*70)
    print("TIER 1 & TIER 2 PLOTS - PLOT_RECOMMENDATIONS.md")
    print("="*70 + "\n")
    
    for break_type in sorted(PLOT_REGISTRY.keys()):
        print(f"\n{break_type.upper()} BREAKS:")
        print("-" * 70)
        
        for subtype in sorted(PLOT_REGISTRY[break_type].keys()):
            print(f"\n  {subtype.upper()}:")
            
            print(f"\n    TIER 1 (Core Metrics):")
            for module, func, desc in PLOT_REGISTRY[break_type][subtype]['tier1']:
                print(f"      • {func}()")
                print(f"        {desc}")
                print(f"        Location: analyses/{module}.py")
            
            print(f"\n    TIER 2 (DGP Visualization):")
            for module, func, desc in PLOT_REGISTRY[break_type][subtype]['tier2']:
                print(f"      • {func}()")
                print(f"        {desc}")
                print(f"        Location: analyses/{module}.py")


def generate_single_plot(module_name, func_name, output_subdir):
    """Generate a single plot by calling the function from the module."""
    try:
        # Import the module dynamically
        module = __import__(f'analyses.{module_name}', fromlist=[func_name])
        plot_func = getattr(module, func_name)
        
        # Create output directory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Generate the plot
        import matplotlib.pyplot as plt
        plot_func()
        
        # The plot functions themselves handle saving, so we just check if it worked
        return True, "Generated"
    except Exception as e:
        return False, str(e)


def generate_plots(break_types, subtypes, tiers, output_dir='figures'):
    """Generate plots for specified break types, subtypes, and tiers."""
    
    print("\n" + "="*70)
    print("TIER 1 & TIER 2 PLOT GENERATION")
    print("="*70 + "\n")
    
    print(f"Break types: {', '.join(break_types)}")
    print(f"Subtypes: {', '.join(subtypes)}")
    print(f"Tiers: {', '.join(tiers)}")
    print(f"Output: {output_dir}/\n")
    
    plot_count = 0
    success_count = 0
    
    for break_type in break_types:
        if break_type not in PLOT_REGISTRY:
            print(f"✗ Unknown break type: {break_type}")
            continue
        
        for subtype in subtypes:
            if subtype not in PLOT_REGISTRY[break_type]:
                print(f"✗ Unknown subtype for {break_type}: {subtype}")
                continue
            
            print(f"\n{break_type.upper()} - {subtype.upper()}:")
            print("-" * 70)
            
            for tier in tiers:
                if tier not in PLOT_REGISTRY[break_type][subtype]:
                    continue
                
                print(f"\n  {tier.upper()}:")
                
                for module, func, desc in PLOT_REGISTRY[break_type][subtype][tier]:
                    output_subdir = os.path.join(output_dir, break_type, subtype)
                    
                    print(f"    • {func}()")
                    print(f"      Description: {desc}")
                    
                    # Generate the plot
                    success, msg = generate_single_plot(module, func, output_subdir)
                    
                    if success:
                        print(f"      Status: ✓ {msg}")
                        success_count += 1
                    else:
                        print(f"      Status: ✗ Error: {msg}")
                    
                    plot_count += 1
    
    print(f"\n" + "="*70)
    print(f"Results: {success_count}/{plot_count} plots generated successfully")
    print("="*70)


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Tier 1 & 2 Plot Generation (Per PLOT_RECOMMENDATIONS.md)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # List all available plots
  %(prog)s --list
  
  # Generate all plots
  %(prog)s --all
  
  # Generate variance plots only
  %(prog)s --variance
  
  # Generate mean single break plots only
  %(prog)s --mean --single
  
  # Generate only Tier 1 plots
  %(prog)s --tier1
"""
    )
    
    parser.add_argument('--list', action='store_true', 
                       help='List all available plots')
    parser.add_argument('--all', action='store_true',
                       help='Generate all plots')
    parser.add_argument('--variance', action='store_true',
                       help='Generate variance break plots')
    parser.add_argument('--mean', action='store_true',
                       help='Generate mean break plots')
    parser.add_argument('--parameter', action='store_true',
                       help='Generate parameter break plots')
    parser.add_argument('--single', action='store_true',
                       help='Generate single break plots only')
    parser.add_argument('--recurring', action='store_true',
                       help='Generate recurring break plots only')
    parser.add_argument('--tier1', action='store_true',
                       help='Generate Tier 1 plots only (core metrics)')
    parser.add_argument('--tier2', action='store_true',
                       help='Generate Tier 2 plots only (DGP visualization)')
    parser.add_argument('--output-dir', default='outputs/figures',
                       help='Output directory for plots (default: outputs/figures/)')
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_plots()
        return
    
    # Determine what to generate
    if args.all or not any([args.variance, args.mean, args.parameter]):
        break_types = ['variance', 'mean', 'parameter']
    else:
        break_types = []
        if args.variance:
            break_types.append('variance')
        if args.mean:
            break_types.append('mean')
        if args.parameter:
            break_types.append('parameter')
    
    if args.single and args.recurring:
        subtypes = ['single', 'recurring']
    elif args.single:
        subtypes = ['single']
    elif args.recurring:
        subtypes = ['recurring']
    else:
        subtypes = ['single', 'recurring']
    
    if args.tier1 and args.tier2:
        tiers = ['tier1', 'tier2']
    elif args.tier1:
        tiers = ['tier1']
    elif args.tier2:
        tiers = ['tier2']
    else:
        tiers = ['tier1', 'tier2']
    
    generate_plots(break_types, subtypes, tiers, args.output_dir)


if __name__ == '__main__':
    main()
