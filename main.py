#!/usr/bin/env python3
"""
Main CLI for Aligned Structural Break Experiments
==================================================
Authors: Aadya Khatavkar, Mahir Baylarov, Bakhodir Izzatulloev
University of Bonn | Winter Semester 2025/26

Unified interface to run aligned experiments for all 3 break types:
  - Variance breaks (single + recurring)
  - Mean breaks (single + recurring)
  - Parameter breaks (single + recurring with persistence levels)

Standardized parameters:
  T = 400 (time series length)
  Tb = 200 (break point for single breaks)
  n_sim = 300 (Monte Carlo replications)

Examples:
  python main.py                    # Run all experiments
  python main.py --quick            # Quick run (n_sim=30, same T and Tb)
  python main.py --variance         # Run variance breaks only
  python main.py --mean             # Run mean breaks only
  python main.py --parameter        # Run parameter breaks only
"""
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Aligned Structural Break Experiments (All 3 Break Types)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Mode: python main.py --quick (T=400, Tb=200, n_sim=30)
Single Break Type: python main.py --variance
Multiple Break Types: python main.py --variance --mean
All Experiments: python main.py

See runner.py for full documentation.
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick run (n_sim=30, same T and Tb)')
    parser.add_argument('--variance', action='store_true', 
                       help='Run variance breaks only')
    parser.add_argument('--mean', action='store_true', 
                       help='Run mean breaks only')
    parser.add_argument('--parameter', action='store_true', 
                       help='Run parameter breaks only')
    
    args = parser.parse_args()
    
    # Build runner.py command
    cmd = [sys.executable, 'runner.py']
    
    if args.quick:
        cmd.append('--quick')
    if args.variance:
        cmd.append('--variance')
    if args.mean:
        cmd.append('--mean')
    if args.parameter:
        cmd.append('--parameter')
    
    # Call the unified runner
    return subprocess.run(cmd).returncode


if __name__ == '__main__':
    raise SystemExit(main())
