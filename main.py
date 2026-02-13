#!/usr/bin/env python3
"""
Main CLI for Structural Break Experiments
==========================================
Simplified interface for users who want complete results.

AUTHORS: Aadya Khatavkar, Mahir Baylarov, Bakhodir Izzatulloev
University of Bonn | Winter Semester 2025/26

QUICK START:
  python main.py          # Run all experiments (full results)
  python main.py --quick  # Quick test (30 simulations instead of 300)
  python main.py --pdf    # Run experiments + generate PDF report

For fine-grained control (run individual break types):
  See runner.py documentation
"""
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Structural Break Experiments - Quick Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python main.py              Run all 3 break types (full Monte Carlo)
  python main.py --quick      Quick test mode (n_sim=30)
  python main.py --pdf        Run experiments + build PDF with tables & figures

For advanced usage (run specific break types):
  python runner.py --variance
  python runner.py --mean
  python runner.py --parameter
  See runner.py for full documentation.
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick run (n_sim=30 instead of 300)')
    parser.add_argument('--pdf', action='store_true',
                       help='Run experiments, then generate PDF report (tables + figures as appendix)')
    
    args = parser.parse_args()
    
    # Build runner.py command for all 3 break types
    cmd = [sys.executable, 'runner.py']
    
    if args.quick:
        cmd.append('--quick')
    
    # Run the experiments (all 3 break types)
    result = subprocess.run(cmd).returncode
    if result != 0:
        return result
    
    # If --pdf flag is set, build tables + figures PDF after experiments
    if args.pdf:
        pdf_cmd = [sys.executable, 'scripts/build_pdfs.py', '--tables', '--figures']
        return subprocess.run(pdf_cmd).returncode
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
