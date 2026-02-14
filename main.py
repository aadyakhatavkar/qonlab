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
    
    # If --pdf flag is set, update paper tables then build PDF
    if args.pdf:
        # Step 1: Generate figures from results
        print("\n" + "="*70)
        print("STEP 1: Generating figures from simulation results...")
        print("="*70)
        fig_cmd = [sys.executable, 'scripts/generate_plots.py']
        result = subprocess.run(fig_cmd).returncode
        if result != 0:
            print("\n⚠️  Figure generation encountered issues, but continuing...")
        
        # Step 2: Update paper tables with latest results
        print("\n" + "="*70)
        print("STEP 2: Updating paper tables with latest experimental results...")
        print("="*70)
        update_cmd = [sys.executable, 'scripts/update_paper_tables.py']
        result = subprocess.run(update_cmd).returncode
        if result != 0:
            print("\n⚠️  Table update encountered issues, but continuing with PDF build...")
        
        # Step 3: Build PDF with tables and figures
        print("\n" + "="*70)
        print("STEP 3: Building PDF with tables and figures...")
        print("="*70)
        pdf_cmd = [sys.executable, 'scripts/build_pdfs.py', '--all']
        return subprocess.run(pdf_cmd).returncode
    
    # No --pdf flag: show helpful next steps
    print("\n" + "="*70)
    print("✅ EXPERIMENTS COMPLETE")
    print("="*70)
    print("\n📁 Results saved to: outputs/tables/")
    print("\n📊 NEXT STEPS:")
    print("   1. Generate figures:")
    print("      pixi run python scripts/generate_plots.py")
    print("")
    print("   2. Build PDF report:")
    print("      pixi run python scripts/build_pdfs.py --all")
    print("")
    print("   Or do both at once (re-run with --pdf):")
    print("      pixi run python main.py --pdf")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
