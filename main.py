#!/usr/bin/env python3
"""
Main CLI for Structural Break Forecasting Project
==================================================

Subcommands:
  variance  - Run variance break Monte Carlo experiments
  mean      - Run mean break Monte Carlo experiments
  parameter - Run parameter break Monte Carlo experiments
  runner    - Run full experiment pipeline with scenarios

Examples:
  python main.py variance --quick
  python main.py mean --n-sim 100
  python main.py parameter --innovation student --df 50
  python main.py runner --scenarios scenarios.json
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Structural Break Forecasting CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py variance --quick
  python main.py mean --n-sim 100 --Tb 150
  python main.py runner --scenarios scenarios.json
        """
    )
    sub = parser.add_subparsers(dest='cmd', help='Available commands')
    
    # Variance subcommand
    var_parser = sub.add_parser('variance', help='Run variance break MC experiments')
    var_parser.add_argument('--quick', action='store_true', help='Quick test (10 reps)')
    var_parser.add_argument('--n-sim', type=int, default=300, help='MC replications')
    var_parser.add_argument('--T', type=int, default=400, help='Sample size')
    var_parser.add_argument('--phi', type=float, default=0.6, help='AR coefficient')
    var_parser.add_argument('--window', type=int, default=100, help='Rolling window size')
    var_parser.add_argument('--horizon', type=int, default=20, help='Forecast horizon')

    # Mean subcommand
    mean_parser = sub.add_parser('mean', help='Run mean break MC experiments')
    mean_parser.add_argument('--quick', action='store_true', help='Quick test (10 reps)')
    mean_parser.add_argument('--n-sim', type=int, default=300, help='MC replications')
    mean_parser.add_argument('--T', type=int, default=400, help='Sample size')
    mean_parser.add_argument('--Tb', type=int, default=200, help='Break point')
    mean_parser.add_argument('--window', type=int, default=60, help='Rolling window size')

    # Parameter subcommand
    param_parser = sub.add_parser('parameter', help='Run parameter break MC experiments')
    param_parser.add_argument('--quick', action='store_true', help='Quick test (10 reps)')
    param_parser.add_argument('--n-sim', type=int, default=300, help='MC replications')
    param_parser.add_argument('--T', type=int, default=400, help='Sample size')
    param_parser.add_argument('--Tb', type=int, default=200, help='Break point')
    param_parser.add_argument('--window', type=int, default=80, help='Rolling window size')
    param_parser.add_argument('--innovation', type=str, default='normal', help='Innovation type: normal or student')
    param_parser.add_argument('--df', type=int, default=5, help='Degrees of freedom for Student-t')
    param_parser.add_argument('--experiment', action='store_true', help='Run full parameter experiments (single + recurring)')

    # Runner subcommand (full pipeline)
    run_parser = sub.add_parser('runner', help='Run full experiment pipeline')
    run_parser.add_argument('--scenarios', type=str, help='JSON scenario file')
    run_parser.add_argument('--quick', action='store_true', help='Quick test mode')
    run_parser.add_argument('--plot', action='store_true', help='Save summary figures')

    args = parser.parse_args()

    if args.cmd == 'variance':
        from scripts.runner import mc_variance_breaks
        
        n_sim = 10 if args.quick else args.n_sim
        T = 100 if args.quick else args.T
        
        print(f"Running variance break MC (n_sim={n_sim}, T={T})...")
        df_point, df_unc = mc_variance_breaks(
            n_sim=n_sim, T=T, phi=args.phi,
            window=args.window, horizon=args.horizon
        )
        print("\nResults:")
        print(df_point.round(4).to_string(index=False))
        return 0

    elif args.cmd == 'mean':
        from scripts.runner import mc_mean_breaks
        
        n_sim = 10 if args.quick else args.n_sim
        T = 100 if args.quick else args.T
        Tb = min(args.Tb, T - 20)
        
        print(f"Running mean break MC (n_sim={n_sim}, T={T}, Tb={Tb})...")
        df = mc_mean_breaks(
            n_sim=n_sim, T=T, Tb=Tb, window=args.window
        )
        print("\nResults:")
        print(df.round(4).to_string(index=False))
        return 0

    elif args.cmd == 'parameter':
        from scripts.runner import mc_unified
        
        n_sim = 10 if args.quick else args.n_sim
        T = 100 if args.quick else args.T
        Tb = min(args.Tb, T - 50)
        
        scenarios = [{
            "name": f"Parameter Break ({args.innovation})",
            "task": "parameter",
            "Tb": Tb,
            "phi1": 0.2,
            "phi2": 0.9,
            "distribution": args.innovation,
            "nu": args.df
        }]
        
        print(f"Running parameter break MC (n_sim={n_sim}, innovation={args.innovation})...")
        df = mc_unified(
            n_sim=n_sim, T=T, window=args.window, task="parameter", scenarios=scenarios
        )
        
        print("\nResults:")
        print(df.round(4).to_string(index=False))
        return 0

    elif args.cmd == 'runner':
        from scripts import runner
        # Pass through to full runner
        sys.argv = ['runner'] + sys.argv[2:]
        runner.main()
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
