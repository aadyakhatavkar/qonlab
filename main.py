#!/usr/bin/env python3
"""Top-level CLI for the project.

Subcommands:
  mc    - run Monte Carlo variance-break analyses
  runner - run variance analysis experiments

This file is the canonical entrypoint to avoid duplicate entrypoints.
"""
import sys
import argparse


def _call_module_main(mod, argv):
    # Call a module's main function with argv (excluding program name)
    try:
        main = getattr(mod, 'main')
    except Exception as e:
        print(f"Module {mod} does not expose a callable main(): {e}")
        return 2
    # Some modules expect to parse sys.argv directly; provide argv slice
    return main()


def main():
    parser = argparse.ArgumentParser(prog='main.py', description='Project CLI')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('mc', help='Run Monte Carlo variance-break experiments')
    sub.add_parser('runner', help='Run variance analysis experiments')

    args, remaining = parser.parse_known_args()

    if args.cmd == 'mc':
        # delegate to analyses.simulations.main
        from analyses import simulations as mc_mod
        # pass through remaining args via sys.argv for the delegated module
        sys.argv = [sys.argv[0]] + remaining
        return _call_module_main(mc_mod, remaining)

    if args.cmd == 'runner':
        # delegate to scripts.runner.main
        from scripts import runner
        sys.argv = [sys.argv[0]] + remaining
        return _call_module_main(runner, remaining)

    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
