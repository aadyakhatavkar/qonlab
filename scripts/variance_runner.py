#!/usr/bin/env python3
"""pixi: small experiment runner for variance-break work

Usage examples:
  scripts/pixi.py --quick
  scripts/pixi.py --n-sim 100 --T 400 --window 100 --horizon 20
  scripts/pixi.py --scenarios scenarios.json

Saves results CSVs to `results/` with timestamped filenames.
"""
import argparse
import json
import os
from datetime import datetime
import sys


def _check_dependencies():
    """Check runtime dependencies and print actionable install instructions if missing."""
    missing = []
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append('numpy')
    try:
        import pandas  # noqa: F401
    except Exception:
        missing.append('pandas')
    try:
        import yfinance  # noqa: F401
    except Exception:
        # yfinance is optional unless using --sp500, but recommend it
        missing.append('yfinance')

    if missing:
        print('\nMissing Python packages detected:', ', '.join(missing))
        print('Install required packages with:')
        print('  python3 -m pip install -r requirements.txt')
        print('or individually:')
        print('  python3 -m pip install ' + ' '.join(missing))
        print('Exiting. Re-run after installing dependencies.')
        sys.exit(1)


# run dependency check early
_check_dependencies()

from analyses.mc import mc_variance_breaks, mc_variance_breaks_grid
from dgps.static import simulate_variance_break
from estimators.ols_like import (
    forecast_dist_arima_rolling,
    log_score_normal,
    interval_coverage,
    rmse_mae_bias,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_and_save(df_point, df_unc, tag=None):
    os.makedirs('figures', exist_ok=True)
    # grouped bar chart for RMSE across models
    df_rmse = df_point[df_point['Metric'] == 'RMSE']
    if df_rmse.empty:
        # fallback: save text snapshot
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.text(0.01, 0.5, df_point.to_string(index=False), family='monospace')
        fname = f"figures/variance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        return fname

    # pivot to have models as columns
    df_wide = df_rmse.pivot(index='Scenario', columns='Metric', values='ARIMA Global')
    # better: rebuild with known model columns
    models = [c for c in df_point.columns if c not in ('Scenario', 'Metric')]
    plot_df = df_rmse.set_index('Scenario')[[m for m in models if m in df_rmse.columns]] if False else None

    # safer approach: collect values per model from df_point
    scenarios = df_rmse['Scenario'].unique()
    model_names = [c for c in df_point.columns if c not in ('Scenario', 'Metric')]
    values = {m: [] for m in model_names}
    for s in scenarios:
        row = df_point[(df_point['Scenario'] == s) & (df_point['Metric'] == 'RMSE')]
        for m in model_names:
            values[m].append(float(row[m].values[0]) if m in row.columns else float('nan'))

    x = np.arange(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(model_names):
        ax.bar(x + i * width, values[m], width, label=m)

    ax.set_xticks(x + width * (len(model_names)-1) / 2)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.set_ylabel('RMSE')
    ax.legend()
    fname = f"figures/variance_rmse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fname


def load_scenarios(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_dfs(df_point, df_unc, tag=None):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tagpart = f"_{tag}" if tag else ""
    base = f"results/variance_{ts}{tagpart}"
    os.makedirs('results', exist_ok=True)
    pfile = base + '_point.csv'
    ufile = base + '_unc.csv'
    df_point.to_csv(pfile, index=False)
    df_unc.to_csv(ufile, index=False)
    return pfile, ufile


def main():
    parser = argparse.ArgumentParser(description='pixi: run variance-break experiments')
    parser.add_argument('--quick', action='store_true', help='short quick run')
    parser.add_argument('--n-sim', type=int, default=200)
    parser.add_argument('--T', type=int, default=400)
    parser.add_argument('--phi', type=float, default=0.6)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--scenarios', type=str, default=None, help='JSON file with scenarios list')
    parser.add_argument('--tag', type=str, default=None, help='Optional tag to add to output filenames')
    parser.add_argument('--plot', action='store_true', help='Save a small summary figure to figures/')
    parser.add_argument('--sp500', action='store_true', help='Fetch S&P500 (^GSPC) and run variance analysis on returns')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date for S&P data (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date for S&P data (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.quick:
        n_sim = min(10, args.n_sim)
        T = min(100, args.T)
        window = min(20, args.window)
        horizon = min(5, args.horizon)
    else:
        n_sim = args.n_sim
        T = args.T
        window = args.window
        horizon = args.horizon

    if args.scenarios:
        scenarios = load_scenarios(args.scenarios)
    else:
        scenarios = [
            {"name": "Single variance break", "Tb": max(2, T//2), "sigma1": 1.0, "sigma2": 2.0, "task": "variance", "owner": "aadya"}
        ]

    # scenarios may contain entries for different tasks/owners. We'll dispatch per scenario.
    all_point = []
    all_unc = []

    for sc in scenarios:
        task = sc.get('task', 'variance')
        owner = sc.get('owner', None)
        tag = sc.get('tag', None) or args.tag
        name = sc.get('name', 'scenario')

        print(f"\nRunning scenario '{name}' (task={task}, owner={owner})")

        if task == 'variance':
            # pass a single-scenario list to mc_variance_breaks so output rows are scenario-scoped
            df_point, df_unc = mc_variance_breaks(n_sim=n_sim, T=T, phi=args.phi, window=window, horizon=horizon, scenarios=[sc])
        elif args.sp500:
            # Run S&P application on returns
            try:
                import yfinance as yf
                print('Fetching S&P500 data...')
                end = args.end
                data = yf.download('^GSPC', start=args.start, end=end, progress=False)
                if data is None or data.empty:
                    print('Failed to fetch S&P data or empty result')
                    continue
                # compute log returns
                prices = data['Adj Close'].dropna()
                returns = np.log(prices).diff().dropna().values
                print(f'Fetched {len(returns)} return observations')
                # Run a small grid analysis on the real returns (use smaller n_sim)
                df_grid = mc_variance_breaks_grid(n_sim=min(20, n_sim), T=len(returns), phi=args.phi, horizon=horizon, window_sizes=[20,50,100], break_magnitudes=[1.5,3.0])
                # save grid
                os.makedirs('results', exist_ok=True)
                outcsv = f"results/sp500_variance_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_grid.to_csv(outcsv, index=False)
                print('Saved S&P variance grid to:', outcsv)
                df_point = pd.DataFrame()
                df_unc = pd.DataFrame()
            except Exception as e:
                print('S&P application failed:', e)
                continue
        elif task == 'parameter':
            # attempt to run parameter-change module if present
            try:
                PC_DIR = os.path.join(ROOT, 'Parameter Change')
                if PC_DIR not in sys.path:
                    sys.path.insert(0, PC_DIR)
                import parameter_change as pc
                # try common entrypoints used in Parameter Change modules
                if hasattr(pc, 'monte_carlo_single_break'):
                    p_point, p_unc = pc.monte_carlo_single_break(n_sim=n_sim, T=T, Tb=sc.get('Tb', T//2), window=window)
                    # standardize to DataFrame rows
                    df_point = pd.DataFrame([{
                        'Scenario': name,
                        'Metric': k,
                        'ARIMA Global': v
                    } for k, v in p_point.items()])
                    df_unc = pd.DataFrame([])
                else:
                    print('Parameter module does not expose expected entrypoint; skipping')
                    continue
            except Exception as e:
                print('Parameter task failed to run:', e)
                continue
        else:
            print('Unknown task', task, '- skipping')
            continue

        # save per-scenario results to results/ with owner/tag
        pfile, ufile = save_dfs(df_point, df_unc, tag=tag or name)
        print('Saved point metrics to:', pfile)
        print('Saved uncertainty metrics to:', ufile)

        all_point.append(df_point.assign(Owner=owner, Task=task))
        all_unc.append(df_unc.assign(Owner=owner, Task=task))

        print('\nSummary (point metrics):')
        print(df_point.round(4).to_string(index=False))

        print('\nSummary (uncertainty metrics):')
        print(df_unc.round(4).to_string(index=False))

        if args.plot:
            try:
                fpath = plot_and_save(df_point, df_unc, tag=tag)
                print('Saved figure to:', fpath)
            except Exception as e:
                print('Plotting failed:', e)

    # optionally concatenate and return summarized combined outputs
    if all_point:
        combined_p = pd.concat(all_point, ignore_index=True)
        combined_u = pd.concat(all_unc, ignore_index=True) if any(len(df)>0 for df in all_unc) else pd.DataFrame()
        return combined_p, combined_u
    return None, None
    if args.plot:
        try:
            fpath = plot_and_save(df_point, df_unc, tag=args.tag)
            print('Saved figure to:', fpath)
        except Exception as e:
            print('Plotting failed:', e)


if __name__ == '__main__':
    main()
