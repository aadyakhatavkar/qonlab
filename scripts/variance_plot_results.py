#!/usr/bin/env python3
"""Load a point/unc CSV pair and produce a few diagnostic plots.
Usage: scripts/plot_results.py results/variance_20260109_163309_point.csv results/variance_20260109_163309_unc.csv
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: plot_results.py <point_csv> <unc_csv>')
        sys.exit(1)
    pfile = sys.argv[1]
    ufile = sys.argv[2]
    dfp = pd.read_csv(pfile)
    dfu = pd.read_csv(ufile)
    os.makedirs('figures', exist_ok=True)

    # RMSE bar plot
    df_rmse = dfp[dfp['Metric'] == 'RMSE']
    models = [c for c in df_rmse.columns if c not in ('Scenario', 'Metric')]
    scenarios = df_rmse['Scenario'].unique()
    x = range(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8,4))
    for i, m in enumerate(models):
        vals = [df_rmse[(df_rmse['Scenario'] == s)][m].values[0] for s in scenarios]
        ax.bar([xi + i*width for xi in x], vals, width, label=m)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(scenarios, rotation=20)
    ax.set_ylabel('RMSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/plot_rmse.png')
    print('Saved figures/plot_rmse.png')

    # Coverage plot (80)
    df_cov80 = dfu[dfu['Metric'] == 'Coverage80']
    models = [c for c in df_cov80.columns if c not in ('Scenario', 'Metric')]
    fig, ax = plt.subplots(figsize=(8,4))
    for i, m in enumerate(models):
        vals = [df_cov80[(df_cov80['Scenario'] == s)][m].values[0] for s in scenarios]
        ax.bar([xi + i*width for xi in x], vals, width, label=m)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(scenarios, rotation=20)
    ax.set_ylabel('Coverage (80%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/plot_coverage80.png')
    print('Saved figures/plot_coverage80.png')
