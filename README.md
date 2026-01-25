# qonlab

Lightweight tools and experiments for variance-break analysis.

Primary guide: follow the professor's master guide (COMPREHENSIVE_MASTER_GUIDE.md).
For layout guidance see the econ-project-templates: https://econ-project-templates.readthedocs.io/

Canonical modules in this repo:

- `dgps/` — data-generating processes and scenario validation
- `estimators/` — forecasting helpers (ARIMA, GARCH) and metrics
- `analyses/` — Monte Carlo runners and grid analyses
- `plots/` — visualization utilities (plots/variance.py)
- `scripts/` — experiment runners (scripts/pixi.py)

Use `main.py` as the canonical CLI entrypoint (subcommands: `mc`, `pixi`).

If you'd like this repo fully converted to the econ-project-templates layout
(src/, docs/, CI), say “apply template” and I will create a branch and implement
the conversion.