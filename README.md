# qonlab

Lightweight tools and experiments for variance-break analysis.




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

## Setup

### Recommended: Using Pixi (reproducible environments)
1. Install Pixi: https://prefix.dev
2. Run: `pixi install`
3. Run experiments: `pixi run python main.py`

This ensures exact dependency versions across all environments.

### Alternative: Using pip
```bash
pip install -r requirements.txt
python main.py
```