# Experiment Catalog

This directory contains configurations and documentation for all Monte Carlo experiments in the research module project.

## Experiment Types

### 1. Mean Break Experiments (`mean_break/`)
Structural breaks in the intercept of AR(1) processes.

### 2. Variance Break Experiments (`variance_break/`)
Structural breaks in the innovation variance (volatility shifts).

### 3. Parameter Break Experiments (`parameter_break/`)
Structural breaks in the AR(1) coefficient.

---

## Reproducibility

All experiments can be reproduced using the following commands:

```bash
# Quick test run (10 replications)
python main.py mc --quick

# Full simulation (200 replications)
python main.py mc --n-sim 200 --T 400 --horizon 20

# Grid search for optimal window (Pesaran 2013 framework)
python main.py mc --grid --n-sim 100
```

---

## Configuration Files

Each experiment has a JSON configuration file specifying:

- **DGP parameters:** T, break location, pre/post-break values
- **Method settings:** window sizes, estimation options
- **Evaluation settings:** metrics, replications, seeds

Example:

```json
{
  "name": "Mean break - moderate shift",
  "task": "mean",
  "T": 400,
  "Tb": 200,
  "mu0": 0.0,
  "mu1": 2.0,
  "phi": 0.6,
  "sigma": 1.0,
  "n_sim": 200,
  "seed": 42
}
```

---

## Output Structure

Results are saved to `results/` with the following structure:

```
results/
├── figures/
│   ├── rmse_comparison.pdf
│   ├── coverage_by_window.pdf
│   └── loss_surface.pdf
├── tables/
│   ├── point_metrics.tex
│   └── uncertainty_metrics.tex
└── raw/
    └── mc_results_YYYYMMDD.csv
```

---

## Simulation Standards

Following best practices from the course:

1. **Seed management:** Reproducible random number generation
2. **Parallel execution:** Optional joblib parallelization
3. **Error handling:** Graceful failure tracking
4. **Documentation:** All parameters recorded with results
