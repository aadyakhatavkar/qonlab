<p align="center">
  <h1 align="center">📈 Structural Break Forecasting</h1>
  <p align="center">
    <strong>A Monte Carlo Study of Time Series Forecasting Under Parameter Instability</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-documentation">Documentation</a> •
    <a href="#-methods">Methods</a> •
    <a href="#-notebooks">Notebooks</a> •
    <a href="#-paper">Paper</a>
  </p>
</p>

---

[![CI](https://github.com/aadyakhatavkar/qonlab/actions/workflows/ci.yml/badge.svg)](https://github.com/aadyakhatavkar/qonlab/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research Module in Econometrics and Statistics**  
University of Bonn | Winter Semester 2025/26  
[Course Website](https://vladislav-morozov.github.io/simulations-course/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Break Types](#-break-types)
- [Methods](#-methods)
- [Metrics](#-metrics)
- [Notebooks](#-notebooks)
- [Paper](#-paper)
- [API Reference](#-api-reference)
- [References](#-references)

---

## 🎯 Overview

This project investigates **forecasting performance under structural breaks** using Monte Carlo simulations. We implement:

- **3 types of structural breaks** (variance, mean, parameter)
- **6 forecasting methods** (ARIMA, GARCH, Markov Switching, etc.)
- **Heavy-tailed distributions** (Student-t with standardization)
- **Optimal window selection** (Pesaran 2013 grid search)
- **Comprehensive evaluation** (RMSE, Coverage, Log-score)

### Research Questions

1. How do forecasting methods perform under different break types?
2. What is the optimal rolling window size for different break magnitudes?
3. Can adaptive methods match oracle specifications?

---

## � Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aadyakhatavkar/qonlab.git
cd qonlab

# Install dependencies
pip install -r requirements.txt
```

### Run Simulations

```bash
# Quick test (10 replications)
python main.py mc --quick

# Full simulation (200 replications)
python main.py mc --n-sim 200 --T 400 --horizon 20

# Grid search for optimal window
python main.py mc --grid
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Break Types

| Break Type | Description | DGP | Parameters |
|------------|-------------|-----|------------|
| **Variance** | Volatility shift | $y_t = \phi y_{t-1} + \sigma_t \varepsilon_t$ | $\sigma_1 \to \sigma_2$ |
| **Mean** | Intercept shift | $y_t = \mu_t + \phi y_{t-1} + \varepsilon_t$ | $\mu_0 \to \mu_1$ |
| **Parameter** | AR coefficient shift | $y_t = \phi_t y_{t-1} + \varepsilon_t$ | $\phi_1 \to \phi_2$ |

### Code Examples

```python
from dgps.static import simulate_variance_break, simulate_mean_break, simulate_parameter_break

# Variance break (volatility doubles at t=200)
y = simulate_variance_break(T=400, variance_Tb=200, variance_sigma1=1.0, variance_sigma2=2.0)

# Mean break (intercept shifts from 0 to 2)
y = simulate_mean_break(T=300, Tb=150, mu0=0.0, mu1=2.0)

# Parameter break (AR coef changes from 0.2 to 0.9)
y = simulate_parameter_break(T=400, Tb=200, phi1=0.2, phi2=0.9)

# Heavy-tailed innovations (Student-t)
y = simulate_variance_break(T=400, distribution='t', nu=3)
```

---

## 🔬 Methods

### Forecasting Methods

| Method | Description | Code |
|--------|-------------|------|
| **Global ARIMA** | Full-sample fit | `forecast_variance_dist_arima_global()` |
| **Rolling ARIMA** | Window-based adaptive | `forecast_variance_dist_arima_rolling()` |
| **GARCH(1,1)** | Conditional variance | `forecast_garch_variance()` |
| **Post-Break ARIMA** | Estimated break point | `forecast_variance_arima_post_break()` |
| **Averaged Window** | Ensemble over windows | `forecast_variance_averaged_window()` |
| **Markov Switching** | Regime-switching | `forecast_markov_switching()` |

### Key Features

- ✅ **Automatic ARIMA order selection** via AIC/BIC
- ✅ **Heavy-tailed distributions** (standardized Student-t)
- ✅ **Optimal window grid search** (Pesaran 2013)
- ✅ **Break point estimation** via SSE minimization

---

## � Metrics

### Point Forecast Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum e_i^2}$ | Penalizes large errors |
| **MAE** | $\frac{1}{N}\sum \|e_i\|$ | Average error magnitude |
| **Bias** | $\frac{1}{N}\sum e_i$ | Systematic over/under-forecasting |

### Uncertainty Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Coverage 80%** | 0.80 | Interval captures 80% of observations |
| **Coverage 95%** | 0.95 | Interval captures 95% of observations |
| **Log-score** | Higher is better | Probabilistic forecast quality |

---

## 📓 Notebooks

| Notebook | Description | 
|----------|-------------|
| [`Variance_Change_Documentation.ipynb`](Variance_Change_Documentation.ipynb) | � Full documentation with theory, examples, and best practices |
| [`variance_workflow.ipynb`](variance_workflow.ipynb) | ⚡ Quick workflow demonstration |

**Run notebooks:**
```bash
jupyter notebook Variance_Change_Documentation.ipynb
```

---

## � Paper

The LaTeX paper is in [`docs/paper/`](docs/paper/):

```bash
cd docs/paper
make          # Compile PDF
make clean    # Remove auxiliary files
```

**Paper sections:**
1. Introduction
2. Data-Generating Processes
3. Forecasting Methods
4. Monte Carlo Design
5. Evaluation Metrics
6. Results
7. Conclusion

---

## 📁 Project Structure

```
qonlab/
├── dgps/                       # Data-Generating Processes
│   └── static.py               # Break simulations + RV functions
├── estimators/                 # Forecasting Methods
│   └── forecasters.py          # ARIMA, GARCH, Markov Switching
├── analyses/                   # Monte Carlo Engines
│   ├── simulations.py          # MC runner
│   └── plots.py                # Visualization
├── scripts/                    # Runners
│   └── runner.py               # Experiment CLI
├── scenarios/                  # Configurations
│   └── example_scenarios.json  # Pre-defined experiments
├── tests/                      # Test suite
├── docs/paper/                 # LaTeX paper
├── *.ipynb                     # Runnable notebooks
├── CHANGES.md                  # Technical changelog
├── RESEARCH.md                 # Research documentation
└── main.py                     # CLI entrypoint
```

---

## � API Reference

### Main CLI

```bash
python main.py mc [OPTIONS]

Options:
  --quick           Quick test (10 reps, small sample)
  --grid            Grid search for optimal window
  --n-sim INT       MC replications (default: 200)
  --T INT           Sample size (default: 400)
  --phi FLOAT       AR(1) coefficient (default: 0.6)
  --window INT      Rolling window size (default: 100)
  --horizon INT     Forecast horizon (default: 20)
  --scenarios FILE  JSON scenario file
```

### Python API

```python
# Monte Carlo simulation
from analyses.simulations import mc_variance_breaks, mc_variance_breaks_grid

df_point, df_unc = mc_variance_breaks(n_sim=200, T=400, horizon=20)

# Grid search
df_grid = mc_variance_breaks_grid(
    window_sizes=[20, 50, 100, 200],
    break_magnitudes=[1.5, 2.0, 3.0, 5.0]
)
```

---

## � References

| Reference | Topic |
|-----------|-------|
| [Pesaran (2013)](https://doi.org/10.1016/B978-0-444-62731-5.00021-9) | Structural breaks in forecasting |
| [Box & Jenkins (1970)](https://www.wiley.com/en-us/Time+Series+Analysis) | ARIMA methodology |
| [Francq & Zakoïan (2019)](https://www.wiley.com/en-us/GARCH+Models) | GARCH models |
| [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1) | GARCH |
| [Hamilton (1989)](https://doi.org/10.2307/1912559) | Markov switching |

---

## 📝 Changelog

See [CHANGES.md](CHANGES.md) for detailed technical changelog.

**Recent updates (Jan 2026):**
- ✅ Unified simulation engine for all break types
- ✅ Student-t distributions with standardization
- ✅ Auto ARIMA order selection (AIC/BIC)
- ✅ Markov switching forecaster
- ✅ Realized volatility functions

---

## 🤝 Contributing

This is a research module project. For questions, contact:

**Aadya Khatavkar**  
📧 s38akhat@uni-bonn.de

---

<p align="center">
  <sub>Built for the <a href="https://vladislav-morozov.github.io/simulations-course/">Research Module in Econometrics and Statistics</a> at University of Bonn</sub>
</p>