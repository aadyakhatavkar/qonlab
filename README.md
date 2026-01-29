<p align="center">
  <h1 align="center">📈 Structural Break Forecasting</h1>
  <p align="center">
    <strong>A Monte Carlo Study of Time Series Forecasting Under Parameter Instability</strong>
  </p>
</p>

---

**Research Module in Econometrics and Statistics**  
University of Bonn | Winter Semester 2025/26  
[Course Website](https://vladislav-morozov.github.io/simulations-course/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Team](#-team)
- [Quick Start](#-quick-start)
- [Break Types](#-break-types)
- [Methods](#-methods)
- [Metrics](#-metrics)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Scenarios](#-scenarios)
- [Paper](#-paper)
- [References](#-references)
- [Changelog](#-changelog)

---

## 🎯 Overview

This project investigates **forecasting performance under structural breaks** using Monte Carlo simulations.

### Research Questions

1. How do forecasting methods perform under different break types?
2. What is the optimal rolling window size for different break magnitudes?
3. Can adaptive methods match oracle specifications?
4. How do heavy-tailed (Student-t) distributions affect results?

### Key Features

- **3 types of structural breaks:** variance, mean, parameter
- **6+ forecasting methods:** ARIMA, GARCH, Markov Switching, etc.
- **Heavy-tailed distributions:** Student-t with standardization
- **Optimal window selection:** Pesaran (2013) grid search
- **Comprehensive evaluation:** RMSE, Coverage, Log-score

---

## 👥 Team

| Section | Owner | Status |
|---------|-------|--------|
| **Variance Breaks** | Aadya | ✅ Integrated |
| **Mean Breaks** | Bakhodir | 🔄 In Progress |
| **Parameter Breaks** | Mahir | 🔄 In Progress |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aadyakhatavkar/qonlab.git
cd qonlab
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

# Custom scenarios
python main.py mc --scenarios scenarios/example_scenarios.json
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Break Types

| Break Type | Description | DGP | Owner |
|------------|-------------|-----|-------|
| **Variance** | Volatility shift | $y_t = \phi y_{t-1} + \sigma_t \varepsilon_t$ | Aadya |
| **Mean** | Intercept shift | $y_t = \mu_t + \phi y_{t-1} + \varepsilon_t$ | Bakhodir |
| **Parameter** | AR coefficient shift | $y_t = \phi_t y_{t-1} + \varepsilon_t$ | Mahir |

### Code Examples

```python
from dgps.static import simulate_variance_break, simulate_mean_break

# Variance break (volatility doubles at t=200)
y = simulate_variance_break(T=400, variance_Tb=200, variance_sigma1=1.0, variance_sigma2=2.0)

# With Student-t innovations
y = simulate_variance_break(T=400, distribution='t', nu=3)

# Mean break
from analyses.mean_simulations import simulate_mean_break_ar1
y = simulate_mean_break_ar1(T=300, Tb=150, mu0=0.0, mu1=2.0)

# Parameter break
from analyses.param_simulations import simulate_parameter_break_ar1
y = simulate_parameter_break_ar1(T=400, Tb=200, phi1=0.2, phi2=0.9)
```

---

## 🔬 Methods

### Forecasting Methods

| Method | Description | Break Type |
|--------|-------------|------------|
| **Global ARIMA** | Full-sample fit with auto-order | All |
| **Rolling ARIMA** | Window-based adaptive | All |
| **GARCH(1,1)** | Conditional variance | Variance |
| **Post-Break ARIMA** | Estimated break point | All |
| **Break Dummy (Oracle)** | Known break date | Mean |
| **Markov Switching** | Regime-switching | All |

### Key Technical Features

- ✅ **Automatic ARIMA order selection** via AIC/BIC
- ✅ **Heavy-tailed distributions** (standardized Student-t)
- ✅ **Optimal window grid search** (Pesaran 2013)
- ✅ **Break point estimation** via SSE minimization
- ✅ **Realized volatility functions** for empirical data

---

## 📈 Metrics

### Point Forecast

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum e_i^2}$ | Penalizes large errors |
| **MAE** | $\frac{1}{N}\sum \|e_i\|$ | Average magnitude |
| **Bias** | $\frac{1}{N}\sum e_i$ | Systematic error |

### Uncertainty

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Coverage 80%** | 0.80 | Interval captures 80% |
| **Coverage 95%** | 0.95 | Interval captures 95% |
| **Log-score** | Higher=better | Probabilistic quality |

---

## 📁 Project Structure

```
qonlab/
├── dgps/                              # Data-Generating Processes
│   ├── static.py                      # Variance breaks + shared utilities
│   ├── mean.py                        # Mean break DGPs
│   ├── parameter.py                   # Parameter break DGPs
│   └── variance.py                    # Variance-specific DGPs
├── estimators/                        # Forecasting Methods
│   ├── forecasters.py                 # Variance: ARIMA, GARCH, Markov
│   ├── mean.py                        # Mean break forecasters
│   └── parameter.py                   # Parameter break forecasters
├── analyses/                          # Monte Carlo Engines
│   ├── simulations.py                 # Variance MC runner
│   ├── mean_simulations.py            # Mean MC runner + forecasters
│   ├── param_simulations.py           # Parameter MC runner + forecasters
│   └── plots.py                       # Visualization
├── legacy/                            # Original standalone scripts
│   ├── legacy_mean_change/            # Bakhodir's original scripts
│   └── legacy_parameter_change/       # Mahir's original scripts
├── scenarios/                         # Experiment configurations
│   └── example_scenarios.json         # Pre-defined scenarios
├── docs/paper/                        # LaTeX paper
├── tests/                             # Test suite
├── Variance_Change_Documentation.ipynb # Runnable documentation
└── main.py                            # CLI entrypoint
```

---

## 🔧 API Reference

### CLI

```bash
python main.py mc [OPTIONS]

Options:
  --quick           Quick test (10 reps)
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
# Variance MC
from analyses.simulations import mc_variance_breaks, mc_variance_breaks_grid
df_point, df_unc = mc_variance_breaks(n_sim=200, T=400, horizon=20)

# Mean MC
from analyses.mean_simulations import mc_mean_breaks
df = mc_mean_breaks(n_sim=200, T=300, Tb=150)

# Parameter MC
from analyses.param_simulations import mc_parameter_breaks_post
err = mc_parameter_breaks_post(n_sim=300, T=400, Tb=200)
```

---

## 📋 Scenarios

Scenarios are defined in `scenarios/example_scenarios.json`:

| Task | Owner | Scenarios |
|------|-------|-----------|
| **variance** | Aadya | Small break, Late break |
| **parameter** | Mahir | Single break (φ₁=0.2 → φ₂=0.9) |
| **mean** | Bakhodir | Moderate, Large, Early, Late |

### Scenario Format

```json
{
  "name": "Single mean break - moderate",
  "task": "mean",
  "T": 400,
  "Tb": 200,
  "mu0": 0.0,
  "mu1": 2.0,
  "owner": "bakhodir"
}
```

---

## 📄 Paper

```bash
cd docs/paper && make
```

**Sections:**
1. Introduction
2. Data-Generating Processes
3. Forecasting Methods
4. Monte Carlo Design
5. Evaluation Metrics
6. Results
7. Conclusion

---

## 📚 References

| Reference | Topic |
|-----------|-------|
| [Pesaran (2013)](https://doi.org/10.1016/B978-0-444-62731-5.00021-9) | Structural breaks in forecasting |
| [Box & Jenkins (1970)](https://www.wiley.com/en-us/Time+Series+Analysis) | ARIMA methodology |
| [Francq & Zakoïan (2019)](https://www.wiley.com/en-us/GARCH+Models) | GARCH models |
| [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1) | GARCH |
| [Hamilton (1989)](https://doi.org/10.2307/1912559) | Markov switching |
| [Bai & Perron (1998)](https://doi.org/10.2307/2998540) | Break detection |

---

## 📝 Changelog

### January 2026

**Structural Changes:**
- Renamed `mean_change/`, `parameter_change/` → `legacy/`
- Created modular `analyses/mean_simulations.py`, `analyses/param_simulations.py`
- Consolidated DGPs into `dgps/` with separate files per break type
- Merged `experiments/` into `scenarios/`

**Technical Features:**
- Student-t distributions with standardization
- Auto ARIMA order selection (AIC/BIC)
- Markov switching forecaster
- Realized volatility functions
- Unified simulation engine for all break types

---

## 🤝 Contact

**Aadya Khatavkar** — s38akhat@uni-bonn.de

---

<p align="center">
  <sub>Built for the <a href="https://vladislav-morozov.github.io/simulations-course/">Research Module</a> at University of Bonn</sub>
</p>