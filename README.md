<p align="center">
  <h1 align="center">📈 Structural Break Forecasting</h1>
  <p align="center">
    <strong>A Monte Carlo Study of Time Series Forecasting Under Parameter Instability</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/release/python-3123/"><img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-blue"></a>
    <a href="https://github.com/aadyakhatavkar/qonlab/actions"><img alt="Tests" src="https://img.shields.io/badge/tests-pytest-brightgreen"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
    <a href="#-status"><img alt="Status" src="https://img.shields.io/badge/status-active-success"></a>
  </p>
</p>

---

**Research Module in Econometrics and Statistics**  
University of Bonn | Winter Semester 2025/26  
[Course Website](https://vladislav-morozov.github.io/simulations-course/)

---

## � Quick Start

### Run All Experiments (Standardized: T=400, Tb=200, n_sim=300)
```bash
python scripts/runner.py              # All 3 break types with variants
python scripts/runner.py --quick      # Quick test (T=150, n_sim=10)
python scripts/runner.py --variance   # Variance breaks only
```

### Example: Variance Break Analysis
```python
from analyses import mc_variance_single_break

results = mc_variance_single_break(n_sim=300, T=400, Tb=200, seed=42)
print(results[['Scenario', 'Method', 'RMSE', 'MAE']])
```

### Project Structure

## ✅ Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Pipeline** | ✅ Functional | All dependencies installed, imports working |
| **Simulations** | ✅ Ready | Variance, mean, parameter break MC engines |
| **Forecasters** | ✅ Complete | ARIMA, GARCH, Markov switching methods |
| **Scenarios** | ✅ Active | 7 scenarios covering all task types |
| **Code Quality** | ✅ Clean | No duplications, symmetric structure |
| **Tests** | 🔄 Coverage | `pytest tests/` for validation |

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
- **Fixed window approach:** Practitioners prefer fixed windows + break detection
- **Comprehensive evaluation:** RMSE, MAE, Bias, Coverage, Log-score

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
# Using scenarios (recommended)
python scripts/runner.py --scenarios scenarios.json

# With plotting
python scripts/runner.py --scenarios scenarios.json --plot

# Quick test
python main.py variance --quick
python main.py mean --quick
python main.py parameter --quick

# Full simulation (200 replications)
python main.py variance --n-sim 200 --T 400 --horizon 20
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
from dgps.variance import simulate_variance_break_ar1
from dgps.mean import simulate_mean_break_ar1
from dgps.parameter import simulate_parameter_break_ar1

# Variance break (volatility doubles at t=200)
y = simulate_variance_break_ar1(T=400, Tb=200, sigma1=1.0, sigma2=2.0, seed=42)

# With Student-t innovations
y = simulate_variance_break_ar1(T=400, Tb=200, distribution='t', nu=3, seed=42)

# Mean break
y = simulate_mean_break_ar1(T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, seed=42)

# Parameter break
y = simulate_parameter_break_ar1(T=400, Tb=200, phi1=0.2, phi2=0.9, seed=42)
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
│   ├── variance.py                    # Variance break DGPs
│   ├── mean.py                        # Mean break DGPs
│   ├── parameter.py                   # Parameter break DGPs (with Student-t)
│   ├── recurring.py                   # Markov-switching AR(1) DGP
│   ├── mean_multiplebreaks.py         # Multiple mean breaks DGP
│   ├── static.py                      # Helper utilities (innovations, transformations)
│   ├── utils.py                       # Scenario validation utilities
│   └── __init__.py                    # Exports all DGP functions
├── estimators/                        # Forecasting Methods
│   ├── variance.py                    # Variance: ARIMA, GARCH, metrics
│   ├── mean.py                        # Mean: ARMA, oracle, break detection, Markov switching
│   ├── parameter.py                   # Parameter: ARMA, Markov switching
│   ├── mean_multiplebreaks.py         # Multiple breaks forecasting
│   └── __init__.py                    # Exports all estimator functions
├── analyses/                          # Monte Carlo Simulations & Plotting
│   ├── simulations.py                 # Unified MC engine (dispatches to task-specific functions)
│   ├── variance_simulations.py        # Variance-specific MC: mc_variance_breaks_post/full
│   ├── mean_simulations.py            # Mean-specific MC: mc_mean_breaks
│   ├── param_simulations.py           # Parameter-specific MC: mc_parameter_breaks_post/full
│   ├── plots_variance.py              # Variance break plots
│   ├── plots_mean.py                  # Mean break plots
│   ├── plots_parameter.py             # Parameter break plots
│   └── __init__.py                    # Exports all simulation and plotting functions
├── scripts/                           # Experiment Runners
│   └── runner.py                      # Full pipeline with scenarios (--scenarios flag)
├── legacy/                            # Archive of original implementations (consolidated into main structure)
│   ├── mean_legacy/                   # Original mean break analysis code
│   ├── parameter_legacy/              # Original parameter break analysis code
│   ├── variance_plot_results.py       # Legacy plotting utilities
│   ├── Bakhodir_latex/                # Bakhodir's original LaTeX paper
│   └── __init__.py
├── scenarios.json                     # Active scenarios config (root level)
├── docs/paper/                        # LaTeX paper with sections
├── tests/                             # Test suite
├── main.py                            # CLI entrypoint
├── task_simulations.py                # PyTask workflow definitions
├── protocols.py                       # Type protocols for DGP & Estimator interfaces
└── requirements.txt                   # Package dependencies
```

---

## 🔧 API Reference

### CLI

```bash
# Variance experiments
python main.py variance --quick
python main.py variance --n-sim 200 --T 400 --horizon 20

# Mean experiments
python main.py mean --quick
python main.py mean --n-sim 100 --T 300 --Tb 150

# Parameter experiments
python main.py parameter --quick
python main.py parameter --innovation student --df 50

# Full pipeline with scenarios (recommended)
python scripts/runner.py --scenarios scenarios.json --plot
```

### Python API

```python
# Unified interface (via scenarios)
from analyses.simulations import mc_variance_breaks
scenarios = [{"task": "variance", "variance_Tb": 200, "variance_sigma1": 1.0, "variance_sigma2": 2.0}]
df_point, df_unc = mc_variance_breaks(scenarios=scenarios, n_sim=200, T=400)

# Task-specific direct calls
from analyses.variance_simulations import mc_variance_breaks_post
errors = mc_variance_breaks_post(n_sim=200, T=400, Tb=200, sigma1=1.0, sigma2=2.0)

from analyses.mean_simulations import mc_mean_breaks
df = mc_mean_breaks(n_sim=200, T=300, Tb=150, mu0=0.0, mu1=2.0)

from analyses.param_simulations import mc_parameter_breaks_post
errors = mc_parameter_breaks_post(n_sim=300, T=400, Tb=200, phi1=0.2, phi2=0.9)
```

---

## 📋 Scenarios

Scenarios are defined in `scenarios.json` (root level):

| Task | Owner | Scenarios |
|------|-------|-----------|
| **variance** | Aadya | Small break (Tb=40), Late break (Tb=80) |
| **parameter** | Mahir | Single break (φ₁=0.2 → φ₂=0.9) |
| **mean** | Bakhodir | Moderate, Large, Early (Tb=100), Late (Tb=300) |

### Scenario Format

```json
{
  "name": "Single variance break small",
  "task": "variance",
  "variance_Tb": 40,
  "variance_sigma1": 1.0,
  "variance_sigma2": 2.0,
  "owner": "aadya",
  "tag": "variance_small"
}
```

### Running Scenarios

```bash
python scripts/runner.py --scenarios scenarios.json --plot
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

### January 29, 2026 — Architecture Refactoring

**File Consolidation (with Mahir & Bakhodir approval):**
- Removed duplicate `legacy/runner.py` (consolidated into `scripts/runner.py`)
- Renamed `legacy/legacy_mean_change/` → `legacy/mean_legacy/` (archived for reference)
- Renamed `legacy/legacy_parameter_change/` → `legacy/parameter_legacy/` (archived for reference)
- All functionality now in main: `analyses/{variance,mean,parameter}_simulations.py`

**File Reorganization:**
- Renamed `estimators/forecasters.py` → `estimators/variance.py` for consistency
- Renamed `analyses/plots.py` → `analyses/plots_variance.py` for consistency  
- Created `analyses/variance_simulations.py` with `mc_variance_breaks_post()` and `mc_variance_breaks_full()` functions
- Moved `scenarios/example_scenarios.json` → `scenarios.json` (root level)
- Cleaned up `scenarios/` folder (removed legacy images, LaTeX files)

**Code Improvements:**
- Refactored `analyses/simulations.py` to use unified MC engine with task-specific dispatch functions
- Updated `protocols.py` with proper type annotations for DGP and Estimator interfaces
- Fixed runner.py: removed `mc_variance_breaks_grid()` (grid search removed per Pesaran 2013 policy favoring fixed windows)
- Updated 8+ import statements across codebase to use new file paths
- Removed duplicate `estimate_variance_break_point()` from `dgps/static.py` (kept in `dgps/variance.py`)

**New Symmetric Structure:**
- All three break types (variance/mean/parameter) now have:
  - `estimators/{variance,mean,parameter}.py` — forecasting methods
  - `analyses/{variance,mean,parameter}_simulations.py` — MC simulation engines  
  - `analyses/plots_{variance,mean,parameter}.py` — visualization functions

**Validation:**
- ✓ Pipeline functional: all dependencies installed, imports working
- ✓ Scenarios load correctly: 7 scenarios covering all 3 task types
- ✓ No code duplications
- ✓ Symmetric file structure established

### Earlier Changes

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