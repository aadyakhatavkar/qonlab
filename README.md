# Structural Break Forecasting

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A Monte Carlo Study of Time Series Forecasting Under Parameter Instability**

University of Bonn | Winter Semester 2025/26  
[Course Website](https://vladislav-morozov.github.io/simulations-course/)

**Authors:** Aadya Khatavkar, Mahir Baylarov, Bakhodir Izzatulloev

---

## Quick Start

### Setup
```bash
# Install pixi (if needed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

### Get Results
```bash
pixi run python main.py --pdf
```
Runs all experiments + generates PDF report with tables & figures. Takes ~20-25 minutes. Output: `outputs/pdf/`

### Quick Test (2 minutes)
```bash
pixi run python main.py --quick
```
Same as above but with n_sim=30 instead of 300 (for testing).

### Full Run (~20 minutes)
```bash
pixi run python main.py
```
All experiments without PDF generation. Results in `outputs/tables/` and `outputs/tex/`.

### Regenerate PDF
```bash
pixi run python scripts/build_pdfs.py --all
```
Builds PDF from existing results (use after experiments are done).

---

## Overview

Evaluates forecasting methods under three types of structural breaks:

- **Variance breaks** — Sudden volatility spikes (e.g., 2008 financial crisis)
- **Mean breaks** — Regime shifts in level (e.g., inflation changes)
- **Parameter breaks** — Changing dynamics via AR coefficients

**Study Design**:
- Time series length: T = 400 (single break at midpoint, Tb = 200)
- Innovation types: Gaussian, Student-t(df=3), Student-t(df=5)
- For recurring breaks: Markov-switching regimes with persistence levels 0.90, 0.95, 0.99
- Monte Carlo: 300 replications per scenario

**Key Research Question**: How do different forecasting strategies react to structural breaks across various simulation scenarios? And how is predictive stability observed?

---

## Advanced Usage

### Run Individual Break Types
```bash
pixi run python runner.py --variance   # Variance breaks only
pixi run python runner.py --mean       # Mean breaks only  
pixi run python runner.py --parameter  # Parameter breaks only
pixi run python runner.py --quick      # Quick test
```

### Step-by-Step Workflow

#### Step 1: Run Specific Experiments
```bash
pixi run python runner.py --variance
# Or combine: pixi run python runner.py --variance --mean --parameter
```
Output: `outputs/tables/*.csv` and `outputs/tex/*.tex`

#### Step 2: Generate Plots
```bash
pixi run python scripts/generate_plots.py
```
Output: `outputs/figures/*.png`

#### Step 3: Compile Tables & Figures to PDF
```bash
pixi run python scripts/build_pdfs.py --all
```
Output: 
- `outputs/pdf/Tables_Results_YYYYMMDD_HHMMSS.pdf` (results)
- `outputs/pdf/Figures_Analysis_YYYYMMDD_HHMMSS.pdf` (plots)

#### Step 4: Create Combined Report (Optional)
```bash
pixi run python scripts/build_pdfs.py --combined
```
Output: `outputs/pdf/Complete_Analysis_YYYYMMDD_HHMMSS.pdf`  
(Executive Summary → TOC → Results Tables → Analysis Plots)

---

## Methods Compared

| Method | Description | Adapts? | Used For |
|--------|-------------|----------|----------|
| Global SARIMA | Baseline: fit entire sample | No | All |
| Rolling SARIMA | Adaptive: last 100 observations | Yes | All |
| GARCH(1,1) | Volatility targeting | Yes | Variance |
| Markov Switching | Hidden regime detection | Yes | Parameter |
| Oracle Dummy | Known break point (benchmark) | Yes | Mean |
| SES / Holt-Winters | Exponential smoothing | Yes | Mean |

**Oracle Dummy** represents the upper bound (perfect break knowledge), while **Global SARIMA** represents the lower bound (complete break ignorance).

---

## Output

All results timestamped and organized in:

- **`outputs/tables/`** — All CSV results (raw metrics from runner: RMSE, MAE, Bias) per method × scenario × innovation type
- **`outputs/tex/`** — LaTeX versions of tables (used for PDF generation)
- **`outputs/figures/`** — Publication-quality plots:
  - Tier 1: Method comparisons (metrics across innovations)
  - Tier 2: DGP visualizations (example time series with breaks)
- **`outputs/pdf/`** — Professional PDF reports:
  - Tables PDF: Latest results from all break types (100+ KB)
  - Combined PDF: Tables + figures in single report (optional)

---

## Contact

| Name | Email |
|------|-------|
| Aadya Khatavkar | s38akhat@uni-bonn.de |
| Mahir Baylarov | s24mbayl@uni-bonn.de |
| Bakhodir Izzatulloev | s36bizza@uni-bonn.de |

Built for the [Research Module](https://vladislav-morozov.github.io/simulations-course/) at University of Bonn