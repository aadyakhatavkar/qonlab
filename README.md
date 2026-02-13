# Structural Break Forecasting

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

### Run Experiments
```bash
pixi run python runner.py              # All 3 break types (T=400, Tb=200, n_sim=300)
pixi run python runner.py --quick      # Quick test (T=400, Tb=200, n_sim=30)
pixi run python runner.py --variance   # Variance breaks only
pixi run python runner.py --mean       # Mean breaks only
pixi run python runner.py --parameter  # Parameter breaks only
```

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

**Key Research Question**: Do simple adaptive methods (rolling windows) perform as well as complex ones (Markov switching) when practitioners don't know a break occurred?

---

## Workflow

**4-step process** from experiments to professional PDF report:

### Step 1: Run Simulations
```bash
pixi run python runner.py
# Output: outputs/csv/*.csv and outputs/tex/*.tex
```

### Step 2: Generate Plots
```bash
python scripts/generate_plots.py --all
# Output: figures/{variance,mean,parameter}/*.png
# See PLOTTING_QUICK_REFERENCE.txt for plot options
```

### Step 3: Compile Tables to PDF
```bash
python scripts/build_pdfs.py --tables
# Output: outputs/pdf/Tables_Results_YYYYMMDD_HHMMSS.pdf
```

### Step 4: Create Combined Report (Optional)
```bash
python scripts/build_pdfs.py --combined
# Output: outputs/pdf/Complete_Analysis_YYYYMMDD_HHMMSS.pdf
# Contains: Executive Summary → TOC → Results Tables → Analysis Plots
```

---

## Methods Compared

| Method | Description | Adapts? |
|--------|-------------|----------|
| Global ARIMA | Baseline: fit entire sample | No |
| Rolling ARIMA | Adaptive: last 50 observations | Yes |
| GARCH(1,1) | Volatility targeting | Yes |
| Markov Switching | Hidden regime detection | Yes |
| Oracle Dummy | Known break point (benchmark) | Yes |

**Oracle Dummy** represents the upper bound (perfect break knowledge), while **Global ARIMA** represents the lower bound (complete break ignorance).

---

## Output

All results timestamped and organized in:

- **`outputs/csv/`** — Raw metrics (RMSE, MAE, Bias, Variance) per method × scenario
- **`outputs/tex/`** — LaTeX tables organized by break type and innovation type
- **`figures/`** — 19 publication-quality plots:
  - Tier 1: Method comparisons (metrics across innovations)
  - Tier 2: DGP visualizations (example time series with breaks)
- **`outputs/pdf/`** — Professional PDF reports:
  - Tables PDF: Results organized by break type (50 KB)
  - Combined PDF: Tables + figures in single report (3.6 MB)

---

## Contact

| Name | Email |
|------|-------|
| Aadya Khatavkar | s38akhat@uni-bonn.de |
| Mahir Baylarov | s24mbayl@uni-bonn.de |
| Bakhodir Izzatulloev | s36bizza@uni-bonn.de |

Built for the [Research Module](https://vladislav-morozov.github.io/simulations-course/) at University of Bonn
