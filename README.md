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

**Authors:** Aadya Khatavkar, Mahir Baylarov, Bakhodir Izzatulloev

---

## � Quick Start

### Setup
```bash
# Install pixi (if needed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

### Run All Experiments (Standardized: T=400, Tb=200, n_sim=300)
```bash
pixi run python runner.py              # All 3 break types with variants
pixi run python runner.py --quick      # Quick test (T=150, n_sim=10)
pixi run python runner.py --variance   # Variance breaks only
pixi run python runner.py --mean       # Mean breaks only
pixi run python runner.py --parameter  # Parameter breaks only
```

### Example: Variance Break Analysis
```python
from analyses import mc_variance_single_break

results = mc_variance_single_break(n_sim=300, T=400, Tb=200, seed=42)
print(results[['Scenario', 'Method', 'RMSE', 'MAE']])
```

## 🎯 Overview

**Why this matters:** Structural breaks are common in real-world time series (stock prices, inflation, GDP). When breaks occur, standard forecasting methods can fail dramatically. This study systematically evaluates which methods handle different break types best.

**Research Questions**

1. How do forecasting methods perform under different break types (variance, mean, parameter)?
2. Do fixed windows outperform adaptive methods after a break?
3. Can simple methods compete with complex ones under model instability?
4. How do heavy-tailed distributions affect forecast robustness?

**Contribution:** Unlike most break-detection literature (which assumes you know a break happened), this study evaluates real-world practitioners' approach: **use fixed windows + model selection**, without prior knowledge of breaks.

---

## 📊 Break Types

| Break Type | Real-World Example | Our Simulation |
|------------|-------------------|----------------|
| **Variance** | Market volatility spikes (2008 crisis) | Sudden jump in $\sigma_t$ at time $T_b$ |
| **Mean** | Inflation regime shift | Sudden jump in $\mu_t$ at time $T_b$ |
| **Parameter** | Changing consumer behavior | Sudden jump in AR coefficient $\phi_t$ at time $T_b$ |

**Study Design:**
- Single break at midpoint ($T_b = T/2$)
- Recurring breaks via Markov-switching regime
- 3 innovation types: Gaussian, Student-t(df=3), Student-t(df=5)
- Persistence levels for parameter breaks: 0.90, 0.95, 0.99

---

## 🔬 Methods Compared

| Method | Logic | Adapts? |
|--------|-------|---------|
| **Global ARIMA** | Fit entire sample | ❌ No |
| **Rolling ARIMA** | Fit last 50 observations | ✅ Adaptive |
| **GARCH(1,1)** | Volatility targeting | ✅ Adaptive |
| **Post-Break ARIMA** | Fit only after estimated break | 🔄 Partial |
| **Markov Switching** | Hidden regime model | ✅ Adaptive |
| **Oracle Break Dummy** | Known break point | ✅ Perfect |

---

## 📈 Output & Results

Each experiment generates a CSV with:
- **Method** — Forecasting method used
- **RMSE, MAE, Bias, Variance** — Error metrics
- **Innovation** — Type of shocks (Gaussian/Student-t)
- **Persistence** — For recurring breaks only

**Output Location:** `results/` folder organized by break type

**Example Output:**
```
Method,RMSE,MAE,Bias,Variance,Innovation
Global ARIMA,2.34,1.89,0.12,5.67,gaussian
Rolling ARIMA,1.45,1.12,-0.05,3.21,gaussian
GARCH(1.1),1.52,1.18,0.08,3.89,gaussian
```

---

## 🚀 Quick Start

### Setup
```bash
# Install pixi (if needed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

### Run Experiments

```bash
pixi run python runner.py              # All 3 break types (25-30 min)
pixi run python runner.py --quick      # Fast test run (2-3 min)
pixi run python runner.py --variance   # Variance only
pixi run python runner.py --mean       # Mean only
pixi run python runner.py --parameter  # Parameter only
```

### View Results
```bash
ls results/                             # Check output files
cat results/variance_20260212_*.csv    # View variance results
```

---

## 🎛️ Reproducibility

All experiments use **seed=42** for reproducibility:
- **T=400** (time series length) or **150** (quick test)
- **Tb=200** (break point) or **75** (quick test)
- **n_sim=300** (replications) or **10** (quick test)

To modify, edit `runner.py` lines 80-85 or pass `--quick` flag for built-in presets.

---

## 📚 Understanding the Mathematics

**Variance Break DGP:**
$$y_t = \phi y_{t-1} + \sigma_t \varepsilon_t, \quad \sigma_t = \begin{cases} \sigma_1 & t < T_b \\ \sigma_2 & t \geq T_b \end{cases}$$

**Mean Break DGP:**
$$y_t = \mu_t + \phi y_{t-1} + \varepsilon_t, \quad \mu_t = \begin{cases} \mu_1 & t < T_b \\ \mu_2 & t \geq T_b \end{cases}$$

**Parameter Break DGP:**
$$y_t = \phi_t y_{t-1} + \varepsilon_t, \quad \phi_t = \begin{cases} \phi_1 & t < T_b \\ \phi_2 & t \geq T_b \end{cases}$$

---

## 🤝 Contact

| Name | eCampus | Email |
|------|--------|-------|
| **Aadya Khatavkar** | s38akhat | s38akhat@uni-bonn.de |
| **Mahir Baylarov** | s24mbayl | s24mbayl@uni-bonn.de |
| **Bakhodir Izzatulloev** | s36bizza | s36bizza@uni-bonn.de |

---

<p align="center">
  <sub>Built for the <a href="https://vladislav-morozov.github.io/simulations-course/">Research Module</a> at University of Bonn</sub>
</p>


