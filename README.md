# Forecasting Under Structural Breaks: A Monte Carlo Study

[![CI](https://github.com/aadyakhatavkar/qonlab/actions/workflows/ci.yml/badge.svg)](https://github.com/aadyakhatavkar/qonlab/actions)

**Research Module in Econometrics and Statistics**  
*Fundamentals of Monte Carlo Simulations in Data Science*  
University of Bonn | Winter Semester 2025/26

---

## 📚 Abstract

This project investigates the forecasting performance of various time series methods in the presence of structural breaks. Using Monte Carlo simulations, we compare global AR(1), rolling-window AR(1), oracle break-dummy specifications, estimated break detection, and Markov switching models. Results indicate that oracle break-dummy models achieve the best performance, while rolling-window estimation offers practical improvements when break dates are unknown.

---

## 📁 Project Structure

```
qonlab/
├── docs/
│   ├── paper/              # LaTeX paper/thesis
│   │   ├── main.tex        # Main document
│   │   ├── bibliography.bib
│   │   └── Makefile
│   └── research_proposal.md
├── dgps/                   # Data-generating processes
│   └── static.py           # Break simulations (mean, variance, parameter)
├── estimators/             # Forecasting methods
│   └── forecasters.py      # ARIMA, GARCH, post-break estimators
├── analyses/               # Monte Carlo engines
│   ├── simulations.py      # MC experiments
│   └── plots.py            # Visualization
├── experiments/            # Experiment configurations
│   └── mean_break_config.json
├── scripts/                # Task-specific runners
├── tests/                  # Test suite
├── scenarios/              # Scenario definitions
└── main.py                 # CLI entrypoint
```

---

## 🚀 Quick Start

### Using Pixi (Recommended)

```bash
# Install Pixi: https://prefix.dev
pixi install
pixi run python main.py mc --quick
```

### Using pip

```bash
pip install -r requirements.txt
python main.py mc --quick
```

---

## 🔬 Running Experiments

### Monte Carlo Simulations

```bash
# Quick test (10 replications, small sample)
python main.py mc --quick

# Full simulation (200 replications)
python main.py mc --n-sim 200 --T 400 --horizon 20

# Grid search for optimal window (Pesaran 2013)
python main.py mc --grid --n-sim 100
```

### Compiling the Paper

```bash
cd docs/paper
make          # Compile PDF
make clean    # Remove auxiliary files
```

---

## 📊 Methods Compared

| Method | Description | Break Knowledge |
|--------|-------------|-----------------|
| Global AR(1) | Full-sample estimation | None |
| Rolling AR(1) | Window-based adaptive | None |
| Break Dummy (Oracle) | Indicator variables | Known |
| Estimated Break | Grid search detection | Estimated |
| Markov Switching | Regime-switching model | Inferred |

---

## 📈 Evaluation Metrics

### Point Forecasts
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Bias**

### Uncertainty Quantification
- **Coverage 80%** and **Coverage 95%**
- **Log-score** (proper scoring rule)

---

## 📖 References

- Pesaran, M. H. (2013). "The Role of Structural Breaks in Forecasting," *Handbook of Economic Forecasting*
- Box, G. E. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Francq, C. & Zakoïan, J. M. (2019). *GARCH Models*

---

## 🎓 Course Information

This project is part of the [Research Module in Econometrics and Statistics](https://vladislav-morozov.github.io/simulations-course/) at the University of Bonn.

**Course Focus:**
- Design, implementation, and interpretation of Monte Carlo simulations
- Evaluating confidence intervals and hypothesis tests
- Assessing predictive algorithms

---

## 📝 License

Academic use. See [University of Bonn policies](https://www.uni-bonn.de).

---

**Author:** Aadya Khatavkar  
**Contact:** s6aakhat@uni-bonn.de