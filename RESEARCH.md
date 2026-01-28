# Research Documentation

## Project: Forecasting Under Structural Breaks

**Research Module:** Fundamentals of Monte Carlo Simulations in Data Science  
**Institution:** University of Bonn  
**Semester:** Winter 2025/26  
**Course Website:** [vladislav-morozov.github.io/simulations-course](https://vladislav-morozov.github.io/simulations-course/)

---

## 📄 Documentation Overview

| Document | Location | Description |
|----------|----------|-------------|
| **Research Proposal** | `docs/research_proposal.md` | Formal research plan and timeline |
| **LaTeX Paper** | `docs/paper/main.tex` | Academic paper (compile with `make`) |
| **Methods PDF** | `RM_methods_explanation.pdf` | Original methods documentation |
| **Changelog** | `CHANGES.md` | Technical implementation log |

---

## 🎯 Research Questions

1. **Primary:** How do different forecasting methods perform under structural breaks?
2. **Secondary:** What is the cost of not knowing break dates in terms of forecast accuracy?
3. **Practical:** Can adaptive methods provide alternatives to oracle specifications?

---

## 🔬 Methodology Summary

### Data-Generating Processes

| Break Type | Location | Implementation |
|------------|----------|----------------|
| Mean break | `dgps/static.py` | `simulate_mean_break()` |
| Variance break | `dgps/static.py` | `simulate_variance_break()` |
| Parameter break | `dgps/static.py` | `simulate_parameter_break()` |

### Forecasting Methods

| Method | Implementation | Break Knowledge |
|--------|----------------|-----------------|
| Global AR(1) | `estimators/forecasters.py` | None |
| Rolling AR(1) | `estimators/forecasters.py` | None |
| GARCH | `estimators/forecasters.py` | None |
| Post-break ARIMA | `estimators/forecasters.py` | Estimated |

### Monte Carlo Engine

| Component | Location | Purpose |
|-----------|----------|---------|
| Main MC runner | `analyses/simulations.py` | Run experiments |
| Grid search | `analyses/simulations.py` | Optimal window selection |
| Visualization | `analyses/plots.py` | Generate figures |

---

## 📊 Evaluation Metrics

### Point Forecast Accuracy
- **RMSE** (Root Mean Squared Error) — penalizes large errors
- **MAE** (Mean Absolute Error) — typical error magnitude
- **Bias** — systematic over/under-forecasting

### Uncertainty Quantification
- **Coverage 80%** — nominal interval accuracy
- **Coverage 95%** — nominal interval accuracy
- **Log-score** — proper scoring rule for probabilistic forecasts

---

## 🚀 Running Experiments

```bash
# Quick test (CI/development)
python main.py mc --quick

# Standard run (200 replications)
python main.py mc --n-sim 200

# Grid search for optimal window
python main.py mc --grid

# Full production run
python main.py mc --n-sim 500 --T 400 --horizon 20
```

---

## 📝 Compiling the Paper

```bash
cd docs/paper
make          # Compile main.pdf
make clean    # Remove auxiliary files
make view     # Open PDF (Linux/Mac)
```

---

## 📚 Key References

1. **Pesaran (2013)** — Structural breaks in forecasting
2. **Box & Jenkins (1970)** — ARIMA methodology
3. **Hamilton (1989)** — Markov switching models
4. **Bai & Perron (1998)** — Multiple structural breaks

---

## 📁 Project Structure

```
qonlab/
├── docs/
│   ├── paper/              # LaTeX thesis/paper
│   │   ├── main.tex
│   │   ├── main.pdf        # Compiled output
│   │   ├── bibliography.bib
│   │   └── Makefile
│   └── research_proposal.md
├── experiments/            # Experiment configurations
│   ├── README.md
│   └── mean_break_config.json
├── dgps/                   # Data-generating processes
├── estimators/             # Forecasting methods
├── analyses/               # Monte Carlo simulations
├── scripts/                # Task runners
├── tests/                  # Test suite
├── CHANGES.md              # Technical changelog
├── RESEARCH.md             # This file
└── README.md               # Quick start guide
```

---

## ✅ Course Deliverables Checklist

- [x] Simulation design and implementation
- [x] Replicable Python code
- [x] Reproducible experiment configurations
- [x] LaTeX paper/thesis
- [ ] Results tables and figures
- [ ] Public presentation
- [ ] Final submission

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-28 | Initial academic structure, LaTeX paper, experiment configs |

---

**Contact:** Aadya Khatavkar — s6aakhat@uni-bonn.de
