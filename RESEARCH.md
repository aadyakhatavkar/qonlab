# Research Documentation

## Project: Structural Break Forecasting — A Monte Carlo Study

**Course:** Fundamentals of Monte Carlo Simulations in Data Science  
**Institution:** University of Bonn  
**Author:** Aadya Khatavkar (s38akhat@uni-bonn.de)  
**Semester:** Winter 2025/26

---

## 🎯 Research Questions

1. How do forecasting methods perform under variance, mean, and parameter breaks?
2. What is the optimal rolling window size for different break magnitudes?
3. How do heavy-tailed (Student-$t$) distributions affect results?
4. Can adaptive methods match oracle specifications?

---

## 📊 Work Completed

### Data-Generating Processes (`dgps/static.py`)

| Function | Break Type | Key Parameters |
|----------|------------|----------------|
| `simulate_variance_break()` | Variance | $\sigma_1, \sigma_2, T_b$, distribution |
| `simulate_mean_break()` | Mean | $\mu_0, \mu_1, T_b$ |
| `simulate_parameter_break()` | Parameter | $\phi_1, \phi_2, T_b$ |
| `simulate_realized_volatility()` | RV | intervals_per_day |
| `_generate_t_innovations()` | — | $\nu$ (degrees of freedom) |
| `estimate_variance_break_point()` | Detection | trim parameter |

### Forecasting Methods (`estimators/forecasters.py`)

| Function | Description |
|----------|-------------|
| `forecast_variance_dist_arima_global()` | Full-sample ARIMA with auto-order |
| `forecast_variance_dist_arima_rolling()` | Rolling-window ARIMA |
| `forecast_garch_variance()` | GARCH(1,1) |
| `forecast_variance_arima_post_break()` | Post-break ARIMA |
| `forecast_variance_averaged_window()` | Ensemble over windows |
| `forecast_markov_switching()` | Markov regime-switching |
| `_auto_select_arima_order()` | AIC/BIC order selection |

### Evaluation Metrics

| Metric | Type | Function |
|--------|------|----------|
| RMSE, MAE, Bias | Point | `variance_rmse_mae_bias()` |
| Coverage 80%, 95% | Uncertainty | `variance_interval_coverage()` |
| Log-score | Uncertainty | `variance_log_score_normal()` |

### Monte Carlo Engine (`analyses/simulations.py`)

| Function | Purpose |
|----------|---------|
| `mc_variance_breaks()` | Main MC simulation |
| `mc_variance_breaks_grid()` | Grid search for optimal window |

### Visualization (`analyses/plots.py`)

- `plot_loss_surfaces()` — RMSE heatmaps
- `plot_logscore_comparison()` — Method × window comparison
- `plot_time_series_example()` — Forecast visualization

---

## 🔧 Key Technical Features

1. **Heavy-tailed distributions**: Student-$t$ with standardization
2. **Automatic ARIMA order selection**: AIC/BIC grid search
3. **Unified simulation engine**: Handles all break types
4. **Realized volatility**: High-frequency data support
5. **Scenario-based configuration**: JSON files

---

## 📓 Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `Variance_Change_Documentation.ipynb` | Full documentation | ✅ Runnable |
| `variance_workflow.ipynb` | Quick demo | ✅ Runnable |

---

## 📄 Paper Structure (`docs/paper/main.tex`)

1. **Introduction** — Research questions, motivation
2. **Data-Generating Processes** — Variance, mean, parameter breaks
3. **Forecasting Methods** — ARIMA, GARCH, Markov, etc.
4. **Monte Carlo Design** — Simulation procedure, grid search
5. **Evaluation Metrics** — RMSE, Coverage, Log-score
6. **Implementation Summary** — Code organization
7. **Results** — Tables (placeholder for simulation output)
8. **Conclusion** — Summary and future work

---

## 🚀 Running Experiments

```bash
# Quick test
python main.py mc --quick

# Full simulation
python main.py mc --n-sim 200 --T 400 --horizon 20

# Grid search
python main.py mc --grid

# Custom scenarios
python main.py mc --scenarios scenarios/example_scenarios.json

# Generate plots
python -m analyses.plots
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📚 References

- Pesaran (2013) — Structural breaks in forecasting
- Francq & Zakoïan (2019) — GARCH models
- Bollerslev (1986) — GARCH
- Box & Jenkins (1970) — ARIMA
- Hamilton (1989) — Markov switching

---

## 🔜 Future Work

1. S&P 500 realized volatility (Thomson Reuters Eikon)
2. Multi-step ahead forecasting
3. ARIMA + GARCH ensembles
4. Online break detection

---

## 📝 Presentation Outline

For the course presentation:

1. **Motivation** (2 min) — Why structural breaks matter
2. **DGPs** (3 min) — Three break types
3. **Methods** (5 min) — ARIMA, GARCH, rolling windows
4. **Monte Carlo Design** (3 min) — Simulation setup
5. **Results** (5 min) — Tables, loss surfaces
6. **Conclusion** (2 min) — Practical implications

---

## ✅ Deliverables Checklist

- [x] DGP implementations
- [x] Forecasting methods
- [x] MC simulation engine
- [x] Evaluation metrics
- [x] Heavy-tailed extensions
- [x] Auto ARIMA selection
- [x] Grid search (Pesaran 2013)
- [x] Visualization utilities
- [x] LaTeX paper
- [x] Documentation notebooks
- [ ] Final simulation results
- [ ] Presentation slides
