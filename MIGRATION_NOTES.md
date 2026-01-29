# Code Migration Summary & Flags

## STATUS: Restructuring Complete

### Sections Created/Moved:

#### 1. **VARIANCE Section** ✓
Location: `dgps/variance.py`, `estimators/forecasters.py` (existing), `analyses/simulations.py`
- ✓ DGP: `simulate_variance_break_ar1()`
- ✓ Estimators: global/rolling ARIMA, GARCH, Markov Switching
- ✓ Metrics: RMSE, MAE, Log Score, Coverage

#### 2. **MEAN Section** ✓
Location: `dgps/mean.py`, `estimators/mean.py`, `analyses/` (via simulations.py)
- ✓ DGP: `simulate_mean_break_ar1()`
- ✓ Estimators: global/rolling AR(1), oracle break dummy, estimated break, Markov Switching
- ✓ Script: `scripts/mean_change/meanchange_singlbreak_scenario.py` (cleaned & refactored)

#### 3. **PARAMETER Section** ✓
Location: `dgps/parameter.py`, `estimators/parameter.py`, `analyses/` (via simulations.py)
- ✓ DGP: `simulate_parameter_break_ar1()`
- ✓ Estimators: global/rolling AR, Markov Switching AR
- ✓ Script: `scripts/parameter_change/parameter_single_break.py` (cleaned & refactored)

---

## FLAGGED: Code Remaining in Scripts (Experiment-Specific)

### 1. **Comparison Scripts - Experiment Logic Only**

#### File: `scripts/mean_change/Comparisonmeanchangewitharticlesuggestedandmyowncase.py`
**FLAG:** Experiment-specific comparison between literature method (Pesaran & Timmermann 2013) and custom method.
- Contains `generate_literature_data()` - Literature baseline replication
- Contains `generate_my_data()` - Custom multi-break scenario
- Contains `evaluate_all()` - Comparison logic with Prophet (external)
**STATUS:** Keep as-is. This is an experimental comparison, not a reusable module.

#### File: `scripts/mean_change/comparionmultiplebreakandsinglebreak.py`
**FLAG:** Comparison between single vs multiple breaks
- Likely contains custom comparison logic and visualizations
**STATUS:** Keep as-is. This is experimental analysis.

#### File: `scripts/mean_change/meanchange_multiplebreak_scenario.py`
**FLAG:** Multiple breaks variant
**STATUS:** Check and apply same extraction pattern if applicable.

#### File: `scripts/mean_change/Meanchange_multiplebreaks_2`
**FLAG:** Appears to be a directory or alternative implementation
**STATUS:** Inspect contents.

#### File: `scripts/parameter_change/parameter_recurring_breaks.py`
**FLAG:** Recurring breaks variant
**STATUS:** Check and apply extraction pattern if reusable DGP or estimators exist.

---

## Next Steps (If Needed):

1. **Inspect remaining scripts** to extract any additional reusable DGPs or estimators
2. **Consolidate multiple-break scenarios** if they contain reusable patterns
3. **Add section-level `__init__.py` files** (optional, for clarity):
   - `analyses/variance.py` - variance-specific Monte Carlo
   - `analyses/mean.py` - mean-specific Monte Carlo
   - `analyses/parameter.py` - parameter-specific Monte Carlo

---

## Module Organization Summary:

```
dgps/
  ├── variance.py      ← DGP for variance breaks
  ├── mean.py          ← DGP for mean breaks
  ├── parameter.py     ← DGP for parameter breaks
  ├── utils.py         ← Scenario validation
  └── __init__.py      ← Updated exports

estimators/
  ├── forecasters.py   ← Variance estimators (existing, updated imports)
  ├── mean.py          ← Mean break estimators (NEW)
  ├── parameter.py     ← Parameter break estimators (NEW)
  └── __init__.py      ← Updated exports

analyses/
  ├── simulations.py   ← Monte Carlo simulations (updated imports)
  ├── plots.py         ← Plotting utilities
  └── (Optional: variance.py, mean.py, parameter.py for section-specific MC)

scripts/
  ├── runner.py        ← Main variance experiment runner
  ├── variance_plot_results.py
  ├── mean_change/     ← Mean break experiments
  │   ├── meanchange_singlbreak_scenario.py (cleaned ✓)
  │   ├── Comparisonmeanchangewitharticlesuggestedandmyowncase.py (flagged)
  │   └── ...
  └── parameter_change/ ← Parameter break experiments
      ├── parameter_single_break.py (cleaned ✓)
      └── parameter_recurring_breaks.py (flagged)
```

---

## Naming Consistency:

- ✓ All variance functions use `variance_break`, `Tb` (break point), `sigma1/sigma2`
- ✓ All mean functions use `mean_break`, `Tb`, `mu0/mu1`
- ✓ All parameter functions use `parameter_break`, `Tb`, `phi1/phi2`
- ✓ Cleared old `static.py` reference, now organized by section
