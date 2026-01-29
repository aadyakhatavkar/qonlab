# Architecture Diagram

## Data Flow: Section-Based Organization

```
┌─────────────────────────────────────────────────────────────────┐
│                         DGPS (Data Generation)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│  │   VARIANCE       │  │      MEAN        │  │    PARAMETER     ││
│  │   (Break point)  │  │   (Mean shift)   │  │   (Coeff change) ││
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤│
│  │ sim_*_ar1()      │  │ sim_*_ar1()      │  │ sim_*_ar1()      ││
│  │ estimate_break() │  │ estimate_break() │  │ [point breaks]   ││
│  │ [point breaks]   │  │ [point breaks]   │  │                  ││
│  └──────────────────┘  └──────────────────┘  └──────────────────┘│
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │             RECURRING (Markov-Switching)                      ││
│  │  sim_markov_switching_ar1() [regime-switching breaks]         ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  UTILS: validate_scenarios() [shared configuration]           ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              ↓
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ESTIMATORS (Forecasting)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│  │   VARIANCE       │  │      MEAN        │  │    PARAMETER     ││
│  │  (forecasters)   │  │  (forecasters)   │  │  (forecasters)   ││
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤│
│  │ ARIMA global     │  │ AR1 global       │  │ AR global        ││
│  │ ARIMA rolling    │  │ AR1 rolling      │  │ AR rolling       ││
│  │ GARCH            │  │ Oracle (Tb)      │  │ MS AR            ││
│  │ MS regression    │  │ Estimated (Tb)   │  │                  ││
│  │ Post-break       │  │ MS regression    │  │                  ││
│  │                  │  │                  │  │                  ││
│  │ Metrics:         │  │ Metrics:         │  │                  ││
│  │ • RMSE/MAE/Bias  │  │ • RMSE/MAE/Bias  │  │ • RMSE/MAE/Bias  ││
│  │ • Coverage       │  │                  │  │                  ││
│  │ • Log score      │  │                  │  │                  ││
│  └──────────────────┘  └──────────────────┘  └──────────────────┘│
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYSES (Simulation)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  mc_variance_breaks()        - Multi-scenario Monte Carlo        │
│  mc_variance_breaks_grid()   - Parameter grid search             │
│                                                                   │
│  [Future: mean.py, parameter.py for section-specific MC]         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SCRIPTS (Experiments)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  scripts/runner.py                  [MAIN VARIANCE RUNNER]       │
│                                                                   │
│  scripts/mean_change/                                            │
│    ├── meanchange_singlbreak_scenario.py    [✓ REFACTORED]      │
│    ├── Comparison*                          [FLAGGED]            │
│    └── ...                                                       │
│                                                                   │
│  scripts/parameter_change/                                       │
│    ├── parameter_single_break.py            [✓ REFACTORED]      │
│    └── parameter_recurring_breaks.py        [✓ REFACTORED]      │
│                                                                   │
│  scripts/variance_plot_results.py           [PLOTTING]           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Import Hierarchy

```
MAIN ENTRY POINT
    ↓
main.py
    ├── analyses.simulations (Monte Carlo)
    │   ├── dgps.{variance, mean, parameter, recurring}
    │   ├── estimators.{forecasters, mean, parameter}
    │   └── dgps.utils
    │
    └── scripts.{runner, mean_change/*, parameter_change/*}
        ├── dgps.{variance, mean, parameter, recurring}
        ├── estimators.{forecasters, mean, parameter}
        └── analyses.simulations

CLEAN SEPARATION:
• Core modules (dgps, estimators) are independent
• Analyses layer orchestrates core modules
• Scripts use analyses + core modules for experiments
```

---

## Naming Convention Summary

```
┌────────────┬─────────────┬──────────┬─────────┬──────────┐
│ SECTION    │ DGP         │ Param 1  │ Param 2 │ Param 3  │
├────────────┼─────────────┼──────────┼─────────┼──────────┤
│ VARIANCE   │ sim_var_*   │ Tb       │ sigma1  │ sigma2   │
│ MEAN       │ sim_mean_*  │ Tb       │ mu0     │ mu1      │
│ PARAMETER  │ sim_param_* │ Tb       │ phi1    │ phi2     │
│ RECURRING  │ sim_ms_*    │ p00, p11 │ phi0    │ phi1     │
└────────────┴─────────────┴──────────┴─────────┴──────────┘
```

---

## File Count Summary

### Modules Created:
- ✓ `dgps/variance.py` (5.5 KB) - Variance break DGPs
- ✓ `dgps/mean.py` (0.5 KB) - Mean break DGP
- ✓ `dgps/parameter.py` (0.5 KB) - Parameter break DGP
- ✓ `dgps/recurring.py` (1.1 KB) - Markov-switching DGP
- ✓ `dgps/utils.py` (2.7 KB) - Scenario validation
- ✓ `estimators/mean.py` (2.9 KB) - Mean forecasters
- ✓ `estimators/parameter.py` (1.3 KB) - Parameter forecasters

### Files Refactored:
- ✓ `scripts/mean_change/meanchange_singlbreak_scenario.py` - Extracted & cleaned
- ✓ `scripts/parameter_change/parameter_single_break.py` - Extracted & cleaned
- ✓ `scripts/parameter_change/parameter_recurring_breaks.py` - Updated imports

### Files Updated (Imports):
- ✓ `estimators/__init__.py` - Added mean & parameter exports
- ✓ `estimators/forecasters.py` - Fixed dgps import
- ✓ `dgps/__init__.py` - Reorganized exports
- ✓ `analyses/simulations.py` - Updated to use new modules
- ✓ `scripts/runner.py` - Updated dgps import

### Flagged (In Scripts - Experiment-specific):
- 🚩 `scripts/mean_change/Comparisonmeanchangewitharticlesuggestedandmyowncase.py`
- 🚩 `scripts/mean_change/comparionmultiplebreakandsinglebreak.py`
- 🚩 `scripts/mean_change/meanchange_multiplebreak_scenario.py`

---

## Directory Tree (Final)

```
qonlab/
├── dgps/
│   ├── __init__.py                   ✓
│   ├── variance.py                   ✓ NEW
│   ├── mean.py                       ✓ NEW
│   ├── parameter.py                  ✓ NEW
│   ├── recurring.py                  ✓ NEW
│   ├── utils.py                      ✓ NEW
│   └── static.py                     [deprecated, kept]
│
├── estimators/
│   ├── __init__.py                   ✓ UPDATED
│   ├── forecasters.py                ✓ UPDATED
│   ├── mean.py                       ✓ NEW
│   ├── parameter.py                  ✓ NEW
│   └── ols_like.py
│
├── analyses/
│   ├── __init__.py
│   ├── simulations.py                ✓ UPDATED
│   ├── plots.py
│   └── [variance.py, mean.py, parameter.py optional future]
│
├── scripts/
│   ├── runner.py                     ✓ UPDATED
│   ├── variance_plot_results.py
│   ├── mean_change/
│   │   ├── meanchange_singlbreak_scenario.py       ✓ CLEANED
│   │   ├── Comparisonmeanchangewitharticlesuggestedandmyowncase.py  🚩
│   │   ├── comparionmultiplebreakandsinglebreak.py  🚩
│   │   └── meanchange_multiplebreak_scenario.py     🚩
│   └── parameter_change/
│       ├── parameter_single_break.py               ✓ CLEANED
│       └── parameter_recurring_breaks.py           ✓ UPDATED
│
├── PROJECT_STRUCTURE.md               ✓ NEW
├── MIGRATION_NOTES.md                 ✓ NEW
├── main.py
├── runner.py
├── protocols.py
└── ... [docs, tests, results, etc.]
```

---

## Section Identification

### Which files belong to VARIANCE?
- `dgps/variance.py`
- `estimators/forecasters.py`
- `analyses/simulations.py::mc_variance_breaks()`
- `scripts/runner.py`

### Which files belong to MEAN?
- `dgps/mean.py`
- `estimators/mean.py`
- `scripts/mean_change/meanchange_singlbreak_scenario.py`
- `scripts/mean_change/*comparison*.py` [flagged]

### Which files belong to PARAMETER?
- `dgps/parameter.py`
- `estimators/parameter.py`
- `scripts/parameter_change/parameter_single_break.py`

### Which files cover RECURRING (Markov-Switching)?
- `dgps/recurring.py`
- `estimators/parameter.py::forecast_markov_switching_ar()`
- `scripts/parameter_change/parameter_recurring_breaks.py`

---

**All sections are now clearly isolated and identifiable. ✓**
