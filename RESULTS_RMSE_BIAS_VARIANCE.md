# COMPREHENSIVE RESULTS: RMSE, BIAS, VARIANCE
## Single vs Recurring Breaks

**Date:** February 12, 2026  
**Status:** Complete for Single & Parameter Recurring; Variance & Mean Recurring in progress

---

## 📊 [1] VARIANCE BREAKS

### SINGLE VARIANCE BREAK

| Method | RMSE | Bias | Variance |
|--------|------|------|----------|
| GARCH | 1.7982 | 0.1178 | 3.2196 |
| SARIMA Global | 1.8893 | 0.2024 | 3.5284 |
| SARIMA PostBreak | 2.8583 | -0.6308 | 7.7717 |
| SARIMA Rolling | 10.9218 | 3.9988 | 103.2952 |

**Best:** GARCH (RMSE=1.798, Var=3.220)  
**Worst:** SARIMA Rolling (RMSE=10.922, Var=103.295)

### RECURRING VARIANCE BREAK

(Simulation in progress - results will be added upon completion)

---

## 📊 [2] PARAMETER BREAKS

### SINGLE PARAMETER BREAK
*Aggregated across Gaussian + Student-t (df=3,5,10) innovations*

| Model | RMSE | Bias | Variance |
|-------|------|------|----------|
| Rolling AR | 1.0093 | -0.0025 | 1.0188 |
| Global AR | 1.0659 | 0.0056 | 1.1362 |
| Markov-Switching AR | 1.1988 | 0.0098 | 1.4370 |

**Best:** Rolling AR (RMSE=1.009, Var=1.019)  
**Worst:** MS AR (RMSE=1.199, Var=1.437)

### RECURRING PARAMETER BREAK

| Model | RMSE | Bias | Variance |
|-------|------|------|----------|
| AR Global | 0.6755 | 0.1877 | 0.4211 |
| AR Rolling | 0.6792 | 0.1261 | 0.4454 |
| Markov-Switching AR | 0.7321 | 0.2084 | 0.4925 |

**Best:** AR Global (RMSE=0.676, Var=0.421)  
**Improvement:** Recurring breaks are ~35% easier than single breaks

---

## 📊 [3] MEAN BREAKS

### SINGLE MEAN BREAK

| Method | RMSE | Bias | Variance |
|--------|------|------|----------|
| ARMA + Estimated Break (grid) | 1.0307 | 0.0184 | 1.0620 |
| ARMA + Break Dummy (oracle) | 1.0497 | -0.1710 | 1.0726 |
| ARMA Rolling (auto) | 1.0725 | 0.0628 | 1.1464 |
| ARMA Global (auto) | 1.1167 | 0.2827 | 1.1670 |

**Best:** Estimated Break w/ Grid (RMSE=1.031, Var=1.062)  
**Oracle Gap:** Only 1.8% higher RMSE than oracle

### RECURRING MEAN BREAK

(Simulation in progress - results will be added upon completion)

---

## 📈 MASTER SUMMARY TABLE

```
Break_Type  Scenario              Method                    RMSE    Bias  Variance
─────────────────────────────────────────────────────────────────────────────────
  VARIANCE    Single        SARIMA Global               1.8893  0.2024    3.5284
  VARIANCE    Single               GARCH               1.7982  0.1178    3.2196 ✓
  VARIANCE    Single     SARIMA PostBreak               2.8583 -0.6308    7.7717
  VARIANCE    Single       SARIMA Rolling              10.9218  3.9988  103.2952

 PARAMETER    Single          Rolling AR               1.0093 -0.0025    1.0188 ✓
 PARAMETER    Single           Global AR               1.0659  0.0056    1.1362
 PARAMETER    Single  Markov-Switching AR              1.1988  0.0098    1.4370

 PARAMETER  Recurring          AR Global               0.6755  0.1877    0.4211 ✓
 PARAMETER  Recurring         AR Rolling               0.6792  0.1261    0.4454
 PARAMETER  Recurring Markov-Switching AR              0.7321  0.2084    0.4925

      MEAN    Single    ARMA + Estimated Break        1.0307  0.0184    1.0620 ✓
      MEAN    Single      ARMA + Oracle Break         1.0497 -0.1710    1.0726
      MEAN    Single           ARMA Rolling           1.0725  0.0628    1.1464
      MEAN    Single            ARMA Global           1.1167  0.2827    1.1670
```

---

## 🎯 KEY INSIGHTS

### Difficulty Ranking (by RMSE)
1. **EASIEST:** Parameter Recurring (RMSE=0.676-0.679)
2. Mean Single (RMSE=1.031-1.117)
3. Variance Single (RMSE=1.798-2.858)
4. **HARDEST:** Parameter Single (RMSE=1.009-1.199)

### Method Performance

**VARIANCE BREAKS:**
- GARCH and SARIMA Global are competitive (~1.79-1.89)
- SARIMA Rolling has catastrophic failure (RMSE=10.92)
- PostBreak method moderate (~2.86)

**PARAMETER BREAKS:**
- Rolling AR superior in single breaks (RMSE=1.009)
- AR Global better for recurring breaks (RMSE=0.676)
- Recurring breaks ~35% easier than single
- MS AR consistently underperforms

**MEAN BREAKS:**
- Estimated break detection effective (gap to oracle: 1.8%)
- Auto methods (rolling/global) degrade ~3-8%
- More variance in this category

---

## 📌 Variance Decomposition

**VARIANCE = RMSE² - Bias²**

**Highest Variance:** SARIMA Rolling on single variance breaks (103.30)
**Lowest Variance:** AR Global on recurring parameter breaks (0.42)

This indicates that parameter recurring breaks have both low error AND low variance - most predictable scenario.

---

*Results include 300 Monte Carlo replications per scenario*  
*Variance calculated as σ² = E[X²] - E[X]²*
