# Scenario Configuration Analysis
**Date:** February 12, 2026  
**Scope:** Complete check of all simulation scenarios and their configurations

---

## 1. CURRENT CONFIGURATION BY SCENARIO

### 1.1 VARIANCE BREAKS

#### Variance Single Break (`analyses/simu_variance_single.py`)
**Function:** `mc_variance_single_break()`

| Parameter | Current | Required |
|-----------|---------|----------|
| `n_sim` | 100 | ✓ Should match across all (recommend 300) |
| `T` | 400 | ✓ MATCHES runner.py (400) |
| `Tb` | 200 | ✓ MATCHES runner.py (200) |
| `window` | 100 | ⚠️ Different from mean (60) and param (80/60) |
| `horizon` | 1 | ✓ One-step ahead |
| `innovation_type` | 'gaussian' | ✓ Supports gaussian + student-t |
| `dof` | None | ✓ Supports dof=3, 5 |
| **Forecast Methods** | 5 methods | **See issue #3 below** |

**Forecast Methods:**
1. SARIMA Global
2. SARIMA Rolling
3. GARCH
4. SARIMA Post-Break
5. SARIMA Avg-Window (windows: 50, 100, 200)

**Issue #1:** Window sizes in Avg-Window [50, 100, 200] are hardcoded, not parameterized ❌

#### Variance Recurring (`analyses/simu_variance_recurring.py`)
**Function:** `mc_variance_recurring()`

| Parameter | Current | Required |
|-----------|---------|----------|
| `n_sim` | 100 | ✗ MISMATCH: Should be 300 |
| `T` | 400 | ✓ MATCHES |
| `p` | 0.95 | ⚠️ Only one persistence level |
| `window` | 100 | ⚠️ DIFFERENT from single break (window=100) |
| `horizon` | 1 | ✓ One-step ahead |

**Forecast Methods:**
1. SARIMA Global
2. SARIMA Rolling
3. SARIMA Avg-Window (hardcoded: 50, 100, 200) ❌
4. MS AR(1)

**Issue #2:** No innovation type support (Gaussian only) ⚠️  
**Issue #3:** Runner.py only calls with default p=0.95, no separate tables ❌

---

### 1.2 MEAN BREAKS

#### Mean Single Break (`analyses/simu_meansingle.py`)
**Function:** `run_mc_single_break_sarima()`

| Parameter | Current | Required |
|-----------|---------|----------|
| `n_sim` | 200 | ✗ MISMATCH: Should be 300 |
| `T` | 300 | ✗ MISMATCH: Should be 400 |
| `Tb` | 150 | ✗ MISMATCH: Should be 200 |
| `window` | 60 | ✗ INCONSISTENT: variance=100, param=80 |
| `horizon` | 1 | ✓ One-step ahead |
| `innovation_type` | 'gaussian' | ✓ Supports gaussian + student-t |
| `dof` | None | ✓ Supports dof=3, 5 |
| `gap_after_break` | 20 | ⚠️ Constraint: t0 = Tb + gap must be < T |

**Forecast Methods:**
1. SARIMA Global
2. SARIMA Rolling (window=60)
3. SARIMA + Break Dummy (oracle Tb)
4. SARIMA + Estimated Break (grid)
5. Simple Exp. Smoothing (SES)

**Issue #4:** n_sim=200, T=300, Tb=150 don't match runner.py expectations (300, 400, 200) ❌

#### Mean Recurring (`analyses/simu_mean_recurring.py`)
**Function:** `mc_mean_recurring()`

| Parameter | Current | Required |
|-----------|---------|----------|
| `n_sim` | 100 | ✗ MISMATCH: Should be 300 |
| `T` | 400 | ✓ MATCHES |
| `p` | 0.95 | ⚠️ Fixed value, no persistence variants |
| `window` | 100 | ⚠️ DIFFERENT from single (60) |
| `horizon` | 1 | ✓ One-step ahead |

**Forecast Methods:**
1. ARIMA Global
2. MS AR(1) (Markov-Switching)

**Issue #5:** Only 2 methods vs 5 in single breaks ⚠️  
**Issue #6:** No innovation type support ⚠️  
**Issue #7:** Runner calls with fixed p=0.95, no separate tables for persistence levels ❌

---

### 1.3 PARAMETER BREAKS

#### Parameter Single Break (`analyses/simu_paramsingle.py`)
**Function:** `monte_carlo_single_break_post()`

| Parameter | Current | Required |
|-----------|---------|----------|
| `n_sim` | 300 | ✓ MATCHES |
| `T` | 400 | ✓ MATCHES runner.py |
| `Tb` | 200 | ✓ MATCHES runner.py |
| `t_post` | 250 | ⚠️ Forecast origin (runner calculates as Tb+30) |
| `window` | 80 | ✗ INCONSISTENT: variance=100, mean=60 |
| `innovation_type` | 'gaussian' | ✓ Supports gaussian + student-t |
| `dof` | None | ✓ Supports dof=3, 5 |

**Forecast Methods:**
1. Global SARIMA
2. Rolling SARIMA
3. MS AR

**Issue #8:** Only 3 methods vs 5 in variance/mean ⚠️

#### Parameter Recurring (`analyses/simu_paramrecurring.py`)
**Function:** `monte_carlo_recurring(p, ...)`

| Parameter | Current | Required |
|-----------|---------|----------|
| `n_sim` | 300 | ✓ MATCHES |
| `T` | 400 | ✓ MATCHES |
| `t0` | 300 | ⚠️ Forecast origin (runner calculates it) |
| `window` | 60 | ✗ INCONSISTENT: variance=100, mean=100 |
| **Persistence levels** | Manual loop in runner | ✓ Correct: 0.90, 0.95, 0.99 |

**Forecast Methods:**
1. Global SARIMA
2. Rolling SARIMA
3. MS AR

**Issue #9:** Window size = 60 is DIFFERENT from recurring variance/mean (100) ❌

---

## 2. KEY FINDINGS & ISSUES

### ❌ CRITICAL ISSUES (Must Fix)

1. **Inconsistent Sample Sizes (n_sim, T, Tb):**
   - Variance Single: n_sim=100, T=400, Tb=200
   - Mean Single: n_sim=200, T=300, Tb=150 ❌ **MISMATCH**
   - Variance Recurring: n_sim=100 ❌ **Should be 300**
   - Mean Recurring: n_sim=100 ❌ **Should be 300**
   - Parameter Single: n_sim=300, T=400, Tb=200 ✓
   - Parameter Recurring: n_sim=300, T=400 ✓
   
   **Required Standard:** n_sim=300, T=400, Tb=200 (or 201 when needed for break point offset)

2. **Inconsistent Window Sizes:**
   - Variance: window=100
   - Mean Single: window=60 ❌
   - Mean Recurring: window=100
   - Parameter Single: window=80 ❌
   - Parameter Recurring: window=60 ❌
   
   **Issue:** Different rolling windows make methods non-comparable!

3. **Missing Innovation/DOF Support:**
   - Variance Recurring: No innovation_type parameter ❌
   - Mean Recurring: No innovation_type parameter ❌
   
   **Result:** Cannot generate separate tables for Gaussian/Student-t(3)/Student-t(5)

4. **Missing Persistence Level Tables:**
   - Runner.py calls `mc_variance_recurring()` only once (fixed p=0.95) ❌
   - Runner.py calls `mc_mean_recurring()` only once (fixed p=0.95) ❌
   - Parameter Recurring correctly has 3 separate calls for p=0.90, 0.95, 0.99 ✓
   
   **Required:** Separate results tables for recurring breaks across persistence levels

5. **Hardcoded Window Sizes:**
   - Variance Avg-Window: `window_sizes=[50, 100, 200]` hardcoded in method
   - Not parameterized, cannot be changed per scenario
   - Inconsistent with other methods

6. **Method Count Inconsistency:**
   - Variance Single: 5 methods
   - Mean Single: 5 methods
   - Parameter Single: 3 methods ❌
   - All recurring: 3-4 methods, but non-comparable across break types

### ⚠️ STRUCTURAL ISSUES

7. **Forecast Origin Consistency:**
   - Mean Single: Uses `gap_after_break=20` (t0 = Tb + 20)
   - Parameter Single: Runner calculates `t_post = Tb + 30`
   - Parameter Recurring: Runner calculates `t0 = min(T-50, ...)`
   - **Issue:** No consistent rule for forecast origin across scenarios

8. **Missing Multiple Break Scenarios:**
   - No `simu_meanmultiple.py` or equivalent for other break types
   - No multiple break experiments at all!
   - Should have: single + multiple + recurring for ALL break types

9. **Metrics Coverage:**
   - ✓ All calculate: RMSE, MAE, Bias, Variance (Good!)
   - ✓ All use `calculate_metrics()` from protocols.py

---

## 3. IMPLEMENTATION GAPS

### What's Working:
- ✓ Metrics: RMSE, MAE, Bias, Variance calculated correctly
- ✓ Parameter Single: Has correct n_sim=300, T=400, Tb=200
- ✓ Parameter Recurring: Has correct persistence level variants (0.90, 0.95, 0.99)
- ✓ One-step ahead forecasting (horizon=1)
- ✓ Innovation types for single breaks (Gaussian, Student-t with dof)

### What's Missing:
- ✗ Unified n_sim, T, Tb across all scenarios
- ✗ Unified window sizes across all scenarios
- ✗ Innovation type support for recurring breaks
- ✗ Persistence level variants for variance/mean recurring
- ✗ Multiple break experiments entirely
- ✗ Consistent forecast origin rules

---

## 4. RECOMMENDED FIXES (Priority Order)

### P0 - Must Fix (Blocking Issues)

1. **Standardize n_sim, T, Tb:**
   ```
   ALL scenarios: n_sim=300, T=400, Tb=200
   Update: simu_meansingle.py, simu_variance_recurring.py, simu_mean_recurring.py
   ```

2. **Standardize window sizes:**
   ```
   ALL scenarios: window=100 (or all=60, but choose one)
   Update: simu_meansingle.py, simu_paramsingle.py, simu_paramrecurring.py
   ```

3. **Add innovation type to recurring breaks:**
   ```
   Add innovation_type, dof parameters to:
   - mc_variance_recurring()
   - mc_mean_recurring()
   - Update runner.py to call with 3 innovation variants each
   ```

4. **Add persistence level variants for variance/mean:**
   ```
   Current: Only p=0.95
   Required: Separate calls for p=0.90, 0.95, 0.99
   Update runner.py similar to parameter recurring
   ```

### P1 - Should Fix (Consistency Issues)

5. **Consistent forecast origin rule:**
   ```
   Single breaks: t0 = Tb + 30 (consistently)
   Recurring: t0 = T - 50 (consistently)
   ```

6. **Parameterize hardcoded window sizes:**
   ```
   Remove: window_sizes=[50, 100, 200] hardcoded
   Add: window_sizes parameter
   Use consistent value: [50, 100, 200] or make configurable
   ```

7. **Add multiple break experiments:**
   ```
   Create: simu_meanmultiple.py, simu_variancemultiple.py
   Copy structure from single breaks but with 2-3 breaks
   ```

### P2 - Nice to Have (Consistency)

8. **Uniform method sets:**
   ```
   Ensure all scenarios have 5-6 comparable forecasting methods
   Parameter single currently has only 3
   ```

---

## 5. SUMMARY TABLE

| Aspect | Current State | Required State | Status |
|--------|--------------|-----------------|--------|
| **Sample Size (n_sim)** | 100-300 mixed | All 300 | ❌ |
| **Time Series Length (T)** | 300-400 mixed | All 400 | ❌ |
| **Break Point (Tb)** | 150-200 mixed | All 200 | ❌ |
| **Window Sizes** | 60-100 mixed | All same (100 or 60) | ❌ |
| **Horizon** | 1 (all) | 1 (all) | ✓ |
| **Innovation Types (Single)** | Gaussian + Student-t | Gaussian + Student-t(3) + Student-t(5) | ⚠️ |
| **DOF Values (Single)** | 3, 5 available | Separate tables for each | ⚠️ |
| **Persistence Levels (Recurring)** | Only 0.95 | 0.90, 0.95, 0.99 | ❌ |
| **Metrics** | RMSE, MAE, Bias, Variance | RMSE, MAE, Bias, Variance | ✓ |
| **Break Types** | Single + Recurring | Single + Recurring + Multiple | ❌ |
| **Forecast Methods** | 3-5 per scenario | 5-6 uniform | ⚠️ |

---

## 6. FILES REQUIRING CHANGES

### Must Update:
1. `analyses/simu_meansingle.py` - Change n_sim, T, Tb, window
2. `analyses/simu_variance_recurring.py` - Change n_sim, add innovation_type, add persistence loop
3. `analyses/simu_mean_recurring.py` - Change n_sim, add innovation_type, add persistence loop
4. `analyses/simu_paramsingle.py` - Change window
5. `analyses/simu_paramrecurring.py` - Change window
6. `runner.py` - Add persistence level loops for variance/mean recurring
7. `analyses/simu_variance_single.py` - Parameterize window_sizes, change n_sim
8. `analyses/simu_meansingle.py` - Parameterize window_sizes if using Avg-Window

### Should Create:
9. `analyses/simu_meanmultiple.py` - For multiple break scenarios
10. `analyses/simu_variancemultiple.py` - For multiple break scenarios
11. `analyses/simu_parammultiple.py` - For multiple break scenarios (optional)

---

## Questions for User Clarification:

1. **Window size**: Should it be 60 or 100? Or different for different break types?
2. **Multiple breaks**: How many breaks? Where should they be placed (e.g., Tb1=130, Tb2=270)?
3. **Recurring persistence**: Should variance/mean recurring also test 0.90 and 0.99, or only 0.95?
4. **Forecast methods**: Should parameter single have 5 methods like variance/mean single?
5. **Forecast origin rule**: Should recurring breaks all use t0 = T - 50, or other?

