# Research Proposal: Forecasting Under Structural Breaks
## Research Module in Econometrics and Statistics
### Fundamentals of Monte Carlo Simulations in Data Science

**Author:** Aadya Khatavkar  
**Supervisor:** Prof. Vladislav Morozov  
**Institution:** University of Bonn  
**Semester:** Winter 2025/26

---

## 1. Research Question

**Primary Question:**  
How do different forecasting methods perform under structural breaks in time series data, and what is the cost of not knowing break dates in terms of forecast accuracy?

**Secondary Questions:**
1. Can adaptive methods (rolling windows, regime switching) provide practical alternatives to oracle specifications when break dates are unknown?
2. What is the optimal window size for rolling-window estimation under different break magnitudes?
3. How do results generalize from mean breaks to variance breaks and parameter breaks?

---

## 2. Motivation

Structural breaks—discrete changes in the parameters of a data-generating process—pose fundamental challenges for time series forecasting. When model parameters shift unexpectedly, forecasts based on historical data can become severely biased. This is particularly relevant in:

- **Macroeconomic forecasting:** Policy changes, financial crises
- **Financial markets:** Volatility regime shifts, market turbulence
- **Climate science:** Trend changes in environmental data

Understanding how different forecasting methods handle structural change is critical for practitioners choosing among available techniques.

---

## 3. Contribution

This project makes the following contributions:

1. **Systematic Monte Carlo comparison** of five forecasting approaches under structural breaks
2. **Quantification of "oracle advantage"**—the performance gap between knowing vs. estimating break dates
3. **Practical guidance** for window size selection in rolling estimation
4. **Replicable Python implementation** following best practices for simulation studies

---

## 4. Methodology

### 4.1 Data-Generating Processes

We consider three types of structural breaks:

| Break Type | Description | DGP |
|------------|-------------|-----|
| **Mean break** | Intercept shift | $y_t = \mu_t + \phi y_{t-1} + u_t$ |
| **Variance break** | Volatility shift | $y_t = \phi y_{t-1} + \sigma_t u_t$ |
| **Parameter break** | AR coefficient shift | $y_t = \phi_t y_{t-1} + u_t$ |

### 4.2 Forecasting Methods

1. **Global AR(1):** Full-sample estimation (benchmark)
2. **Rolling AR(1):** Window-based adaptive estimation
3. **Break Dummy (Oracle):** Known break dates via indicator variables
4. **Estimated Break (Grid Search):** Data-driven break detection
5. **Markov Switching:** Regime-switching model (if numerically stable)

### 4.3 Monte Carlo Design

- **Replications:** $N = 200-500$
- **Sample sizes:** $T \in \{100, 200, 400\}$
- **Break locations:** Mid-sample ($T/2$), early ($T/4$), late ($3T/4$)
- **Break magnitudes:** Small, medium, large

### 4.4 Evaluation Metrics

- **Point forecast:** RMSE, MAE, Bias
- **Uncertainty quantification:** Coverage (80%, 95%), Log-score

---

## 5. Expected Results

Based on preliminary simulations, we expect:

1. **Oracle break dummies** will achieve lowest errors (benchmark performance)
2. **Rolling AR(1)** will outperform global AR(1) when breaks are present
3. **Estimated break methods** will show intermediate performance (cost of break date uncertainty)
4. **Optimal window size** will depend on break magnitude and time since break

---

## 6. Timeline

| Week | Task |
|------|------|
| 1-2 | Literature review; finalize simulation design |
| 3-4 | Implement DGPs and forecasting methods |
| 5-6 | Run main Monte Carlo experiments |
| 7-8 | Analyze results; create figures and tables |
| 9-10 | Write paper sections; prepare presentation |
| 11-12 | Revisions and final submission |

---

## 7. Deliverables

1. **Term paper** (15-20 pages) documenting methodology and findings
2. **Public presentation** of key results
3. **Replicable code repository** with:
   - Documented Python modules
   - Reproducible experiment configurations
   - Automated testing via CI/CD

---

## 8. References

- Pesaran, M. H. (2013). "The Role of Structural Breaks in Forecasting," *Handbook of Economic Forecasting*
- Box, G. E. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple Structural Changes"

---

## 9. Course Alignment

This project directly addresses the course objectives:

> "The role of simulations in evaluating statistical methods... Evaluating confidence intervals and hypothesis tests... Assessing predictive algorithms."

The simulation framework evaluates:
- ✅ Forecasting method performance (predictive algorithms)
- ✅ Uncertainty quantification (confidence intervals)
- ✅ Hypothesis testing (break detection)

---

**Status:** Proposal approved  
**Repository:** [github.com/aadyakhatavkar/qonlab](https://github.com/aadyakhatavkar/qonlab)
