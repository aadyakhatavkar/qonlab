# Scenarios

This directory contains JSON configuration files for Monte Carlo experiments.

## File: `example_scenarios.json`

Contains pre-defined scenarios for all break types:

| Task | Scenarios |
|------|-----------|
| **variance** | Small break, Late break |
| **parameter** | Single break (φ₁=0.2 → φ₂=0.9) |
| **mean** | Moderate, Large, Early, Late breaks |

## Usage

```bash
# Run with custom scenarios
python main.py mc --scenarios scenarios/example_scenarios.json

# Quick test
python main.py mc --quick
```

## Scenario Format

```json
{
  "name": "Scenario name",
  "task": "variance|mean|parameter",
  "T": 400,
  "Tb": 200,
  "owner": "name",
  "tag": "identifier"
}
```

### Task-Specific Parameters

**Variance:**
- `variance_Tb`, `variance_sigma1`, `variance_sigma2`

**Mean:**
- `Tb`, `mu0`, `mu1`, `phi`, `sigma`

**Parameter:**
- `Tb`, `phi1`, `phi2`, `sigma`

## Simulation Standards

1. **Seed management:** Reproducible random number generation
2. **Parallel execution:** Optional joblib parallelization
3. **Error handling:** Graceful failure tracking
4. **Documentation:** All parameters recorded with results
