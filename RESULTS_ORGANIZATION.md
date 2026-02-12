# Results Organization Guide

## Directory Structure

All simulation results are organized in the `results/` folder with the following structure:

```
results/
├── csv/
│   ├── variance_single_*.csv
│   ├── variance_recurring_*.csv
│   ├── mean_single_*.csv
│   ├── mean_recurring_*.csv
│   ├── parameter_single_*.csv
│   └── parameter_recurring_*.csv
│
├── tex/
│   ├── variance_single_*.tex
│   ├── variance_recurring_*.tex
│   ├── mean_single_*.tex
│   ├── mean_recurring_*.tex
│   ├── parameter_single_*.tex
│   └── parameter_recurring_*.tex
│
└── combined/
    ├── aligned_breaks_*.csv      (All results merged)
    └── aligned_breaks_*.tex      (All results merged - LaTeX)
```

## File Types

### Individual Results (csv/ and tex/ folders)

Each experiment variant generates both CSV and LaTeX table formats:

- **Single Breaks**: 9 files (3 per break type × 3 innovations)
  - `variance_single_TIMESTAMP_Gaussian.{csv,tex}`
  - `variance_single_TIMESTAMP_Student-tdf5.{csv,tex}`
  - `variance_single_TIMESTAMP_Student-tdf3.{csv,tex}`
  - Similar for `mean_single_*` and `parameter_single_*`

- **Recurring Breaks**: 5 files
  - `variance_recurring_TIMESTAMP_MarkovSwitching.{csv,tex}`
  - `mean_recurring_TIMESTAMP_MarkovSwitching.{csv,tex}`
  - `parameter_recurring_TIMESTAMP_p090.{csv,tex}`
  - `parameter_recurring_TIMESTAMP_p095.{csv,tex}`
  - `parameter_recurring_TIMESTAMP_p099.{csv,tex}`

### Combined Results (combined/ folder)

- `aligned_breaks_TIMESTAMP.csv` - All results merged in CSV format
- `aligned_breaks_TIMESTAMP.tex` - All results merged in LaTeX format (raw)

## File Naming Convention

Files are named with the following pattern:

```
{break_type}_{variant}_{timestamp}_{innovation|persistence}.{format}
```

- **break_type**: variance, mean, or parameter
- **variant**: single or recurring
- **timestamp**: YYYYMMDD_HHMMSS (prevents overwriting)
- **innovation/persistence**: Gaussian, Student-t(df=X), or pY## (e.g., p090)
- **format**: csv or tex

## Usage

### Access Individual Results

```bash
# View all variance single break results
ls results/csv/variance_single_*.csv

# View specific innovation type
cat results/csv/variance_single_*_Gaussian.csv
```

### Use Results in LaTeX Documents

```latex
% Include a single result table
\input{results/tex/variance_single_TIMESTAMP_Gaussian.tex}

% Or combine all results
\input{results/combined/aligned_breaks_TIMESTAMP.tex}
```

### Generate PDF with All Results

```bash
# Compile all results into a professional PDF
pixi run compile-pdf

# Output: docs/simulation_results_latest.pdf
```

## Timestamping Strategy

Each run generates timestamped files to:
- Prevent overwriting previous results
- Track multiple experimental runs
- Allow version control and comparison

Use the latest timestamp to access the most recent results.

## Best Practices

1. **Don't modify CSV/TEX files directly** - they're auto-generated
2. **Use latest timestamp** when referencing results
3. **Archive old results** by moving to a separate folder if needed
4. **Keep combined/ folder clean** - only latest merged results needed
5. **Version important runs** by copying to a separate archive folder

