# PDF Compilation Guide

## Overview
After running your simulation experiments, you can automatically compile all generated LaTeX tables into a professional PDF document.

## Quick Start

### Option 1: Using Pixi Task (Recommended)
```bash
pixi run compile-pdf
```

### Option 2: Direct Python Script
```bash
PYTHONPATH=. pixi run python scripts/compile_results_pdf.py
```

### Option 3: System Python (if pdflatex installed)
```bash
python scripts/compile_results_pdf.py
```

## Output

The PDF is generated in the `docs/` folder:

- **Timestamped version**: `docs/simulation_results_YYYYMMDD_HHMMSS.pdf`
- **Latest symlink**: `docs/simulation_results_latest.pdf` (always points to most recent)

Example files:
```
docs/simulation_results_20260212_235322.pdf  (126.8 KB, 6 pages)
docs/simulation_results_latest.pdf → simulation_results_20260212_235322.pdf
```

## PDF Contents

The generated PDF includes:

### Main Sections
- **Table of Contents** with navigation
- **Variance Single Breaks** - Results for variance breaks with 3 innovation types
- **Mean Single Breaks** - Results for mean breaks with 3 innovation types  
- **Parameter Single Breaks** - Results for parameter breaks with 3 innovation types

### Appendices
1. **Simulation Configuration**
   - Number of simulations: 300
   - Time series length: 400
   - Break point: 200
   - Rolling window: 70
   - Forecast horizon: 1-step ahead

2. **Innovation Types**
   - Gaussian: Standard normal
   - Student-t(df=5): Heavy-tailed with 5 df
   - Student-t(df=3): Heavy-tailed with 3 df

3. **Methods**
   - GARCH: Conditional volatility modeling
   - SARIMA: Seasonal ARIMA (full-sample, rolling)
   - SARIMAX: SARIMA with exogenous break indicator
   - SES: Simple exponential smoothing
   - MS-AR: Markov-switching autoregressive

## Workflow Integration

### Full Pipeline
```bash
# 1. Run experiments
pixi run python runner.py

# 2. Compile results to PDF
pixi run compile-pdf

# 3. Open PDF in docs folder
open docs/simulation_results_latest.pdf
```

### Automated (Single Command)
```bash
pixi run python runner.py && pixi run compile-pdf
```

## Requirements

- **LaTeX Installation**: `pdflatex` must be available
  - On Ubuntu/Debian: `sudo apt-get install texlive-latex-base`
  - On macOS: `brew install --cask mactex` or `tlmgr install pdflatex`
  - On Windows: Install [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

- **Python Packages**: All included in `pixi.toml`

## Troubleshooting

### "pdflatex not found"
Install LaTeX distribution:
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base

# macOS
brew install --cask mactex

# Verify installation
which pdflatex
```

### PDF not created
- Check that `results/tex/` contains .tex files from your run
- Verify sufficient disk space (PDFs are ~130 KB each)
- Check logs for LaTeX compilation errors

### To enable detailed debugging
Edit `scripts/compile_results_pdf.py` and change the compile function to preserve temporary files

## File Organization

```
qonlab/
├── scripts/
│   └── compile_results_pdf.py        # PDF compilation script
├── results/
│   ├── csv/                          # Experiment result CSVs
│   └── tex/                          # Generated LaTeX tables
└── docs/
    ├── simulation_results_latest.pdf  # Most recent output
    ├── simulation_results_*.pdf       # Timestamped archives
    ├── paper/
    ├── presentation/
    └── slides/
```

## Customization

### Modify PDF Structure
Edit `scripts/compile_results_pdf.py`:
- `create_master_tex()` - Control LaTeX template
- `organize_files_by_type()` - Change section organization
- Document preamble - Adjust formatting, fonts, margins

### Metadata
Add/modify in the LaTeX template:
```python
\title{Your Title}
\author{Author Names}
\date{\today}
```

## Performance

- **Compilation time**: ~5-10 seconds
- **PDF size**: ~127 KB
- **Pages**: 6 (with 18 tables + TOC + appendix)

Scales linearly with number of results (each table adds ~3-4 KB).

