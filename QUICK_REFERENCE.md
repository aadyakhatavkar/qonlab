# Quick Reference - Complete Workflow

## 📋 Standard Workflow

**4-Step Process** from experiments to professional report:

```bash
# Step 1: Run simulations (generates results and LaTeX tables)
pixi run python runner.py

# Step 2: Generate plots (Tier 1 metrics + Tier 2 DGP visualization)
python scripts/generate_plots.py --all

# Step 3: Compile tables to PDF
python scripts/build_pdfs.py --tables

# Step 4: Create combined report (tables + figures in one PDF)
python scripts/build_pdfs.py --combined
```

---

## Plotting Commands

```bash
# List all 13 available plots
python scripts/generate_plots.py --list

# Generate all plots
python scripts/generate_plots.py --all

# Generate by break type
python scripts/generate_plots.py --variance
python scripts/generate_plots.py --mean
python scripts/generate_plots.py --parameter

# Generate by subtype
python scripts/generate_plots.py --single
python scripts/generate_plots.py --recurring

# Generate by tier
python scripts/generate_plots.py --tier1    # Core metrics (RMSE, MAE, etc)
python scripts/generate_plots.py --tier2    # DGP visualization

# Combinations
python scripts/generate_plots.py --variance --tier1
python scripts/generate_plots.py --mean --single --tier2
python scripts/generate_plots.py --parameter --recurring
```

## PDF Compilation Commands

### Main Commands
```bash
# Tables PDF only (50 KB)
python scripts/build_pdfs.py --tables
# Output: outputs/pdf/Tables_Results_YYYYMMDD_HHMMSS.pdf

# Figures PDF only (5.3 MB)
python scripts/build_pdfs.py --figures
# Output: outputs/pdf/Figures_Tier1-2_YYYYMMDD_HHMMSS.pdf

# Combined report (tables + figures, 3.6 MB) - RECOMMENDED
python scripts/build_pdfs.py --combined
# Output: outputs/pdf/Complete_Analysis_YYYYMMDD_HHMMSS.pdf

# Generate all three at once
python scripts/build_pdfs.py --all
python scripts/build_pdfs.py  # (default, same as --all)
```

### Inspection
```bash
# List all PDF files created
python scripts/build_pdfs.py --list

# Show file sizes
ls -lh outputs/pdf/

# Open latest combined PDF
open outputs/pdf/Complete_Analysis_*.pdf
```

---

## View Results

```bash
# Show all compiled PDFs
ls -lh outputs/pdf/

# Show figure count
find outputs/figures -name "*.png" | wc -l

# Show table count
find outputs/tex -name "*.tex" | wc -l

# Open latest figures PDF
open outputs/pdf/Figures_Tier1-2_*.pdf

# Download latest PDF
scp user@host:outputs/pdf/Figures_Tier1-2_*.pdf .
```

## Color Scheme Reference

Use in custom plots:
```python
from scripts.unified_plots import COLOR_PALETTE

ax.plot(x, y, color=COLOR_PALETTE['primary'])      # Blue
ax.bar(x, y, color=COLOR_PALETTE['warning'])       # Orange
ax.scatter(x, y, color=COLOR_PALETTE['success'])   # Green
```

Colors:
- `'primary'` → Blue (#2E86AB)
- `'accent'` → Purple (#A23B72)
- `'success'` → Green (#06A77D)
- `'warning'` → Orange (#F18F01)
- `'danger'` → Red (#C73E1D)
- `'innovation_gaussian'` → Blue (#1f77b4)
- `'innovation_t3'` → Orange (#ff7f0e)
- `'innovation_t5'` → Green (#2ca02c)

## Key Features

✅ **Tier 1 plots**: Method comparison, uncertainty metrics, innovation effects  
✅ **Tier 2 plots**: Time series with breaks, DGP visualization  
✅ **DOF labels**: Automatically shown for innovation types (Gaussian, t-df3, t-df5)  
✅ **One-command PDFs**: Timestamped, organized by break type  
✅ **New PDF each time**: Not overwritten, stacked in outputs/pdf/  
✅ **Consistent colors**: Single palette used everywhere  
✅ **No duplication**: All plot functions in scripts/unified_plots.py  

## Files

| File | Purpose |
|------|---------|
| `scripts/unified_plots.py` | All plot functions (centralized) |
| `scripts/build_pdfs.py` | PDF compilation (tables & figures) |
| `outputs/pdf/` | Output directory for timestamped PDFs |
| `outputs/csv/` | Source CSV tables |
| `outputs/tex/` | Source LaTeX files |

## Workflow

```bash
# 1. Generate plots
python scripts/generate_plots.py --all

# 2. Compile to PDF
python scripts/build_pdfs.py --figures

# 3. Check location
ls -lh docs/compilations/

# 4. Done! New PDF created with timestamp
# Output: docs/compilations/Figures_Tier1-2_20260213_130000.pdf (5.3 MB)
```

---

For more details, see:
- UNIFIED_PLOTTING_GUIDE.md (comprehensive guide)
- TIER_PLOTS_USAGE.md (Tier 1 & 2 details)
- IMPLEMENTATION_COMPLETE.md (features overview)
