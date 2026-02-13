# Unified Plotting System Guide

## Overview

The unified plotting system generates **coherent, publication-quality visualizations** from Monte Carlo simulation results. All plots share a consistent visual style for professional appearance.

## Architecture

### Unified Styling (Applied to All Plots)

```python
# Professional Color Scheme
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'accent': '#A23B72',       # Accent purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
}
```

All figures use:
- **DPI**: 300 (publication quality)
- **Font**: Sans-serif, 11pt body, 13pt titles
- **Grid**: Light gridlines (alpha=0.3) with dashed style
- **Spines**: Top and right spines removed
- **Edge colors**: Black 0.5pt for all bar patches
- **Line width**: 1.5pt for all lines

### Core Plotting Functions

#### 1. `plot_method_comparison_metrics(df, break_type, save_path=None)`
**Purpose**: Core comparison across forecasting methods  
**Generates**: 4-panel figure (2×2) with RMSE, MAE, Bias, Variance  
**Features**:
- Sorted by ascending metric value
- Value labels on bars
- Consistent color scheme
- Grid aligned for readability

**Example**:
```
python scripts/generate_plots.py --break-type mean
# Outputs: figures/mean/mean_method_comparison.png
```

#### 2. `plot_uncertainty_metrics(df, break_type, save_path=None)`
**Purpose**: Uncertainty quantification (variance scenarios only)  
**Generates**: 3-panel figure with Coverage80, Coverage95, LogScore  
**Features**:
- Target lines at 80% and 95% (red dashed)
- Coverage80 in red, Coverage95 in teal
- LogScore in green (sorted ascending)
- Automatically skipped if data not available

**Example**:
```
python scripts/generate_plots.py --break-type variance
# Outputs: figures/variance/variance_uncertainty.png
```

#### 3. `plot_rmse_comparison_all_innovations(df_all_innovations, break_type, save_path=None)`
**Purpose**: Compare methods across innovation types  
**Generates**: Grouped bar chart (methods on x-axis, colors by innovation)  
**Features**:
- Gaussian, Student-t(df=3), Student-t(df=5) distinguished by color
- Methods sorted lexicographically
- X-axis labels rotated 45°
- Automatically skipped if only one innovation type

**Example**:
```
python scripts/generate_plots.py --break-type parameter
# Outputs: figures/parameter/parameter_innovation_comparison.png (if multiple innovations)
```

## Usage

### Generate Latest Results Plots

```bash
pixi run python scripts/generate_plots.py --latest
```

Automatically finds most recent `results/aligned_breaks_*.csv` and generates all plots.

### Generate Plots for Specific Break Type

```bash
# Single break type
pixi run python scripts/generate_plots.py --break-type variance

# Mean break
pixi run python scripts/generate_plots.py --break-type mean

# Parameter break
pixi run python scripts/generate_plots.py --break-type parameter

# All break types (default)
pixi run python scripts/generate_plots.py --break-type all
```

### Custom Results File

```bash
pixi run python scripts/generate_plots.py --results results/custom_results.csv --output-dir figures
```

### List All Options

```bash
pixi run python scripts/generate_plots.py --help
```

## Output Structure

```
figures/
├── mean/
│   ├── mean_method_comparison.png         # 4-panel metrics
│   └── mean_innovation_comparison.png      # Grouped bar chart (if multiple innovations)
├── variance/
│   ├── variance_method_comparison.png      # 4-panel metrics
│   ├── variance_uncertainty.png            # Coverage + LogScore
│   └── variance_innovation_comparison.png  # Grouped bar chart (if multiple innovations)
└── parameter/
    ├── parameter_method_comparison.png     # 4-panel metrics
    └── parameter_innovation_comparison.png # Grouped bar chart (if multiple innovations)
```

## Visual Consistency Checklist

- ✅ All bar charts use primary blue (#2E86AB) by default
- ✅ Coverage bars use red (#FF6B6B) for 80%, teal (#4ECDC4) for 95%
- ✅ LogScore uses success green (#06A77D)
- ✅ All fonts use sans-serif, consistent sizes
- ✅ Grid lines applied uniformly
- ✅ Target lines (coverage) are dark red dashed (linewidth=2.5)
- ✅ All axes labeled consistently
- ✅ Titles bold and descriptive
- ✅ DPI = 300 for all saved figures
- ✅ White background for all figures

## Integration with Runner

The `runner.py` script can generate plots automatically:

```bash
# Run simulations AND generate plots
pixi run python runner.py --generate-plots

# Generate plots only (from existing results)
pixi run python runner.py --plots-only
```

## Extending the System

To add new coherent plots:

1. Use `apply_unified_style()` at the start
2. Choose colors from `COLORS` dict (or `COVERAGE_COLORS` for coverage metrics)
3. Use consistent font sizes (11pt body, 12pt labels, 13pt titles)
4. Apply grid with `ax.grid(True, axis='y', alpha=0.3, linestyle='--')`
5. Remove top/right spines with `ax.set_axisbelow(True)`
6. Save with `dpi=300, bbox_inches='tight', facecolor='white'`

### Template

```python
def plot_new_metric(df, break_type, save_path=None):
    """Plot new metric coherently."""
    apply_unified_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Your plotting code here, using COLORS from color scheme
    
    ax.set_xlabel('X Label', fontsize=12)
    ax.set_ylabel('Y Label', fontsize=12)
    ax.set_title(f'{break_type.title()} - Metric', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    
    plt.close()
```

## Troubleshooting

### No plots generated

- Check that results file exists: `ls results/aligned_breaks_*.csv`
- Verify column names match expected metrics (RMSE, MAE, Bias, Variance, etc.)
- Check output directory is writable: `ls -l figures/`

### Coverage metrics not showing

- Ensure `--break-type variance` is specified
- Verify results CSV contains Coverage80, Coverage95, LogScore columns

### Innovation comparison missing

- Check that results include multiple innovation types (Gaussian, Student-t variants)
- Verify `Innovation` column exists in CSV

## Performance Notes

- Typical plot generation: ~0.5-1 second per figure
- All plots are matplotlib-based (no external graphing libraries)
- Memory footprint: ~50-100MB for typical results files
