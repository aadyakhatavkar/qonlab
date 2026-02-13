#!/usr/bin/env python3
"""
Compile Latest Figures and Tables into PDFs
==============================================
Creates timestamped PDFs of latest figures and tables.
Organized by break type and tier.
Placed in outputs/pdf/

Usage:
    python scripts/build_pdfs.py --figures      # Latest figures only
    python scripts/build_pdfs.py --tables       # Latest tables only
    python scripts/build_pdfs.py --all          # Both figures and tables
    python scripts/build_pdfs.py --list         # List recent files
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import glob
import shutil

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage, PageBreak, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from PyPDF2 import PdfMerger, PdfReader, PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# =========================================================
# CONFIGURATION
# =========================================================

REPO_ROOT = Path(__file__).parent.parent
FIGURES_DIR = REPO_ROOT / 'outputs' / 'figures'
RESULTS_DIR = REPO_ROOT / 'outputs'  # CSV and TEX files now directly in outputs/
COMPILATIONS_DIR = REPO_ROOT / 'outputs' / 'pdf'

# Create compilations directory if it doesn't exist
COMPILATIONS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DATE_READABLE = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def find_latest_files(directory, pattern='*', max_age_hours=None, limit=None):
    """Find latest files matching pattern."""
    if not os.path.exists(directory):
        return []
    
    files = []
    for root, dirs, filenames in os.walk(directory):
        for fname in filenames:
            # Check if file matches pattern
            matches = False
            if pattern == '*':
                matches = True
            elif pattern.startswith('*'):
                matches = fname.endswith(pattern[1:])
            else:
                matches = fname.endswith(pattern)
            
            if matches:
                fpath = Path(root) / fname
                files.append((fpath, fpath.stat().st_mtime))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    
    if limit:
        files = files[:limit]
    
    return [f[0] for f in files]


def get_latest_tables():
    """Get only the latest CSV files per scenario, keeping all innovation variants."""
    import os
    from collections import defaultdict
    
    csv_files = []
    for root, dirs, files in os.walk(str(RESULTS_DIR)):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    
    if not csv_files:
        return []
    
    # Group files by break_type + mode + variant
    # Keep latest of each combo (Gaussian, Student-tdf3, Student-tdf5, MarkovSwitching, etc.)
    scenarios = defaultdict(list)
    
    for csv_path in csv_files:
        filename = csv_path.stem
        
        if '_results' in filename:
            # Skip aggregated results files
            continue
        else:
            # Detailed: mean_single_20260213_211638_Gaussian.csv
            # Extract break_type + mode + variant
            parts = filename.split('_')
            if len(parts) >= 3:
                break_type = parts[0]  # mean, variance, parameter
                mode = parts[1]  # single, recurring
                variant = parts[-1]  # Gaussian, Student-tdf3, Student-tdf5, MarkovSwitching, p09, etc.
                scenario_key = f"{break_type}_{mode}_{variant}"
            else:
                scenario_key = filename
        
        scenarios[scenario_key].append(csv_path)
    
    # Keep only latest per scenario
    latest_files = []
    for scenario_key, files in scenarios.items():
        # Sort by modification time, get the newest
        latest = max(files, key=lambda f: f.stat().st_mtime)
        latest_files.append(latest)
    
    return sorted(latest_files)


def convert_csv_to_latex():
    """Convert CSV metrics files to LaTeX tables with actual data-driven captions."""
    import pandas as pd
    import numpy as np
    
    # Get only latest tables (avoid duplicates)
    csv_files = get_latest_tables()
    
    tex_files = []
    for csv_path in csv_files:
        try:
            filename = csv_path.stem
            
            # Skip aligned_breaks files (they contain NaN values for incomplete scenarios)
            if 'aligned_breaks' in filename:
                continue
            
            df = pd.read_csv(csv_path)
            
            # Extract metadata from CSV
            n_sim = int(df['N'].iloc[0]) if 'N' in df.columns else '?'
            n_methods = len(df)
            
            # Remove metadata columns that shouldn't appear in table
            drop_cols = []
            if 'Break Type' in df.columns:
                drop_cols.append('Break Type')
            if 'Persistence' in df.columns:
                drop_cols.append('Persistence')
            if 'Innovation' in df.columns:
                drop_cols.append('Innovation')
            if 'Successes' in df.columns:
                drop_cols.append('Successes')
            if 'Failures' in df.columns:
                drop_cols.append('Failures')
            if 'N' in df.columns:
                drop_cols.append('N')
            
            # For variance tables, remove Coverage80/Coverage95 (keep LogScore for uncertainty quantification)
            if 'variance' in filename:
                if 'Coverage80' in df.columns:
                    drop_cols.append('Coverage80')
                if 'Coverage95' in df.columns:
                    drop_cols.append('Coverage95')
            
            # For mean and parameter tables, remove Coverage80/Coverage95/LogScore
            if 'mean' in filename or 'parameter' in filename:
                if 'Coverage80' in df.columns:
                    drop_cols.append('Coverage80')
                if 'Coverage95' in df.columns:
                    drop_cols.append('Coverage95')
                if 'LogScore' in df.columns:
                    drop_cols.append('LogScore')
            
            df_display = df.drop(columns=drop_cols, errors='ignore')
            
            # Fill blank LogScore/Coverage values with proper format
            for col in ['LogScore', 'Coverage80', 'Coverage95']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].fillna(np.nan)
            
            # Create meaningful caption based on filename and data
            if 'variance' in filename:
                if 'recurring' in filename:
                    # For variance recurring, don't include p= in caption (it's not for variance)
                    caption = f"Variance Recurring: {n_sim} simulations"
                else:
                    # Check for innovation type
                    if 'Student-tdf5' in filename or 'tdf5' in filename:
                        caption = f"Variance Single Break (Student-t df=5): {n_sim} simulations"
                    elif 'Student-tdf3' in filename or 'tdf3' in filename:
                        caption = f"Variance Single Break (Student-t df=3): {n_sim} simulations"
                    elif 'Gaussian' in filename:
                        caption = f"Variance Single Break (Gaussian): {n_sim} simulations"
                    else:
                        caption = f"Variance Single Break: {n_sim} simulations"
            elif 'mean' in filename:
                if 'recurring' in filename:
                    caption = f"Mean Recurring: {n_sim} simulations"
                else:
                    # Check for innovation type
                    if 'Student-tdf5' in filename or 'tdf5' in filename:
                        caption = f"Mean Single Break (Student-t df=5): {n_sim} simulations"
                    elif 'Student-tdf3' in filename or 'tdf3' in filename:
                        caption = f"Mean Single Break (Student-t df=3): {n_sim} simulations"
                    elif 'Gaussian' in filename:
                        caption = f"Mean Single Break (Gaussian): {n_sim} simulations"
                    else:
                        caption = f"Mean Single Break: {n_sim} simulations"
            elif 'parameter' in filename:
                if 'recurring' in filename:
                    p_val = filename.split('p')[-1] if 'p' in filename else '?'
                    caption = f"Parameter Recurring (p={p_val}): {n_sim} simulations"
                else:
                    # Check for innovation type
                    if 'Student-tdf5' in filename or 'tdf5' in filename:
                        caption = f"Parameter Single Break (Student-t df=5): {n_sim} simulations"
                    elif 'Student-tdf3' in filename or 'tdf3' in filename:
                        caption = f"Parameter Single Break (Student-t df=3): {n_sim} simulations"
                    elif 'Gaussian' in filename:
                        caption = f"Parameter Single Break (Gaussian): {n_sim} simulations"
                    else:
                        caption = f"Parameter Single Break: {n_sim} simulations"
            else:
                caption = f"{filename}: {n_sim} simulations"
            
            # Convert to LaTeX with dynamic caption
            latex_content = df_display.to_latex(
                index=False,
                float_format='%.4f',
                caption=caption,
                label=f"tab:{filename}"
            )
            
            # Wrap table with size adjustment and centering for better page fit
            latex_content = latex_content.replace(
                '\\begin{table}',
                '\\begin{table}[H]\n\\centering\n\\small'
            )
            latex_content = latex_content.replace(
                '\\begin{tabular}',
                '\\begin{tabular}'
            )
            latex_content = latex_content.replace(
                '\\end{tabular}',
                '\\end{tabular}'
            )
            latex_content = latex_content.replace(
                '\\end{table}',
                '\\normalsize\n\\end{table}'
            )
            
            # Create corresponding .tex file in same directory
            tex_path = csv_path.with_suffix('.tex')
            with open(tex_path, 'w') as f:
                f.write(latex_content)
            
            tex_files.append(tex_path)
        except Exception as e:
            print(f"Warning: Could not convert {csv_path.name}: {e}")
    
    return tex_files


def organize_figures_by_type():
    """Organize recent PNG files by break type and tier."""
    files = find_latest_files(FIGURES_DIR, '*.png', limit=50)
    
    organized = {
        'variance': {'tier1': [], 'tier2': []},
        'mean': {'tier1': [], 'tier2': []},
        'parameter': {'tier1': [], 'tier2': []},
    }
    
    for fpath in files:
        fname = fpath.name
        parent_dir = fpath.parent.name
        
        # Determine break type
        if 'variance' in fname or parent_dir == 'variance':
            break_type = 'variance'
        elif 'mean' in fname or parent_dir == 'mean':
            break_type = 'mean'
        elif 'parameter' in fname or parent_dir == 'parameter':
            break_type = 'parameter'
        else:
            continue
        
        # Determine tier
        if 'tier1' in fname or 'comparison' in fname or 'logscore' in fname or 'metric' in fname:
            tier = 'tier1'
        elif 'tier2' in fname or 'example' in fname or 'dgp' in fname or 'timeseries' in fname:
            tier = 'tier2'
        else:
            tier = 'tier1'  # Default
        
        organized[break_type][tier].append(fpath)
    
    return organized


def organize_tables_by_type(csv_files=None):
    """Organize LaTeX files by break type (using only deduped latest CSVs)."""
    import re
    
    # If csv_files provided, convert to .tex files
    if csv_files is None:
        files = find_latest_files(RESULTS_DIR, '*.tex', limit=100)
    else:
        # Convert provided CSV paths to .tex files, skip aligned_breaks
        files = [csv_path.with_suffix('.tex') for csv_path in csv_files if 'aligned_breaks' not in csv_path.name]
    
    organized = {
        'variance': [],
        'mean': [],
        'parameter': [],
        'other': [],
    }
    
    for fpath in files:
        if not fpath.exists():
            continue
        
        fname = fpath.name.lower()
        
        # Skip aligned_breaks files
        if 'aligned_breaks' in fname:
            continue
            
        # Classify by break type
        if 'variance' in fname:
            organized['variance'].append(fpath)
        elif 'mean' in fname:
            organized['mean'].append(fpath)
        elif 'parameter' in fname:
            organized['parameter'].append(fpath)
        else:
            organized['other'].append(fpath)
    
    return organized


def create_figure_pdf_native(figures_dict, output_path):
    """
    Create PDF of figures using reportlab (native Python).
    This avoids external dependencies like ImageMagick.
    """
    if not HAS_REPORTLAB or not HAS_PIL:
        print("⚠ reportlab or PIL not available. Install with: pip install reportlab pillow")
        return False
    
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage, PageBreak, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=12,
        alignment=1  # Center
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#A23B72'),
        spaceAfter=10,
        alignment=1
    )
    
    story = []
    
    # Title page
    story.append(Paragraph("Tier 1 & Tier 2 Analysis Plots", title_style))
    story.append(Paragraph(f"Generated: {DATE_READABLE}", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Add figures organized by break type
    for break_type in ['variance', 'mean', 'parameter']:
        if not figures_dict.get(break_type):
            continue
        
        story.append(PageBreak())
        story.append(Paragraph(f"{break_type.upper()} BREAKS", title_style))
        
        for tier in ['tier1', 'tier2']:
            figs = figures_dict[break_type].get(tier, [])
            if not figs:
                continue
            
            story.append(Paragraph(f"{tier.upper()} Plots", subtitle_style))
            
            for fig_path in figs[:5]:  # Limit to 5 per section
                try:
                    # Get image size
                    img = Image.open(fig_path)
                    width, height = img.size
                    aspect = height / width
                    
                    # Scale to fit page
                    max_width = 6.5*inch
                    img_width = max_width
                    img_height = max_width * aspect
                    
                    # Add image
                    story.append(RLImage(str(fig_path), width=img_width, height=img_height))
                    story.append(Spacer(1, 0.3*inch))
                    story.append(Paragraph(fig_path.stem, styles['Normal']))
                    story.append(Spacer(1, 0.5*inch))
                    
                except Exception as e:
                    print(f"  ⚠ Skipping {fig_path.name}: {e}")
    
    # Build PDF
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        doc.build(story)
        return True
    except Exception as e:
        print(f"  ✗ Error building PDF: {e}")
        return False


def sort_tables_hierarchical(tables_dict):
    """
    Reorganize tables in hierarchical order:
    1. Mean
       1.1 Single (by DOF: Gaussian, tdf3, tdf5)
       1.2 Recurring
    2. Parameter
       2.1 Single (by DOF: Gaussian, tdf3, tdf5)
       2.2 Recurring (by persistence: p09, p095, p099)
    3. Variance
       3.1 Single (by DOF: Gaussian, tdf3, tdf5)
       3.2 Recurring
    4. Combined
    """
    ordered = {
        'mean': [],
        'parameter': [],
        'variance': [],
        'other': []
    }
    
    # Sort within each break type by name
    for break_type in ['mean', 'parameter', 'variance', 'other']:
        if break_type in tables_dict:
            ordered[break_type] = sorted(tables_dict[break_type], key=lambda x: x.name)
    
    # Custom sorting for single tables (by innovation type: Gaussian, tdf3, tdf5)
    def innovation_sort_key(path):
        name = path.name.lower()
        if 'gaussian' in name:
            return 0
        elif 'tdf3' in name:
            return 1
        elif 'tdf5' in name:
            return 2
        else:
            return 3
    
    # Custom sorting for recurring/persistence (p09, p095, p099)
    def persistence_sort_key(path):
        name = path.name.lower()
        if 'p09' in name or 'p=0.9' in name:
            return 0
        elif 'p095' in name or 'p=0.95' in name:
            return 1
        elif 'p099' in name or 'p=0.99' in name:
            return 2
        else:
            return 3
    
    # Re-sort mean tables
    mean_single = [t for t in ordered['mean'] if 'single' in t.name]
    mean_recurring = [t for t in ordered['mean'] if 'recurring' in t.name or 'markov' in t.name]
    ordered['mean'] = sorted(mean_single, key=innovation_sort_key) + sorted(mean_recurring)
    
    # Re-sort parameter tables
    param_single = [t for t in ordered['parameter'] if 'single' in t.name]
    param_recurring = [t for t in ordered['parameter'] if 'recurring' in t.name or 'p0' in t.name]
    ordered['parameter'] = sorted(param_single, key=innovation_sort_key) + sorted(param_recurring, key=persistence_sort_key)
    
    # Re-sort variance tables
    var_single = [t for t in ordered['variance'] if 'single' in t.name]
    var_recurring = [t for t in ordered['variance'] if 'recurring' in t.name or 'markov' in t.name]
    ordered['variance'] = sorted(var_single, key=innovation_sort_key) + sorted(var_recurring)
    
    return ordered


def extract_summary_statistics_from_tables():
    """
    Extract summary statistics from all CSV result files for dynamic executive summary.
    Returns a dictionary with key findings for the executive summary.
    """
    import pandas as pd
    
    summary = {
        'n_sim': 5,
        'total_results': 0,
        'best_overall_rmse': float('inf'),
        'best_overall_method': 'Unknown',
        'persistence_levels': set(),
        'dof_values': set(),
        'avg_coverage_95': None,
        'avg_logscore': None,
        'best_by_break_type': {},  # {'mean': {'rmse': X, 'method': Y, 'dof': Z}, ...}
        'best_by_persistence': {},  # {'p0.90': {'rmse': X, 'method': Y}, ...}
    }
    
    csv_files = list(RESULTS_DIR.glob('**/*.csv'))
    if not csv_files:
        return summary
    
    try:
        coverage_95_list = []
        logscore_list = []
        break_type_data = {'mean': [], 'parameter': [], 'variance': []}
        persistence_data = {'p0.90': [], 'p0.95': [], 'p0.99': []}
        
        # Get n_sim from first CSV
        first_df = pd.read_csv(csv_files[0])
        if 'N' in first_df.columns:
            summary['n_sim'] = int(first_df['N'].iloc[0])
        
        # Process all CSV files
        for csv_path in csv_files:
            fname = csv_path.stem.lower()
            df = pd.read_csv(csv_path)
            summary['total_results'] += len(df)
            
            # Extract persistence levels from filename
            if 'recurring' in fname:
                if 'p09' in fname or '_p0.9_' in fname:
                    summary['persistence_levels'].add('0.90')
                    persistence_data['p0.90'].extend(df['RMSE'].dropna().tolist() if 'RMSE' in df.columns else [])
                if 'p095' in fname or '_p0.95_' in fname:
                    summary['persistence_levels'].add('0.95')
                    persistence_data['p0.95'].extend(df['RMSE'].dropna().tolist() if 'RMSE' in df.columns else [])
                if 'p099' in fname or '_p0.99_' in fname:
                    summary['persistence_levels'].add('0.99')
                    persistence_data['p0.99'].extend(df['RMSE'].dropna().tolist() if 'RMSE' in df.columns else [])
            
            # Extract DOF from filename
            dof = None
            if 'gaussian' in fname:
                dof = 'Gaussian'
                summary['dof_values'].add('Gaussian')
            if 'tdf5' in fname or 'student-t(df=5)' in fname or 't_df5' in fname:
                dof = 'Student-t(df=5)'
                summary['dof_values'].add('5')
            if 'tdf3' in fname or 'student-t(df=3)' in fname or 't_df3' in fname:
                dof = 'Student-t(df=3)'
                summary['dof_values'].add('3')
            
            # Extract break type
            break_type = None
            if 'mean_' in fname:
                break_type = 'mean'
            elif 'parameter_' in fname:
                break_type = 'parameter'
            elif 'variance_' in fname:
                break_type = 'variance'
            
            # Find best RMSE overall
            if 'RMSE' in df.columns:
                min_rmse_idx = df['RMSE'].idxmin()
                current_rmse = df['RMSE'].iloc[min_rmse_idx]
                if current_rmse < summary['best_overall_rmse']:
                    summary['best_overall_rmse'] = current_rmse
                    summary['best_overall_method'] = df['Method'].iloc[min_rmse_idx]
                
                # Track best by break type
                if break_type:
                    break_type_data[break_type].append({
                        'rmse': current_rmse,
                        'method': df['Method'].iloc[min_rmse_idx],
                        'dof': dof or 'Unknown',
                        'all_rmses': df['RMSE'].dropna().tolist()
                    })
            
            # Collect coverage and logscore
            if 'Coverage95' in df.columns:
                coverage_95_list.extend(df['Coverage95'].dropna().tolist())
            if 'LogScore' in df.columns:
                logscore_list.extend(df['LogScore'].dropna().tolist())
        
        # Calculate best by break type
        for break_type in ['mean', 'parameter', 'variance']:
            if break_type_data[break_type]:
                best_item = min(break_type_data[break_type], key=lambda x: x['rmse'])
                summary['best_by_break_type'][break_type] = {
                    'rmse': best_item['rmse'],
                    'method': best_item['method'],
                    'dof': best_item['dof']
                }
        
        # Calculate best by persistence level
        for level in ['p0.90', 'p0.95', 'p0.99']:
            if persistence_data[level]:
                avg_rmse = sum(persistence_data[level]) / len(persistence_data[level])
                summary['best_by_persistence'][level] = {
                    'rmse': min(persistence_data[level]),
                    'avg_rmse': avg_rmse
                }
        
        # Calculate averages
        if coverage_95_list:
            summary['avg_coverage_95'] = sum(coverage_95_list) / len(coverage_95_list)
        if logscore_list:
            summary['avg_logscore'] = sum(logscore_list) / len(logscore_list)
    
    except Exception as e:
        print(f"Warning: Could not extract summary statistics: {e}")
    
    return summary


def generate_executive_summary_latex(summary_stats):
    """
    Generate LaTeX code for executive summary section based on extracted statistics.
    """
    # Format numbers
    best_rmse_str = f"{summary_stats['best_overall_rmse']:.4f}" if summary_stats['best_overall_rmse'] != float('inf') else "N/A"
    persistence_str = ", ".join(sorted(summary_stats['persistence_levels'])) if summary_stats['persistence_levels'] else "N/A"
    dof_str = ", ".join(sorted(summary_stats['dof_values'])) if summary_stats['dof_values'] else "N/A"
    coverage_str = f"{summary_stats['avg_coverage_95']:.4f}" if summary_stats['avg_coverage_95'] is not None else "N/A"
    logscore_str = f"{summary_stats['avg_logscore']:.4f}" if summary_stats['avg_logscore'] is not None else "N/A"
    
    # Build key findings
    key_findings = []
    
    # Best overall method
    key_findings.append(f"\\item \\textbf{{Best Overall Method:}} {summary_stats['best_overall_method']} achieves lowest RMSE of {best_rmse_str}")
    
    # Best by break type
    for break_type in ['mean', 'parameter', 'variance']:
        if break_type in summary_stats['best_by_break_type']:
            info = summary_stats['best_by_break_type'][break_type]
            key_findings.append(f"\\item \\textbf{{{break_type.title()} Break Performance:}} Best RMSE = {info['rmse']:.4f} ({info['dof']})")
    
    # Best by persistence level
    if summary_stats['best_by_persistence']:
        for level, info in sorted(summary_stats['best_by_persistence'].items()):
            key_findings.append(f"\\item \\textbf{{Persistence {level}:}} Best RMSE = {info['rmse']:.4f}")
    
    # Coverage and logscore if available
    if summary_stats['avg_coverage_95'] is not None:
        key_findings.append(f"\\item \\textbf{{Predictive Performance:}} Average Coverage@95\\%: {coverage_str}, Average LogScore: {logscore_str}")
    
    key_findings_str = "\n".join(key_findings) if key_findings else "\\item \\textbf{Analysis Status:} Results compiled from all available simulations"
    
    # Build LaTeX executive summary
    exec_summary = r"""
\textbf{Study Overview:}
\begin{itemize}
\item \textbf{Simulation Design:} Monte Carlo experiments with """ + str(summary_stats['n_sim']) + r""" replications per scenario
\item \textbf{Time Series Length:} T = 400 observations, break point Tb = 200
\item \textbf{Break Types Analyzed:} Variance, Mean, and Parameter breaks
\item \textbf{Innovation Types:} """ + dof_str + r"""
\item \textbf{Persistence Levels:} """ + persistence_str + r"""
\item \textbf{Total Scenarios:} """ + str(summary_stats['total_results']) + r""" forecast results across 3 break types
\end{itemize}

\textbf{Key Findings:}
\begin{itemize}
""" + key_findings_str + r"""
\end{itemize}
"""
    
    return exec_summary


def create_table_pdf_from_tex(tables_dict, output_path):
    """
    Create PDF from LaTeX table files.
    Requires pdflatex or xelatex.
    """
    import subprocess
    import tempfile
    
    # Reorganize tables in hierarchical order
    tables_dict = sort_tables_hierarchical(tables_dict)
    
    # Extract summary statistics dynamically from all CSV files
    summary_stats = extract_summary_statistics_from_tables()
    
    # Generate executive summary LaTeX
    exec_summary_latex = generate_executive_summary_latex(summary_stats)
    
    # Create a master LaTeX file with DYNAMIC content
    latex_content = r"""
\documentclass[12pt]{article}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\geometry{margin=0.75in, headheight=15pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\textit{Analysis Results - Structural Breaks}}
\fancyfoot[C]{\thepage}

% Reduce spacing in tables
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{0.9}

\title{\Large\textbf{Structural Break Analysis Results}}
\author{}
\date{""" + DATE_READABLE + r"""}

\begin{document}
\maketitle
\vspace{0.3cm}

\section*{Executive Summary}

""" + exec_summary_latex + r"""

\vspace{0.5cm}
\newpage

"""
    
    # Organize tables by break type in specific order: mean, parameter, variance
    for break_type in ['mean', 'parameter', 'variance']:
        tables = tables_dict.get(break_type, [])
        if not tables:
            continue
        
        # Main section
        title = break_type.title()
        latex_content += f"\n\\section{{{title} Break}}\n\n"
        
        # Separate single and recurring
        single_tables = [t for t in tables if 'single' in t.name]
        recurring_tables = [t for t in tables if 'recurring' in t.name or 'p0' in t.name]
        
        # Subsection 1: Single Break
        if single_tables:
            latex_content += f"\\subsection{{Single Break}}\n\n"
            for table_path in sorted(single_tables):
                try:
                    with open(table_path, 'r') as f:
                        latex_content += f.read()
                        latex_content += "\n"
                except Exception as e:
                    pass
        
        # Subsection 2: Recurring Break (or Persistence for parameter)
        if recurring_tables:
            if break_type == 'parameter':
                subsection_title = "Persistence Results"
            else:
                subsection_title = "Recurring Break"
            
            latex_content += f"\\subsection{{{subsection_title}}}\n\n"
            for table_path in sorted(recurring_tables):
                try:
                    with open(table_path, 'r') as f:
                        latex_content += f.read()
                        latex_content += "\n"
                except Exception as e:
                    pass
    
    # Add combined results at the end with page break
    other_tables = tables_dict.get('other', [])
    if other_tables:
        latex_content += r"\newpage" + "\n\n"
        latex_content += r"\section{Combined Results}" + "\n\n"
        for table_path in other_tables:
            try:
                with open(table_path, 'r') as f:
                    content = f.read()
                    # Remove table wrapper to allow longtable page breaks
                    content = content.replace(r'\begin{table}', '')
                    content = content.replace(r'\end{table}', '')
                    content = content.replace(r'\caption{Complete Structural Break Forecasting Results}', r'\caption{Complete Structural Break Forecasting Results}\\')
                    content = content.replace(r'\label{tab:all_results}', '')
                    
                    # Replace \begin{tabular} with \begin{longtable} to allow page breaks
                    content = content.replace(r'\begin{tabular}', r'\begin{longtable}')
                    content = content.replace(r'\end{tabular}', r'\end{longtable}')
                    
                    # Remove Successes and Failures columns from header and data
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        # Remove "Successes & Failures &" from header
                        line = line.replace(' Successes & Failures &', '')
                        # Remove the corresponding data columns (they're typically NaN values)
                        # Split by & and reconstruct without the successes/failures columns
                        if '&' in line and 'Successes' not in line and 'Failures' not in line:
                            # Count & to identify column positions
                            parts = line.split('&')
                            if len(parts) >= 11:  # If we have enough columns
                                # Remove columns 9 and 10 (Successes and Failures, 0-indexed)
                                parts_new = parts[:8] + parts[10:]
                                line = '&'.join(parts_new)
                        new_lines.append(line)
                    
                    content = '\n'.join(new_lines)
                    latex_content += content
                    latex_content += "\n\n"
            except Exception as e:
                print(f"Error processing combined results: {e}")
    
    latex_content += r"\end{document}"
    
    # Write to temp file and compile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tex_file = tmpdir / 'tables.tex'
        
        with open(tex_file, 'w') as f:
            f.write(latex_content)
        
        # Try to compile with pdflatex
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(tmpdir), str(tex_file)],
                capture_output=True,
                timeout=30
            )
            
            pdf_src = tmpdir / 'tables.pdf'
            if pdf_src.exists():
                shutil.copy(pdf_src, output_path)
                return True
            else:
                print(f"  ⚠ PDF file not created. pdflatex output:")
                print(result.stdout.decode('utf-8', errors='ignore')[-500:])
                print(result.stderr.decode('utf-8', errors='ignore')[-500:])
        except Exception as e:
            print(f"  ⚠ pdflatex not available or failed: {e}")
    
    return False


def compile_combined():
    """Compile tables and figures into a single comprehensive PDF."""
    import subprocess
    import tempfile
    
    print("\n" + "="*70)
    print("COMPILING TABLES AND FIGURES INTO SINGLE PDF")
    print("="*70 + "\n")
    
    tables = organize_tables_by_type()
    figs = organize_figures_by_type()
    
    # Count files
    total_tables = sum(len(tables[bt]) for bt in tables)
    total_figs = sum(len(figs[bt][tier]) for bt in figs for tier in figs[bt])
    
    print(f"Found {total_tables} tables and {total_figs} figures")
    
    if total_tables == 0:
        print("✗ No tables found")
        return
    
    # Extract summary statistics dynamically from all CSV files
    summary_stats = extract_summary_statistics_from_tables()
    
    # Generate executive summary LaTeX
    exec_summary_latex = generate_executive_summary_latex(summary_stats)
    
    # Create LaTeX document with tables and figures
    latex_content = r"""
\documentclass[12pt]{article}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\geometry{margin=0.75in, headheight=15pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\textit{Analysis Results - Structural Breaks}}
\fancyfoot[C]{\thepage}

\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{0.9}

\title{\Large\textbf{Structural Break Analysis Results}}
\author{}
\date{""" + DATE_READABLE + r"""}

\begin{document}
\maketitle
\vspace{0.3cm}

\section*{Executive Summary}

""" + exec_summary_latex + r"""

\vspace{0.5cm}
\newpage

"""
    
    # Add Tables - properly ordered by break type
    section_num = 1
    for break_type in ['mean', 'parameter', 'variance']:
        tables_list = tables.get(break_type, [])
        if not tables_list:
            continue
        
        title = break_type.title()
        latex_content += f"\n\\section{{{section_num}. {title} Break}}\n\n"
        
        single_tables = [t for t in tables_list if 'single' in t.name]
        recurring_tables = [t for t in tables_list if 'recurring' in t.name or 'p0' in t.name]
        
        subsection_num = 1
        if single_tables:
            latex_content += f"\\subsection{{{section_num}.{subsection_num} Single Break}}\n\n"
            for table_path in sorted(single_tables):
                try:
                    with open(table_path, 'r') as f:
                        latex_content += f.read()
                        latex_content += "\n"
                except:
                    pass
            subsection_num += 1
        
        if recurring_tables:
            subsection_title = "Persistence Results" if break_type == 'parameter' else "Recurring Break"
            latex_content += f"\\subsection{{{section_num}.{subsection_num} {subsection_title}}}\n\n"
            for table_path in sorted(recurring_tables):
                try:
                    with open(table_path, 'r') as f:
                        latex_content += f.read()
                        latex_content += "\n"
                except:
                    pass
        
        section_num += 1
    
    # Add Combined Results (Table 15 under Section 4)
    other_tables = tables.get('other', [])
    if other_tables:
        latex_content += r"\section{4. Combined Results}" + "\n\n"
        for table_path in other_tables:
            try:
                with open(table_path, 'r') as f:
                    content = f.read()
                    content = content.replace(r'\begin{table}', '')
                    content = content.replace(r'\end{table}', '')
                    content = content.replace(r'\caption{Complete Structural Break Forecasting Results}', r'\caption{Complete Structural Break Forecasting Results}\\')
                    content = content.replace(r'\label{tab:all_results}', '')
                    content = content.replace(r'\begin{tabular}', r'\begin{longtable}')
                    content = content.replace(r'\end{tabular}', r'\end{longtable}')
                    
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        line = line.replace(' Successes & Failures &', '')
                        if '&' in line and 'Successes' not in line and 'Failures' not in line:
                            parts = line.split('&')
                            if len(parts) >= 11:
                                parts_new = parts[:8] + parts[10:]
                                line = '&'.join(parts_new)
                        new_lines.append(line)
                    
                    content = '\n'.join(new_lines)
                    latex_content += content
                    latex_content += "\n\n"
            except:
                pass
    
    # Add Figures section at the END (Section 5)
    if total_figs > 0:
        latex_content += r"\newpage" + "\n\n"
        latex_content += r"\section{5. Analysis Plots}" + "\n\n"
        latex_content += r"{\small" + "\n"
        
        section_num = 5
        subsection_num = 1
        for break_type in ['mean', 'parameter', 'variance']:
            if figs[break_type]['tier1'] or figs[break_type]['tier2']:
                latex_content += f"\n\\subsection{{{section_num}.{subsection_num} {break_type.title()} Break}}\n\n"
                
                # Tier 1 plots
                if figs[break_type]['tier1']:
                    latex_content += "\\textbf{Tier 1 - Method Comparison}\n\n"
                    for fig_path in figs[break_type]['tier1'][:3]:
                        try:
                            latex_content += f"\\includegraphics[width=5.5cm]{{{fig_path}}}\n\n"
                        except:
                            pass
                
                # Tier 2 plots
                if figs[break_type]['tier2']:
                    latex_content += "\\textbf{Tier 2 - DGP Visualization}\n\n"
                    for fig_path in figs[break_type]['tier2'][:2]:
                        try:
                            latex_content += f"\\includegraphics[width=5.5cm]{{{fig_path}}}\n\n"
                        except:
                            pass
                
                subsection_num += 1
        
        latex_content += r"}" + "\n"
    
    latex_content += r"\end{document}"
    
    # Compile with pdflatex
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tex_file = tmpdir / 'combined.tex'
        
        with open(tex_file, 'w') as f:
            f.write(latex_content)
        
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(tmpdir), str(tex_file)],
                capture_output=True,
                timeout=60
            )
            
            pdf_src = tmpdir / 'combined.pdf'
            if pdf_src.exists():
                output_name = f"Complete_Analysis_{TIMESTAMP}.pdf"
                output_path = COMPILATIONS_DIR / output_name
                shutil.copy(pdf_src, output_path)
                size_mb = output_path.stat().st_size / (1024*1024)
                print(f"✓ Successfully created: {output_path}")
                print(f"  Size: {size_mb:.2f} MB")
                print(f"  Location: {output_path}")
                return True
            else:
                print(f"⚠ PDF file not created. pdflatex output:")
                print(result.stdout.decode('utf-8', errors='ignore')[-500:])
        except Exception as e:
            print(f"  ⚠ pdflatex failed: {e}")
    
    return False


def list_recent_files():
    """List recent figures and tables."""
    print("\n" + "="*70)
    print("RECENT FIGURES")
    print("="*70)
    
    figs = organize_figures_by_type()
    for break_type in ['variance', 'mean', 'parameter']:
        for tier in ['tier1', 'tier2']:
            files = figs[break_type][tier]
            if files:
                print(f"\n{break_type.upper()} - {tier.upper()}:")
                for f in files[:5]:
                    print(f"  {f.name}")
    
    print("\n" + "="*70)
    print("RECENT TABLES")
    print("="*70)
    
    tables = organize_tables_by_type()
    for break_type in ['variance', 'mean', 'parameter', 'other']:
        files = tables[break_type]
        if files:
            print(f"\n{break_type.upper()}:")
            for f in files[:5]:
                print(f"  {f.name}")


# =========================================================
# MAIN COMPILATION FUNCTIONS
# =========================================================

def compile_figures():
    """Compile latest figures into PDF."""
    print("\n" + "="*70)
    print("COMPILING FIGURES INTO PDF")
    print("="*70 + "\n")
    
    figs = organize_figures_by_type()
    
    # Count figures
    total = sum(len(figs[bt][tier]) for bt in figs for tier in figs[bt])
    print(f"Found {total} figures")
    
    for break_type in ['variance', 'mean', 'parameter']:
        count = sum(len(figs[break_type][tier]) for tier in figs[break_type])
        if count > 0:
            print(f"  {break_type}: {count} figures")
    
    if total == 0:
        print("✗ No figures found in figures/ directory")
        return
    
    output_name = f"Figures_Tier1-2_{TIMESTAMP}.pdf"
    output_path = COMPILATIONS_DIR / output_name
    
    print(f"\nCreating PDF: {output_name}")
    
    if create_figure_pdf_native(figs, output_path):
        size_mb = output_path.stat().st_size / (1024*1024)
        print(f"✓ Successfully created: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Location: {output_path}")
    else:
        print(f"✗ Failed to create PDF")


def compile_tables():
    """Compile latest tables into PDF."""
    print("\n" + "="*70)
    print("COMPILING TABLES INTO PDF")
    print("="*70 + "\n")
    
    # First: Get only latest CSV files (avoid duplicates)
    print("Converting CSV → LaTeX (using latest files only)...")
    latest_csv = get_latest_tables()
    print(f"Found {len(latest_csv)} latest CSV tables")
    
    # Convert to LaTeX
    convert_csv_to_latex()
    
    # Organize using only the deduplicated CSV files
    tables = organize_tables_by_type(csv_files=latest_csv)
    
    # Count tables
    total = sum(len(tables[bt]) for bt in tables)
    print(f"Found {total} tables")
    
    for break_type in ['variance', 'mean', 'parameter', 'other']:
        count = len(tables[break_type])
        if count > 0:
            print(f"  {break_type}: {count} tables")
    
    if total == 0:
        print("✗ No tables found in outputs/tables/ directory")
        return
    
    output_name = f"Tables_Results_{TIMESTAMP}.pdf"
    output_path = COMPILATIONS_DIR / output_name
    
    print(f"\nCreating PDF: {output_name}")
    
    if create_table_pdf_from_tex(tables, output_path):
        size_mb = output_path.stat().st_size / (1024*1024)
        print(f"✓ Successfully created: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Location: {output_path}")
    else:
        print(f"⚠ Could not create table PDF (pdflatex may not be installed)")
        print(f"  Install with: sudo apt install texlive-latex-base")


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compile latest figures and tables into PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Compile figures only
  %(prog)s --figures
  
  # Compile tables only
  %(prog)s --tables
  
  # Compile combined document (tables + figures)
  %(prog)s --combined
  
  # Compile all three (default)
  %(prog)s --all
  
  # List recent files
  %(prog)s --list
  
OUTPUT LOCATION:
  PDFs are saved to: outputs/pdf/
  Format: Figures_Tier1-2_YYYYMMDD_HHMMSS.pdf
  Format: Tables_Results_YYYYMMDD_HHMMSS.pdf
  Format: Complete_Analysis_YYYYMMDD_HHMMSS.pdf (combined tables + figures)
"""
    )
    
    parser.add_argument('--figures', action='store_true', 
                       help='Compile figures only')
    parser.add_argument('--tables', action='store_true',
                       help='Compile tables only')
    parser.add_argument('--combined', action='store_true',
                       help='Compile combined PDF with tables and figures')
    parser.add_argument('--all', action='store_true',
                       help='Compile figures, tables, and combined (default)')
    parser.add_argument('--list', action='store_true',
                       help='List recent figures and tables')
    
    args = parser.parse_args()
    
    # Handle list
    if args.list:
        list_recent_files()
        return
    
    # Determine what to compile
    if args.all or not (args.figures or args.tables or args.combined):
        do_figures = True
        do_tables = True
        do_combined = True
    else:
        do_figures = args.figures
        do_tables = args.tables
        do_combined = args.combined
    
    # Check what files exist
    tables_exist = any(RESULTS_DIR.glob('**/*.csv'))
    figures_exist = any(FIGURES_DIR.glob('**/*.png'))
    
    # Smart validation: tell user what exists, what's missing, and what to do
    print("\n" + "="*70)
    print("FILE STATUS CHECK")
    print("="*70)
    
    status_msg = []
    if tables_exist:
        latest_tables = sorted(RESULTS_DIR.glob('**/*.csv'))[-5:]  # Show 5 latest
        status_msg.append(f"✓ Tables found ({len(list(RESULTS_DIR.glob('**/*.csv')))} files)")
    else:
        status_msg.append("✗ No tables found")
    
    if figures_exist:
        latest_figures = sorted(FIGURES_DIR.glob('**/*.png'))[-5:]
        status_msg.append(f"✓ Figures found ({len(list(FIGURES_DIR.glob('**/*.png')))} files)")
    else:
        if do_figures or do_combined:
            status_msg.append("✗ No figures found (plots generation is disabled)")
    
    for msg in status_msg:
        print(f"  {msg}")
    print()
    
    # Determine if we can proceed
    can_do_tables = tables_exist
    can_do_figures = figures_exist
    can_do_combined = tables_exist and figures_exist
    
    # Check if requested tasks can be completed
    if (do_tables and not can_do_tables) or (do_figures and not can_do_figures) or (do_combined and not can_do_combined):
        print("⚠ Missing files for requested compilation:\n")
        
        if do_tables and not can_do_tables:
            print("  • Tables required but not found")
            print("    → Run: pixi run python main.py --quick  (30 runs)")
            print("    → Or:  pixi run python main.py         (300 runs)\n")
        
        if do_figures and not can_do_figures:
            print("  • Figures required but not available")
            print("    → Run: pixi run python scripts/generate_plots.py --all\n")
        
        if do_combined and not can_do_combined:
            print("  • Combined PDF requires both tables and figures")
            if not tables_exist:
                print("    • Tables missing → pixi run python main.py")
            if not figures_exist:
                print("    • Figures missing → pixi run python scripts/generate_plots.py --all")
            print()
        
        return
    
    print(f"Compilation started at: {DATE_READABLE}")
    print(f"Output directory: {COMPILATIONS_DIR}\n")
    
    if do_figures:
        compile_figures()
    
    if do_tables:
        compile_tables()
    
    if do_combined:
        compile_combined()
    
    # Show where files were saved
    print(f"\n" + "="*70)
    print("Compilation complete!")
    print("="*70)
    print(f"\nAll PDFs saved to: {COMPILATIONS_DIR}")
    print("\nYou can find your files:")
    pdf_files = sorted(COMPILATIONS_DIR.glob('*.pdf'), key=lambda x: x.stat().st_mtime, reverse=True)
    if pdf_files:
        for pdf in pdf_files[:3]:  # Show 3 most recent
            size_mb = pdf.stat().st_size / (1024*1024)
            print(f"  • {pdf.name} ({size_mb:.2f} MB)")
    print()


if __name__ == '__main__':
    main()
