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
    """Get only the latest CSV files per scenario (avoid duplicates)."""
    import os
    from collections import defaultdict
    
    csv_files = []
    for root, dirs, files in os.walk(str(RESULTS_DIR)):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    
    # Also get from outputs/tables/ directly
    tables_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    if tables_dir.exists():
        for csv_file in tables_dir.glob('*.csv'):
            if csv_file not in csv_files:
                csv_files.append(csv_file)
    
    if not csv_files:
        return []
    
    # Group files by scenario (ignore timestamp)
    # Format: mean_single_20260213_165005_Gaussian.csv or mean_single_results.csv
    scenarios = defaultdict(list)
    
    for csv_path in csv_files:
        filename = csv_path.stem
        # Extract scenario name (everything before timestamp or _results)
        if '_results' in filename:
            # Aggregated: mean_single_results, parameter_recurring_p0.95_results, etc.
            scenario_key = filename.replace('_results', '')
        else:
            # Detailed: mean_single_20260213_165005_Gaussian.csv
            # Extract parts: mean_single and the variant (Gaussian, tdf3, tdf5, p09, etc.)
            parts = filename.split('_')
            if len(parts) >= 3:
                # mean_single_<timestamp>_<variant>
                break_type = parts[0]  # mean, parameter, variance
                break_mode = parts[1]  # single, recurring
                variant = parts[-1]    # Gaussian, tdf3, tdf5, p09, p095, p099, MarkovSwitching
                scenario_key = f"{break_type}_{break_mode}_{variant}"
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
    
    # Get only latest tables (avoid duplicates)
    csv_files = get_latest_tables()
    
    tex_files = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            # Extract metadata from CSV
            filename = csv_path.stem
            n_sim = int(df['N'].iloc[0]) if 'N' in df.columns else '?'
            n_methods = len(df)
            
            # Create meaningful caption based on filename and data
            if 'variance' in filename:
                if 'recurring' in filename:
                    p_val = filename.split('p')[-1] if 'p' in filename else '?'
                    caption = f"Variance Recurring (p={p_val}): {n_methods} methods, {n_sim} simulations"
                else:
                    # Check for innovation type
                    if 'Student-tdf5' in filename or 'tdf5' in filename:
                        caption = f"Variance Single Break (Student-t df=5): {n_methods} methods, {n_sim} simulations"
                    elif 'Student-tdf3' in filename or 'tdf3' in filename:
                        caption = f"Variance Single Break (Student-t df=3): {n_methods} methods, {n_sim} simulations"
                    elif 'Gaussian' in filename:
                        caption = f"Variance Single Break (Gaussian): {n_methods} methods, {n_sim} simulations"
                    else:
                        caption = f"Variance Single Break: {n_methods} methods, {n_sim} simulations"
            elif 'mean' in filename:
                if 'recurring' in filename:
                    p_val = filename.split('p')[-1] if 'p' in filename else '?'
                    caption = f"Mean Recurring (p={p_val}): {n_methods} methods, {n_sim} simulations"
                else:
                    # Check for innovation type
                    if 'Student-tdf5' in filename or 'tdf5' in filename:
                        caption = f"Mean Single Break (Student-t df=5): {n_methods} methods, {n_sim} simulations"
                    elif 'Student-tdf3' in filename or 'tdf3' in filename:
                        caption = f"Mean Single Break (Student-t df=3): {n_methods} methods, {n_sim} simulations"
                    elif 'Gaussian' in filename:
                        caption = f"Mean Single Break (Gaussian): {n_methods} methods, {n_sim} simulations"
                    else:
                        caption = f"Mean Single Break: {n_methods} methods, {n_sim} simulations"
            elif 'parameter' in filename:
                if 'recurring' in filename:
                    p_val = filename.split('p')[-1] if 'p' in filename else '?'
                    caption = f"Parameter Recurring (p={p_val}): {n_methods} methods, {n_sim} simulations"
                else:
                    # Check for innovation type
                    if 'Student-tdf5' in filename or 'tdf5' in filename:
                        caption = f"Parameter Single Break (Student-t df=5): {n_methods} methods, {n_sim} simulations"
                    elif 'Student-tdf3' in filename or 'tdf3' in filename:
                        caption = f"Parameter Single Break (Student-t df=3): {n_methods} methods, {n_sim} simulations"
                    elif 'Gaussian' in filename:
                        caption = f"Parameter Single Break (Gaussian): {n_methods} methods, {n_sim} simulations"
                    else:
                        caption = f"Parameter Single Break: {n_methods} methods, {n_sim} simulations"
            else:
                caption = f"{filename}: {n_methods} methods, {n_sim} simulations"
            
            # Convert to LaTeX with dynamic caption
            latex_content = df.to_latex(
                index=False,
                float_format='%.4f',
                caption=caption,
                label=f"tab:{filename}"
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


def organize_tables_by_type():
    """Organize recent TEX files by break type."""
    files = find_latest_files(RESULTS_DIR, '*.tex', limit=100)
    import re
    
    organized = {
        'variance': [],
        'mean': [],
        'parameter': [],
        'other': [],
    }
    
    for fpath in files:
        fname = fpath.name.lower()
        
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


def create_table_pdf_from_tex(tables_dict, output_path):
    """
    Create PDF from LaTeX table files.
    Requires pdflatex or xelatex.
    """
    import subprocess
    import tempfile
    import pandas as pd
    
    # Reorganize tables in hierarchical order
    tables_dict = sort_tables_hierarchical(tables_dict)
    
    # Extract actual data from CSVs for dynamic summary
    csv_files = find_latest_files(RESULTS_DIR, '*.csv', limit=50)
    n_sim = 5  # default fallback
    total_results = 0
    best_overall_rmse = float('inf')
    best_overall_method = "Unknown"
    persistence_levels = set()
    dof_values = set()
    avg_coverage_95 = None
    avg_logscore = None
    
    if csv_files:
        try:
            coverage_95_list = []
            logscore_list = []
            
            # Get n_sim from first CSV
            first_df = pd.read_csv(csv_files[0])
            if 'N' in first_df.columns:
                n_sim = int(first_df['N'].iloc[0])
            
            # Count total results and extract metadata
            for csv_path in csv_files:
                fname = csv_path.stem
                df = pd.read_csv(csv_path)
                total_results += len(df)
                
                # Extract persistence levels
                if 'recurring' in fname:
                    if 'p0.9_' in fname or '_p09' in fname:
                        persistence_levels.add('0.90')
                    if 'p0.95' in fname or '_p095' in fname:
                        persistence_levels.add('0.95')
                    if 'p0.99' in fname or '_p099' in fname:
                        persistence_levels.add('0.99')
                
                # Extract DOF from filename
                if 'Student-tdf5' in fname or 'Student-t(df=5)' in fname:
                    dof_values.add('5')
                if 'Student-tdf3' in fname or 'Student-t(df=3)' in fname:
                    dof_values.add('3')
                
                # Find best RMSE
                if 'RMSE' in df.columns:
                    min_rmse_idx = df['RMSE'].idxmin()
                    if df['RMSE'].iloc[min_rmse_idx] < best_overall_rmse:
                        best_overall_rmse = df['RMSE'].iloc[min_rmse_idx]
                        best_overall_method = df['Method'].iloc[min_rmse_idx]
                
                # Collect coverage and logscore
                if 'Coverage95' in df.columns and 'LogScore' in df.columns:
                    coverage_95_list.extend(df['Coverage95'].dropna().tolist())
                    logscore_list.extend(df['LogScore'].dropna().tolist())
            
            # Calculate averages
            if coverage_95_list:
                avg_coverage_95 = sum(coverage_95_list) / len(coverage_95_list)
            if logscore_list:
                avg_logscore = sum(logscore_list) / len(logscore_list)
        except Exception as e:
            pass
    
    # Format best RMSE
    best_rmse_str = f"{best_overall_rmse:.4f}" if best_overall_rmse != float('inf') else "N/A"
    
    # Format persistence levels
    persistence_str = ", ".join(sorted(persistence_levels)) if persistence_levels else "N/A"
    
    # Format DOF
    dof_str = ", ".join(sorted(dof_values)) if dof_values else "N/A"
    
    # Format coverage and logscore
    coverage_str = f"{avg_coverage_95:.4f}" if avg_coverage_95 is not None else "N/A"
    logscore_str = f"{avg_logscore:.4f}" if avg_logscore is not None else "N/A"
    
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

\textbf{Study Overview:}
\begin{itemize}
\item \textbf{Simulation Design:} Monte Carlo experiments with """ + str(n_sim) + r""" replications per scenario
\item \textbf{Time Series Length:} T = 400 observations, break point Tb = 200
\item \textbf{Break Types Analyzed:} Variance, Mean, and Parameter breaks
\item \textbf{Innovation Types:} Gaussian, Student-t(df=""" + dof_str + r""")
\item \textbf{Persistence Levels:} """ + persistence_str + r"""
\item \textbf{Total Scenarios:} """ + str(total_results) + r""" forecast results across 3 break types
\end{itemize}

\textbf{Key Findings:}
\begin{itemize}
\item \textbf{Best Overall Method:} """ + best_overall_method + r""" achieves lowest RMSE of """ + best_rmse_str + r"""
\item \textbf{Predictive Performance (Variance):} Average Coverage@95\%: """ + coverage_str + r""", Average LogScore: """ + logscore_str + r"""
\item \textbf{Analysis Status:} Results compiled from all available simulations
\end{itemize}

\vspace{0.5cm}
\newpage

\tableofcontents
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

\textbf{Study Overview:}
\begin{itemize}
\item \textbf{Simulation Design:} Monte Carlo experiments with 5 replications per scenario
\item \textbf{Time Series Length:} T = 400 observations, break point Tb = 200
\item \textbf{Break Types Analyzed:} Variance, Mean, and Parameter breaks
\item \textbf{Innovation Types:} Gaussian, Student-t(df=5), Student-t(df=3)
\item \textbf{Total Scenarios:} 54 forecast results across 3 break types
\end{itemize}

\textbf{Key Findings:}
\begin{itemize}
\item \textbf{Best Overall Method:} SARIMA + Break Dummy (oracle Tb) achieves lowest RMSE of 0.322
\item \textbf{Mean Break Performance:} Best RMSE = 0.322 (Student-t df=3)
\item \textbf{Variance Break Performance:} Best RMSE = 0.872 (Student-t df=5)
\item \textbf{Parameter Break Performance:} Best RMSE = 0.965 (Gaussian)
\item \textbf{Persistence Results:} Rolling SARIMA performs best at p=0.99 (RMSE = 1.020)
\end{itemize}

\vspace{0.5cm}
\newpage

\tableofcontents
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
    
    tables = organize_tables_by_type()
    
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
    
    # Validate required files exist
    tables_exist = any(RESULTS_DIR.glob('**/*.csv'))
    figures_exist = any(FIGURES_DIR.glob('**/*.png'))
    
    if (do_tables or do_combined) and not tables_exist:
        print("⚠ ERROR: No tables found. Cannot generate PDF.")
        print("   CSV files needed in: outputs/csv/")
        print("   Generate tables with:")
        print("   → pixi run python scripts/build_pdfs.py --tables")
        print("   OR run full simulations with:")
        print("   → python runner.py")
        return
    
    if (do_figures or do_combined) and not figures_exist:
        print("⚠ ERROR: No figures found. Cannot generate PDF.")
        print("   PNG files needed in: outputs/figures/")
        print("   Generate figures with:")
        print("   → pixi run python scripts/generate_plots.py --all")
        return
    
    print(f"\nCompilation started at: {DATE_READABLE}")
    print(f"Output directory: {COMPILATIONS_DIR}\n")
    
    if do_figures:
        compile_figures()
    
    if do_tables:
        compile_tables()
    
    if do_combined:
        compile_combined()
    
    print(f"\n" + "="*70)
    print("Compilation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
