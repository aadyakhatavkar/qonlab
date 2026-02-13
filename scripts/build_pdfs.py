#!/usr/bin/env python3
"""
Compile Latest Figures and Tables into PDFs
==============================================
Creates timestamped PDFs of latest figures and tables.
Organized by break type and tier.
Placed in bld/pdf/

Usage:
    python scripts/compile_figures.py --figures      # Latest figures only
    python scripts/compile_figures.py --tables       # Latest tables only
    python scripts/compile_figures.py --all          # Both figures and tables
    python scripts/compile_figures.py --list         # List recent files
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
FIGURES_DIR = REPO_ROOT / 'figures'
RESULTS_DIR = REPO_ROOT / 'bld'
COMPILATIONS_DIR = REPO_ROOT / 'bld' / 'pdf'

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
    """Organize recent TEX files by break type, keeping LATEST run (no old duplicates from different days/runs)."""
    files = find_latest_files(RESULTS_DIR, '*.tex', limit=100)
    import re
    from datetime import datetime, timedelta
    
    organized = {
        'variance': [],
        'mean': [],
        'parameter': [],
        'other': [],
    }
    
    # Find the MOST RECENT file overall to determine the latest run window
    latest_global_time = None
    
    for fpath in files:
        fname = fpath.name
        match = re.search(r'_(\d{8})_(\d{6})', fname)
        if not match:
            continue
        
        try:
            dt = datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
            if latest_global_time is None or dt > latest_global_time:
                latest_global_time = dt
        except:
            continue
    
    if latest_global_time is None:
        return organized
    
    # Include files from the latest run (within 1 hour window of latest file)
    cutoff_time = latest_global_time - timedelta(hours=1)
    
    for fpath in files:
        fname = fpath.name
        match = re.search(r'_(\d{8})_(\d{6})', fname)
        if not match:
            continue
        
        try:
            dt = datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
            if dt >= cutoff_time:
                # This file is from the latest run window
                if 'variance' in fname:
                    organized['variance'].append(fpath)
                elif 'mean' in fname:
                    organized['mean'].append(fpath)
                elif 'parameter' in fname or 'param' in fname:
                    organized['parameter'].append(fpath)
                else:
                    organized['other'].append(fpath)
        except:
            continue
    
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


def create_table_pdf_from_tex(tables_dict, output_path):
    """
    Create PDF from LaTeX table files.
    Requires pdflatex or xelatex.
    """
    import subprocess
    import tempfile
    
    # Create a master LaTeX file
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
    
    tables = organize_tables_by_type()
    
    # Count tables
    total = sum(len(tables[bt]) for bt in tables)
    print(f"Found {total} tables")
    
    for break_type in ['variance', 'mean', 'parameter', 'other']:
        count = len(tables[break_type])
        if count > 0:
            print(f"  {break_type}: {count} tables")
    
    if total == 0:
        print("✗ No tables found in bld/ directory")
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
  PDFs are saved to: bld/pdf/
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
