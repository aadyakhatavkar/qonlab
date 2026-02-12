#!/usr/bin/env python
"""
Compile all generated LaTeX tables into a single PDF document.
Places PDF in docs/ folder for easy distribution.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import re


def extract_metadata_from_filename(filename):
    """Extract break type and innovation from filename."""
    # Format: {break_type}_{variant}_{timestamp}_{innovation}.tex
    # e.g., variance_single_20260212_232254_Gaussian.tex
    match = re.match(r'(\w+)_(\w+)_\d+_(.+)\.tex', filename)
    if match:
        break_type = match.group(1)  # variance, mean, parameter
        variant = match.group(2)      # single, recurring
        innovation = match.group(3)   # Gaussian, Student-t...
        return break_type, variant, innovation
    return None, None, None


def read_table_content(filepath):
    """Read the table content from a tex file."""
    with open(filepath, 'r') as f:
        return f.read().strip()


def organize_files_by_type():
    """Organize tex files by break type."""
    tex_dir = Path(__file__).parent.parent / "results" / "tex"
    organized = {}
    
    for tex_file in sorted(tex_dir.glob("*.tex")):
        break_type, variant, innovation = extract_metadata_from_filename(tex_file.name)
        if break_type is None:
            continue
        
        key = f"{break_type.title()} ({variant.title()} Breaks)"
        if key not in organized:
            organized[key] = []
        
        organized[key].append((innovation, tex_file))
    
    return organized


def create_master_tex():
    """Create master LaTeX document with all tables."""
    organized = organize_files_by_type()
    
    # Start document
    content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage[hidelinks]{hyperref}
\usepackage{longtable}

\title{Monte Carlo Simulation Results}
\author{qonlab}
\date{\today}

\begin{document}

\maketitle

\tableofcontents
\newpage

"""
    
    # Add sections for each break type
    for section_idx, (section_title, files) in enumerate(sorted(organized.items()), 1):
        # Section header
        content += f"\n\\section{{{section_title}}}\n\n"
        content += f"This section contains results for {section_title.lower()}.\n\n"
        
        # Add each table
        for innovation, filepath in sorted(files):
            table_label = filepath.stem.replace("_", "-")
            subsection = f"{section_title} - {innovation}"
            content += f"\\subsection{{{subsection}}}\n\n"
            
            # Read and include the table
            table_content = read_table_content(filepath)
            content += table_content + "\n\n"
    
    # Add appendix with metadata
    content += r"""
\appendix

\section{Simulation Configuration}

\begin{itemize}
    \item Number of simulations: 300
    \item Time series length: 400
    \item Break point: 200 (single breaks)
    \item Rolling window size: 70
    \item Forecast horizon: 1-step ahead
\end{itemize}

\section{Innovation Types}

\begin{itemize}
    \item \textbf{Gaussian}: Standard normal distribution
    \item \textbf{Student-t(df=5)}: Heavy-tailed with 5 degrees of freedom
    \item \textbf{Student-t(df=3)}: Heavier-tailed with 3 degrees of freedom
\end{itemize}

\section{Methods}

\begin{itemize}
    \item \textbf{GARCH}: Generalized Autoregressive Conditional Heteroscedasticity
    \item \textbf{SARIMA Global}: Full-sample SARIMA fit
    \item \textbf{SARIMA Rolling}: Rolling-window SARIMA forecasts
    \item \textbf{SARIMA Avg-Window}: Average of rolling-window predictions
    \item \textbf{SARIMAX}: SARIMA with exogenous break dummy variable
    \item \textbf{SES}: Simple Exponential Smoothing
    \item \textbf{MS-AR}: Markov-switching autoregressive model
\end{itemize}

\end{document}
"""
    
    return content


def compile_to_pdf(tex_content, output_path):
    """Compile LaTeX content to PDF."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tex_file = tmpdir / "results.tex"
        
        # Write tex file
        with open(tex_file, 'w') as f:
            f.write(tex_content)
        
        # Compile with pdflatex (run twice for TOC)
        for run in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(tmpdir), str(tex_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check for actual errors (pdflatex returns 0 even with warnings)
            # but will have "Fatal error" or "! Error" for real issues
            if "Fatal error" in result.stdout or "! Error" in result.stdout:
                print(f"LaTeX compilation failed on run {run + 1}")
                print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                return False
        
        # Copy PDF to output location
        pdf_file = tmpdir / "results.pdf"
        if pdf_file.exists():
            shutil.copy(pdf_file, output_path)
            return True
        else:
            print("PDF file was not created")
            return False


def main():
    """Main execution."""
    print("=" * 60)
    print("Compiling Results PDF")
    print("=" * 60)
    
    # Check if pdflatex is available
    result = subprocess.run(['which', 'pdflatex'], capture_output=True)
    if result.returncode != 0:
        print("❌ pdflatex not found. Install with: apt-get install texlive-latex-base")
        return False
    
    print("✓ pdflatex found")
    
    # Check tex files exist
    tex_dir = Path(__file__).parent.parent / "results" / "tex"
    tex_files = list(tex_dir.glob("*.tex"))
    
    if not tex_files:
        print("❌ No tex files found in results/tex/")
        return False
    
    print(f"✓ Found {len(tex_files)} tex files")
    
    # Create master tex
    print("\n📝 Creating master LaTeX document...")
    tex_content = create_master_tex()
    
    # Ensure docs directory exists
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = docs_dir / f"simulation_results_{timestamp}.pdf"
    
    print(f"\n📊 Compiling to PDF...")
    print(f"   Output: {output_path}")
    
    success = compile_to_pdf(tex_content, output_path)
    
    if success:
        print(f"\n✅ PDF created successfully!")
        print(f"   Location: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Also create a symlink to latest
        latest_link = docs_dir / "simulation_results_latest.pdf"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_path.name)
        print(f"   Symlink: simulation_results_latest.pdf")
        
        return True
    else:
        print(f"\n❌ PDF compilation failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
