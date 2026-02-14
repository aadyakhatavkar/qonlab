#!/usr/bin/env python3
"""
Automatic Data Transfer Pipeline: Update Paper Tables with Latest Results
==========================================================================

This script automates the data transfer pipeline:
1. Finds latest CSV result files from experiments
2. Extracts data and formats as LaTeX tables
3. Injects formatted tables into combined_paper.tex
4. Maintains paper structure and readability

Usage:
    python scripts/update_paper_tables.py                # Update all tables
    python scripts/update_paper_tables.py --dry-run      # Show changes without applying
    python scripts/update_paper_tables.py --table variance_single  # Update specific table type

This is called automatically by main.py --pdf or can be run standalone.
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

# Configuration
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / 'outputs' / 'tables'
PAPER_PATH = REPO_ROOT / 'documents' / 'paper' / 'combined_paper.tex'

# Mapping of table types to their patterns in LaTeX
TABLE_CONFIGS = {
    'variance_single_Gaussian': {
        'label': 'tab:variance_single_.*_Gaussian',
        'csv_pattern': '*variance_single*Gaussian*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance', 'Coverage80', 'Coverage95', 'LogScore'],
        'caption_template': 'Variance Single Break (Gaussian): {} simulations',
    },
    'variance_single_Student-tdf3': {
        'label': 'tab:variance_single_.*_Student-tdf3',
        'csv_pattern': '*variance_single*Student-tdf3*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance', 'Coverage80', 'Coverage95', 'LogScore'],
        'caption_template': 'Variance Single Break (Student-t df=3): {} simulations',
    },
    'variance_single_Student-tdf5': {
        'label': 'tab:variance_single_.*_Student-tdf5',
        'csv_pattern': '*variance_single*Student-tdf5*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance', 'Coverage80', 'Coverage95', 'LogScore'],
        'caption_template': 'Variance Single Break (Student-t df=5): {} simulations',
    },
    'variance_recurring_p0.95': {
        'label': 'tab:variance_recurring.*p0.95',
        'csv_pattern': '*variance_recurring*p0.95*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance', 'Coverage80', 'Coverage95', 'LogScore'],
        'caption_template': 'Variance Recurring (p=0.95): {} simulations',
    },
    'mean_single_Gaussian': {
        'label': 'tab:mean_single_.*_Gaussian',
        'csv_pattern': '*mean_single*Gaussian*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance'],
        'caption_template': 'Mean Single Break (Gaussian): {} simulations',
    },
    'mean_single_Student-tdf3': {
        'label': 'tab:mean_single_.*_Student-tdf3',
        'csv_pattern': '*mean_single*Student-tdf3*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance'],
        'caption_template': 'Mean Single Break (Student-t df=3): {} simulations',
    },
    'mean_single_Student-tdf5': {
        'label': 'tab:mean_single_.*_Student-tdf5',
        'csv_pattern': '*mean_single*Student-tdf5*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance'],
        'caption_template': 'Mean Single Break (Student-t df=5): {} simulations',
    },
    'parameter_single_Gaussian': {
        'label': 'tab:parameter_single_.*_Gaussian',
        'csv_pattern': '*parameter_single*Gaussian*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance'],
        'caption_template': 'Parameter Single Break (Gaussian): {} simulations',
    },
    'parameter_single_Student-tdf3': {
        'label': 'tab:parameter_single_.*_Student-tdf3',
        'csv_pattern': '*parameter_single*Student-tdf3*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance'],
        'caption_template': 'Parameter Single Break (Student-t df=3): {} simulations',
    },
    'parameter_single_Student-tdf5': {
        'label': 'tab:parameter_single_.*_Student-tdf5',
        'csv_pattern': '*parameter_single*Student-tdf5*.csv',
        'columns': ['Method', 'RMSE', 'MAE', 'Bias', 'Variance'],
        'caption_template': 'Parameter Single Break (Student-t df=5): {} simulations',
    },
}


def find_latest_csv(pattern):
    """Find the latest CSV file matching the pattern."""
    import glob
    matches = list(RESULTS_DIR.glob(pattern))
    if not matches:
        return None
    # Filter out '_results' files which are aggregates
    matches = [m for m in matches if '_results' not in m.name]
    if not matches:
        return None
    # Return most recently modified
    return max(matches, key=lambda p: p.stat().st_mtime)


def extract_table_data(csv_path, columns):
    """Extract data from CSV and format as LaTeX table rows."""
    try:
        df = pd.read_csv(csv_path)
        
        # Get number of simulations from first row if available
        n_sim = int(df['N'].iloc[0]) if 'N' in df.columns else '300'
        
        # Select only the required columns
        df_display = df[columns].copy()
        
        # Format numeric columns to 4 decimal places
        for col in df_display.columns[1:]:  # Skip method name
            if df_display[col].dtype in ['float64', 'float32']:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
        
        # Convert to LaTeX table rows
        rows = []
        for idx, row in df_display.iterrows():
            row_str = ' & '.join(str(val) for val in row)
            rows.append(row_str + ' \\\\')
        
        return n_sim, rows
    except Exception as e:
        print(f"Error extracting data from {csv_path}: {e}")
        return None, None


def generate_latex_table(table_config, csv_path):
    """Generate a complete LaTeX table for the given data."""
    n_sim, rows = extract_table_data(csv_path, table_config['columns'])
    
    if rows is None:
        return None
    
    # Build the table
    table_lines = [
        '\\begin{table}[H]',
        '\\centering',
        '\\small',
        f"\\caption{{{table_config['caption_template'].format(n_sim)}}}",
        f"\\label{{{table_config['label'].split()[0]}}}",  # Use simplified label
    ]
    
    # Determine column format based on number of columns
    n_cols = len(table_config['columns'])
    col_format = 'l' + 'r' * (n_cols - 1)
    table_lines.append(f'\\begin{{tabular}}{{{col_format}}}')
    table_lines.append('\\toprule')
    
    # Header row
    header = ' & '.join(table_config['columns']) + ' \\\\'
    table_lines.append(header)
    table_lines.append('\\midrule')
    
    # Data rows
    table_lines.extend(rows)
    
    table_lines.append('\\bottomrule')
    table_lines.append('\\end{tabular}')
    table_lines.append('\\normalsize')
    table_lines.append('\\end{table}')
    
    return '\n'.join(table_lines)


def replace_table_in_paper(paper_content, table_key, new_table_latex, dry_run=False):
    """Replace a table in the paper with updated LaTeX."""
    table_config = TABLE_CONFIGS[table_key]
    
    # Find the table using regex pattern
    label_pattern = table_config['label']
    
    # Pattern to match entire table from \begin{table} to \end{table}
    # This is more flexible than relying on exact label matching
    pattern = r'\\begin\{table\}\[H\].*?\\label\{' + label_pattern + r'\}.*?\\end\{table\}'
    
    # Try a more flexible search if regex doesn't work
    if not re.search(pattern, paper_content, re.DOTALL):
        print(f"⚠️  Could not find table with label pattern: {label_pattern}")
        return paper_content
    
    # Replace the table
    updated_content = re.sub(
        pattern,
        new_table_latex,
        paper_content,
        count=1,
        flags=re.DOTALL
    )
    
    if dry_run:
        if updated_content != paper_content:
            print(f"  ✓ Would update {table_key}")
        else:
            print(f"  ✗ No match found for {table_key}")
    else:
        if updated_content != paper_content:
            print(f"  ✓ Updated {table_key}")
        else:
            print(f"  ✗ No match found for {table_key}")
    
    return updated_content


def update_paper_tables(dry_run=False, table_filter=None):
    """Main function to update all paper tables."""
    print("\n" + "="*70)
    print("DATA TRANSFER PIPELINE: Update Paper Tables with Latest Results")
    print("="*70)
    
    # Read current paper
    with open(PAPER_PATH, 'r') as f:
        paper_content = f.read()
    
    updated_content = paper_content
    tables_updated = 0
    tables_skipped = 0
    
    # Process each table configuration
    for table_key, table_config in TABLE_CONFIGS.items():
        # Skip if filter is applied and doesn't match
        if table_filter and table_filter not in table_key:
            continue
        
        print(f"\n📊 Processing: {table_key}")
        
        # Find latest CSV for this table
        csv_path = find_latest_csv(table_config['csv_pattern'])
        
        if csv_path is None:
            print(f"  ⚠️  No CSV file found matching: {table_config['csv_pattern']}")
            tables_skipped += 1
            continue
        
        print(f"  📁 Using: {csv_path.name}")
        
        # Generate LaTeX table
        new_table = generate_latex_table(table_config, csv_path)
        
        if new_table is None:
            print(f"  ✗ Failed to generate LaTeX table")
            tables_skipped += 1
            continue
        
        # Replace in paper
        updated_content = replace_table_in_paper(updated_content, table_key, new_table, dry_run)
        tables_updated += 1
    
    # Write updated paper
    if not dry_run and tables_updated > 0:
        with open(PAPER_PATH, 'w') as f:
            f.write(updated_content)
        print(f"\n✅ Wrote updated paper: {PAPER_PATH}")
    elif dry_run:
        print(f"\n🔍 Dry-run mode: No changes written")
    
    print(f"\n📈 Summary: {tables_updated} updated, {tables_skipped} skipped")
    print("="*70 + "\n")
    
    return 0 if tables_updated > 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description='Update paper tables with latest experimental results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python scripts/update_paper_tables.py              # Update all tables
  python scripts/update_paper_tables.py --dry-run    # Preview changes
  python scripts/update_paper_tables.py --table variance_single  # Update specific type
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without writing to file')
    parser.add_argument('--table', type=str, default=None,
                       help='Only update tables matching this substring (e.g., "variance_single")')
    
    args = parser.parse_args()
    
    return update_paper_tables(dry_run=args.dry_run, table_filter=args.table)


if __name__ == '__main__':
    sys.exit(main())
