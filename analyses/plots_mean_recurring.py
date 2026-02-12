"""
Mean Recurring Break: Results Plotting
=======================================
Visualization for Markov-switching mean break experiments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_recurring_results(df, figsize=(12, 8)):
    """
    Plot mean recurring break results.
    
    Parameters:
        df: Results DataFrame with columns: Method, RMSE, MAE, Bias, Variance
        figsize: Figure size
    """
    if len(df) == 0:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Mean Recurring Break: Markov-Switching Results", fontsize=14, fontweight='bold')
    
    # RMSE
    ax = axes[0, 0]
    ax.barh(df['Method'], df['RMSE'], color='steelblue', alpha=0.8)
    ax.set_xlabel('RMSE')
    ax.set_title('Root Mean Squared Error')
    ax.grid(axis='x', alpha=0.3)
    
    # MAE
    ax = axes[0, 1]
    ax.barh(df['Method'], df['MAE'], color='coral', alpha=0.8)
    ax.set_xlabel('MAE')
    ax.set_title('Mean Absolute Error')
    ax.grid(axis='x', alpha=0.3)
    
    # Bias
    ax = axes[1, 0]
    ax.barh(df['Method'], df['Bias'], color='lightgreen', alpha=0.8)
    ax.set_xlabel('Bias')
    ax.set_title('Mean Bias')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Variance
    ax = axes[1, 1]
    ax.barh(df['Method'], df['Variance'], color='plum', alpha=0.8)
    ax.set_xlabel('Variance')
    ax.set_title('Forecast Error Variance')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mean_recurring_example(y, Tb, figsize=(12, 6)):
    """
    Plot example Markov-switching mean break time series.
    
    Parameters:
        y: Simulated time series
        Tb: Break point
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    T = len(y)
    ax.plot(y, linewidth=1.5, label='Series', color='navy', alpha=0.7)
    ax.axvline(Tb, color='red', linestyle='--', linewidth=2, label=f'Break point (t={Tb})')
    ax.fill_between([0, Tb], ax.get_ylim()[0], ax.get_ylim()[1], 
                     alpha=0.1, color='blue', label='Regime 0')
    ax.fill_between([Tb, T], ax.get_ylim()[0], ax.get_ylim()[1], 
                     alpha=0.1, color='orange', label='Regime 1 (Markov-switching)')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Example: Mean Break with Markov-Switching')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


__all__ = ['plot_mean_recurring_results', 'plot_mean_recurring_example']
