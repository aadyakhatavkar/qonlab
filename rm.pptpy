"""
Research Module: Monte Carlo Simulation under Structural Breaks
Topic: Mean Change Comparison (ARIMA Global, ARIMA Rolling, Markov Switching)
Scenario: Single and Multiple Structural Breaks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress convergence warnings for cleaner output in VS Code
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
NUM_SIMULATIONS = 20  # Number of Monte Carlo iterations
T = 300               # Total time series length
BURN_IN = 50          # Warm-up period to stabilize initial conditions
ROLLING_WINDOW = 60   # Window size for the adaptive ARIMA Rolling model

def generate_data(scenario='single'):
    """
    Generates time series data with structural breaks in the mean.
    - Single Break: Mean shifts from 0 to 3 at the midpoint.
    - Multiple Breaks: Mean shifts 0 -> 4 -> -2 at intervals of T/3.
    """
    # Create random noise with burn-in
    total_len = T + BURN_IN
    data = np.random.normal(0, 1, total_len)
    
    if scenario == 'single':
        break_point = total_len // 2
        data[break_point:] += 3
        
    elif scenario == 'multiple':
        b1 = total_len // 3
        b2 = 2 * total_len // 3
        data[b1:b2] += 4
        data[b2:] -= 2
        
    # Return data minus the burn-in period
    return data[BURN_IN:]

def evaluate_models(data):
    """
    Fits three competing models and calculates RMSE, MAE, and Bias.
    """
    actual = data
    results = {}

    # --- Method A: ARIMA Global (Stable/Linear) ---
    try:
        model_global = ARIMA(data, order=(1, 0, 0)).fit()
        pred_global = model_global.fittedvalues
    except Exception:
        pred_global = np.zeros(T)

    # --- Method B: ARIMA Rolling (Adaptive/Linear) ---
    pred_rolling = []
    for i in range(T):
        if i < ROLLING_WINDOW:
            # Not enough data for window, use mean of available data
            pred_rolling.append(np.mean(data[:i+1]))
        else:
            # Use a rolling window to estimate the current local mean
            window = data[i-ROLLING_WINDOW:i]
            pred_rolling.append(np.mean(window))
    pred_rolling = np.array(pred_rolling)

    # --- Method C: Markov Switching (Regime Detection/Non-Linear) ---
    try:
        # Fits a 2-regime model with switching mean and variance
        ms_model = MarkovRegression(data, k_regimes=2, switching_variance=True).fit(disp=False)
        pred_ms = ms_model.predict()
    except Exception:
        pred_ms = np.zeros(T)

    # Calculate Evaluation Metrics
    model_preds = {
        'ARIMA Global': pred_global, 
        'ARIMA Rolling': pred_rolling, 
        'Markov Switching': pred_ms
    }
    
    for name, pred in model_preds.items():
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        bias = np.mean(pred - actual)
        results[name] = [rmse, mae, bias]
        
    return results

# --- 2. MAIN MONTE CARLO EXECUTION ---
if __name__ == "__main__":
    scenarios = ['single', 'multiple']
    final_output = []

    print("Starting Monte Carlo Simulation...")
    print("-" * 40)

    for sc in scenarios:
        print(f"Running Scenario: {sc.upper()} BREAK")
        simulation_results = []
        
        for i in range(NUM_SIMULATIONS):
            series = generate_data(scenario=sc)
            metrics = evaluate_models(series)
            
            # Convert dict to DataFrame and transpose
            df_iter = pd.DataFrame(metrics, index=['RMSE', 'MAE', 'Bias']).T
            simulation_results.append(df_iter)
        
        # Aggregate results by taking the mean across all simulations
        summary_df = pd.concat(simulation_results).groupby(level=0).mean()
        summary_df['Scenario'] = sc
        final_output.append(summary_df)
        
        print(summary_df.round(4))
        print(f"Best Method for {sc}: {summary_df['RMSE'].idxmin()}")
        print("-" * 40)

    # --- 3. EXPORT TO CSV ---
    full_report = pd.concat(final_output)
    full_report.to_csv("monte_carlo_results.csv")
    print("Simulation Complete. Results saved to 'monte_carlo_results.csv'.")

    # --- 4. DATA VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(generate_data('single'), color='blue', label='Single Break')
    plt.title("Sample Data: Single Mean Change")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(generate_data('multiple'), color='orange', label='Multiple Breaks')
    plt.title("Sample Data: Multiple Mean Changes")
    plt.legend()
    
    plt.tight_layout()
    plt.show()