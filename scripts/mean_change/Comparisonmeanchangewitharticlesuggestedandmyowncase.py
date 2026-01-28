import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- 1. GLOBAL CONFIGURATION ---
T = 300
NUM_SIMULATIONS = 15  # Set to 50 for final submission
BURN_IN = 50

# ---------------------------------------------------------
# STAGE 1: REPLICATION (PESARAN & TIMMERMANN, 2013)
# ---------------------------------------------------------
def generate_literature_data():
    """Strictly follows the Discrete Break logic from the article."""
    data = np.zeros(T + BURN_IN)
    data[:150] = np.random.normal(0, 1, 150)
    data[150:] = 4.0 + np.random.normal(0, 1, (T + BURN_IN - 150))
    return data[BURN_IN:]

# ---------------------------------------------------------
# STAGE 2: MY CASE (YOUR ORIGINAL EXTENSION)
# ---------------------------------------------------------
def generate_my_data():
    """My unique scenario: Volatility shifts + Multiple breaks."""
    data = np.zeros(T + BURN_IN)
    for t in range(1, T + BURN_IN):
        if t < 100: mu, vol = 0, 1.0
        elif t < 200: mu, vol = 5.0, 2.5 # My defined High Volatility
        else: mu = -2.0; vol = 0.8       # My defined Recovery
        data[t] = mu + np.random.normal(0, vol)
    return data[BURN_IN:]

# ---------------------------------------------------------
# STAGE 3: EVALUATION & COMPARISON
# ---------------------------------------------------------
def evaluate_all(data):
    df_p = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=len(data), freq='D'), 'y': data})
    
    # ARIMA (Baseline)
    m_global = ARIMA(data, order=(1,1,1)).fit()
    p_global = m_global.fittedvalues
    
    # Markov Switching (Literature Method)
    try:
        m_ms = MarkovRegression(data, k_regimes=2, switching_variance=True).fit(disp=False)
        p_ms = m_ms.predict()
    except: p_ms = np.zeros(len(data))
    
    # Prophet (My Modern Method)
    m_prophet = Prophet(changepoint_prior_scale=0.5).fit(df_p)
    p_prophet = m_prophet.predict(df_p)['yhat'].values
    
    return {
        'ARIMA Global': [np.sqrt(mean_squared_error(data, p_global)), mean_absolute_error(data, p_global)],
        'Markov Switching': [np.sqrt(mean_squared_error(data, p_ms)), mean_absolute_error(data, p_ms)],
        'Prophet (My Method)': [np.sqrt(mean_squared_error(data, p_prophet)), mean_absolute_error(data, p_prophet)]
    }

# --- EXECUTION ---
res_literature = []
res_my_case = []

print("Running Monte Carlo Simulations...")
for _ in range(NUM_SIMULATIONS):
    res_literature.append(evaluate_all(generate_literature_data()))
    res_my_case.append(evaluate_all(generate_my_data()))

def summarize(res_list):
    df = pd.concat([pd.DataFrame(r, index=['RMSE', 'MAE']).T for r in res_list])
    return df.groupby(level=0).mean()

summary_lit = summarize(res_literature)
summary_my = summarize(res_my_case)

# --- PROFESSIONAL OUTPUT ---
print("\n" + "="*50)
print("RESULTS 1: REPLICATION OF PESARAN & TIMMERMANN (2013)")
print(summary_lit.round(4))
print("\nRESULTS 2: MY UNIQUE RESEARCH EXTENSION")
print(summary_my.round(4))
print("="*50)

# --- 4 GRAPHICS FOR YOUR REPORT ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Graphic 1: Literature Pattern
axes[0, 0].plot(generate_literature_data(), color='navy')
axes[0, 0].set_title("Graphic 1: Literature Data (Pesaran & Timmermann)")
axes[0, 0].axvline(x=100, color='red', linestyle='--')

# Graphic 2: My Case Pattern
axes[0, 1].plot(generate_my_data(), color='darkgreen')
axes[0, 1].set_title("Graphic 2: My Case Data (Complex Shifts)")
axes[0, 1].axvline(x=100, color='orange', linestyle='--')
axes[0, 1].axvline(x=200, color='orange', linestyle='--')

# Graphic 3: Performance (Literature Scenario)
summary_lit['RMSE'].plot(kind='bar', ax=axes[1, 0], color=['grey', 'blue', 'purple'])
axes[1, 0].set_title("Graphic 3: Method Comparison on Literature Scenarios")
axes[1, 0].set_ylabel("RMSE")

# Graphic 4: Performance (My Case Scenario)
summary_my['RMSE'].plot(kind='bar', ax=axes[1, 1], color=['grey', 'blue', 'green'])
axes[1, 1].set_title("Graphic 4: Method Comparison on My Research Scenarios")
axes[1, 1].set_ylabel("RMSE")

plt.tight_layout()
plt.show()