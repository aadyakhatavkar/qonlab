import numpy as np

def simulate_mean_break_ar1(
    T=300, Tb=150, mu0=0.0, mu1=2.0, phi=0.6, sigma=1.0, seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the intercept (mean).
    y_t = mu_t + phi * y_{t-1} + eps_t
    mu_t = mu0 for t <= Tb, mu1 for t > Tb
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = rng.normal(0.0, sigma, size=T)

    for t in range(1, T):
        mu = mu0 if t <= Tb else mu1
        y[t] = mu + phi * y[t-1] + eps[t]

    return y
