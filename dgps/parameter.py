import numpy as np

def simulate_parameter_break_ar1(
    T=400, Tb=200, phi1=0.2, phi2=0.9, sigma=1.0, seed=None
):
    """
    Simulate AR(1) series with a deterministic change in the autoregressive parameter phi.
    y_t = phi_t * y_{t-1} + eps_t
    phi_t = phi1 for t <= Tb, phi2 for t > Tb
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = rng.normal(0.0, sigma, size=T)

    for t in range(1, T):
        phi = phi1 if t <= Tb else phi2
        y[t] = phi * y[t-1] + eps[t]

    return y
