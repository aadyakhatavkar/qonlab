import numpy as np


def simulate_variance_break(
    T=400, Tb=200, phi=0.6, mu=0.0, sigma1=1.0, sigma2=2.0, seed=None
):
    if not (1 <= Tb < T):
        raise ValueError(f"Tb must satisfy 1 <= Tb < T (got Tb={Tb}, T={T})")

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = np.zeros(T)

    eps[:Tb] = rng.normal(0.0, sigma1, size=Tb)
    eps[Tb:] = rng.normal(0.0, sigma2, size=T - Tb)

    for t in range(1, T):
        y[t] = mu + phi * (y[t - 1] - mu) + eps[t]
    return y


def _validate_scenarios(scenarios, T):
    if scenarios is None:
        return [{"name": "Single variance break", "Tb": max(1, T // 2), "sigma1": 1.0, "sigma2": 2.0}]

    validated = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            raise ValueError("Each scenario must be a dict")
        for key in ("name", "Tb", "sigma1", "sigma2"):
            if key not in sc:
                raise ValueError(f"Scenario missing required key: {key}")
        Tb = int(sc["Tb"])
        if Tb >= T:
            Tb = T - 1
        if Tb < 1:
            Tb = 1
        validated.append({"name": sc["name"], "Tb": Tb, "sigma1": float(sc["sigma1"]), "sigma2": float(sc["sigma2"])})
    return validated
