"""
DGP Utilities
=============
Utility functions for data-generating processes.
"""
import numpy as np


def validate_scenarios(scenarios, T):
    """
    Validate and normalize scenario definitions for Monte Carlo simulations.
    
    Parameters:
        scenarios: List of scenario dicts or None for default
        T: Sample size
    
    Returns:
        List of validated scenario dicts
    """
    if scenarios is None:
        return [{
            "name": "Single variance break", 
            "variance_Tb": max(1, T // 2), 
            "variance_sigma1": 1.0, 
            "variance_sigma2": 2.0, 
            "task": "variance"
        }]

    validated = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            raise ValueError("Each scenario must be a dict")
        
        task = sc.get("task", "variance")
        
        if task == "variance":
            # Mapping old keys to new variance-prefix keys for backwards compatibility
            key_map = {"Tb": "variance_Tb", "sigma1": "variance_sigma1", "sigma2": "variance_sigma2"}
            for old_k, new_k in key_map.items():
                if new_k not in sc and old_k in sc:
                    sc[new_k] = sc[old_k]
                    
            for key in ("name", "variance_Tb", "variance_sigma1", "variance_sigma2"):
                if key not in sc:
                    raise ValueError(f"Scenario missing required key: {key} for variance task")
            
            v_Tb = int(sc["variance_Tb"])
            if v_Tb >= T: v_Tb = T - 1
            if v_Tb < 1: v_Tb = 1
            
            validated.append({
                "name": sc["name"], 
                "task": "variance",
                "variance_Tb": v_Tb, 
                "variance_sigma1": float(sc["variance_sigma1"]), 
                "variance_sigma2": float(sc["variance_sigma2"]),
                "distribution": sc.get("distribution", "normal"),
                "nu": sc.get("nu", 3)
            })
        elif task == "mean":
            for key in ("name", "Tb", "mu0", "mu1"):
                if key not in sc:
                    raise ValueError(f"Scenario missing required key: {key} for mean task")
            validated.append({
                "name": sc["name"],
                "task": "mean",
                "Tb": int(sc["Tb"]),
                "mu0": float(sc["mu0"]),
                "mu1": float(sc["mu1"]),
                "phi": float(sc.get("phi", 0.6)),
                "sigma": float(sc.get("sigma", 1.0))
            })
        elif task == "parameter":
            for key in ("name", "Tb", "phi1", "phi2"):
                if key not in sc:
                    raise ValueError(f"Scenario missing required key: {key} for parameter task")
            validated.append({
                "name": sc["name"],
                "task": "parameter",
                "Tb": int(sc["Tb"]),
                "phi1": float(sc["phi1"]),
                "phi2": float(sc["phi2"]),
                "sigma": float(sc.get("sigma", 1.0))
            })
        else:
            validated.append(sc)

    return validated
