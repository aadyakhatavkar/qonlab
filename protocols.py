from typing import Protocol, Tuple, List, Dict, Any, Union
import numpy as np


class DGPProtocol(Protocol):
    """Protocol for Data-Generating Processes."""
    
    def simulate(self, T: int, seed: int | None = None) -> np.ndarray:
        """Simulate a time series."""
        ...


class EstimatorProtocol(Protocol):
    """Protocol for Forecasting Methods."""
    
    def fit(self, y: np.ndarray) -> None:
        """Fit/train on data."""
        ...

    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray | None]:
        """Generate forecasts for specified horizon.
        
        Returns:
            mean: Point forecast of shape (horizon,)
            variance: Forecast variance of shape (horizon,), or None if not available
        """
        ...


def calculate_metrics(errors: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    Compute RMSE, MAE, Bias, and Variance from forecast errors.
    
    Parameters:
        errors: Array-like of forecast errors (actual - predicted)
    
    Returns:
        dict: Metrics containing RMSE, MAE, Bias, and Variance.
    """
    e = np.asarray(errors)
    e = e[~np.isnan(e)]
    if len(e) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "Variance": np.nan}
    
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e)),
        "Variance": float(np.var(e))
    }


def validate_scenarios(scenarios: List[Dict[str, Any]], T: int) -> List[Dict[str, Any]]:
    """
    Validate and normalize scenario definitions for Monte Carlo simulations.
    
    Parameters:
        scenarios: List of scenario dicts or None for default
        T: Sample size
    
    Returns:
        List of validated scenario dicts
    """
    if scenarios is None:
        return []

    validated = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            raise ValueError("Each scenario must be a dict")
        
        task = sc.get("task", "mean")
        
        # Ensure name exists
        if "name" not in sc:
            sc["name"] = f"{task.capitalize()} Scenario"
            
        # Basic alignment for all tasks
        sc.setdefault("distribution", "normal")
        sc.setdefault("nu", 5)
        
        # Specific task validation could be added here, but we'll keep it flexible
        validated.append(sc)

    return validated


__all__ = ["DGPProtocol", "EstimatorProtocol", "calculate_metrics", "validate_scenarios"]
