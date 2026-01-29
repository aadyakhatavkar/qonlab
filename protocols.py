from typing import Protocol, Tuple
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


__all__ = ["DGPProtocol", "EstimatorProtocol"]
