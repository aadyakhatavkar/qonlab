from typing import Protocol
import numpy as np


class DGPProtocol(Protocol):
    def simulate(self, *args, **kwargs) -> np.ndarray: ...


class EstimatorProtocol(Protocol):
    def fit(self, y: np.ndarray): ...

    def predict(self, horizon: int): ...

__all__ = ["DGPProtocol", "EstimatorProtocol"]
