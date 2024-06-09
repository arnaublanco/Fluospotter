"""Model class, to be extended by specific types of models."""

from typing import List
import numpy as np
from ..datasets import Dataset


class Model:
    """Base class to be subclassed by predictors for specific types of data."""

    def __init__(self) -> None:
        pass

    def train(self, dataset: Dataset) -> None:
        """Training loop."""
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Evaluation function."""
        pass