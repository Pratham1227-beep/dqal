"""
Base interfaces and protocols for DQAL (Data-Quality-Aware Learning).
"""

from __future__ import annotations
from typing import Protocol, Any, Dict, Optional, Union, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    """
    Protocol defining the unified interface for wrapped machine learning models.
    Supports scikit-learn, PyTorch, or custom user models.
    """

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate point predictions for given inputs."""
        ...

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Generate prediction probabilities (if supported)."""
        ...

    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Return feature importance map {feature_name: importance_weight}, if available."""
        ...


@runtime_checkable
class SubScorer(Protocol):
    """
    Protocol for individual data quality signal extractors.
    Each sub-scorer evaluates a specific dimension of data quality and returns a score in [0, 1].
    (0 = degraded / high anomaly, 1 = pristine quality).
    """

    name: str

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame]) -> None:
        """Capture baseline distribution from training data."""
        ...

    def score(
        self,
        X_batch: Union[np.ndarray, pd.DataFrame],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Evaluate batch against baseline.
        Returns quality score in [0.0, 1.0] where 1.0 is highest quality (zero defect/drift).
        """
        ...
