"""
Learned-Weights Scorer (Meta-Model) for DQAL.
Maps raw data quality signals to empirical accuracy drop based on historical degradation episodes.
"""

from __future__ import annotations
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from dqal.config import DQALConfig
from dqal.scorer import QualityScorer


class LearnedQualityScorer:
    """
    Learned meta-model scorer.
    Instead of fixed static weights, trains a regressor to map:
    (missing_signal, drift_signal, outlier_signal, missing_rate, drift_mag, outlier_rate)
    --> empirical accuracy drop.

    Outputs learned Quality Score Q_learned = 1.0 - predicted_accuracy_drop.
    """

    FEATURE_KEYS = [
        "missing_quality",
        "drift_quality",
        "outlier_quality",
        "missing_rate",
        "drift_magnitude",
        "outlier_rate",
    ]

    def __init__(self, base_scorer: QualityScorer, model_type: str = "ridge"):
        self.base_scorer = base_scorer
        self.model_type = model_type
        if model_type == "gbr":
            self.meta_regressor = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        else:
            self.meta_regressor = Ridge(alpha=1.0, positive=True)
        self.is_fitted: bool = False

    def _extract_feature_vector(self, signals: Dict[str, float]) -> np.ndarray:
        return np.array([signals.get(k, 0.0) for k in self.FEATURE_KEYS]).reshape(1, -1)

    def fit_meta_model(
        self,
        episodes_signals: List[Dict[str, float]],
        observed_accuracy_drops: List[float],
    ) -> LearnedQualityScorer:
        """
        Train the meta-regressor on degradation episodes.
        """
        if len(episodes_signals) < 5:
            raise ValueError("Need at least 5 historical episodes to train meta-scorer.")

        X_meta = np.array([
            [sig.get(k, 0.0) for k in self.FEATURE_KEYS]
            for sig in episodes_signals
        ])
        y_meta = np.asarray(observed_accuracy_drops, dtype=np.float64)

        self.meta_regressor.fit(X_meta, y_meta)
        self.is_fitted = True
        return self

    def score(
        self,
        X_batch: Union[np.ndarray, pd.DataFrame],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute learned Quality Score Q_learned.
        """
        # Get raw signals from base scorer
        q_fixed, signals = self.base_scorer.score(X_batch, feature_weights)

        if not self.is_fitted:
            # Fallback to fixed aggregation if meta-regressor not yet trained
            signals["Q_learned"] = q_fixed
            signals["predicted_accuracy_drop"] = 1.0 - q_fixed
            return q_fixed, signals

        feat_vec = self._extract_feature_vector(signals)
        pred_drop = float(self.meta_regressor.predict(feat_vec)[0])
        pred_drop = float(np.clip(pred_drop, 0.0, 1.0))

        Q_learned = float(np.clip(1.0 - pred_drop, 0.0, 1.0))
        signals["Q_learned"] = Q_learned
        signals["predicted_accuracy_drop"] = pred_drop
        signals["Q_fixed"] = q_fixed
        signals["Q"] = Q_learned

        return Q_learned, signals

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> LearnedQualityScorer:
        with open(filepath, "rb") as f:
            return pickle.load(f)
