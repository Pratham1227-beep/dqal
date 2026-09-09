"""
Wrapped Model Adapter and Main DQAL Wrapper.
Enables plug-and-play quality assurance for scikit-learn, PyTorch, and custom ML models.
"""

from __future__ import annotations
import uuid
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np
import pandas as pd

from dqal.base import ModelAdapter
from dqal.config import DQALConfig
from dqal.scorer import QualityScorer
from dqal.trigger import TriggerGate, QualityDecision
from dqal.logger import TelemetryLogger


@dataclass
class DQALResult:
    """Encapsulates prediction, gating decision, and quality telemetry."""
    predictions: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]
    decision: str  # "SERVE" | "FLAG" | "ABSTAIN"
    Q: float
    signals: Dict[str, float]
    model_version: str
    served: bool
    batch_id: str
    audit_meta: Dict[str, Any]

    def summary(self, compact: bool = False) -> str:
        """
        Return a clear, human-readable summary of the batch quality score and gating decision.

        Args:
            compact: If True, returns a streamlined key-value output.
                     If False (default), returns a structured report card.
        """
        # Interpretation of decision using normal, easy-to-understand terms
        decision_map = {
            "PASSED": "[PASSED] Safe - Quality check passed",
            "SERVE": "[PASSED] Safe - Quality check passed",
            "PASS": "[PASSED] Safe - Quality check passed",
            "WARNING": "[WARNING] Caution - Degraded data quality detected",
            "FLAG": "[WARNING] Caution - Degraded data quality detected",
            "WARN": "[WARNING] Caution - Degraded data quality detected",
            "BLOCKED": "[BLOCKED] Action Required - Severe degradation; fallback used",
            "ABSTAIN": "[BLOCKED] Action Required - Severe degradation; fallback used",
            "BLOCK": "[BLOCKED] Action Required - Severe degradation; fallback used",
        }
        dec_desc = decision_map.get(
            self.decision.upper() if isinstance(self.decision, str) else str(self.decision),
            f"[{self.decision}] Status: {self.decision}"
        )

        # Quality rating
        if self.Q >= 0.80:
            quality_rating = "High Quality"
        elif self.Q >= 0.50:
            quality_rating = "Degraded / Warning"
        else:
            quality_rating = "Severely Degraded"

        def _format_signal(key: str, val: float) -> Tuple[str, str]:
            pct = val * 100
            k = key.lower()
            if "missing" in k:
                name = "Data Completeness"
                desc = "No missing values" if val >= 0.99 else f"{(1.0 - val) * 100:.1f}% missing"
            elif "drift" in k:
                name = "Covariate Stability"
                desc = "Minimal drift" if val >= 0.85 else "Moderate drift" if val >= 0.50 else "High drift"
            elif "outlier" in k or "anomaly" in k:
                name = "Inlier Adherence"
                desc = "Normal distribution" if val >= 0.85 else "Mild outliers" if val >= 0.50 else "Severe outliers"
            else:
                name = key.replace("_", " ").title()
                desc = "Normal" if val >= 0.80 else "Degraded"
            return name, f"{pct:5.1f}% ({val:.3f}) - {desc}"

        # Sample predictions
        preds_str = "None"
        if self.predictions is not None:
            if hasattr(self.predictions, "tolist"):
                preds_list = self.predictions.tolist()
            else:
                preds_list = list(self.predictions)
            sample = preds_list[:5]
            preds_str = f"{sample} (first {len(sample)} of {len(preds_list)} samples)"

        if compact:
            lines = [
                f"Batch Quality Score (Q): {self.Q:.3f} ({self.Q * 100:.1f}% - {quality_rating})",
                f"Gating Decision:         {dec_desc}",
                "Sub-Signals (1.0 = Pristine):",
            ]
            for key, val in self.signals.items():
                name, desc = _format_signal(key, val)
                lines.append(f"  • {name:<22}: {desc}")
            lines.append(f"Sample Predictions:      {preds_str}")
            return "\n".join(lines)

        # Full Report Card
        width = 62
        lines = [
            "=" * width,
            "                   DQAL Batch Quality Report".center(width).rstrip(),
            "=" * width,
            f"Gating Decision : {dec_desc}",
            f"Quality Score Q : {self.Q:.3f} / 1.000 ({self.Q * 100:.1f}% - {quality_rating})",
            f"Model Version   : {self.model_version}",
            f"Batch ID        : {self.batch_id}",
            "",
            "Sub-Signals (1.0 = Pristine, 0.0 = Degraded):",
        ]
        for key, val in self.signals.items():
            name, desc = _format_signal(key, val)
            lines.append(f"  • {name:<22}: {desc}")

        lines.append("")
        lines.append(f"Sample Predictions: {preds_str}")
        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class SklearnModelAdapter:
    """Adapter for scikit-learn estimators."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
        return self.estimator.predict(X_arr)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            if isinstance(X, pd.DataFrame):
                X_arr = X.to_numpy()
            else:
                X_arr = np.asarray(X)
            return self.estimator.predict_proba(X_arr)
        return None

    def feature_importances(self) -> Optional[Dict[str, float]]:
        if hasattr(self.estimator, "feature_importances_"):
            vals = self.estimator.feature_importances_
            return {f"feature_{i}": float(v) for i, v in enumerate(vals)}
        elif hasattr(self.estimator, "coef_"):
            coef = np.abs(self.estimator.coef_).ravel()
            return {f"feature_{i}": float(v) for i, v in enumerate(coef)}
        return None


class PyTorchModelAdapter:
    """Adapter for PyTorch nn.Module models."""

    def __init__(self, module: Any, device: str = "cpu"):
        self.module = module
        self.device = device

    def _to_tensor(self, X: Union[np.ndarray, pd.DataFrame]):
        import torch
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        return torch.tensor(arr, device=self.device)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        import torch
        self.module.eval()
        with torch.no_grad():
            tensor = self._to_tensor(X)
            out = self.module(tensor)
            if out.ndim > 1 and out.shape[1] > 1:
                preds = torch.argmax(out, dim=1).cpu().numpy()
            else:
                # Binary classification or regression
                if torch.is_floating_point(out):
                    preds = (out.squeeze() > 0.5).long().cpu().numpy()
                else:
                    preds = out.squeeze().cpu().numpy()
            return preds

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        import torch
        self.module.eval()
        with torch.no_grad():
            tensor = self._to_tensor(X)
            out = self.module(tensor)
            if out.ndim > 1 and out.shape[1] > 1:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                return probs
            elif out.ndim == 1 or out.shape[1] == 1:
                p1 = torch.sigmoid(out).squeeze().cpu().numpy()
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
        return None

    def feature_importances(self) -> Optional[Dict[str, float]]:
        return None


def adapt_model(model: Any) -> ModelAdapter:
    """Factory creating appropriate ModelAdapter for given estimator."""
    if hasattr(model, "predict") and not hasattr(model, "forward"):
        return SklearnModelAdapter(model)
    elif hasattr(model, "forward") or "torch.nn" in str(type(model)):
        return PyTorchModelAdapter(model)
    elif isinstance(model, ModelAdapter):
        return model
    else:
        # Fallback adapter if it has predict
        if hasattr(model, "predict"):
            return SklearnModelAdapter(model)
        raise ValueError(f"Unsupported model type: {type(model)}. Must implement ModelAdapter protocol.")


class DQAL:
    """
    Data-Quality-Aware Learning Wrapper.
    Wraps any machine learning model to provide real-time quality scoring, gating, and telemetry.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[Union[DQALConfig, str]] = None,
        model_version: str = "v1.0.0",
        logger: Optional[TelemetryLogger] = None,
    ):
        if isinstance(config, str):
            self.config = DQALConfig.from_yaml(config)
        elif isinstance(config, DQALConfig):
            self.config = config
        else:
            self.config = DQALConfig()

        self.model = adapt_model(model)
        self.raw_model = model
        self.model_version = model_version
        self.scorer = QualityScorer(self.config)
        self.trigger = TriggerGate(self.config)
        self.logger = logger or TelemetryLogger(
            db_path=self.config.logging.db_path,
            log_raw_features=self.config.logging.log_raw_features,
        )
        self.is_fitted: bool = False

    def fit_baseline(self, X_train: Union[np.ndarray, pd.DataFrame]) -> DQAL:
        """
        Fit the baseline distributions from clean training data.
        CRITICAL: Must be called before running .predict().
        """
        self.scorer.fit(X_train)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch_id: Optional[str] = None,
        fallback_prediction: Optional[np.ndarray] = None,
    ) -> DQALResult:
        """
        Evaluate batch data quality, gate inference, and log telemetry.

        Args:
            X: Input feature batch (DataFrame or numpy array)
            batch_id: Optional identifier for tracking
            fallback_prediction: Default predictions to return if gating decision is ABSTAIN

        Returns:
            DQALResult containing predictions, gating decision, and quality metrics.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "DQAL baseline is not fitted! Please call dqal.fit_baseline(X_train) before predict()."
            )

        batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        n_samples = len(X) if hasattr(X, "__len__") else 1

        # Extract feature weights if configured
        feature_weights = None
        if self.config.scorer_settings.use_feature_importances:
            feature_weights = self.model.feature_importances()

        # 1. Compute Quality Score Q and sub-scores
        Q, signals = self.scorer.score(X, feature_weights=feature_weights)

        # 2. Gate decision
        decision, audit_meta = self.trigger.decide(Q)

        # 3. Model execution
        predictions: Optional[np.ndarray] = None
        probabilities: Optional[np.ndarray] = None
        served = False

        if decision in (
            QualityDecision.PASSED,
            QualityDecision.WARNING,
            "PASSED",
            "WARNING",
            "SERVE",
            "FLAG",
        ):
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            served = True
        else:  # BLOCKED / ABSTAIN
            predictions = fallback_prediction
            probabilities = None
            served = False

        # 4. Log telemetry
        self.logger.log_batch(
            batch_id=batch_id,
            Q=Q,
            signals=signals,
            decision=decision.value,
            model_version=self.model_version,
            n_samples=n_samples,
            raw_features=X if self.config.logging.log_raw_features else None,
            extra_meta=audit_meta,
        )

        return DQALResult(
            predictions=predictions,
            probabilities=probabilities,
            decision=decision.value,
            Q=Q,
            signals=signals,
            model_version=self.model_version,
            served=served,
            batch_id=batch_id,
            audit_meta=audit_meta,
        )

    def save_package(self, path: str) -> None:
        """Save wrapped model and baseline scorer together."""
        state = {
            "model_version": self.model_version,
            "config": self.config.to_dict(),
            "raw_model": self.raw_model,
            "scorer": self.scorer,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_package(cls, path: str) -> DQAL:
        """Load saved DQAL instance and baseline."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        config = DQALConfig.from_dict(state["config"])
        instance = cls(
            model=state["raw_model"],
            config=config,
            model_version=state["model_version"],
        )
        instance.scorer = state["scorer"]
        instance.is_fitted = state["is_fitted"]
        return instance
