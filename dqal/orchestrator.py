"""
Retraining Orchestrator for DQAL.
Monitors low-quality batches, validates label availability, retrains models,
and executes safe, validated promotion with rollback support.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dqal.config import DQALConfig
from dqal.wrapper import DQAL
from dqal.logger import TelemetryLogger


@dataclass
class RetrainCandidate:
    """Queued retrain data buffer."""
    X: pd.DataFrame
    y: Optional[np.ndarray] = None
    accumulated_flag_count: int = 0
    accumulated_abstain_count: int = 0


@dataclass
class RetrainResult:
    """Outcome of retraining attempt."""
    success: bool
    promoted: bool
    old_version: str
    new_version: str
    val_score_old: float
    val_score_new: float
    reason: str
    details: Dict[str, Any]


class RetrainOrchestrator:
    """
    Manages selective model retraining when data degradation accumulates.
    
    Safety features:
    1. Volume verification (requires threshold of degraded records).
    2. Label availability check (defends against unlabelled pseudo-retraining).
    3. Validated promotion (candidate must demonstrably outperform active model).
    4. Version history & instant rollback.
    """

    def __init__(
        self,
        dqal_instance: DQAL,
        config: Optional[DQALConfig] = None,
        model_builder: Optional[Callable[[], Any]] = None,
        eval_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        self.dqal = dqal_instance
        self.config = config or dqal_instance.config
        self.logger = dqal_instance.logger
        self.model_builder = model_builder
        self.eval_metric_fn = eval_metric_fn or accuracy_score

        self.last_checked_log_id: int = 0
        self.version_history: Dict[str, Any] = {
            self.dqal.model_version: copy.deepcopy(self.dqal.raw_model)
        }
        self.active_version: str = self.dqal.model_version
        self.pending_queue: List[Dict[str, Any]] = []

    def should_trigger_retrain(self) -> Tuple[bool, str]:
        """
        Check if accumulated degraded batches meet the retrain trigger threshold.
        """
        flags, abstains = self.logger.count_flagged_or_abstained_since(self.last_checked_log_id)
        total_degraded = flags + abstains

        min_batches = self.config.retrain.min_flagged_batches
        if total_degraded >= min_batches:
            return True, f"Accumulated {total_degraded} degraded batches (threshold: {min_batches})"
        return False, f"Degraded batches ({total_degraded}) below threshold ({min_batches})"

    def execute_retrain(
        self,
        X_train_new: Union[np.ndarray, pd.DataFrame],
        y_train_new: Optional[np.ndarray],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: np.ndarray,
        new_version_tag: Optional[str] = None,
    ) -> RetrainResult:
        """
        Attempt model retraining and perform validated promotion.

        Args:
            X_train_new: New or combined training feature set
            y_train_new: Verified ground-truth labels
            X_val: Held-out validation feature set
            y_val: Ground-truth validation targets
            new_version_tag: Optional custom version tag
        """
        old_version = self.active_version

        # 1. Label Availability Check
        if self.config.retrain.require_labels and y_train_new is None:
            reason = "Retrain blocked: Verified ground-truth labels are unavailable."
            self.logger.log_retrain_event(
                old_version=old_version,
                new_version=old_version,
                trigger_reason=reason,
                val_score_old=0.0,
                val_score_new=0.0,
                promoted=False,
                details={"status": "queued_waiting_labels"},
            )
            return RetrainResult(
                success=False,
                promoted=False,
                old_version=old_version,
                new_version=old_version,
                val_score_old=0.0,
                val_score_new=0.0,
                reason=reason,
                details={"status": "waiting_for_labels"},
            )

        # 2. Volume Check
        n_samples = len(X_train_new)
        if n_samples < self.config.retrain.min_samples:
            reason = f"Retrain blocked: insufficient sample count ({n_samples} < {self.config.retrain.min_samples})"
            return RetrainResult(
                success=False,
                promoted=False,
                old_version=old_version,
                new_version=old_version,
                val_score_old=0.0,
                val_score_new=0.0,
                reason=reason,
                details={"n_samples": n_samples},
            )

        # 3. Fit Candidate Model
        if self.model_builder is not None:
            candidate_model = self.model_builder()
        else:
            candidate_model = copy.deepcopy(self.dqal.raw_model)

        candidate_model.fit(X_train_new, y_train_new)

        # 4. Evaluate Active vs Candidate Model on Held-out Validation Slice
        val_preds_old = self.dqal.model.predict(X_val)
        val_preds_new = candidate_model.predict(X_val)

        val_score_old = float(self.eval_metric_fn(y_val, val_preds_old))
        val_score_new = float(self.eval_metric_fn(y_val, val_preds_new))

        min_delta = self.config.retrain.min_improvement_delta
        score_diff = val_score_new - val_score_old

        # Determine version tag
        if not new_version_tag:
            try:
                major, minor, patch = self.active_version.lstrip("v").split(".")
                new_version_tag = f"v{major}.{int(minor)+1}.0"
            except Exception:
                new_version_tag = f"{self.active_version}_retrained"

        # 5. Promotion Gate
        if score_diff >= min_delta:
            # Candidate beats active model -> Promote!
            self.version_history[new_version_tag] = candidate_model
            self.active_version = new_version_tag

            # Update DQAL instance with new model and updated baseline
            self.dqal.raw_model = candidate_model
            self.dqal.model = copy.deepcopy(candidate_model)
            from dqal.wrapper import adapt_model
            self.dqal.model = adapt_model(candidate_model)
            self.dqal.model_version = new_version_tag
            self.dqal.fit_baseline(X_train_new)
            self.dqal.trigger.reset()

            reason = f"Candidate promoted! Val score improved from {val_score_old:.4f} to {val_score_new:.4f} (+{score_diff:.4f})"
            promoted = True
        else:
            reason = f"Candidate rejected: Val score change {score_diff:.4f} < required delta {min_delta:.4f} (old: {val_score_old:.4f}, new: {val_score_new:.4f})"
            promoted = False

        self.logger.log_retrain_event(
            old_version=old_version,
            new_version=new_version_tag if promoted else old_version,
            trigger_reason=reason,
            val_score_old=val_score_old,
            val_score_new=val_score_new,
            promoted=promoted,
            details={"score_diff": score_diff, "min_delta": min_delta},
        )

        return RetrainResult(
            success=True,
            promoted=promoted,
            old_version=old_version,
            new_version=new_version_tag if promoted else old_version,
            val_score_old=val_score_old,
            val_score_new=val_score_new,
            reason=reason,
            details={"score_diff": score_diff},
        )

    def rollback_to(self, version_tag: str) -> bool:
        """Rollback active model to a previous version."""
        if version_tag not in self.version_history:
            return False
        model = self.version_history[version_tag]
        self.active_version = version_tag
        self.dqal.raw_model = model
        from dqal.wrapper import adapt_model
        self.dqal.model = adapt_model(model)
        self.dqal.model_version = version_tag
        return True
