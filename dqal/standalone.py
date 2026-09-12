"""
Standalone dataset quality checker for DQAL.
Enables shift-left data quality testing, pre-commit checks, and EDA validation without requiring a trained model.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
import pandas as pd

from dqal.config import DQALConfig
from dqal.scorer import QualityScorer
from dqal.trigger import TriggerGate, QualityDecision


def load_dataset(source: Union[str, os.PathLike, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """
    Load a dataset from a DataFrame, NumPy array, or file path (.csv, .tsv, .parquet, .json).
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, np.ndarray):
        if source.ndim == 1:
            source = source.reshape(-1, 1)
        cols = [f"feature_{i}" for i in range(source.shape[1])]
        return pd.DataFrame(source, columns=cols)

    path = str(source)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    elif ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        # Fallback to read_csv
        return pd.read_csv(path)


@dataclass
class QualityReport:
    """
    Encapsulates dataset quality analysis, gating decision, and sub-signal breakdown.
    Designed for standalone dataset validation, pre-flight checks, and CI gates.
    """
    decision: QualityDecision
    Q: float
    signals: Dict[str, float]
    n_samples: int
    n_features: int
    feature_names: List[str]
    audit_meta: Dict[str, Any]

    @property
    def passed(self) -> bool:
        """True if the data quality passed all checks without warnings or blocking."""
        return self.decision == QualityDecision.PASSED

    @property
    def warning(self) -> bool:
        """True if mild or moderate quality degradation was detected."""
        return self.decision == QualityDecision.WARNING

    @property
    def blocked(self) -> bool:
        """True if severe quality degradation was detected."""
        return self.decision == QualityDecision.BLOCKED

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to a dictionary for JSON logging or API responses."""
        return {
            "decision": str(self.decision.value if hasattr(self.decision, "value") else self.decision),
            "quality_score": round(self.Q, 4),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "signals": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.signals.items()},
            "passed": self.passed,
            "warning": self.warning,
            "blocked": self.blocked,
        }

    def summary(self, compact: bool = False) -> str:
        """
        Return a clear, human-readable summary of the data quality checks.

        Args:
            compact: If True, returns a condensed key-value report.
                     If False (default), returns a formatted report card.
        """
        decision_map = {
            QualityDecision.PASSED: "[PASSED] Safe - Quality checks passed",
            QualityDecision.WARNING: "[WARNING] Caution - Data degradation detected",
            QualityDecision.BLOCKED: "[BLOCKED] Action Required - Severe data degradation detected",
        }
        dec_desc = decision_map.get(
            self.decision,
            f"[{self.decision}] Status: {self.decision}"
        )

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

        signal_keys = ["missing_quality", "drift_quality", "outlier_quality"]
        active_signals = {k: self.signals[k] for k in signal_keys if k in self.signals}
        if not active_signals:
            active_signals = {k: v for k, v in self.signals.items() if "quality" in k}

        if compact:
            lines = [
                f"Data Quality Score (Q): {self.Q:.3f} ({self.Q * 100:.1f}% - {quality_rating})",
                f"Validation Status:      {dec_desc}",
                f"Dataset Shape:          {self.n_samples} samples x {self.n_features} features",
                "Sub-Signals (1.0 = Pristine):",
            ]
            for key, val in active_signals.items():
                name, desc = _format_signal(key, val)
                lines.append(f"  • {name:<22}: {desc}")
            return "\n".join(lines)

        width = 62
        lines = [
            "=" * width,
            "                   DQAL Data Quality Report".center(width).rstrip(),
            "=" * width,
            f"Validation Status: {dec_desc}",
            f"Quality Score Q  : {self.Q:.3f} / 1.000 ({self.Q * 100:.1f}% - {quality_rating})",
            f"Dataset Shape    : {self.n_samples} samples x {self.n_features} features",
            "",
            "Sub-Signals (1.0 = Pristine, 0.0 = Degraded):",
        ]
        for key, val in active_signals.items():
            name, desc = _format_signal(key, val)
            lines.append(f"  • {name:<22}: {desc}")

        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def check_quality(
    data: Union[str, os.PathLike, pd.DataFrame, np.ndarray],
    baseline: Optional[Union[str, os.PathLike, pd.DataFrame, np.ndarray]] = None,
    config: Optional[Union[DQALConfig, str]] = None,
) -> QualityReport:
    """
    Validate data quality of a dataset against an optional baseline distribution.

    Args:
        data: Target dataset (DataFrame, NumPy array, or path to CSV/Parquet/JSON).
        baseline: Baseline training dataset (DataFrame, NumPy array, or path).
                  If None, the baseline is fitted on the data itself (zero-drift reference).
        config: Optional DQALConfig instance or path to YAML config file.

    Returns:
        QualityReport: Comprehensive report card with status, score Q, and sub-signal breakdown.

    Example:
        >>> from dqal import check_quality
        >>> report = check_quality("data/incoming.csv", baseline="data/train.csv")
        >>> print(report.summary())
        >>> if not report.passed:
        ...     raise SystemExit(1)
    """
    if isinstance(config, str):
        cfg = DQALConfig.from_yaml(config)
    elif isinstance(config, DQALConfig):
        cfg = config
    else:
        cfg = DQALConfig()

    # For standalone single-shot checks, evaluate immediately without stream smoothing
    cfg.hysteresis.consecutive_batches = 1
    cfg.hysteresis.deadband = 0.0

    df_target = load_dataset(data)
    df_baseline = load_dataset(baseline) if baseline is not None else df_target

    scorer = QualityScorer(cfg)
    scorer.fit(df_baseline)

    trigger = TriggerGate(cfg)

    Q, signals = scorer.score(df_target)
    decision, audit_meta = trigger.decide(Q)

    return QualityReport(
        decision=decision,
        Q=Q,
        signals=signals,
        n_samples=len(df_target),
        n_features=len(df_target.columns),
        feature_names=list(df_target.columns),
        audit_meta=audit_meta,
    )
