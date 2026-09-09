"""
Configuration schemas and loaders for DQAL.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


@dataclass
class WeightsConfig:
    missingness: float = 0.3
    drift: float = 0.4
    outlier: float = 0.3

    def normalize(self) -> None:
        total = self.missingness + self.drift + self.outlier
        if total > 0:
            self.missingness /= total
            self.drift /= total
            self.outlier /= total


@dataclass
class ThresholdsConfig:
    abstain_below: float = 0.5
    flag_below: float = 0.8


@dataclass
class HysteresisConfig:
    deadband: float = 0.03
    consecutive_batches: int = 2


@dataclass
class ScorerSettingsConfig:
    psi_bins: int = 10
    categorical_min_freq: int = 5
    outlier_method: str = "isolation_forest"  # "isolation_forest" | "mahalanobis"
    outlier_contamination: float = 0.05
    use_feature_importances: bool = False


@dataclass
class LoggingConfig:
    db_path: str = "dqal_telemetry.db"
    log_raw_features: bool = False


@dataclass
class RetrainConfig:
    min_flagged_batches: int = 5
    min_samples: int = 100
    require_labels: bool = True
    min_improvement_delta: float = 0.01


@dataclass
class DQALConfig:
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    scorer_settings: ScorerSettingsConfig = field(default_factory=ScorerSettingsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    mode: str = "batch"  # "batch" or "row"
    batch_size: int = 100

    @classmethod
    def from_yaml(cls, path: str) -> DQALConfig:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DQALConfig:
        weights = WeightsConfig(**d.get("weights", {}))
        weights.normalize()
        thresholds = ThresholdsConfig(**d.get("thresholds", {}))
        hysteresis = HysteresisConfig(**d.get("hysteresis", {}))
        scorer_settings = ScorerSettingsConfig(**d.get("scorer_settings", {}))
        logging = LoggingConfig(**d.get("logging", {}))
        retrain = RetrainConfig(**d.get("retrain", {}))
        return cls(
            weights=weights,
            thresholds=thresholds,
            hysteresis=hysteresis,
            scorer_settings=scorer_settings,
            logging=logging,
            retrain=retrain,
            mode=d.get("mode", "batch"),
            batch_size=d.get("batch_size", 100),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {
                "missingness": self.weights.missingness,
                "drift": self.weights.drift,
                "outlier": self.weights.outlier,
            },
            "thresholds": {
                "abstain_below": self.thresholds.abstain_below,
                "flag_below": self.thresholds.flag_below,
            },
            "hysteresis": {
                "deadband": self.hysteresis.deadband,
                "consecutive_batches": self.hysteresis.consecutive_batches,
            },
            "scorer_settings": {
                "psi_bins": self.scorer_settings.psi_bins,
                "categorical_min_freq": self.scorer_settings.categorical_min_freq,
                "outlier_method": self.scorer_settings.outlier_method,
                "outlier_contamination": self.scorer_settings.outlier_contamination,
                "use_feature_importances": self.scorer_settings.use_feature_importances,
            },
            "logging": {
                "db_path": self.logging.db_path,
                "log_raw_features": self.logging.log_raw_features,
            },
            "retrain": {
                "min_flagged_batches": self.retrain.min_flagged_batches,
                "min_samples": self.retrain.min_samples,
                "require_labels": self.retrain.require_labels,
                "min_improvement_delta": self.retrain.min_improvement_delta,
            },
            "mode": self.mode,
            "batch_size": self.batch_size,
        }
