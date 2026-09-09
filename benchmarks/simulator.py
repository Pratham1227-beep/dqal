"""
Degradation Simulator for DQAL Benchmarks.
Simulates a multi-phase production timeline with progressive missingness, distribution drift, and noise.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class DegradationPhase:
    name: str
    num_batches: int
    missing_prob: float = 0.0
    drift_mean_shift: float = 0.0
    drift_scale_factor: float = 1.0
    noise_outlier_prob: float = 0.0
    noise_scale: float = 1.0


class DegradationSimulator:
    """
    Generates realistic degradation timelines by progressively corrupting clean data.
    """

    def __init__(self, X_clean: pd.DataFrame, y_clean: np.ndarray, batch_size: int = 50, random_state: int = 42):
        self.X_clean = X_clean.copy()
        self.y_clean = np.asarray(y_clean).copy()
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)

    def generate_timeline(self, phases: Optional[List[DegradationPhase]] = None) -> List[Tuple[pd.DataFrame, np.ndarray, str]]:
        """
        Generate chronological batches across sequential degradation phases.
        Returns list of (X_batch, y_batch, phase_name).
        """
        if phases is None:
            phases = [
                DegradationPhase(name="Pristine (Baseline)", num_batches=10, missing_prob=0.0, drift_mean_shift=0.0, noise_outlier_prob=0.0),
                DegradationPhase(name="Mild Sensor Drift", num_batches=10, missing_prob=0.02, drift_mean_shift=0.8, noise_outlier_prob=0.02),
                DegradationPhase(name="Missing Fields Spike", num_batches=10, missing_prob=0.25, drift_mean_shift=1.2, noise_outlier_prob=0.05),
                DegradationPhase(name="Severe Noise & Outliers", num_batches=10, missing_prob=0.35, drift_mean_shift=2.0, noise_outlier_prob=0.30, noise_scale=5.0),
                DegradationPhase(name="Catastrophic Drift", num_batches=10, missing_prob=0.50, drift_mean_shift=3.5, noise_outlier_prob=0.45, noise_scale=8.0),
                DegradationPhase(name="Post-Retrain Recovery", num_batches=10, missing_prob=0.01, drift_mean_shift=0.1, noise_outlier_prob=0.01),
            ]

        batches: List[Tuple[pd.DataFrame, np.ndarray, str]] = []
        n_total = len(self.X_clean)

        for phase in phases:
            for _ in range(phase.num_batches):
                # Sample batch from clean pool
                indices = self.rng.choice(n_total, size=self.batch_size, replace=True)
                batch_X = self.X_clean.iloc[indices].copy().reset_index(drop=True)
                batch_y = self.y_clean[indices].copy()

                # 1. Inject Numeric Drift (mean shift + scale)
                numeric_cols = batch_X.select_dtypes(include=[np.number]).columns
                if phase.drift_mean_shift > 0 or phase.drift_scale_factor != 1.0:
                    for col in numeric_cols:
                        col_std = self.X_clean[col].std() if self.X_clean[col].std() > 0 else 1.0
                        shift = phase.drift_mean_shift * col_std
                        batch_X[col] = (batch_X[col] * phase.drift_scale_factor) + shift

                # 2. Inject Outlier Noise
                if phase.noise_outlier_prob > 0:
                    outlier_mask = self.rng.rand(len(batch_X)) < phase.noise_outlier_prob
                    if np.any(outlier_mask):
                        for col in numeric_cols:
                            col_std = self.X_clean[col].std() if self.X_clean[col].std() > 0 else 1.0
                            noise = self.rng.randn(np.sum(outlier_mask)) * (phase.noise_scale * col_std)
                            batch_X.loc[outlier_mask, col] += noise

                # 3. Inject Missingness / Nulls
                if phase.missing_prob > 0:
                    for col in batch_X.columns:
                        null_mask = self.rng.rand(len(batch_X)) < phase.missing_prob
                        batch_X.loc[null_mask, col] = np.nan

                batches.append((batch_X, batch_y, phase.name))

        return batches
