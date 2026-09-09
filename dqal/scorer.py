"""
Data Quality Scorer module for DQAL.
Implements individual sub-scorers (Missingness, Drift, Outliers) and aggregation into Quality Score Q.
"""

from __future__ import annotations
import pickle
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet, EmpiricalCovariance

from dqal.config import DQALConfig


def _to_dataframe(X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """Standardize input into pandas DataFrame with string column names."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        df.columns = [str(c) for c in df.columns]
        return df
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    col_names = [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=col_names)


class MissingnessScorer:
    """
    Sub-scorer for evaluating missing / null data rate in inference batches.
    Returns quality score in [0.0, 1.0] where 1.0 means zero missing values.
    """

    name = "missingness"

    def __init__(self):
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame]) -> None:
        df = _to_dataframe(X_train)
        self.feature_names = list(df.columns)
        self.is_fitted = True

    def score(
        self,
        X_batch: Union[np.ndarray, pd.DataFrame],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Returns:
            (quality_score, raw_missing_rate)
            quality_score = 1.0 - raw_missing_rate (in [0, 1])
        """
        df = _to_dataframe(X_batch)
        if df.empty:
            return 1.0, 0.0

        # Check for missing features expected from training
        missing_cols = set(self.feature_names) - set(df.columns)
        null_counts = df.isnull().sum()

        if feature_weights and len(self.feature_names) > 0:
            total_weight = sum(feature_weights.get(c, 1.0) for c in self.feature_names)
            if total_weight <= 0:
                total_weight = 1.0
            col_rates = []
            for col in self.feature_names:
                w = feature_weights.get(col, 1.0)
                if col in missing_cols:
                    rate = 1.0
                else:
                    rate = float(null_counts.get(col, 0)) / len(df)
                col_rates.append(rate * w)
            raw_missing_rate = float(sum(col_rates) / total_weight)
        else:
            # Simple global missing rate
            total_elements = df.size + (len(missing_cols) * len(df))
            if total_elements == 0:
                raw_missing_rate = 0.0
            else:
                total_nulls = int(null_counts.sum()) + (len(missing_cols) * len(df))
                raw_missing_rate = float(total_nulls / total_elements)

        raw_missing_rate = np.clip(raw_missing_rate, 0.0, 1.0)
        quality = float(1.0 - raw_missing_rate)
        return quality, raw_missing_rate


class DriftScorer:
    """
    Sub-scorer for detecting statistical covariate shift / distribution drift.
    - Continuous features: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test.
    - Categorical features: Chi-squared test and Jensen-Shannon divergence.
    Returns quality score in [0.0, 1.0] where 1.0 means zero drift vs baseline.
    """

    name = "drift"

    def __init__(self, psi_bins: int = 10, categorical_min_freq: int = 5):
        self.psi_bins = psi_bins
        self.categorical_min_freq = categorical_min_freq
        self.continuous_baselines: Dict[str, Dict[str, Any]] = {}
        self.categorical_baselines: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame]) -> None:
        df = _to_dataframe(X_train)
        self.continuous_baselines.clear()
        self.categorical_baselines.clear()

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            # Determine if continuous or categorical
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
                # Continuous feature
                # Calculate quantile bin edges for PSI
                quantiles = np.linspace(0, 1, self.psi_bins + 1)
                bin_edges = np.percentile(series, quantiles * 100)
                # Ensure strictly increasing bins by adding tiny jitter if duplicated
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 2:
                    bin_edges = np.array([series.min() - 1e-5, series.max() + 1e-5])

                # Baseline frequencies per bin
                counts, _ = np.histogram(series, bins=bin_edges)
                probs = (counts + 1e-4) / (np.sum(counts) + 1e-4 * len(counts))

                self.continuous_baselines[col] = {
                    "raw_values": series.to_numpy(),
                    "bin_edges": bin_edges,
                    "expected_probs": probs,
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1) if len(series) > 1 else 1.0),
                }
            else:
                # Categorical or low-cardinality discrete feature
                val_counts = series.value_counts(normalize=True).to_dict()
                self.categorical_baselines[col] = {str(k): float(v) for k, v in val_counts.items()}

        self.is_fitted = True

    def _calc_feature_psi(self, actual: np.ndarray, base_meta: Dict[str, Any]) -> float:
        """Compute Population Stability Index for a numeric feature with robust Laplace smoothing."""
        bin_edges = base_meta["bin_edges"]
        expected_probs = base_meta["expected_probs"]
        n_bins = len(expected_probs)
        n_actual = len(actual)

        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        
        # Laplace smoothing: add 1 / n_bins smoothing pseudocount
        pseudocount = max(0.5, float(n_actual) * 0.02)
        actual_probs = (actual_counts + pseudocount) / (n_actual + pseudocount * n_bins)
        
        # Symmetrized stable PSI
        expected_smoothed = (expected_probs * n_actual + pseudocount) / (n_actual + pseudocount * n_bins)
        psi_val = np.sum((actual_probs - expected_smoothed) * np.log(actual_probs / expected_smoothed))
        return float(max(0.0, np.nan_to_num(psi_val, nan=0.0)))

    def _calc_feature_js(self, actual_series: pd.Series, base_dist: Dict[str, float]) -> float:
        """Compute Jensen-Shannon distance for categorical feature."""
        actual_counts = actual_series.astype(str).value_counts(normalize=True).to_dict()
        all_keys = set(base_dist.keys()).union(set(actual_counts.keys()))

        p = np.array([base_dist.get(k, 1e-5) for k in all_keys])
        q = np.array([actual_counts.get(k, 1e-5) for k in all_keys])
        p = p / np.sum(p)
        q = q / np.sum(q)

        # jensenshannon returns distance in [0, 1]
        js_dist = jensenshannon(p, q)
        return float(np.nan_to_num(js_dist, nan=0.0))

    def score(
        self,
        X_batch: Union[np.ndarray, pd.DataFrame],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Returns:
            (quality_score, mean_psi_or_js_drift)
            quality_score is mapped via exp(-3.0 * mean_drift) into [0, 1].
            PSI < 0.1 -> Quality > 0.74 (stable)
            PSI = 0.25 -> Quality ~ 0.47 (moderate drift)
            PSI > 0.5 -> Quality < 0.22 (severe drift)
        """
        if not self.is_fitted:
            raise RuntimeError("DriftScorer must be fitted with baseline data before scoring.")

        df = _to_dataframe(X_batch)
        if df.empty:
            return 1.0, 0.0

        drift_values: List[float] = []
        weights: List[float] = []

        # Numeric features PSI
        for col, base_meta in self.continuous_baselines.items():
            if col not in df.columns:
                drift_values.append(1.0)  # Complete missing feature penalty
                weights.append(feature_weights.get(col, 1.0) if feature_weights else 1.0)
                continue

            vals = df[col].dropna().to_numpy()
            if len(vals) < 2:
                # Empty or single-value corrupted feature cannot match baseline distribution
                drift_values.append(1.0)
                weights.append(feature_weights.get(col, 1.0) if feature_weights else 1.0)
                continue

            psi = self._calc_feature_psi(vals, base_meta)
            w = feature_weights.get(col, 1.0) if feature_weights else 1.0
            drift_values.append(psi)
            weights.append(w)

        # Categorical features JS
        for col, base_dist in self.categorical_baselines.items():
            if col not in df.columns:
                drift_values.append(1.0)
                weights.append(feature_weights.get(col, 1.0) if feature_weights else 1.0)
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            js = self._calc_feature_js(series, base_dist)
            w = feature_weights.get(col, 1.0) if feature_weights else 1.0
            drift_values.append(js)
            weights.append(w)

        if not drift_values:
            return 1.0, 0.0

        w_arr = np.array(weights)
        if np.sum(w_arr) == 0:
            w_arr = np.ones_like(w_arr)
        weighted_drift = float(np.average(drift_values, weights=w_arr))

        # Monotonic mapping from drift magnitude -> quality in [0, 1]
        # At drift=0 -> 1.0, drift=0.1 -> 0.82 (stable), drift=0.5 -> 0.36, drift=1.0 -> 0.135
        quality = float(np.exp(-2.0 * weighted_drift))
        quality = float(np.clip(quality, 0.0, 1.0))

        return quality, weighted_drift


class OutlierScorer:
    """
    Sub-scorer for detecting anomalous records / noise.
    Uses Isolation Forest or Empirical/Robust Mahalanobis distance.
    Returns quality score in [0.0, 1.0] where 1.0 means zero outlier noise.
    """

    name = "outlier"

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
    ):
        self.method = method
        self.contamination = contamination
        self.iso_forest: Optional[IsolationForest] = None
        self.cov_estimator: Optional[EmpiricalCovariance] = None
        self.feature_columns: List[str] = []
        self.baseline_mean: Optional[np.ndarray] = None
        self.impute_values: Dict[str, float] = {}
        self.score_q10: float = -0.5
        self.score_q90: float = 0.5
        self.is_fitted: bool = False

    def _prepare_numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        numeric_df = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                fill_val = self.impute_values.get(col, 0.0)
                numeric_df[col] = series.fillna(fill_val)
            else:
                numeric_df[col] = self.impute_values.get(col, 0.0)
        return numeric_df.to_numpy()

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame]) -> None:
        df = _to_dataframe(X_train)
        # Select numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = list(df.columns)
        self.feature_columns = numeric_cols

        # Baseline impute values
        self.impute_values = {
            col: float(pd.to_numeric(df[col], errors="coerce").median())
            if not pd.to_numeric(df[col], errors="coerce").dropna().empty
            else 0.0
            for col in self.feature_columns
        }

        X_num = self._prepare_numeric_matrix(df)

        if self.method == "mahalanobis":
            try:
                self.cov_estimator = MinCovDet().fit(X_num)
            except Exception:
                self.cov_estimator = EmpiricalCovariance().fit(X_num)
            train_dist = self.cov_estimator.mahalanobis(X_num)
            self.score_q90 = float(np.percentile(train_dist, 90))
        else:
            self.iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            self.iso_forest.fit(X_num)
            train_scores = self.iso_forest.decision_function(X_num)
            self.score_q10 = float(np.percentile(train_scores, 5))
            self.score_q90 = float(np.percentile(train_scores, 95))

        self.is_fitted = True

    def score(
        self,
        X_batch: Union[np.ndarray, pd.DataFrame],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Returns:
            (quality_score, outlier_rate_or_anomaly_fraction)
        """
        if not self.is_fitted:
            raise RuntimeError("OutlierScorer must be fitted with baseline data before scoring.")

        df = _to_dataframe(X_batch)
        if df.empty:
            return 1.0, 0.0

        X_num = self._prepare_numeric_matrix(df)

        if self.method == "mahalanobis" and self.cov_estimator is not None:
            distances = self.cov_estimator.mahalanobis(X_num)
            # Higher distance -> lower quality
            threshold = self.score_q90 * 1.5
            outliers = (distances > threshold).astype(float)
            outlier_rate = float(np.mean(outliers))
            # Smooth quality
            norm_dist = np.mean(distances) / (self.score_q90 + 1e-5)
            quality = float(1.0 / (1.0 + np.maximum(0.0, norm_dist - 1.0)))
        else:
            if self.iso_forest is None:
                return 1.0, 0.0
            decision_scores = self.iso_forest.decision_function(X_num)
            preds = self.iso_forest.predict(X_num)  # -1 = outlier, 1 = inlier
            outlier_rate = float(np.mean(preds == -1))

            # Calibrate decision scores into [0, 1] quality
            # Inlier scores are centered around positive values; outliers are negative.
            scaled = (decision_scores - self.score_q10) / (self.score_q90 - self.score_q10 + 1e-5)
            mean_sample_quality = float(np.mean(np.clip(0.2 + 0.8 * scaled, 0.0, 1.0)))
            # Penalize by outlier rate
            quality = float(np.clip(mean_sample_quality * (1.0 - outlier_rate), 0.0, 1.0))

        return quality, outlier_rate


class QualityScorer:
    """
    Main aggregator Quality Scorer.
    Composes MissingnessScorer, DriftScorer, and OutlierScorer.
    Computes unified Quality Score Q in [0.0, 1.0].
    """

    def __init__(self, config: Optional[DQALConfig] = None):
        self.config = config or DQALConfig()
        self.missingness_scorer = MissingnessScorer()
        self.drift_scorer = DriftScorer(
            psi_bins=self.config.scorer_settings.psi_bins,
            categorical_min_freq=self.config.scorer_settings.categorical_min_freq,
        )
        self.outlier_scorer = OutlierScorer(
            method=self.config.scorer_settings.outlier_method,
            contamination=self.config.scorer_settings.outlier_contamination,
        )
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.baseline_stats: Dict[str, Any] = {}

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame]) -> QualityScorer:
        """
        Fit all sub-scorers on training baseline.
        Captures full feature distributions and metadata.
        """
        df = _to_dataframe(X_train)
        self.feature_names = list(df.columns)

        self.missingness_scorer.fit(df)
        self.drift_scorer.fit(df)
        self.outlier_scorer.fit(df)

        # Store baseline summary statistics
        self.baseline_stats = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "features": self.feature_names,
            "numeric_means": {
                c: float(df[c].mean())
                for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            },
            "numeric_stds": {
                c: float(df[c].std(ddof=1) if len(df) > 1 else 1.0)
                for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            },
        }
        self.is_fitted = True
        return self

    def score(
        self,
        X_batch: Union[np.ndarray, pd.DataFrame],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute aggregate Quality Score Q and sub-scorer telemetry.

        Returns:
            (Q, signals_dict)
            Q: float in [0.0, 1.0]
            signals_dict: raw and quality values for missingness, drift, outliers
        """
        if not self.is_fitted:
            raise RuntimeError(
                "QualityScorer is not fitted! You MUST fit the baseline with scorer.fit(X_train) "
                "before scoring batches."
            )

        # Normalize weights
        w_miss = self.config.weights.missingness
        w_drift = self.config.weights.drift
        w_outlier = self.config.weights.outlier
        total_w = w_miss + w_drift + w_outlier
        if total_w <= 0:
            w_miss, w_drift, w_outlier = 0.333, 0.334, 0.333
        else:
            w_miss /= total_w
            w_drift /= total_w
            w_outlier /= total_w

        # Compute sub-scores
        q_miss, raw_miss = self.missingness_scorer.score(X_batch, feature_weights)
        q_drift, raw_drift = self.drift_scorer.score(X_batch, feature_weights)
        q_outlier, raw_outlier = self.outlier_scorer.score(X_batch, feature_weights)

        # Fixed-weight linear aggregation
        Q = float(w_miss * q_miss + w_drift * q_drift + w_outlier * q_outlier)
        Q = float(np.clip(Q, 0.0, 1.0))

        signals = {
            "Q": Q,
            "missing_quality": q_miss,
            "drift_quality": q_drift,
            "outlier_quality": q_outlier,
            "missing_rate": raw_miss,
            "drift_magnitude": raw_drift,
            "outlier_rate": raw_outlier,
            "weight_missingness": w_miss,
            "weight_drift": w_drift,
            "weight_outlier": w_outlier,
        }

        return Q, signals

    def save_baseline(self, filepath: str) -> None:
        """Persist fitted baseline scorer to disk (pickle)."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted baseline scorer.")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_baseline(cls, filepath: str) -> QualityScorer:
        """Load fitted baseline scorer from disk."""
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, QualityScorer):
            raise TypeError(f"Loaded object is not a QualityScorer instance: {type(obj)}")
        return obj
