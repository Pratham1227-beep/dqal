"""
Unit tests for DQAL Data Quality Scorers (Missingness, Drift, Outliers, Aggregator).
"""

import numpy as np
import pandas as pd
import pytest

from dqal.scorer import MissingnessScorer, DriftScorer, OutlierScorer, QualityScorer
from dqal.config import DQALConfig


@pytest.fixture
def clean_data():
    rng = np.random.RandomState(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    # Add a categorical column
    df["cat_col"] = rng.choice(["A", "B", "C"], size=200, p=[0.6, 0.3, 0.1])
    return df


def test_missingness_scorer_clean_vs_corrupted(clean_data):
    scorer = MissingnessScorer()
    scorer.fit(clean_data)

    # Clean batch
    q_clean, raw_clean = scorer.score(clean_data)
    assert q_clean == 1.0
    assert raw_clean == 0.0

    # Batch with 50% missing values in 2 columns
    corrupted = clean_data.copy()
    corrupted.iloc[:100, 0] = np.nan
    corrupted.iloc[:100, 1] = np.nan

    q_corrupt, raw_corrupt = scorer.score(corrupted)
    assert q_corrupt < q_clean
    assert raw_corrupt > 0.0


def test_missingness_weighted(clean_data):
    scorer = MissingnessScorer()
    scorer.fit(clean_data)

    corrupted = clean_data.copy()
    corrupted.loc[:, "feat_0"] = np.nan

    weights_high = {"feat_0": 10.0, "feat_1": 1.0, "feat_2": 1.0, "feat_3": 1.0, "feat_4": 1.0, "cat_col": 1.0}
    weights_low = {"feat_0": 0.1, "feat_1": 1.0, "feat_2": 1.0, "feat_3": 1.0, "feat_4": 1.0, "cat_col": 1.0}

    q_high, _ = scorer.score(corrupted, feature_weights=weights_high)
    q_low, _ = scorer.score(corrupted, feature_weights=weights_low)

    # When missing feature has high weight, quality should be lower
    assert q_high < q_low


def test_drift_scorer_numeric_psi(clean_data):
    scorer = DriftScorer(psi_bins=10)
    scorer.fit(clean_data)

    # Clean batch sampled from same distribution
    rng = np.random.RandomState(99)
    clean_batch = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(100, 5)),
        columns=[f"feat_{i}" for i in range(5)]
    )
    clean_batch["cat_col"] = rng.choice(["A", "B", "C"], size=100, p=[0.6, 0.3, 0.1])

    q_clean, drift_clean = scorer.score(clean_batch)
    assert q_clean > 0.60
    assert drift_clean < 0.25

    # Severely drifted batch (mean shift by 4.0 std)
    drifted_batch = pd.DataFrame(
        rng.normal(loc=4.0, scale=2.0, size=(100, 5)),
        columns=[f"feat_{i}" for i in range(5)]
    )
    drifted_batch["cat_col"] = rng.choice(["A", "B", "C"], size=100, p=[0.1, 0.1, 0.8])

    q_drift, drift_mag = scorer.score(drifted_batch)
    assert q_drift < q_clean
    assert drift_mag > drift_clean


def test_outlier_scorer_isolation_forest(clean_data):
    scorer = OutlierScorer(method="isolation_forest", contamination=0.05)
    scorer.fit(clean_data)

    # Clean batch
    q_clean, outlier_clean = scorer.score(clean_data.iloc[:50])
    assert q_clean > 0.50
    assert outlier_clean < 0.20

    # Injected extreme anomalies
    anomalies = clean_data.iloc[:50].copy()
    anomalies.iloc[:, :5] += 100.0  # massive outliers

    q_anom, outlier_rate = scorer.score(anomalies)
    assert q_anom < q_clean
    assert outlier_rate > 0.70


def test_quality_scorer_aggregator_and_persistence(clean_data, tmp_path):
    config = DQALConfig()
    config.weights.missingness = 0.3
    config.weights.drift = 0.4
    config.weights.outlier = 0.3

    scorer = QualityScorer(config)
    scorer.fit(clean_data)

    Q_clean, signals_clean = scorer.score(clean_data.iloc[:50])
    assert 0.0 <= Q_clean <= 1.0
    assert Q_clean > 0.70
    assert "missing_quality" in signals_clean
    assert "drift_quality" in signals_clean
    assert "outlier_quality" in signals_clean

    # Test baseline persistence
    tmp_path_file = str(tmp_path / "baseline_scorer.pkl")
    scorer.save_baseline(tmp_path_file)
    loaded = QualityScorer.load_baseline(tmp_path_file)
    Q_loaded, _ = loaded.score(clean_data.iloc[:50])
    assert abs(Q_clean - Q_loaded) < 1e-5
