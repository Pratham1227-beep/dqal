"""
Unit tests for standalone dataset quality checker and QualityReport.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from dqal import check_quality, QualityReport
from dqal.trigger import QualityDecision


@pytest.fixture
def clean_data():
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        rng.randn(100, 4),
        columns=["feat_a", "feat_b", "feat_c", "feat_d"],
    )


def test_check_quality_clean(clean_data):
    report = check_quality(clean_data)

    assert isinstance(report, QualityReport)
    assert report.decision in (QualityDecision.PASSED, "PASSED")
    assert report.passed is True
    assert report.blocked is False
    assert report.Q >= 0.75
    assert report.n_samples == 100
    assert report.n_features == 4
    assert len(report.feature_names) == 4

    # Check summary string
    summary = report.summary()
    assert "DQAL Data Quality Report" in summary
    assert "PASSED" in summary

    compact_summary = report.summary(compact=True)
    assert "Data Quality Score (Q)" in compact_summary


def test_check_quality_with_baseline(clean_data):
    # Holdout batch from clean distribution
    test_batch = clean_data.iloc[:50].copy()
    report = check_quality(test_batch, baseline=clean_data)
    assert report.passed is True
    assert report.blocked is False
    assert report.Q >= 0.80


def test_check_quality_severe_missingness(clean_data):
    corrupted = clean_data.copy()
    # Introduce 80% nulls
    corrupted.iloc[:, :] = np.nan

    report = check_quality(corrupted, baseline=clean_data)
    assert report.blocked is True
    assert report.passed is False
    assert report.Q < 0.50
    assert report.signals["missing_quality"] < 0.10


def test_check_quality_file_input(clean_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        test_csv = os.path.join(tmpdir, "test.csv")

        clean_data.to_csv(train_csv, index=False)
        clean_data.iloc[:40].to_csv(test_csv, index=False)

        report = check_quality(test_csv, baseline=train_csv)
        assert report.passed is True
        assert report.n_samples == 40
        assert report.n_features == 4


def test_quality_report_to_dict(clean_data):
    report = check_quality(clean_data)
    d = report.to_dict()

    assert "decision" in d
    assert "quality_score" in d
    assert "n_samples" in d
    assert "n_features" in d
    assert "signals" in d
    assert d["passed"] is True
