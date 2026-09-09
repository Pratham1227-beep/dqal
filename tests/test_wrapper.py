"""
Unit tests for DQAL Model Adapters and unified wrapper.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from dqal.wrapper import DQAL, SklearnModelAdapter, PyTorchModelAdapter
from dqal.config import DQALConfig


@pytest.fixture
def sample_dataset():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_sklearn_adapter(sample_dataset):
    X, y = sample_dataset
    clf = LogisticRegression().fit(X, y)
    adapter = SklearnModelAdapter(clf)

    preds = adapter.predict(X[:10])
    probs = adapter.predict_proba(X[:10])
    importances = adapter.feature_importances()

    assert len(preds) == 10
    assert probs is not None
    assert probs.shape == (10, 2)
    assert importances is not None
    assert len(importances) == 4


def test_pytorch_adapter(sample_dataset):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    X, y = sample_dataset
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    adapter = PyTorchModelAdapter(model)
    preds = adapter.predict(X[:10])
    probs = adapter.predict_proba(X[:10])

    assert len(preds) == 10
    assert probs is not None
    assert probs.shape == (10, 2)


def test_dqal_wrapper_serve_and_abstain(sample_dataset, tmp_path):
    X, y = sample_dataset
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    db_path = str(tmp_path / "test_wrapper.db")

    config = DQALConfig()
    config.logging.db_path = db_path
    config.hysteresis.consecutive_batches = 1

    dqal = DQAL(model=clf, config=config, model_version="v1.0.0")
    dqal.fit_baseline(X)

    # 1. Clean batch -> PASSED (legacy: SERVE)
    res_clean = dqal.predict(X[:30])
    assert res_clean.decision in ("PASSED", "SERVE")
    assert res_clean.served is True
    assert res_clean.predictions is not None
    assert len(res_clean.predictions) == 30

    # 2. Catastrophically degraded batch -> BLOCKED (legacy: ABSTAIN)
    corrupted = np.full_like(X[:30], np.nan)
    res_corrupt = dqal.predict(corrupted, fallback_prediction=np.zeros(30))
    assert res_corrupt.decision in ("BLOCKED", "ABSTAIN")
    assert res_corrupt.served is False
    assert np.array_equal(res_corrupt.predictions, np.zeros(30))


def test_dqal_save_and_load_package(sample_dataset, tmp_path):
    X, y = sample_dataset
    clf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    pkg_path = str(tmp_path / "test_pkg.pkl")

    dqal = DQAL(model=clf, model_version="v2.0.0")
    dqal.fit_baseline(X)

    dqal.save_package(pkg_path)

    loaded_dqal = DQAL.load_package(pkg_path)
    assert loaded_dqal.model_version == "v2.0.0"
    assert loaded_dqal.is_fitted is True

    res = loaded_dqal.predict(X[:10])
    assert res.predictions is not None


def test_dqal_result_summary(sample_dataset):
    X, y = sample_dataset
    clf = LogisticRegression().fit(X, y)
    dqal = DQAL(model=clf, model_version="v1.0.0")
    dqal.fit_baseline(X)

    res = dqal.predict(X[:20])

    # Default report format
    summary_text = res.summary()
    assert "DQAL Batch Quality Report" in summary_text
    assert "Gating Decision" in summary_text
    assert "[PASSED]" in summary_text
    assert "Quality Score Q" in summary_text
    assert "Data Completeness" in summary_text
    assert "Covariate Stability" in summary_text
    assert "Inlier Adherence" in summary_text
    assert "Sample Predictions" in summary_text

    # Compact format
    compact_text = res.summary(compact=True)
    assert "Batch Quality Score (Q):" in compact_text
    assert "Gating Decision:" in compact_text
    assert "Sub-Signals (1.0 = Pristine):" in compact_text

    # str(res) returns default summary
    assert str(res) == summary_text

