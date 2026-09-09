"""
Unit tests for DQAL Retrain Orchestrator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from dqal.wrapper import DQAL
from dqal.config import DQALConfig
from dqal.orchestrator import RetrainOrchestrator


@pytest.fixture
def clean_synthetic():
    rng = np.random.RandomState(42)
    X = rng.randn(150, 4)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_orchestrator_label_safety_and_promotion(clean_synthetic, tmp_path):
    X, y = clean_synthetic
    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:], y[100:]

    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

    db_path = str(tmp_path / "test_orchestrator.db")
    config = DQALConfig()
    config.logging.db_path = db_path
    config.retrain.min_samples = 20
    config.retrain.min_improvement_delta = 0.00  # any non-negative improvement

    dqal = DQAL(model=clf, config=config, model_version="v1.0.0")
    dqal.fit_baseline(X_train)

    orchestrator = RetrainOrchestrator(dqal_instance=dqal, config=config)

    # 1. Test safety block when labels are missing
    res_no_labels = orchestrator.execute_retrain(
        X_train_new=X_train,
        y_train_new=None,
        X_val=X_val,
        y_val=y_val,
    )
    assert res_no_labels.success is False
    assert res_no_labels.promoted is False
    assert "Verified ground-truth labels are unavailable" in res_no_labels.reason

    # 2. Test successful retrain and promotion
    res_retrain = orchestrator.execute_retrain(
        X_train_new=X_train,
        y_train_new=y_train,
        X_val=X_val,
        y_val=y_val,
        new_version_tag="v1.1.0",
    )
    assert res_retrain.success is True
    assert res_retrain.promoted is True
    assert orchestrator.active_version == "v1.1.0"
    assert dqal.model_version == "v1.1.0"

    # 3. Test rollback
    rolled_back = orchestrator.rollback_to("v1.0.0")
    assert rolled_back is True
    assert orchestrator.active_version == "v1.0.0"
    assert dqal.model_version == "v1.0.0"
