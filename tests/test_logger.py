"""
Unit tests for DQAL SQLite Telemetry Logger.
"""

import os
import pandas as pd
import pytest
from dqal.logger import TelemetryLogger


def test_logger_lifecycle_and_privacy(tmp_path):
    db_path = str(tmp_path / "test_telemetry.db")
    logger = TelemetryLogger(db_path=db_path, log_raw_features=False)

    # Log a batch
    logger.log_batch(
        batch_id="batch_001",
        Q=0.88,
        signals={"missing_quality": 1.0, "drift_quality": 0.9, "outlier_quality": 0.8},
        decision="SERVE",
        model_version="v1.0.0",
        n_samples=50,
        raw_features=pd.DataFrame({"secret_col": [1, 2, 3]}),
    )

    df = logger.get_telemetry_df()
    assert len(df) == 1
    assert df.iloc[0]["Q"] == 0.88
    assert df.iloc[0]["decision"] == "SERVE"
    # Privacy check: raw_features_json must be None because log_raw_features is False
    assert df.iloc[0]["raw_features_json"] is None

    # Log retrain event
    logger.log_retrain_event(
        old_version="v1.0.0",
        new_version="v1.1.0",
        trigger_reason="Test retrain",
        val_score_old=0.80,
        val_score_new=0.92,
        promoted=True,
    )

    retrain_df = logger.get_retrain_events_df()
    assert len(retrain_df) == 1
    assert retrain_df.iloc[0]["promoted"] == 1
    assert retrain_df.iloc[0]["new_version"] == "v1.1.0"
