"""
Unit tests for DQAL Learned Scorer (Meta-regressor).
"""

import numpy as np
import pandas as pd
import pytest
from dqal.scorer import QualityScorer
from dqal.learned_scorer import LearnedQualityScorer


def test_learned_scorer_training():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    base_scorer = QualityScorer()
    base_scorer.fit(X)

    learned = LearnedQualityScorer(base_scorer=base_scorer, model_type="ridge")

    # Generate synthetic historical episodes
    episodes_signals = []
    accuracy_drops = []
    for _ in range(10):
        miss_q = float(rng.uniform(0.2, 1.0))
        drift_q = float(rng.uniform(0.2, 1.0))
        outlier_q = float(rng.uniform(0.2, 1.0))
        sig = {
            "missing_quality": miss_q,
            "drift_quality": drift_q,
            "outlier_quality": outlier_q,
            "missing_rate": 1.0 - miss_q,
            "drift_magnitude": 1.0 - drift_q,
            "outlier_rate": 1.0 - outlier_q,
        }
        episodes_signals.append(sig)
        # Drop is higher when quality is low
        drop = float(1.0 - (0.3 * miss_q + 0.4 * drift_q + 0.3 * outlier_q))
        accuracy_drops.append(drop)

    learned.fit_meta_model(episodes_signals, accuracy_drops)
    assert learned.is_fitted is True

    # Test scoring a new batch
    test_batch = rng.randn(20, 4)
    Q_learned, signals = learned.score(test_batch)
    assert 0.0 <= Q_learned <= 1.0
    assert "Q_learned" in signals
    assert "predicted_accuracy_drop" in signals
