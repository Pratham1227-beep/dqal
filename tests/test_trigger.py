"""
Unit tests for DQAL Trigger Gate (3-tier state machine & hysteresis dead-band).
"""

import pytest
from dqal.config import DQALConfig
from dqal.trigger import TriggerGate, QualityDecision


def test_trigger_basic_decisions():
    config = DQALConfig()
    config.thresholds.abstain_below = 0.5
    config.thresholds.flag_below = 0.8
    config.hysteresis.consecutive_batches = 1
    config.hysteresis.deadband = 0.0

    gate = TriggerGate(config)

    # Q = 0.90 -> SERVE
    dec, _ = gate.decide(0.90)
    assert dec == QualityDecision.SERVE

    # Q = 0.65 -> FLAG
    dec, _ = gate.decide(0.65)
    assert dec == QualityDecision.FLAG

    # Q = 0.30 -> ABSTAIN
    dec, _ = gate.decide(0.30)
    assert dec == QualityDecision.ABSTAIN


def test_trigger_hysteresis_consecutive_buffer():
    config = DQALConfig()
    config.thresholds.abstain_below = 0.5
    config.thresholds.flag_below = 0.8
    config.hysteresis.consecutive_batches = 3  # Requires 3 consecutive confirmations
    config.hysteresis.deadband = 0.0

    gate = TriggerGate(config)
    assert gate.current_state == QualityDecision.SERVE

    # Single dip to 0.70 (FLAG territory) should NOT immediately switch state
    dec1, meta1 = gate.decide(0.70)
    assert dec1 == QualityDecision.SERVE
    assert meta1["pending_state"] == "FLAG"
    assert meta1["consecutive_count"] == 1

    # Second dip
    dec2, meta2 = gate.decide(0.70)
    assert dec2 == QualityDecision.SERVE
    assert meta2["consecutive_count"] == 2

    # Third consecutive dip confirms transition
    dec3, meta3 = gate.decide(0.70)
    assert dec3 == QualityDecision.FLAG
    assert meta3["transitioned"] is True


def test_trigger_hysteresis_deadband():
    config = DQALConfig()
    config.thresholds.flag_below = 0.80
    config.hysteresis.deadband = 0.05
    config.hysteresis.consecutive_batches = 1

    gate = TriggerGate(config)
    assert gate.current_state == QualityDecision.SERVE

    # High thresh is 0.80. Deadband is 0.05.
    # While in SERVE, dropping requires falling below (0.80 - 0.05) = 0.75.
    # So Q = 0.78 should stay SERVE because of deadband.
    dec, _ = gate.decide(0.78)
    assert dec == QualityDecision.SERVE

    # Q = 0.72 drops below 0.75 -> transitions to FLAG
    dec2, _ = gate.decide(0.72)
    assert dec2 == QualityDecision.FLAG
