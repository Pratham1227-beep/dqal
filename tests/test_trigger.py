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

    # Q = 0.90 -> PASSED (SERVE)
    dec, _ = gate.decide(0.90)
    assert dec == QualityDecision.PASSED
    assert dec == QualityDecision.SERVE
    assert dec == "PASSED"

    # Q = 0.65 -> WARNING (FLAG)
    dec, _ = gate.decide(0.65)
    assert dec == QualityDecision.WARNING
    assert dec == QualityDecision.FLAG
    assert dec == "WARNING"

    # Q = 0.30 -> BLOCKED (ABSTAIN)
    dec, _ = gate.decide(0.30)
    assert dec == QualityDecision.BLOCKED
    assert dec == QualityDecision.ABSTAIN
    assert dec == "BLOCKED"


def test_trigger_hysteresis_consecutive_buffer():
    config = DQALConfig()
    config.thresholds.abstain_below = 0.5
    config.thresholds.flag_below = 0.8
    config.hysteresis.consecutive_batches = 3  # Requires 3 consecutive confirmations
    config.hysteresis.deadband = 0.0

    gate = TriggerGate(config)
    assert gate.current_state in (QualityDecision.PASSED, QualityDecision.SERVE)

    # Single dip to 0.70 (WARNING/FLAG territory) should NOT immediately switch state
    dec1, meta1 = gate.decide(0.70)
    assert dec1 == QualityDecision.PASSED
    assert meta1["pending_state"] in ("WARNING", "FLAG")
    assert meta1["consecutive_count"] == 1

    # Second dip
    dec2, meta2 = gate.decide(0.70)
    assert dec2 == QualityDecision.PASSED
    assert meta2["consecutive_count"] == 2

    # Third consecutive dip confirms transition
    dec3, meta3 = gate.decide(0.70)
    assert dec3 == QualityDecision.WARNING
    assert meta3["transitioned"] is True


def test_trigger_hysteresis_deadband():
    config = DQALConfig()
    config.thresholds.flag_below = 0.80
    config.hysteresis.deadband = 0.05
    config.hysteresis.consecutive_batches = 1

    gate = TriggerGate(config)
    assert gate.current_state == QualityDecision.PASSED

    # High thresh is 0.80. Deadband is 0.05.
    # While in PASSED, dropping requires falling below (0.80 - 0.05) = 0.75.
    # So Q = 0.78 should stay PASSED because of deadband.
    dec, _ = gate.decide(0.78)
    assert dec == QualityDecision.PASSED

    # Q = 0.72 drops below 0.75 -> transitions to WARNING
    dec2, _ = gate.decide(0.72)
    assert dec2 == QualityDecision.WARNING
