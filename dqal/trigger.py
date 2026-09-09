"""
Trigger Logic and 3-Tier State Machine for DQAL.
Implements PASSED / WARNING / BLOCKED decision gating with hysteresis dead-band and persistence smoothing.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
from dqal.config import DQALConfig, ThresholdsConfig, HysteresisConfig


class QualityDecision(str, Enum):
    """
    3-tier gating decision for batch data quality.

    Standard terms:
    - PASSED:  High quality (Q > flag_below). Predictions executed normally.
    - WARNING: Mild/moderate degradation (abstain_below < Q <= flag_below). Flagged with telemetry.
    - BLOCKED: Critical quality failure (Q <= abstain_below). Execution halted, fallback used.
    """
    PASSED = "PASSED"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"

    # Backward-compatible aliases for legacy SERVE / FLAG / ABSTAIN
    SERVE = "PASSED"
    FLAG = "WARNING"
    ABSTAIN = "BLOCKED"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            mapping = {
                "SERVE": cls.PASSED,
                "FLAG": cls.WARNING,
                "ABSTAIN": cls.BLOCKED,
                "PASS": cls.PASSED,
                "WARN": cls.WARNING,
                "BLOCK": cls.BLOCKED,
            }
            if value.upper() in mapping:
                return mapping[value.upper()]
        return super()._missing_(value)


class TriggerGate:
    """
    Explainable, auditable 3-tier state machine for gating model predictions.

    - PASSED (SERVE): High quality (Q > flag_below). Predictions served normally.
    - WARNING (FLAG): Degraded quality (abstain_below < Q <= flag_below). Predictions served with warning/log.
    - BLOCKED (ABSTAIN): Severe quality degradation (Q <= abstain_below). Predictions withheld or fallback used.

    Includes Hysteresis dead-band and consecutive confirmation buffer to eliminate rapid state flapping.
    """

    def __init__(self, config: Optional[DQALConfig] = None):
        self.config = config or DQALConfig()
        self.current_state: QualityDecision = QualityDecision.PASSED
        self.pending_state: Optional[QualityDecision] = None
        self.consecutive_count: int = 0
        self.decision_history: List[Dict[str, Any]] = []

    def _raw_decision(self, Q: float) -> QualityDecision:
        """Raw threshold check without hysteresis."""
        if Q > self.config.thresholds.flag_below:
            return QualityDecision.PASSED
        elif Q > self.config.thresholds.abstain_below:
            return QualityDecision.WARNING
        else:
            return QualityDecision.BLOCKED

    def decide(self, Q: float) -> Tuple[QualityDecision, Dict[str, Any]]:
        """
        Evaluate Quality Score Q and return state with audit metadata.

        Returns:
            (decision, metadata_dict)
        """
        deadband = self.config.hysteresis.deadband
        req_consecutive = self.config.hysteresis.consecutive_batches

        low_thresh = self.config.thresholds.abstain_below
        high_thresh = self.config.thresholds.flag_below

        # Determine target state considering deadband around current state
        if self.current_state == QualityDecision.PASSED:
            # Drop to WARNING requires falling below (high_thresh - deadband)
            if Q <= (low_thresh - deadband):
                target_state = QualityDecision.BLOCKED
            elif Q <= (high_thresh - deadband):
                target_state = QualityDecision.WARNING
            else:
                target_state = QualityDecision.PASSED

        elif self.current_state == QualityDecision.WARNING:
            # Upgrade to PASSED requires exceeding (high_thresh + deadband)
            # Downgrade to BLOCKED requires falling below (low_thresh - deadband)
            if Q > (high_thresh + deadband):
                target_state = QualityDecision.PASSED
            elif Q <= (low_thresh - deadband):
                target_state = QualityDecision.BLOCKED
            else:
                target_state = QualityDecision.WARNING

        else:  # current_state == QualityDecision.BLOCKED
            # Upgrade to WARNING requires exceeding (low_thresh + deadband)
            # Upgrade to PASSED requires exceeding (high_thresh + deadband)
            if Q > (high_thresh + deadband):
                target_state = QualityDecision.PASSED
            elif Q > (low_thresh + deadband):
                target_state = QualityDecision.WARNING
            else:
                target_state = QualityDecision.BLOCKED

        # Consecutive confirmation check
        if target_state != self.current_state:
            if target_state == self.pending_state:
                self.consecutive_count += 1
            else:
                self.pending_state = target_state
                self.consecutive_count = 1

            if self.consecutive_count >= req_consecutive:
                # Confirmed transition
                old_state = self.current_state
                self.current_state = target_state
                self.pending_state = None
                self.consecutive_count = 0
                transitioned = True
            else:
                transitioned = False
        else:
            self.pending_state = None
            self.consecutive_count = 0
            transitioned = False

        audit_meta = {
            "Q": Q,
            "decision": self.current_state.value,
            "raw_decision": self._raw_decision(Q).value,
            "pending_state": self.pending_state.value if self.pending_state else None,
            "consecutive_count": self.consecutive_count,
            "required_consecutive": req_consecutive,
            "transitioned": transitioned,
            "deadband": deadband,
            "threshold_abstain": low_thresh,
            "threshold_flag": high_thresh,
        }

        self.decision_history.append(audit_meta)
        return self.current_state, audit_meta

    def reset(self) -> None:
        """Reset state machine to initial PASSED state."""
        self.current_state = QualityDecision.PASSED
        self.pending_state = None
        self.consecutive_count = 0
        self.decision_history.clear()

