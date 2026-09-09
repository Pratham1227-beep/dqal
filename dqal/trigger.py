"""
Trigger Logic and 3-Tier State Machine for DQAL.
Implements SERVE / FLAG / ABSTAIN decision gating with hysteresis dead-band and persistence smoothing.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
from dqal.config import DQALConfig, ThresholdsConfig, HysteresisConfig


class QualityDecision(str, Enum):
    SERVE = "SERVE"
    FLAG = "FLAG"
    ABSTAIN = "ABSTAIN"


class TriggerGate:
    """
    Explainable, auditable 3-tier state machine for gating model predictions.

    - SERVE: High quality (Q > flag_below). Predictions served normally.
    - FLAG: Degraded quality (abstain_below < Q <= flag_below). Predictions served with warning/log.
    - ABSTAIN: Severe quality degradation (Q <= abstain_below). Predictions withheld or fallback used.

    Includes Hysteresis dead-band and consecutive confirmation buffer to eliminate rapid state flapping.
    """

    def __init__(self, config: Optional[DQALConfig] = None):
        self.config = config or DQALConfig()
        self.current_state: QualityDecision = QualityDecision.SERVE
        self.pending_state: Optional[QualityDecision] = None
        self.consecutive_count: int = 0
        self.decision_history: List[Dict[str, Any]] = []

    def _raw_decision(self, Q: float) -> QualityDecision:
        """Raw threshold check without hysteresis."""
        if Q > self.config.thresholds.flag_below:
            return QualityDecision.SERVE
        elif Q > self.config.thresholds.abstain_below:
            return QualityDecision.FLAG
        else:
            return QualityDecision.ABSTAIN

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
        if self.current_state == QualityDecision.SERVE:
            # Drop to FLAG requires falling below (high_thresh - deadband)
            if Q <= (low_thresh - deadband):
                target_state = QualityDecision.ABSTAIN
            elif Q <= (high_thresh - deadband):
                target_state = QualityDecision.FLAG
            else:
                target_state = QualityDecision.SERVE

        elif self.current_state == QualityDecision.FLAG:
            # Upgrade to SERVE requires exceeding (high_thresh + deadband)
            # Downgrade to ABSTAIN requires falling below (low_thresh - deadband)
            if Q > (high_thresh + deadband):
                target_state = QualityDecision.SERVE
            elif Q <= (low_thresh - deadband):
                target_state = QualityDecision.ABSTAIN
            else:
                target_state = QualityDecision.FLAG

        else:  # current_state == QualityDecision.ABSTAIN
            # Upgrade to FLAG requires exceeding (low_thresh + deadband)
            # Upgrade to SERVE requires exceeding (high_thresh + deadband)
            if Q > (high_thresh + deadband):
                target_state = QualityDecision.SERVE
            elif Q > (low_thresh + deadband):
                target_state = QualityDecision.FLAG
            else:
                target_state = QualityDecision.ABSTAIN

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
        """Reset state machine to initial SERVE state."""
        self.current_state = QualityDecision.SERVE
        self.pending_state = None
        self.consecutive_count = 0
        self.decision_history.clear()
