"""
DQAL — Data-Quality-Aware Learning
A real-time, model-agnostic quality assurance and selective retraining framework for ML pipelines.
"""

from dqal.config import DQALConfig
from dqal.base import ModelAdapter, SubScorer
from dqal.scorer import QualityScorer, MissingnessScorer, DriftScorer, OutlierScorer
from dqal.trigger import TriggerGate, QualityDecision
from dqal.logger import TelemetryLogger
from dqal.wrapper import DQAL, DQALResult, adapt_model
from dqal.orchestrator import RetrainOrchestrator, RetrainResult
from dqal.learned_scorer import LearnedQualityScorer
from dqal.standalone import check_quality, QualityReport

__version__ = "0.2.0"
__all__ = [
    "DQAL",
    "DQALConfig",
    "DQALResult",
    "QualityDecision",
    "TriggerGate",
    "QualityScorer",
    "MissingnessScorer",
    "DriftScorer",
    "OutlierScorer",
    "TelemetryLogger",
    "RetrainOrchestrator",
    "RetrainResult",
    "LearnedQualityScorer",
    "ModelAdapter",
    "SubScorer",
    "adapt_model",
    "check_quality",
    "QualityReport",
]
