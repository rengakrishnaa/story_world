# runtime/control/contracts.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class EpisodeState(str, Enum):
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    DEGRADED = "DEGRADED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BeatState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    intent: str
    created_at: datetime
    policies: Dict[str, Any]


@dataclass(frozen=True)
class BeatSpec:
    beat_id: str
    episode_id: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ExecutionObservation:
    episode_id: str
    beat_id: str
    attempt_id: str
    success: bool
    confidence: float
    failure_type: Optional[str]
    explanation: Optional[str]
    artifacts: Dict[str, str]
    metrics: Dict[str, Any]

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "beat_id": self.beat_id,
            "attempt_id": self.attempt_id,
            "success": self.success,
            "confidence": self.confidence,
            "failure_type": self.failure_type,
            "explanation": self.explanation,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }
