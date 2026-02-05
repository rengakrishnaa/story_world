"""
State Transition Model

Represents a video-triggered state transition in the world graph.
This is the atomic unit of computation where video acts as the function
that transforms WorldState_t → WorldState_t+1.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class TransitionStatus(Enum):
    """Status of a state transition."""
    PENDING = "pending"       # Transition initiated, video not yet rendered
    RENDERING = "rendering"   # Video is being generated
    OBSERVING = "observing"   # Video rendered, observation in progress
    COMPLETED = "completed"   # Transition complete, new state available
    FAILED = "failed"         # Transition failed
    REJECTED = "rejected"     # Quality check failed, needs retry
    BLOCKED = "blocked"       # Epistemic halt - missing evidence prevents evaluation


class ActionOutcome(Enum):
    """Outcome of an action as observed in video."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


@dataclass
class TransitionMetrics:
    """Metrics for a state transition."""
    render_latency_ms: int = 0
    observation_latency_ms: int = 0
    total_latency_ms: int = 0
    render_cost_usd: float = 0.0
    quality_score: float = 0.0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionMetrics":
        return cls(**data)


@dataclass
class ContinuityError:
    """A continuity error detected in video observation."""
    error_type: str  # "character_missing", "location_mismatch", etc.
    description: str
    severity: float  # 0-1
    affected_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContinuityError":
        return cls(**data)


@dataclass
class StateTransition:
    """
    Represents a video-triggered state transition.
    
    This is the core computational unit:
    - Input: source state + action (beat)
    - Process: video generation + observation
    - Output: target state + metrics
    
    Video is the function, not the product.
    """
    transition_id: str
    episode_id: str
    source_node_id: str
    target_node_id: Optional[str] = None  # Set after completion
    
    # Action that triggered this transition
    beat_id: str = ""
    action_description: str = ""
    
    # Video artifact
    video_uri: Optional[str] = None
    video_duration_sec: float = 0.0
    
    # Observation results
    observation_json: Optional[str] = None
    action_outcome: ActionOutcome = ActionOutcome.UNKNOWN
    continuity_errors: List[ContinuityError] = field(default_factory=list)
    
    # State
    status: TransitionStatus = TransitionStatus.PENDING
    error_message: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Metrics
    metrics: TransitionMetrics = field(default_factory=TransitionMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "episode_id": self.episode_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "beat_id": self.beat_id,
            "action_description": self.action_description,
            "video_uri": self.video_uri,
            "video_duration_sec": self.video_duration_sec,
            "observation_json": self.observation_json,
            "action_outcome": self.action_outcome.value,
            "continuity_errors": [e.to_dict() for e in self.continuity_errors],
            "status": self.status.value,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateTransition":
        return cls(
            transition_id=data["transition_id"],
            episode_id=data["episode_id"],
            source_node_id=data["source_node_id"],
            target_node_id=data.get("target_node_id"),
            beat_id=data.get("beat_id", ""),
            action_description=data.get("action_description", ""),
            video_uri=data.get("video_uri"),
            video_duration_sec=data.get("video_duration_sec", 0.0),
            observation_json=data.get("observation_json"),
            action_outcome=ActionOutcome(data.get("action_outcome", "unknown")),
            continuity_errors=[
                ContinuityError.from_dict(e) 
                for e in data.get("continuity_errors", [])
            ],
            status=TransitionStatus(data.get("status", "pending")),
            error_message=data.get("error_message"),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) 
                if data.get("completed_at") else None
            ),
            metrics=TransitionMetrics.from_dict(data.get("metrics", {})),
        )
    
    def mark_rendering(self) -> None:
        """Mark transition as rendering."""
        self.status = TransitionStatus.RENDERING
    
    def mark_observing(self, video_uri: str, video_duration_sec: float = 0.0) -> None:
        """Mark transition as observing (video ready, analyzing)."""
        self.status = TransitionStatus.OBSERVING
        self.video_uri = video_uri
        self.video_duration_sec = video_duration_sec
    
    def mark_completed(
        self,
        target_node_id: str,
        observation: Dict[str, Any],
        action_outcome: ActionOutcome,
        quality_score: float,
        continuity_errors: Optional[List[ContinuityError]] = None,
    ) -> None:
        """Mark transition as successfully completed."""
        self.status = TransitionStatus.COMPLETED
        self.target_node_id = target_node_id
        self.observation_json = json.dumps(observation)
        self.action_outcome = action_outcome
        self.metrics.quality_score = quality_score
        self.continuity_errors = continuity_errors or []
        self.completed_at = datetime.utcnow()
        
        # Calculate total latency
        self.metrics.total_latency_ms = int(
            (self.completed_at - self.created_at).total_seconds() * 1000
        )
    
    def mark_failed(self, error_message: str) -> None:
        """Mark transition as failed."""
        self.status = TransitionStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
    
    def mark_rejected(self, reason: str) -> None:
        """Mark transition as rejected (quality check failed)."""
        self.status = TransitionStatus.REJECTED
        self.error_message = reason
        self.metrics.retry_count += 1
    
    def is_terminal(self) -> bool:
        """Check if transition is in a terminal state."""
        return self.status in (
            TransitionStatus.COMPLETED,
            TransitionStatus.FAILED,
        )
    
    def needs_retry(self) -> bool:
        """Check if transition needs retry."""
        return self.status == TransitionStatus.REJECTED
    
    def get_observation(self) -> Optional[Dict[str, Any]]:
        """Parse and return observation dict."""
        if self.observation_json:
            return json.loads(self.observation_json)
        return None


def create_transition(
    episode_id: str,
    source_node_id: str,
    beat_id: str,
    action_description: str = "",
) -> StateTransition:
    """
    Factory function to create a new state transition.
    
    Args:
        episode_id: Episode ID
        source_node_id: Source node in world graph
        beat_id: Beat ID that triggered this transition
        action_description: Human-readable action description
        
    Returns:
        New StateTransition in PENDING status
    """
    return StateTransition(
        transition_id=str(uuid.uuid4()),
        episode_id=episode_id,
        source_node_id=source_node_id,
        beat_id=beat_id,
        action_description=action_description,
    )
