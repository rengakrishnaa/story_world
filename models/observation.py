"""
Observation Model

Structured observations extracted from video by the VideoObserverAgent.
These observations drive state transitions in the world graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionOutcome(Enum):
    """Outcome of an action as observed in video."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class EmotionState(Enum):
    """Detected emotional states."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DETERMINED = "determined"
    DEFEATED = "defeated"
    EXCITED = "excited"
    CONFUSED = "confused"


class ContinuityErrorType(Enum):
    """Types of continuity errors."""
    CHARACTER_MISSING = "character_missing"
    CHARACTER_DUPLICATE = "character_duplicate"
    LOCATION_MISMATCH = "location_mismatch"
    OBJECT_MISMATCH = "object_mismatch"
    POSE_DISCONTINUITY = "pose_discontinuity"
    LIGHTING_INCONSISTENT = "lighting_inconsistent"
    STYLE_DRIFT = "style_drift"
    TEMPORAL_ARTIFACT = "temporal_artifact"


@dataclass
class ContinuityError:
    """A continuity error detected in video observation."""
    error_type: ContinuityErrorType
    description: str
    severity: float  # 0-1, higher = worse
    frame_range: Optional[tuple] = None  # (start_frame, end_frame)
    affected_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "description": self.description,
            "severity": self.severity,
            "frame_range": self.frame_range,
            "affected_entities": self.affected_entities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContinuityError":
        return cls(
            error_type=ContinuityErrorType(data["error_type"]),
            description=data["description"],
            severity=data["severity"],
            frame_range=tuple(data["frame_range"]) if data.get("frame_range") else None,
            affected_entities=data.get("affected_entities", []),
        )


@dataclass
class CharacterObservation:
    """Observation of a single character in video."""
    character_id: str
    visible: bool = True
    confidence: float = 1.0
    
    # Spatial
    position: Optional[Dict[str, float]] = None  # normalized 0-1 x, y
    bounding_box: Optional[Dict[str, float]] = None  # x, y, width, height
    facing_direction: Optional[str] = None  # left, right, camera, away
    
    # Pose & Motion
    pose: Optional[str] = None  # standing, sitting, running, fighting, etc.
    motion_intensity: float = 0.0  # 0-1, how much movement
    
    # Expression
    emotion: Optional[EmotionState] = None
    expression_intensity: float = 0.5
    
    # Appearance
    appearance_consistent: bool = True
    appearance_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "visible": self.visible,
            "confidence": self.confidence,
            "position": self.position,
            "bounding_box": self.bounding_box,
            "facing_direction": self.facing_direction,
            "pose": self.pose,
            "motion_intensity": self.motion_intensity,
            "emotion": self.emotion.value if self.emotion else None,
            "expression_intensity": self.expression_intensity,
            "appearance_consistent": self.appearance_consistent,
            "appearance_notes": self.appearance_notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterObservation":
        return cls(
            character_id=data["character_id"],
            visible=data.get("visible", True),
            confidence=data.get("confidence", 1.0),
            position=data.get("position"),
            bounding_box=data.get("bounding_box"),
            facing_direction=data.get("facing_direction"),
            pose=data.get("pose"),
            motion_intensity=data.get("motion_intensity", 0.0),
            emotion=EmotionState(data["emotion"]) if data.get("emotion") else None,
            expression_intensity=data.get("expression_intensity", 0.5),
            appearance_consistent=data.get("appearance_consistent", True),
            appearance_notes=data.get("appearance_notes"),
        )


@dataclass
class EnvironmentObservation:
    """Observation of the environment/scene in video."""
    location_id: Optional[str] = None
    location_description: Optional[str] = None
    
    # Temporal
    time_of_day: Optional[str] = None  # dawn, morning, noon, afternoon, dusk, night
    
    # Atmosphere
    weather: Optional[str] = None
    lighting: Optional[str] = None
    mood: Optional[str] = None
    
    # Objects
    objects_detected: List[str] = field(default_factory=list)
    
    # Quality
    background_consistent: bool = True
    background_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "location_description": self.location_description,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "lighting": self.lighting,
            "mood": self.mood,
            "objects_detected": self.objects_detected,
            "background_consistent": self.background_consistent,
            "background_notes": self.background_notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentObservation":
        return cls(**data)


@dataclass
class ActionObservation:
    """Observation of the action/event in video."""
    action_description: str
    outcome: ActionOutcome = ActionOutcome.UNKNOWN
    outcome_confidence: float = 0.5
    
    # What happened
    action_type: Optional[str] = None  # attack, dialogue, movement, etc.
    interaction_type: Optional[str] = None  # character-character, character-object, etc.
    participants: List[str] = field(default_factory=list)
    
    # Narrative impact
    narrative_beat_achieved: bool = False
    narrative_implications: List[str] = field(default_factory=list)
    suggested_next_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_description": self.action_description,
            "outcome": self.outcome.value,
            "outcome_confidence": self.outcome_confidence,
            "action_type": self.action_type,
            "interaction_type": self.interaction_type,
            "participants": self.participants,
            "narrative_beat_achieved": self.narrative_beat_achieved,
            "narrative_implications": self.narrative_implications,
            "suggested_next_action": self.suggested_next_action,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionObservation":
        return cls(
            action_description=data["action_description"],
            outcome=ActionOutcome(data.get("outcome", "unknown")),
            outcome_confidence=data.get("outcome_confidence", 0.5),
            action_type=data.get("action_type"),
            interaction_type=data.get("interaction_type"),
            participants=data.get("participants", []),
            narrative_beat_achieved=data.get("narrative_beat_achieved", False),
            narrative_implications=data.get("narrative_implications", []),
            suggested_next_action=data.get("suggested_next_action"),
        )


@dataclass
class QualityMetrics:
    """Quality metrics for the video observation."""
    # Overall
    overall_quality: float = 0.5  # 0-1
    
    # Visual
    visual_clarity: float = 0.5
    motion_smoothness: float = 0.5
    temporal_coherence: float = 0.5
    style_consistency: float = 0.5
    
    # Content
    action_clarity: float = 0.5  # Was the action clear?
    character_recognizability: float = 0.5
    narrative_coherence: float = 0.5
    
    # Technical
    artifacts_detected: int = 0
    frame_drops: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetrics":
        return cls(**data)
    
    def compute_overall(self) -> float:
        """Compute weighted overall quality score."""
        weights = {
            "visual_clarity": 0.15,
            "motion_smoothness": 0.15,
            "temporal_coherence": 0.2,
            "style_consistency": 0.15,
            "action_clarity": 0.15,
            "character_recognizability": 0.1,
            "narrative_coherence": 0.1,
        }
        score = sum(
            getattr(self, k) * v for k, v in weights.items()
        )
        # Penalize artifacts
        artifact_penalty = min(0.3, self.artifacts_detected * 0.05)
        self.overall_quality = max(0, score - artifact_penalty)
        return self.overall_quality


@dataclass
class ObservationResult:
    """
    Complete structured observation from VideoObserverAgent.
    
    This is the output of video → observation computation.
    It feeds into the decision loop and world state graph.
    """
    observation_id: str
    video_uri: str
    beat_id: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    observation_latency_ms: int = 0
    
    # Content observations
    characters: Dict[str, CharacterObservation] = field(default_factory=dict)
    environment: Optional[EnvironmentObservation] = None
    action: Optional[ActionObservation] = None
    
    # Quality assessment
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    continuity_errors: List[ContinuityError] = field(default_factory=list)
    
    # Confidence & source
    confidence: float = 0.5
    observer_type: str = "gemini"  # "gemini" or "local"
    model_version: str = "unknown"
    
    # Raw response (for training)
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "video_uri": self.video_uri,
            "beat_id": self.beat_id,
            "created_at": self.created_at.isoformat(),
            "observation_latency_ms": self.observation_latency_ms,
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "environment": self.environment.to_dict() if self.environment else None,
            "action": self.action.to_dict() if self.action else None,
            "quality": self.quality.to_dict(),
            "continuity_errors": [e.to_dict() for e in self.continuity_errors],
            "confidence": self.confidence,
            "observer_type": self.observer_type,
            "model_version": self.model_version,
            "raw_response": self.raw_response,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservationResult":
        characters = {}
        for k, v in data.get("characters", {}).items():
            characters[k] = CharacterObservation.from_dict(v)
        
        return cls(
            observation_id=data["observation_id"],
            video_uri=data["video_uri"],
            beat_id=data.get("beat_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            observation_latency_ms=data.get("observation_latency_ms", 0),
            characters=characters,
            environment=EnvironmentObservation.from_dict(data["environment"]) if data.get("environment") else None,
            action=ActionObservation.from_dict(data["action"]) if data.get("action") else None,
            quality=QualityMetrics.from_dict(data.get("quality", {})),
            continuity_errors=[ContinuityError.from_dict(e) for e in data.get("continuity_errors", [])],
            confidence=data.get("confidence", 0.5),
            observer_type=data.get("observer_type", "gemini"),
            model_version=data.get("model_version", "unknown"),
            raw_response=data.get("raw_response"),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ObservationResult":
        return cls.from_dict(json.loads(json_str))
    
    def get_quality_score(self) -> float:
        """Get overall quality score."""
        return self.quality.overall_quality
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if quality meets threshold."""
        return self.get_quality_score() >= threshold
    
    def has_continuity_errors(self) -> bool:
        """Check if any continuity errors detected."""
        return len(self.continuity_errors) > 0
    
    def get_severe_errors(self, severity_threshold: float = 0.7) -> List[ContinuityError]:
        """Get continuity errors above severity threshold."""
        return [e for e in self.continuity_errors if e.severity >= severity_threshold]
    
    def to_world_state_update(self) -> Dict[str, Any]:
        """
        Convert observation to world state update format.
        Used for WorldStateGraph.transition().
        """
        update = {
            "characters": {},
            "narrative_flags": {},
            "custom_state": {
                "last_observation_id": self.observation_id,
                "last_action_outcome": self.action.outcome.value if self.action else None,
            },
        }
        
        # Convert character observations to character states
        for char_id, char_obs in self.characters.items():
            update["characters"][char_id] = {
                "character_id": char_id,
                "emotion": char_obs.emotion.value if char_obs.emotion else None,
                "position": char_obs.position,
                "pose": char_obs.pose,
                "visible": char_obs.visible,
            }
        
        # Update environment
        if self.environment:
            update["environment"] = {
                "location_id": self.environment.location_id or "unknown",
                "time_of_day": self.environment.time_of_day,
                "weather": self.environment.weather,
                "lighting": self.environment.lighting,
            }
        
        # Update narrative flags from action
        if self.action:
            update["narrative_flags"]["last_action"] = self.action.action_description
            update["narrative_flags"]["beat_achieved"] = self.action.narrative_beat_achieved
            if self.action.narrative_implications:
                for impl in self.action.narrative_implications:
                    update["narrative_flags"][f"implication_{hash(impl) % 1000}"] = impl
        
        return update


@dataclass
class TaskContext:
    """
    Context for an observation task.
    Used to determine adaptive quality thresholds.
    """
    task_type: str = "storytelling"  # storytelling, simulation, training, qa
    beat_id: Optional[str] = None
    episode_id: Optional[str] = None
    
    # Risk assessment
    is_branch_point: bool = False
    downstream_beats: int = 0
    
    # Expected content
    expected_characters: List[str] = field(default_factory=list)
    expected_action: Optional[str] = None
    expected_location: Optional[str] = None
    
    # Previous state (for continuity checking)
    previous_observation: Optional[ObservationResult] = None
    previous_world_state: Optional[Dict[str, Any]] = None
    
    # Quality requirements
    required_confidence: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "beat_id": self.beat_id,
            "episode_id": self.episode_id,
            "is_branch_point": self.is_branch_point,
            "downstream_beats": self.downstream_beats,
            "expected_characters": self.expected_characters,
            "expected_action": self.expected_action,
            "expected_location": self.expected_location,
            "required_confidence": self.required_confidence,
        }
