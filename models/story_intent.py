"""
Story Intent Model

Represents the hierarchical narrative intent structure:
- MACRO: Fixed high-level story intent (overall narrative goals)
- MESO: Flexible story beats (can adapt to circumstances)  
- MICRO: Fully reactive action selection (responds to current state)

Design Philosophy:
"Great storytelling = predictable destination + surprising journey"
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Intent Levels
# ============================================================================

class IntentLevel(Enum):
    """Hierarchy levels of narrative intent."""
    MACRO = "macro"   # Fixed high-level story goals
    MESO = "meso"     # Flexible story beats
    MICRO = "micro"   # Reactive action selection


class IntentStatus(Enum):
    """Status of an intent."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NarrativeGoalType(Enum):
    """Types of high-level narrative goals."""
    CHARACTER_ARC = "character_arc"       # Character development journey
    PLOT_MILESTONE = "plot_milestone"     # Story plot checkpoints
    THEMATIC = "thematic"                 # Thematic expression
    EMOTIONAL = "emotional"               # Emotional impact targets
    WORLD_BUILDING = "world_building"     # World state evolution


# ============================================================================
# MACRO Layer: High-Level Intent (Fixed)
# ============================================================================

@dataclass
class MacroIntent:
    """
    High-level story intent. These are FIXED and define the destination.
    
    Examples:
    - "Hero must eventually confront the villain"
    - "Romance leads must end up together"
    - "Mystery must be solved"
    """
    intent_id: str
    goal_type: NarrativeGoalType
    description: str
    
    # Target state - what must eventually be true
    target_state: Dict[str, Any] = field(default_factory=dict)
    
    # Characters involved
    characters: List[str] = field(default_factory=list)
    
    # Priority (1-10, higher = more important)
    priority: int = 5
    
    # Constraints
    must_occur_by: Optional[str] = None  # Beat ID deadline
    cannot_occur_before: Optional[str] = None
    
    # Dependencies
    requires_intents: List[str] = field(default_factory=list)
    
    # Status
    status: IntentStatus = IntentStatus.PENDING
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "goal_type": self.goal_type.value,
            "description": self.description,
            "target_state": self.target_state,
            "characters": self.characters,
            "priority": self.priority,
            "must_occur_by": self.must_occur_by,
            "cannot_occur_before": self.cannot_occur_before,
            "requires_intents": self.requires_intents,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacroIntent":
        return cls(
            intent_id=data["intent_id"],
            goal_type=NarrativeGoalType(data.get("goal_type", "plot_milestone")),
            description=data.get("description", ""),
            target_state=data.get("target_state", {}),
            characters=data.get("characters", []),
            priority=data.get("priority", 5),
            must_occur_by=data.get("must_occur_by"),
            cannot_occur_before=data.get("cannot_occur_before"),
            requires_intents=data.get("requires_intents", []),
            status=IntentStatus(data.get("status", "pending")),
        )
    
    def is_satisfied(self, world_state: Dict[str, Any]) -> bool:
        """Check if target state conditions are met."""
        for key, expected in self.target_state.items():
            actual = world_state.get(key)
            if actual != expected:
                return False
        return True


# ============================================================================
# MESO Layer: Story Beats (Flexible)
# ============================================================================

@dataclass
class StoryBeat:
    """
    Mid-level story beat. These are FLEXIBLE - they define waypoints
    but can be reordered, replaced, or adapted.
    
    Examples:
    - "Hero trains with mentor"
    - "Couple has first major conflict"
    - "Clue is discovered"
    """
    beat_id: str
    description: str
    
    # What this beat should achieve
    objectives: List[str] = field(default_factory=list)
    
    # Expected state changes
    expected_state_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Characters involved
    characters: List[str] = field(default_factory=list)
    
    # Location
    location: Optional[str] = None
    
    # Contributes to macro intents
    contributes_to: List[str] = field(default_factory=list)
    
    # Ordering
    suggested_position: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)
    
    # Flexibility
    is_optional: bool = False
    alternatives: List[str] = field(default_factory=list)
    
    # Status
    status: IntentStatus = IntentStatus.PENDING
    actual_video_uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "description": self.description,
            "objectives": self.objectives,
            "expected_state_changes": self.expected_state_changes,
            "characters": self.characters,
            "location": self.location,
            "contributes_to": self.contributes_to,
            "suggested_position": self.suggested_position,
            "depends_on": self.depends_on,
            "is_optional": self.is_optional,
            "alternatives": self.alternatives,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryBeat":
        return cls(
            beat_id=data["beat_id"],
            description=data.get("description", ""),
            objectives=data.get("objectives", []),
            expected_state_changes=data.get("expected_state_changes", {}),
            characters=data.get("characters", []),
            location=data.get("location"),
            contributes_to=data.get("contributes_to", []),
            suggested_position=data.get("suggested_position"),
            depends_on=data.get("depends_on", []),
            is_optional=data.get("is_optional", False),
            alternatives=data.get("alternatives", []),
            status=IntentStatus(data.get("status", "pending")),
        )


# ============================================================================
# MICRO Layer: Actions (Reactive)
# ============================================================================

class ActionType(Enum):
    """Types of micro-level actions."""
    DIALOGUE = "dialogue"
    MOVEMENT = "movement"
    GESTURE = "gesture"
    EMOTION_SHIFT = "emotion_shift"
    INTERACTION = "interaction"
    ENVIRONMENTAL = "environmental"
    COMBAT = "combat"
    CUSTOM = "custom"


@dataclass
class MicroAction:
    """
    Low-level action. Fully REACTIVE - selected based on current state.
    
    Examples:
    - "Character smiles warmly"
    - "Character steps back defensively"
    - "Objects fall due to earthquake"
    """
    action_id: str
    action_type: ActionType
    description: str
    
    # Actor and targets
    actor: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    
    # Context requirements
    requires_state: Dict[str, Any] = field(default_factory=dict)
    
    # Effects
    state_effects: Dict[str, Any] = field(default_factory=dict)
    
    # Animation/rendering hints
    motion_intensity: float = 0.5
    duration_hint_sec: float = 2.0
    camera_hint: Optional[str] = None
    
    # Priority for selection
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "actor": self.actor,
            "targets": self.targets,
            "requires_state": self.requires_state,
            "state_effects": self.state_effects,
            "motion_intensity": self.motion_intensity,
            "duration_hint_sec": self.duration_hint_sec,
            "camera_hint": self.camera_hint,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MicroAction":
        return cls(
            action_id=data["action_id"],
            action_type=ActionType(data.get("action_type", "custom")),
            description=data.get("description", ""),
            actor=data.get("actor"),
            targets=data.get("targets", []),
            requires_state=data.get("requires_state", {}),
            state_effects=data.get("state_effects", {}),
            motion_intensity=data.get("motion_intensity", 0.5),
            duration_hint_sec=data.get("duration_hint_sec", 2.0),
            camera_hint=data.get("camera_hint"),
        )


# ============================================================================
# Complete Intent Structure
# ============================================================================

@dataclass
class StoryIntentGraph:
    """
    Complete hierarchical intent structure for an episode.
    
    Structure:
        MacroIntents (fixed goals)
            ↓
        StoryBeats (flexible waypoints)
            ↓
        MicroActions (reactive selection)
    """
    episode_id: str
    
    # Layers
    macro_intents: Dict[str, MacroIntent] = field(default_factory=dict)
    story_beats: Dict[str, StoryBeat] = field(default_factory=dict)
    action_templates: Dict[str, MicroAction] = field(default_factory=dict)
    
    # Current state
    current_beat_id: Optional[str] = None
    completed_beats: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_macro_intent(self, intent: MacroIntent) -> None:
        """Add a fixed high-level intent."""
        self.macro_intents[intent.intent_id] = intent
    
    def add_story_beat(self, beat: StoryBeat) -> None:
        """Add a flexible story beat."""
        self.story_beats[beat.beat_id] = beat
    
    def add_action_template(self, action: MicroAction) -> None:
        """Add an action template for reactive selection."""
        self.action_templates[action.action_id] = action
    
    def get_pending_beats(self) -> List[StoryBeat]:
        """Get beats that are pending and have satisfied dependencies."""
        pending = []
        for beat in self.story_beats.values():
            if beat.status != IntentStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = all(
                self.story_beats.get(dep_id, StoryBeat(dep_id, "")).status == IntentStatus.COMPLETED
                for dep_id in beat.depends_on
            )
            
            if deps_satisfied:
                pending.append(beat)
        
        # Sort by suggested position
        return sorted(pending, key=lambda b: b.suggested_position or 999)
    
    def get_next_beat(self) -> Optional[StoryBeat]:
        """Get the next beat to execute."""
        pending = self.get_pending_beats()
        return pending[0] if pending else None
    
    def mark_beat_complete(self, beat_id: str, video_uri: Optional[str] = None) -> None:
        """Mark a beat as completed."""
        if beat_id in self.story_beats:
            beat = self.story_beats[beat_id]
            beat.status = IntentStatus.COMPLETED
            beat.actual_video_uri = video_uri
            self.completed_beats.append(beat_id)
            
            # Check if any macro intents are completed
            self._check_macro_completions()
    
    def _check_macro_completions(self) -> None:
        """Check and update macro intent completions."""
        for intent in self.macro_intents.values():
            if intent.status == IntentStatus.COMPLETED:
                continue
            
            # Check if all contributing beats are done
            contributing = [
                beat for beat in self.story_beats.values()
                if intent.intent_id in beat.contributes_to
            ]
            
            if all(b.status == IntentStatus.COMPLETED for b in contributing):
                intent.status = IntentStatus.COMPLETED
                intent.completed_at = datetime.utcnow()
    
    def get_active_characters(self) -> Set[str]:
        """Get characters involved in current/pending beats."""
        chars = set()
        if self.current_beat_id and self.current_beat_id in self.story_beats:
            chars.update(self.story_beats[self.current_beat_id].characters)
        
        for beat in self.get_pending_beats()[:3]:  # Look ahead 3 beats
            chars.update(beat.characters)
        
        return chars
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "macro_intents": {k: v.to_dict() for k, v in self.macro_intents.items()},
            "story_beats": {k: v.to_dict() for k, v in self.story_beats.items()},
            "action_templates": {k: v.to_dict() for k, v in self.action_templates.items()},
            "current_beat_id": self.current_beat_id,
            "completed_beats": self.completed_beats,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryIntentGraph":
        graph = cls(episode_id=data["episode_id"])
        
        for k, v in data.get("macro_intents", {}).items():
            graph.macro_intents[k] = MacroIntent.from_dict(v)
        
        for k, v in data.get("story_beats", {}).items():
            graph.story_beats[k] = StoryBeat.from_dict(v)
        
        for k, v in data.get("action_templates", {}).items():
            graph.action_templates[k] = MicroAction.from_dict(v)
        
        graph.current_beat_id = data.get("current_beat_id")
        graph.completed_beats = data.get("completed_beats", [])
        
        return graph


# ============================================================================
# Factory Functions
# ============================================================================

def create_macro_intent(
    description: str,
    goal_type: NarrativeGoalType = NarrativeGoalType.PLOT_MILESTONE,
    characters: Optional[List[str]] = None,
    target_state: Optional[Dict[str, Any]] = None,
    priority: int = 5,
) -> MacroIntent:
    """Create a new macro intent."""
    return MacroIntent(
        intent_id=f"macro-{uuid.uuid4().hex[:8]}",
        goal_type=goal_type,
        description=description,
        characters=characters or [],
        target_state=target_state or {},
        priority=priority,
    )


def create_story_beat(
    description: str,
    objectives: Optional[List[str]] = None,
    characters: Optional[List[str]] = None,
    location: Optional[str] = None,
    contributes_to: Optional[List[str]] = None,
) -> StoryBeat:
    """Create a new story beat."""
    return StoryBeat(
        beat_id=f"beat-{uuid.uuid4().hex[:8]}",
        description=description,
        objectives=objectives or [],
        characters=characters or [],
        location=location,
        contributes_to=contributes_to or [],
    )


def create_micro_action(
    description: str,
    action_type: ActionType = ActionType.CUSTOM,
    actor: Optional[str] = None,
    state_effects: Optional[Dict[str, Any]] = None,
) -> MicroAction:
    """Create a new micro action."""
    return MicroAction(
        action_id=f"action-{uuid.uuid4().hex[:8]}",
        action_type=action_type,
        description=description,
        actor=actor,
        state_effects=state_effects or {},
    )
