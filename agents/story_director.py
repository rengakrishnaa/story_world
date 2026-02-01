"""
Story Director Agent

Three-tier hierarchical divergence control for storytelling.
Fixed intent at top, flexible beats in middle, reactive actions at bottom.

Design Philosophy:
- MACRO: "What must happen" - immutable narrative destinations
- MESO: "How it happens" - adaptable story beats
- MICRO: "What happens now" - purely reactive moment-to-moment

"The destination is sacred, the journey is creative."
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid

from models.story_intent import (
    StoryIntentGraph,
    MacroIntent,
    StoryBeat,
    MicroAction,
    IntentLevel,
    IntentStatus,
    NarrativeGoalType,
    ActionType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Director Configuration
# ============================================================================

@dataclass
class DirectorConfig:
    """Configuration for StoryDirector."""
    # Macro layer
    allow_macro_reordering: bool = False  # Macro goals are FIXED
    
    # Meso layer
    max_beat_adaptations: int = 3
    allow_beat_substitution: bool = True
    allow_beat_insertion: bool = True
    
    # Micro layer
    action_selection_top_k: int = 5
    action_randomness: float = 0.2  # For variety
    
    # Observation integration
    adapt_on_quality_failure: bool = True
    adapt_on_continuity_error: bool = True
    
    # Lookahead
    planning_horizon: int = 3  # How many beats to plan ahead


# ============================================================================
# Director State
# ============================================================================

@dataclass
class DirectorState:
    """Current state of the director."""
    episode_id: str
    intent_graph: StoryIntentGraph
    
    # Progress
    beats_completed: int = 0
    beats_adapted: int = 0
    beats_failed: int = 0
    
    # Current execution
    current_beat: Optional[StoryBeat] = None
    current_actions: List[MicroAction] = field(default_factory=list)
    
    # History
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "beats_completed": self.beats_completed,
            "beats_adapted": self.beats_adapted,
            "beats_failed": self.beats_failed,
            "current_beat_id": self.current_beat.beat_id if self.current_beat else None,
            "adaptation_count": len(self.adaptation_history),
        }


# ============================================================================
# Decisions
# ============================================================================

class DecisionType:
    """Types of director decisions."""
    PROCEED = "proceed"       # Continue with current beat
    ADAPT = "adapt"          # Modify current beat
    SUBSTITUTE = "substitute" # Replace beat with alternative
    INSERT = "insert"        # Add new beat
    SKIP = "skip"           # Skip optional beat
    RETRY = "retry"         # Retry with different actions
    COMPLETE = "complete"    # Episode complete


@dataclass
class DirectorDecision:
    """A decision from the story director."""
    decision_type: str
    beat_id: Optional[str] = None
    actions: List[MicroAction] = field(default_factory=list)
    
    # Reasoning
    reason: str = ""
    triggered_by: str = ""  # What caused this decision
    
    # State changes to expect
    expected_state_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Rendering hints
    suggested_style: Optional[str] = None
    priority_characters: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "beat_id": self.beat_id,
            "actions": [a.to_dict() for a in self.actions],
            "reason": self.reason,
            "triggered_by": self.triggered_by,
            "expected_state_changes": self.expected_state_changes,
        }


# ============================================================================
# Story Director
# ============================================================================

class StoryDirector:
    """
    Hierarchical story director with three-tier divergence control.
    
    Responsibilities:
    1. Maintain narrative coherence (macro stability)
    2. Adapt story beats based on world state (meso flexibility)
    3. Select moment-to-moment actions (micro reactivity)
    
    Usage:
        intent_graph = StoryIntentGraph(episode_id="ep-001")
        # ... populate with macro intents and beats ...
        
        director = StoryDirector(intent_graph)
        
        while not director.is_complete():
            decision = director.next_decision(current_world_state)
            # Execute the beat...
            director.record_outcome(beat_id, observation, quality_result)
    """
    
    def __init__(
        self,
        intent_graph: StoryIntentGraph,
        config: Optional[DirectorConfig] = None,
    ):
        self.config = config or DirectorConfig()
        self.state = DirectorState(
            episode_id=intent_graph.episode_id,
            intent_graph=intent_graph,
        )
        
        logger.info(
            f"[story_director] initialized for {intent_graph.episode_id}: "
            f"{len(intent_graph.macro_intents)} macro, "
            f"{len(intent_graph.story_beats)} beats"
        )
    
    def is_complete(self) -> bool:
        """Check if all macro intents are satisfied."""
        for intent in self.state.intent_graph.macro_intents.values():
            if intent.status not in (IntentStatus.COMPLETED, IntentStatus.SKIPPED):
                return False
        return True
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        total_beats = len(self.state.intent_graph.story_beats)
        return {
            "is_complete": self.is_complete(),
            "beats_completed": self.state.beats_completed,
            "beats_total": total_beats,
            "beats_adapted": self.state.beats_adapted,
            "progress_pct": (self.state.beats_completed / max(1, total_beats)) * 100,
        }
    
    def next_decision(
        self,
        world_state: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> DirectorDecision:
        """
        Get the next director decision based on current state.
        
        Args:
            world_state: Current world state
            observation: Latest observation from video
            
        Returns:
            DirectorDecision with beat and actions
        """
        world_state = world_state or {}
        
        # Check if complete
        if self.is_complete():
            return DirectorDecision(
                decision_type=DecisionType.COMPLETE,
                reason="All macro intents satisfied",
            )
        
        # Get next pending beat
        next_beat = self.state.intent_graph.get_next_beat()
        
        if not next_beat:
            # No more beats, check if we need to generate new ones
            return self._handle_no_pending_beats(world_state)
        
        # Check if beat should be adapted based on world state
        adaptation = self._evaluate_adaptation(next_beat, world_state, observation)
        
        if adaptation:
            return adaptation
        
        # Select actions for the beat
        actions = self._select_actions(next_beat, world_state)
        
        # Update state
        self.state.current_beat = next_beat
        self.state.current_actions = actions
        self.state.intent_graph.current_beat_id = next_beat.beat_id
        
        return DirectorDecision(
            decision_type=DecisionType.PROCEED,
            beat_id=next_beat.beat_id,
            actions=actions,
            reason="Next beat in sequence",
            expected_state_changes=next_beat.expected_state_changes,
            priority_characters=next_beat.characters,
        )
    
    def _handle_no_pending_beats(
        self,
        world_state: Dict[str, Any],
    ) -> DirectorDecision:
        """Handle case when no pending beats remain."""
        # Check for unsatisfied macro intents
        unsatisfied = [
            intent for intent in self.state.intent_graph.macro_intents.values()
            if intent.status == IntentStatus.PENDING
        ]
        
        if unsatisfied:
            # Need to insert new beats
            highest_priority = max(unsatisfied, key=lambda i: i.priority)
            
            new_beat = self._generate_beat_for_intent(highest_priority, world_state)
            self.state.intent_graph.add_story_beat(new_beat)
            
            return DirectorDecision(
                decision_type=DecisionType.INSERT,
                beat_id=new_beat.beat_id,
                actions=self._select_actions(new_beat, world_state),
                reason=f"Generated beat to satisfy: {highest_priority.description}",
                triggered_by="macro_unsatisfied",
            )
        
        return DirectorDecision(
            decision_type=DecisionType.COMPLETE,
            reason="No pending beats and all intents resolved",
        )
    
    def _evaluate_adaptation(
        self,
        beat: StoryBeat,
        world_state: Dict[str, Any],
        observation: Optional[Dict[str, Any]],
    ) -> Optional[DirectorDecision]:
        """Evaluate if beat needs adaptation based on current state."""
        # Check character availability
        missing_chars = [
            c for c in beat.characters
            if world_state.get("characters", {}).get(c, {}).get("available") is False
        ]
        
        if missing_chars and self.config.allow_beat_substitution:
            # Try to substitute with alternative
            if beat.alternatives:
                alt_id = beat.alternatives[0]
                if alt_id in self.state.intent_graph.story_beats:
                    alt_beat = self.state.intent_graph.story_beats[alt_id]
                    
                    self._record_adaptation(
                        beat.beat_id,
                        "substituted",
                        f"Missing characters: {missing_chars}",
                    )
                    
                    return DirectorDecision(
                        decision_type=DecisionType.SUBSTITUTE,
                        beat_id=alt_beat.beat_id,
                        actions=self._select_actions(alt_beat, world_state),
                        reason=f"Substituted due to missing characters: {missing_chars}",
                        triggered_by="character_unavailable",
                    )
        
        # Check location accessibility
        if beat.location:
            location_accessible = world_state.get("locations", {}).get(
                beat.location, {}
            ).get("accessible", True)
            
            if not location_accessible and beat.is_optional:
                self._record_adaptation(beat.beat_id, "skipped", "Location inaccessible")
                beat.status = IntentStatus.SKIPPED
                
                return DirectorDecision(
                    decision_type=DecisionType.SKIP,
                    beat_id=beat.beat_id,
                    reason=f"Skipped optional beat: location {beat.location} inaccessible",
                    triggered_by="location_inaccessible",
                )
        
        # Check observation-based adaptation
        if observation and self.config.adapt_on_continuity_error:
            errors = observation.get("continuity_errors", [])
            if errors and self.state.beats_adapted < self.config.max_beat_adaptations:
                # Adapt the beat to address continuity
                return self._adapt_beat_for_continuity(beat, errors, world_state)
        
        return None
    
    def _adapt_beat_for_continuity(
        self,
        beat: StoryBeat,
        errors: List[Dict[str, Any]],
        world_state: Dict[str, Any],
    ) -> DirectorDecision:
        """Adapt beat to address continuity errors."""
        # Create adapted version of the beat
        adapted_beat = StoryBeat(
            beat_id=f"{beat.beat_id}-adapted",
            description=f"{beat.description} (adapted)",
            objectives=beat.objectives.copy(),
            characters=beat.characters.copy(),
            location=beat.location,
            contributes_to=beat.contributes_to.copy(),
            depends_on=[beat.beat_id] if beat.beat_id in self.state.intent_graph.story_beats else [],
        )
        
        # Add corrective objectives
        for error in errors:
            if error.get("error_type") == "character_missing":
                affected = error.get("affected_entities", [])
                adapted_beat.objectives.append(f"Re-establish {', '.join(affected)}")
        
        self.state.intent_graph.add_story_beat(adapted_beat)
        self._record_adaptation(beat.beat_id, "adapted", f"Continuity errors: {len(errors)}")
        
        return DirectorDecision(
            decision_type=DecisionType.ADAPT,
            beat_id=adapted_beat.beat_id,
            actions=self._select_actions(adapted_beat, world_state),
            reason=f"Adapted to fix {len(errors)} continuity errors",
            triggered_by="continuity_error",
        )
    
    def _select_actions(
        self,
        beat: StoryBeat,
        world_state: Dict[str, Any],
    ) -> List[MicroAction]:
        """Select micro actions for a beat based on current state."""
        actions = []
        
        # Score all action templates
        scored = []
        for action in self.state.intent_graph.action_templates.values():
            score = self._score_action(action, beat, world_state)
            if score > 0:
                scored.append((score, action))
        
        # Sort by score
        scored.sort(key=lambda x: -x[0])
        
        # Select top-k
        for score, action in scored[:self.config.action_selection_top_k]:
            action_copy = MicroAction(
                action_id=f"{action.action_id}-{uuid.uuid4().hex[:4]}",
                action_type=action.action_type,
                description=action.description,
                actor=action.actor or (beat.characters[0] if beat.characters else None),
                targets=action.targets,
                state_effects=action.state_effects.copy(),
                motion_intensity=action.motion_intensity,
                duration_hint_sec=action.duration_hint_sec,
            )
            action_copy.relevance_score = score
            actions.append(action_copy)
        
        # If no matching templates, generate basic actions
        if not actions:
            actions = self._generate_basic_actions(beat)
        
        return actions
    
    def _score_action(
        self,
        action: MicroAction,
        beat: StoryBeat,
        world_state: Dict[str, Any],
    ) -> float:
        """Score an action's relevance to current beat and state."""
        score = 0.0
        
        # Check actor relevance
        if action.actor in beat.characters:
            score += 1.0
        
        # Check state requirements
        requirements_met = True
        for key, expected in action.requires_state.items():
            actual = world_state.get(key)
            if actual != expected:
                requirements_met = False
                break
        
        if not requirements_met:
            return 0.0
        
        # Check if action contributes to beat objectives
        action_desc_lower = action.description.lower()
        for obj in beat.objectives:
            if any(word in action_desc_lower for word in obj.lower().split()):
                score += 0.5
        
        # Action type relevance
        type_relevance = {
            ActionType.DIALOGUE: 0.3,
            ActionType.MOVEMENT: 0.2,
            ActionType.GESTURE: 0.2,
            ActionType.INTERACTION: 0.4,
            ActionType.EMOTION_SHIFT: 0.3,
        }
        score += type_relevance.get(action.action_type, 0.1)
        
        return score
    
    def _generate_basic_actions(self, beat: StoryBeat) -> List[MicroAction]:
        """Generate basic actions when no templates match."""
        actions = []
        
        for i, char in enumerate(beat.characters[:3]):
            action = MicroAction(
                action_id=f"basic-{uuid.uuid4().hex[:4]}",
                action_type=ActionType.GESTURE,
                description=f"{char} participates in {beat.description}",
                actor=char,
                motion_intensity=0.5,
            )
            actions.append(action)
        
        return actions
    
    def _generate_beat_for_intent(
        self,
        intent: MacroIntent,
        world_state: Dict[str, Any],
    ) -> StoryBeat:
        """Generate a new beat to satisfy a macro intent."""
        return StoryBeat(
            beat_id=f"generated-{uuid.uuid4().hex[:8]}",
            description=f"Achieve: {intent.description}",
            objectives=[intent.description],
            characters=intent.characters.copy(),
            contributes_to=[intent.intent_id],
            expected_state_changes=intent.target_state.copy(),
        )
    
    def _record_adaptation(
        self,
        beat_id: str,
        adaptation_type: str,
        reason: str,
    ) -> None:
        """Record an adaptation for history."""
        self.state.adaptation_history.append({
            "beat_id": beat_id,
            "type": adaptation_type,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.state.beats_adapted += 1
    
    def record_outcome(
        self,
        beat_id: str,
        observation: Optional[Dict[str, Any]] = None,
        quality_result: Optional[Dict[str, Any]] = None,
        video_uri: Optional[str] = None,
    ) -> None:
        """
        Record the outcome of beat execution.
        
        Args:
            beat_id: ID of executed beat
            observation: Observation from video
            quality_result: Quality evaluation result
            video_uri: URI of rendered video
        """
        beat = self.state.intent_graph.story_beats.get(beat_id)
        if not beat:
            logger.warning(f"[story_director] unknown beat: {beat_id}")
            return
        
        # Check if quality is acceptable
        is_acceptable = True
        if quality_result:
            is_acceptable = quality_result.get("is_acceptable", True)
        
        if is_acceptable:
            self.state.intent_graph.mark_beat_complete(beat_id, video_uri)
            self.state.beats_completed += 1
            logger.info(f"[story_director] beat completed: {beat_id}")
        else:
            beat.status = IntentStatus.FAILED
            self.state.beats_failed += 1
            logger.warning(f"[story_director] beat failed quality: {beat_id}")
    
    def get_render_hints(self) -> Dict[str, Any]:
        """Get hints for video rendering based on current state."""
        beat = self.state.current_beat
        if not beat:
            return {}
        
        return {
            "characters": beat.characters,
            "location": beat.location,
            "objectives": beat.objectives,
            "actions": [a.to_dict() for a in self.state.current_actions],
            "style_hints": {
                "motion_intensity": max(
                    (a.motion_intensity for a in self.state.current_actions),
                    default=0.5
                ),
            },
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_story_director(
    episode_id: str,
    script: Optional[str] = None,
) -> StoryDirector:
    """
    Factory to create StoryDirector from episode ID.
    
    Args:
        episode_id: Episode ID
        script: Optional script to parse into intents
        
    Returns:
        Configured StoryDirector
    """
    intent_graph = StoryIntentGraph(episode_id=episode_id)
    return StoryDirector(intent_graph)


def load_story_director(data: Dict[str, Any]) -> StoryDirector:
    """Load StoryDirector from serialized data."""
    intent_graph = StoryIntentGraph.from_dict(data.get("intent_graph", {}))
    return StoryDirector(intent_graph)
