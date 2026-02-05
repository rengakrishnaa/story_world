"""
Goal Graph Model

Infrastructure-agnostic replacement for "StoryIntentGraph".
Represents the hierarchical goal structure of a generic simulation.

Hierarchy:
- GOAL: Fixed high-level objective (SimulationGoal)
- PROPOSAL: Flexible plan segment (ActionProposal)
- ACTION: Reactive atomic execution (MicroAction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from datetime import datetime
import uuid
import json

from models.simulation_goal import SimulationGoal, GoalType, GoalStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Hierarchy Levels
# ============================================================================

class HierarchyLevel(Enum):
    """Hierarchy levels of simulation control."""
    GOAL = "goal"       # High-level objective
    PROPOSAL = "proposal" # Mid-level plan segment
    ACTION = "action"   # Low-level execution


class ProposalStatus(Enum):
    """Status of an action proposal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALIDATED = "invalidated" # New Phase 7 state


# ============================================================================
# Mid-Level: Action Proposals (Flexible)
# ============================================================================

@dataclass
class ActionProposal:
    """
    Mid-level action proposal.
    Replaces "StoryBeat". Represents a proposed segment of simulation.
    Flexible: can be reordered, replaced, or adapted.
    """
    proposal_id: str
    description: str
    
    # What this proposal should achieve
    objectives: List[str] = field(default_factory=list)
    
    # Expected state changes
    expected_state_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Resources/Entities involved
    participants: List[str] = field(default_factory=list) # Was "characters"
    
    # Location/Context
    context_location: Optional[str] = None # Was "location"
    
    # Contributes to goals
    contributes_to: List[str] = field(default_factory=list)
    
    # Ordering
    suggested_position: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)
    
    # Flexibility
    is_optional: bool = False
    alternatives: List[str] = field(default_factory=list)
    
    # Status
    status: ProposalStatus = ProposalStatus.PENDING
    actual_video_uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "description": self.description,
            "objectives": self.objectives,
            "expected_state_changes": self.expected_state_changes,
            "participants": self.participants,
            "context_location": self.context_location,
            "contributes_to": self.contributes_to,
            "suggested_position": self.suggested_position,
            "depends_on": self.depends_on,
            "is_optional": self.is_optional,
            "alternatives": self.alternatives,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionProposal":
        return cls(
            proposal_id=data["proposal_id"],
            description=data.get("description", ""),
            objectives=data.get("objectives", []),
            expected_state_changes=data.get("expected_state_changes", {}),
            participants=data.get("participants", []),
            context_location=data.get("context_location") or data.get("location"),
            contributes_to=data.get("contributes_to", []),
            suggested_position=data.get("suggested_position"),
            depends_on=data.get("depends_on", []),
            is_optional=data.get("is_optional", False),
            alternatives=data.get("alternatives", []),
            status=ProposalStatus(data.get("status", "pending")),
        )


# ============================================================================
# Low-Level: Micro Actions
# ============================================================================

class ActionType(Enum):
    """Types of atomic actions."""
    DIALOGUE = "dialogue"
    MOVEMENT = "movement"
    GESTURE = "gesture"
    INTERACTION = "interaction"
    ENVIRONMENTAL = "environmental"
    COMBAT = "combat"
    WAIT = "wait"           # Epistemically rational inaction
    CUSTOM = "custom"


@dataclass
class MicroAction:
    """Atomic action definition."""
    action_id: str
    action_type: ActionType
    description: str
    
    actor: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    
    requires_state: Dict[str, Any] = field(default_factory=dict)
    state_effects: Dict[str, Any] = field(default_factory=dict)
    
    motion_intensity: float = 0.5
    duration_hint_sec: float = 2.0
    camera_hint: Optional[str] = None
    
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
# Goal Graph (Complete Plan)
# ============================================================================

@dataclass
class GoalGraph:
    """
    Complete hierarchical plan for a simulation episode.
    Replaces "StoryIntentGraph".
    """
    episode_id: str
    
    # Layers
    goals: Dict[str, SimulationGoal] = field(default_factory=dict)
    proposals: Dict[str, ActionProposal] = field(default_factory=dict)
    action_templates: Dict[str, MicroAction] = field(default_factory=dict)
    
    # State
    current_proposal_id: Optional[str] = None
    completed_proposals: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_goal(self, goal: SimulationGoal) -> None:
        """Add a simulation goal."""
        self.goals[goal.goal_id] = goal
    
    def add_proposal(self, proposal: ActionProposal) -> None:
        """Add an action proposal."""
        self.proposals[proposal.proposal_id] = proposal
    
    def add_action_template(self, action: MicroAction) -> None:
        self.action_templates[action.action_id] = action
    
    def get_pending_proposals(self) -> List[ActionProposal]:
        """Get executable proposals."""
        pending = []
        for prop in self.proposals.values():
            if prop.status != ProposalStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = all(
                self.proposals.get(dep_id, ActionProposal(dep_id, "")).status == ProposalStatus.COMPLETED
                for dep_id in prop.depends_on
            )
            
            if deps_satisfied:
                pending.append(prop)
        
        return sorted(pending, key=lambda p: p.suggested_position or 999)
    
    def get_next_proposal(self) -> Optional[ActionProposal]:
        pending = self.get_pending_proposals()
        return pending[0] if pending else None
    
    def mark_proposal_complete(self, proposal_id: str, video_uri: Optional[str] = None) -> None:
        if proposal_id in self.proposals:
            prop = self.proposals[proposal_id]
            prop.status = ProposalStatus.COMPLETED
            prop.actual_video_uri = video_uri
            self.completed_proposals.append(proposal_id)
            self._check_goal_completions()
    
    def _check_goal_completions(self) -> None:
        for goal in self.goals.values():
            if goal.status == GoalStatus.COMPLETED:
                continue
            
            # Check if contributing proposals done
            contributing = [
                p for p in self.proposals.values()
                if goal.goal_id in p.contributes_to
            ]
            
            if contributing and all(p.status == ProposalStatus.COMPLETED for p in contributing):
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.utcnow()
    
    def get_active_participants(self) -> Set[str]:
        chars = set()
        if self.current_proposal_id and self.current_proposal_id in self.proposals:
            chars.update(self.proposals[self.current_proposal_id].participants)
        for prop in self.get_pending_proposals()[:3]:
            chars.update(prop.participants)
        return chars
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "proposals": {k: v.to_dict() for k, v in self.proposals.items()},
            "action_templates": {k: v.to_dict() for k, v in self.action_templates.items()},
            "current_proposal_id": self.current_proposal_id,
            "completed_proposals": self.completed_proposals,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalGraph":
        graph = cls(episode_id=data["episode_id"])
        
        for k, v in data.get("goals", {}).items():
            graph.goals[k] = SimulationGoal.from_dict(v)
        
        for k, v in data.get("proposals", {}).items():
            graph.proposals[k] = ActionProposal.from_dict(v)
            
        for k, v in data.get("action_templates", {}).items():
            graph.action_templates[k] = MicroAction.from_dict(v)
            
        graph.current_proposal_id = data.get("current_proposal_id")
        graph.completed_proposals = data.get("completed_proposals", [])
        return graph


# ============================================================================
# Factory Functions
# ============================================================================

def create_action_proposal(
    description: str,
    objectives: Optional[List[str]] = None,
    participants: Optional[List[str]] = None,
    context_location: Optional[str] = None,
    contributes_to: Optional[List[str]] = None,
) -> ActionProposal:
    """Factory for action proposal."""
    return ActionProposal(
        proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
        description=description,
        objectives=objectives or [],
        participants=participants or [],
        context_location=context_location,
        contributes_to=contributes_to or [],
    )

def create_micro_action(
    description: str,
    action_type: ActionType = ActionType.CUSTOM,
    actor: Optional[str] = None,
    state_effects: Optional[Dict[str, Any]] = None,
) -> MicroAction:
    return MicroAction(
        action_id=f"action-{uuid.uuid4().hex[:8]}",
        action_type=action_type,
        description=description,
        actor=actor,
        state_effects=state_effects or {},
    )
