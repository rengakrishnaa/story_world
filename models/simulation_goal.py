"""
Simulation Goal Model

Infrastructure-agnostic replacement for "MacroIntent".
Represents immutable objectives that the Policy Engine attempts to satisfy.

Key Difference from StoryIntent:
- Goals can be INVALIDATED by world state changes.
- Goals can be ABANDONED based on cost/benefit analysis.
- Goals can be SUSPENDED due to high uncertainty.
- No narrative "fixing" - if a goal is impossible, it fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Lifecycle status of a simulation goal."""
    PENDING = "pending"         # Initial state
    ACTIVE = "active"           # Currently being pursued
    COMPLETED = "completed"     # Successfully achieved
    
    # Failure Modes
    FAILED = "failed"           # Execution failed (can retry)
    IMPOSSIBLE = "impossible"   # Constraint discovered prevents goal (Observer verified)
    INVALIDATED = "invalidated" # World state change made goal irrelevant/nonsensical
    ABANDONED = "abandoned"     # Policy Engine chose to stop (cost > value)
    SUSPENDED = "suspended"     # Uncertainty too high, awaiting more evidence


class GoalType(Enum):
    """Types of simulation goals (Neutralized terminology)."""
    STATE_TARGET = "state_target"       # Reach a specific world state
    CONSTRAINT_TEST = "constraint_test" # Verify a physical constraint
    INVARIANCE_TEST = "invariance_test" # Ensure something does NOT happen
    EXPLORATION = "exploration"         # Map unknown state space
    OPTIMIZATION = "optimization"       # Maximize/minimize a metric


@dataclass
class SimulationGoal:
    """
    A specific objective for the simulation.
    Replaces MacroIntent with rigorous state lifecycle.
    """
    goal_id: str
    description: str
    goal_type: GoalType
    
    # Target condition
    target_state_subset: Dict[str, Any] = field(default_factory=dict)
    
    # Constraints
    hard_constraints: List[str] = field(default_factory=list)  # Must never be violated
    max_cost_threshold: float = 100.0
    
    # Status
    status: GoalStatus = GoalStatus.PENDING
    status_reason: Optional[str] = None  # Why is it impossible/abandoned?
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Evidence linking (Provenance)
    terminating_observation_id: Optional[str] = None  # Validates IMPOSSIBLE status
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "target_state_subset": self.target_state_subset,
            "hard_constraints": self.hard_constraints,
            "max_cost_threshold": self.max_cost_threshold,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "terminating_observation_id": self.terminating_observation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationGoal":
        goal = cls(
            goal_id=data["goal_id"],
            description=data.get("description", ""),
            goal_type=GoalType(data.get("goal_type", "state_target")),
            target_state_subset=data.get("target_state_subset", {}),
            hard_constraints=data.get("hard_constraints", []),
            max_cost_threshold=data.get("max_cost_threshold", 100.0),
            status=GoalStatus(data.get("status", "pending")),
            status_reason=data.get("status_reason"),
            terminating_observation_id=data.get("terminating_observation_id"),
        )
        if "created_at" in data:
            goal.created_at = datetime.fromisoformat(data["created_at"])
        if "completed_at" in data and data["completed_at"]:
            goal.completed_at = datetime.fromisoformat(data["completed_at"])
        return goal


def create_simulation_goal(
    description: str,
    target_state: Dict[str, Any],
    goal_type: GoalType = GoalType.STATE_TARGET,
) -> SimulationGoal:
    """Factory for simulation goals."""
    return SimulationGoal(
        goal_id=f"goal-{uuid.uuid4().hex[:8]}",
        description=description,
        target_state_subset=target_state,
        goal_type=goal_type,
    )
