"""
Episode Outcome Types - Video-Native Compliance

Defines first-class failure states for episodes:
- GOAL_ACHIEVED: Episode completed successfully
- GOAL_IMPOSSIBLE: Observer detected physics/world contradiction
- GOAL_ABANDONED: Budget exhausted without solution
- DEAD_STATE: No valid actions remain (REQUIRES physics constraint)
- CONSTRAINT_VIOLATED: Hard simulation limit hit
- UNCERTAIN_TERMINATION: Stopped with only epistemic constraints (video_unavailable, etc.)
  - video_unavailable must NEVER be terminal by itself; use this instead
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json

# Constraints that indicate missing evidence, not physics. Never conclude physics from these.
# Naming: insufficient_physical_evidence = solver block (missing turn radius, mass, etc.)
#         observer_unavailable = confidence penalty (observer infra failed, solver-only)
EPISTEMIC_CONSTRAINTS = frozenset({
    "video_unavailable", "insufficient_evidence", "insufficient_physical_evidence",
    "observer_unavailable", "missing_video", "observer_exception",
    "json_parse_error", "observer_no_observation", "observation_occluded",
})


def is_epistemic_only(constraints: List[str]) -> bool:
    """True if all constraints are epistemic (missing evidence). No physics learned."""
    if not constraints:
        return True
    normalized = {c.strip().lower().replace("-", "_") for c in constraints if c}
    return normalized.issubset(EPISTEMIC_CONSTRAINTS)


def has_physics_constraint(constraints: List[str]) -> bool:
    """True if at least one constraint indicates discovered physics (not epistemic)."""
    return not is_epistemic_only(constraints)


# ============================================================================
# Episode Outcome (First-Class Failure States)
# ============================================================================

class EpisodeOutcome(Enum):
    """
    First-class episode outcomes including failure states.
    
    Video-Native Principle: Episodes CAN fail. Success is not mandatory.
    """
    # Success states
    GOAL_ACHIEVED = "goal_achieved"
    
    # Failure states (NEW - critical for video-native compliance)
    GOAL_IMPOSSIBLE = "goal_impossible"      # Physics/world contradicts goal
    GOAL_ABANDONED = "goal_abandoned"        # Budget exhausted, no solution found
    DEAD_STATE = "dead_state"                # No valid actions remain (requires physics constraint)
    CONSTRAINT_VIOLATED = "constraint_violated"  # Simulation hit hard limit
    UNCERTAIN_TERMINATION = "uncertain_termination"  # Stopped with unresolved uncertainty
    EPISTEMICALLY_INCOMPLETE = "epistemically_incomplete"  # Halted: missing evidence, constraints unevaluable
    
    # In-progress states
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    
    @property
    def is_terminal(self) -> bool:
        """True if episode has reached a final state."""
        return self in {
            EpisodeOutcome.GOAL_ACHIEVED,
            EpisodeOutcome.GOAL_IMPOSSIBLE,
            EpisodeOutcome.GOAL_ABANDONED,
            EpisodeOutcome.DEAD_STATE,
            EpisodeOutcome.CONSTRAINT_VIOLATED,
            EpisodeOutcome.UNCERTAIN_TERMINATION,
            EpisodeOutcome.EPISTEMICALLY_INCOMPLETE,
        }
    
    @property
    def is_success(self) -> bool:
        """True if episode succeeded."""
        return self == EpisodeOutcome.GOAL_ACHIEVED
    
    @property
    def is_failure(self) -> bool:
        """True if episode failed (not just incomplete). Epistemic halt is NOT failure."""
        return self in {
            EpisodeOutcome.GOAL_IMPOSSIBLE,
            EpisodeOutcome.GOAL_ABANDONED,
            EpisodeOutcome.DEAD_STATE,
            EpisodeOutcome.CONSTRAINT_VIOLATED,
            EpisodeOutcome.UNCERTAIN_TERMINATION,
        }


# ============================================================================
# Observer Verdict (Constraint Authority)
# ============================================================================

class ObserverVerdict(Enum):
    """
    Observer verdicts with authority to block intent.
    
    Video-Native Principle: Observer can declare actions impossible
    and force episode termination.
    """
    # Normal outcomes
    VALID = "valid"              # Action succeeded as intended
    DEGRADED = "degraded"        # Action partially succeeded
    FAILED = "failed"            # Action failed but recoverable
    
    # Blocking outcomes (NEW - critical for video-native compliance)
    IMPOSSIBLE = "impossible"    # Action is physically impossible
    CONTRADICTS = "contradicts"  # Action contradicts established state
    BLOCKS_INTENT = "blocks"     # Action makes macro goal unreachable
    
    # Epistemic outcomes (Phase 7)
    UNCERTAIN = "uncertain"      # Information insufficient/conflicting
    
    @property
    def allows_continuation(self) -> bool:
        """True if episode can continue after this verdict. UNCERTAIN = explore more, not abort."""
        return self in {
            ObserverVerdict.VALID,
            ObserverVerdict.DEGRADED,
            ObserverVerdict.FAILED,
            ObserverVerdict.UNCERTAIN,  # Video ambiguous -> reduce confidence, explore more
        }
    
    @property
    def forces_termination(self) -> bool:
        """True if this verdict should terminate the episode."""
        return self in {
            ObserverVerdict.IMPOSSIBLE,
            ObserverVerdict.CONTRADICTS,
            ObserverVerdict.BLOCKS_INTENT,
        }


# ============================================================================
# Termination Reason
# ============================================================================

@dataclass
class TerminationReason:
    """Detailed explanation of why an episode terminated."""
    outcome: EpisodeOutcome
    verdict: Optional[ObserverVerdict] = None
    
    # What caused termination
    trigger: str = ""  # e.g., "observer_veto", "budget_exhausted"
    description: str = ""  # Human-readable explanation
    
    # Which intent/goal was affected
    blocked_intent_id: Optional[str] = None
    blocked_beat_id: Optional[str] = None
    
    # Evidence
    physics_violation: Optional[str] = None  # e.g., "Character cannot fly"
    state_contradiction: Optional[str] = None  # e.g., "Character already dead"
    
    # Metadata
    terminated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "verdict": self.verdict.value if self.verdict else None,
            "trigger": self.trigger,
            "description": self.description,
            "blocked_intent_id": self.blocked_intent_id,
            "blocked_beat_id": self.blocked_beat_id,
            "physics_violation": self.physics_violation,
            "state_contradiction": self.state_contradiction,
            "terminated_at": self.terminated_at.isoformat(),
        }


# ============================================================================
# State-Centric Episode Result
# ============================================================================

@dataclass
class EpisodeResult:
    """
    State-centric episode result.
    
    Video-Native Principle: Primary output is state delta, not video.
    Video is optional debug artifact.
    """
    episode_id: str
    outcome: EpisodeOutcome
    
    # Primary outputs (state-first)
    state_delta: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    total_cost_usd: float = 0.0
    branches_created: int = 0
    
    # Termination details
    termination_reason: Optional[TerminationReason] = None
    
    # Metrics (execution scaffolding; success = observer-validated transitions)
    beats_attempted: int = 0
    beats_completed: int = 0
    beats_failed: int = 0
    total_observations: int = 0
    
    # Discovered constraints (from observer verdicts)
    constraints_discovered: List[str] = field(default_factory=list)
    
    # Debug artifacts (OPTIONAL - video is not primary)
    debug: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.outcome.is_success
    
    @property
    def is_terminal(self) -> bool:
        return self.outcome.is_terminal
    
    @property
    def is_failure(self) -> bool:
        return self.outcome.is_failure
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "outcome": self.outcome.value,
            "is_success": self.is_success,
            "is_terminal": self.is_terminal,
            
            # State-first (cost alias for API contract)
            "state_delta": self.state_delta,
            "confidence": self.confidence,
            "total_cost_usd": self.total_cost_usd,
            "cost": self.total_cost_usd,
            "branches_created": self.branches_created,
            "constraints_discovered": self.constraints_discovered,
            
            # Termination
            "termination_reason": (
                self.termination_reason.to_dict()
                if self.termination_reason else None
            ),
            
            # Metrics
            "metrics": {
                "beats_attempted": self.beats_attempted,
                "beats_completed": self.beats_completed,
                "beats_failed": self.beats_failed,
                "total_observations": self.total_observations,
            },
            
            # Debug (optional)
            "debug": self.debug if self.debug else None,
        }
    
    def with_video_debug(self, video_uri: str, retention_hours: int = 24) -> "EpisodeResult":
        """Add video as optional debug artifact."""
        self.debug["video_uri"] = video_uri
        self.debug["video_retention_hours"] = retention_hours
        self.debug["video_is_debug_only"] = True
        return self


# ============================================================================
# Constraint Violation
# ============================================================================

@dataclass
class ConstraintViolation:
    """A detected violation that may block execution."""
    violation_type: str  # "physics", "continuity", "state", "budget"
    severity: str  # "warning", "error", "fatal"
    description: str
    
    # What triggered this
    beat_id: Optional[str] = None
    observation_id: Optional[str] = None
    
    # Impact
    blocks_current_beat: bool = False
    blocks_macro_intent: bool = False
    forces_episode_termination: bool = False
    
    # Suggested resolution
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "severity": self.severity,
            "description": self.description,
            "beat_id": self.beat_id,
            "blocks_current_beat": self.blocks_current_beat,
            "blocks_macro_intent": self.blocks_macro_intent,
            "forces_episode_termination": self.forces_episode_termination,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_success_result(
    episode_id: str,
    state_delta: Dict[str, Any],
    cost_usd: float = 0.0,
    confidence: float = 1.0,
) -> EpisodeResult:
    """Create a successful episode result."""
    return EpisodeResult(
        episode_id=episode_id,
        outcome=EpisodeOutcome.GOAL_ACHIEVED,
        state_delta=state_delta,
        confidence=confidence,
        total_cost_usd=cost_usd,
    )


def create_failure_result(
    episode_id: str,
    outcome: EpisodeOutcome,
    reason: str,
    trigger: str = "unknown",
) -> EpisodeResult:
    """Create a failed episode result."""
    return EpisodeResult(
        episode_id=episode_id,
        outcome=outcome,
        termination_reason=TerminationReason(
            outcome=outcome,
            trigger=trigger,
            description=reason,
        ),
    )


def create_impossible_result(
    episode_id: str,
    physics_violation: str,
    blocked_intent_id: Optional[str] = None,
) -> EpisodeResult:
    """Create result for physically impossible action."""
    return EpisodeResult(
        episode_id=episode_id,
        outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
        termination_reason=TerminationReason(
            outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
            verdict=ObserverVerdict.IMPOSSIBLE,
            trigger="observer_veto",
            description=f"Action physically impossible: {physics_violation}",
            physics_violation=physics_violation,
            blocked_intent_id=blocked_intent_id,
        ),
    )


def create_dead_state_result(
    episode_id: str,
    reason: str,
) -> EpisodeResult:
    """Create result for dead-state termination."""
    return EpisodeResult(
        episode_id=episode_id,
        outcome=EpisodeOutcome.DEAD_STATE,
        termination_reason=TerminationReason(
            outcome=EpisodeOutcome.DEAD_STATE,
            trigger="no_valid_actions",
            description=reason,
        ),
    )
