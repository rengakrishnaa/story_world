"""
Policy Engine Agent

Infrastructure-agnostic replacement for "StoryDirector".
Responsible for high-level decision making and goal pursuit.

Core Responsibilities:
1. Select active goals (from GoalGraph)
2. Generate/Select Action Proposals (Proposals)
3. React to Observer Verdicts (Success, Failure, Impossible, Uncertain)
4. Manage Goal Lifecycle (Pending -> Active -> Completed/Failed/Abandoned)
"""

from __future__ import annotations

import logging
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from models.goal_graph import (
    GoalGraph, SimulationGoal, ActionProposal, 
    ActionType, HierarchyLevel, ProposalStatus,
    create_action_proposal, GoalStatus
)
from models.episode_outcome import ObserverVerdict

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Type of decision made by the engine."""
    PROPOSAL = "proposal"       # Execute a proposal
    RETRY = "retry"             # Retry current proposal
    ABANDON = "abandon"         # Abandon current goal/proposal
    COMPLETE = "complete"       # Episode complete
    WAIT = "wait"               # Wait for more info


@dataclass
class PolicyDecision:
    """Decision made by the Policy Engine."""
    decision_type: DecisionType
    decision_id: str
    
    # Context
    goal_id: Optional[str] = None
    proposal_id: Optional[str] = None
    
    # Execution details
    actions: List[Any] = field(default_factory=list) # MicroActions
    reasoning: str = ""
    
    # Meta
    is_branch_point: bool = False
    downstream_impact: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type.value,
            "decision_id": self.decision_id,
            "goal_id": self.goal_id,
            "proposal_id": self.proposal_id,
            "reasoning": self.reasoning,
        }


class PolicyEngine:
    """
    High-level decision maker for the simulation.
    Replaces "StoryDirector".
    
    Operates on:
    - GoalGraph (Objectives)
    - WorldState (Current Reality)
    - ObservationResult (Feedback)
    """
    
    SYSTEM_PROMPT = """
You are the Policy Engine for a high-fidelity physics simulation.
Your job is to guide the simulation towards specified GOALS while respecting
PHYSICAL CONSTRAINTS and CAUSAL REALITY.

You have a set of active GOALS and a set of pending ACTION PROPOSALS.
You must choose the next PROPOSAL to execute to advance the GOALS.

CRITICAL RULES:
1. REALITY IS ABSOLUTE. If an observer reports an action is IMPOSSIBLE, you MUST NOT retry it identically.
2. GOALS ARE NOT PROMISES. If a goal is proven impossible, you must ABANDON it.
3. MINIMIZE NARRATIVE BIAS. Do not prioritize "drama" or "story". Prioritize CAUSALITY and GOAL COMPLETION.
4. HANDLE UNCERTAINTY. If the state is UNCERTAIN, propose actions to resolve uncertainty (EXPLORATION).

Input:
- Current Goals
- Pending Proposals
- World State
- Last Action Outcome (Verdict)

Output:
- Next Proposal ID
- Modifications (if any)
- Reasoning
"""

    def __init__(
        self,
        goal_graph: GoalGraph,
        model_name: str = "gemini-pro",
    ):
        self.goal_graph = goal_graph
        self.model_name = model_name
        self.history: List[Dict[str, Any]] = []
        
        # State tracking
        self.decisions_made = 0
        self.decisions_failed = 0
        self.goals_abandoned = 0
        
        logger.info(
            f"[policy_engine] initialized for {goal_graph.episode_id}: "
            f"{len(goal_graph.goals)} goals, "
            f"{len(goal_graph.proposals)} proposals"
        )
    
    def is_complete(self) -> bool:
        """Check if all goals are resolved (Completed, Failed, Abandoned)."""
        pending_goals = [
            g for g in self.goal_graph.goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.SUSPENDED)
        ]
        return len(pending_goals) == 0
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        total_goals = len(self.goal_graph.goals)
        completed = sum(1 for g in self.goal_graph.goals.values() if g.status == GoalStatus.COMPLETED)
        return {
            "is_complete": self.is_complete(),
            "goals_total": total_goals,
            "goals_completed": completed,
            "goals_abandoned": self.goals_abandoned,
            "decisions_made": self.decisions_made,
        }
    
    def next_decision(
        self,
        world_state: Optional[Dict[str, Any]] = None,
        last_observation: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """
        Get the next policy decision based on current state.
        
        Args:
            world_state: Current world state
            last_observation: Latest observation (optional)
            
        Returns:
            PolicyDecision
        """
        world_state = world_state or {}
        
        # 1. Process feedback from last observation (if any)
        if last_observation and self.goal_graph.current_proposal_id:
            self._process_observation_feedback(last_observation)
        
        # 2. Check if complete
        if self.is_complete():
            return PolicyDecision(
                decision_type=DecisionType.COMPLETE,
                decision_id=str(uuid.uuid4()),
                reasoning="All goals resolved",
            )
        
        # 3. Get next pending proposal
        next_proposal = self.goal_graph.get_next_proposal()
        
        if not next_proposal:
            return self._handle_no_pending_proposals(world_state)
        
        # 4. Evaluate proposal viability (Logic, not Drama)
        viability = self._evaluate_proposal_viability(next_proposal, world_state)
        
        if not viability["viable"]:
            # If strictly impossible, check if we can substitute or must abandon
            return self._handle_inviable_proposal(next_proposal, viability["reason"])
            
        # 5. Select micro-actions
        actions = self._select_actions(next_proposal, world_state)
        
        # Update state
        self.goal_graph.current_proposal_id = next_proposal.proposal_id
        next_proposal.status = ProposalStatus.ACTIVE
        self.decisions_made += 1
        
        return PolicyDecision(
            decision_type=DecisionType.PROPOSAL,
            decision_id=str(uuid.uuid4()),
            proposal_id=next_proposal.proposal_id,
            goal_id=next_proposal.contributes_to[0] if next_proposal.contributes_to else None,
            actions=actions,
            reasoning="Executing next proposal in sequence",
            is_branch_point=bool(next_proposal.alternatives),
            downstream_impact=len(next_proposal.depends_on),
        )

    def _process_observation_feedback(self, observation: Dict[str, Any]) -> None:
        """Process feedback to update goal status."""
        # Check verdict
        verdict = observation.get("verdict")
        disagreement = observation.get("disagreement_score", 0.0)
        
        current_prop_id = self.goal_graph.current_proposal_id
        if not current_prop_id:
            return
            
        prop = self.goal_graph.proposals.get(current_prop_id)
        if not prop:
            return

        # Phase 7: Handle IMPOSSIBLE / UNCERTAIN
        if verdict == "impossible" or verdict == "contradicts":
            logger.warning(f"[policy_engine] Proposal {current_prop_id} deemed IMPOSSIBLE")
            prop.status = ProposalStatus.FAILED
            
            # Impact on Goal?
            for goal_id in prop.contributes_to:
                goal = self.goal_graph.goals.get(goal_id)
                if goal:
                    # If this was a critical path, goal might be impossible
                    if prop.is_optional:
                         pass # Just skip
                    else:
                        # For now, mark goal as FAILED (will retry logic handle it? or abandon?)
                        # Strict mode: Abandon goal if physically impossible
                        goal.status = GoalStatus.IMPOSSIBLE
                        goal.status_reason = f"Critical proposal {prop.proposal_id} impossible: {observation.get('impossible_reason')}"
                        self.goals_abandoned += 1
                        logger.info(f"[policy_engine] Goal {goal_id} marked IMPOSSIBLE")

        elif verdict == "uncertain" or disagreement > 0.4:
            logger.warning(f"[policy_engine] High uncertainty for {current_prop_id}")
            # Could trigger exploration beat
            pass

    def _handle_no_pending_proposals(self, world_state: Dict[str, Any]) -> PolicyDecision:
        """Generate new proposals if goals assume unfinished."""
        # Find active/pending goals
        pending_goals = [
            g for g in self.goal_graph.goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.ACTIVE)
        ]
        
        if pending_goals:
            target_goal = pending_goals[0] # Priority logic here
            
            # Generate a bridge proposal
            new_proposal = self._generate_proposal_for_goal(target_goal, world_state)
            self.goal_graph.add_proposal(new_proposal)
            
            return PolicyDecision(
                decision_type=DecisionType.PROPOSAL,
                decision_id=str(uuid.uuid4()),
                proposal_id=new_proposal.proposal_id,
                goal_id=target_goal.goal_id,
                actions=self._select_actions(new_proposal, world_state),
                reasoning=f"Generated proposal for {target_goal.description}",
            )
            
        return PolicyDecision(
            decision_type=DecisionType.COMPLETE,
            decision_id=str(uuid.uuid4()),
            reasoning="No pending proposals or active goals",
        )

    def _evaluate_proposal_viability(
        self,
        proposal: ActionProposal,
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if proposal is viable in current world state."""
        # Check participants
        unavailable = []
        for p in proposal.participants:
            char_state = world_state.get("characters", {}).get(p, {})
            if char_state.get("available") is False:
                unavailable.append(p)
        
        if unavailable:
            return {"viable": False, "reason": f"Participants unavailable: {unavailable}"}
            
        return {"viable": True}

    def _handle_inviable_proposal(self, proposal: ActionProposal, reason: str) -> PolicyDecision:
        """Handle inviable proposal (substitute or fail)."""
        if proposal.alternatives:
            alt_id = proposal.alternatives[0]
            if alt_id in self.goal_graph.proposals:
                alt_prop = self.goal_graph.proposals[alt_id]
                proposal.status = ProposalStatus.SKIPPED
                return self.next_decision() # Recurse
        
        if proposal.is_optional:
            proposal.status = ProposalStatus.SKIPPED
            return self.next_decision()
            
        # Critical failure -> Abandon goal
        proposal.status = ProposalStatus.FAILED
        return PolicyDecision(
            decision_type=DecisionType.ABANDON,
            decision_id=str(uuid.uuid4()),
            proposal_id=proposal.proposal_id,
            reasoning=f"Critical proposal inviable: {reason}",
        )

    def _select_actions(
        self,
        proposal: ActionProposal,
        world_state: Dict[str, Any],
    ) -> List[Any]:
        """Select micro-actions for proposal."""
        # (Simplified logic from StoryDirector)
        actions = []
        # Try to find matching templates
        # If none, generate basic
        if not actions:
             actions = self._generate_basic_actions(proposal)
        return actions

    def _generate_basic_actions(self, proposal: ActionProposal) -> List[Any]:
        """Generate basic placeholder actions."""
        actions = []
        from models.goal_graph import MicroAction, ActionType
        
        for i, char in enumerate(proposal.participants[:3]):
             action = MicroAction(
                 action_id=f"basic-{uuid.uuid4().hex[:4]}",
                 action_type=ActionType.GESTURE,
                 description=f"{char} acts: {proposal.description}",
                 actor=char,
                 motion_intensity=0.5
             )
             actions.append(action)
        return actions

    def _generate_proposal_for_goal(
        self,
        goal: SimulationGoal,
        world_state: Dict[str, Any]
    ) -> ActionProposal:
        """Generate a proposal to advance a goal."""
        return create_action_proposal(
            description=f"Advance goal: {goal.description}",
            objectives=[goal.description],
            participants=[], # Would need logic to select
            contributes_to=[goal.goal_id]
        )

    def record_outcome(
        self,
        beat_id: str, # Legacy param name support
        observation: Optional[Dict[str, Any]] = None,
        quality_result: Optional[Dict[str, Any]] = None,
        video_uri: Optional[str] = None,
        proposal_id: Optional[str] = None
    ) -> None:
        """Record outcome."""
        pid = proposal_id or beat_id
        prop = self.goal_graph.proposals.get(pid)
        if not prop:
            return
            
        is_success = True
        if quality_result and not quality_result.get("is_acceptable", True):
            is_success = False
        
        if is_success:
            self.goal_graph.mark_proposal_complete(pid, video_uri)
            logger.info(f"[policy_engine] Proposal completed: {pid}")
        else:
            # Handle failure (retry logic handled by Loop usually, but here we track status)
            # If max retries reached externally, it stays FAILED
            pass

    def get_render_hints(self) -> Dict[str, Any]:
        """Get render hints."""
        pid = self.goal_graph.current_proposal_id
        prop = self.goal_graph.proposals.get(pid) if pid else None
        
        if not prop:
            return {}
            
        return {
            "characters": prop.participants,
            "location": prop.context_location,
            "objectives": prop.objectives,
        }

# ============================================================================
# Convenience Functions
# ============================================================================

def create_policy_engine(
    episode_id: str,
) -> PolicyEngine:
    """Factory."""
    graph = GoalGraph(episode_id=episode_id)
    return PolicyEngine(graph)

