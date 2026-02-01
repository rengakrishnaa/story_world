"""
Budget Controller

Elastic, value-driven regeneration budget system.
Retries are computation, not failure. Budget is allocated based on expected value.

Design Principles:
1. Budget is elastic, not fixed
2. Allocation based on expected downstream value
3. Diminishing returns tracked per-beat
4. Episode-level and beat-level budgets
"""

from __future__ import annotations

import os
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Budget Configuration
# ============================================================================

@dataclass
class BudgetConfig:
    """Configuration for budget controller."""
    # Episode-level limits
    max_episode_budget_usd: float = 10.0
    max_episode_retries: int = 100
    
    # Beat-level limits
    max_beat_retries: int = 5
    base_retry_budget_usd: float = 0.10
    
    # Cost assumptions
    render_cost_usd: float = float(os.getenv("RENDER_COST_USD", "0.05"))
    observation_cost_usd: float = 0.001
    
    # Diminishing returns
    diminishing_returns_factor: float = 0.7  # Each retry worth 70% of previous
    min_expected_value: float = 0.1  # Minimum value to justify retry
    
    # Value multipliers
    branch_point_multiplier: float = 2.0
    downstream_factor: float = 0.1  # Per downstream beat
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Budget State
# ============================================================================

@dataclass
class BeatBudgetState:
    """Budget state for a single beat."""
    beat_id: str
    initial_budget_usd: float
    spent_usd: float = 0.0
    attempts: int = 0
    successful: bool = False
    
    # Value tracking
    initial_expected_value: float = 1.0
    current_expected_value: float = 1.0
    
    # History of attempt outcomes
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def remaining_budget_usd(self) -> float:
        return max(0, self.initial_budget_usd - self.spent_usd)
    
    @property
    def can_retry(self) -> bool:
        return (
            not self.successful
            and self.remaining_budget_usd > 0
            and self.current_expected_value >= 0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "initial_budget_usd": self.initial_budget_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_budget_usd,
            "attempts": self.attempts,
            "successful": self.successful,
            "current_expected_value": self.current_expected_value,
        }


@dataclass
class EpisodeBudgetState:
    """Budget state for entire episode."""
    episode_id: str
    total_budget_usd: float
    spent_usd: float = 0.0
    total_attempts: int = 0
    successful_beats: int = 0
    failed_beats: int = 0
    
    # Beat-level tracking
    beat_states: Dict[str, BeatBudgetState] = field(default_factory=dict)
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def remaining_budget_usd(self) -> float:
        return max(0, self.total_budget_usd - self.spent_usd)
    
    @property
    def utilization_rate(self) -> float:
        if self.total_budget_usd == 0:
            return 0
        return self.spent_usd / self.total_budget_usd
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "total_budget_usd": self.total_budget_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_budget_usd,
            "utilization_rate": self.utilization_rate,
            "total_attempts": self.total_attempts,
            "successful_beats": self.successful_beats,
            "failed_beats": self.failed_beats,
            "beat_count": len(self.beat_states),
        }


# ============================================================================
# Budget Decision
# ============================================================================

class BudgetDecision(Enum):
    """Decisions from budget controller."""
    PROCEED = "proceed"
    RETRY = "retry"
    ACCEPT_DEGRADED = "accept_degraded"
    ABORT = "abort"


@dataclass
class BudgetResult:
    """Result of budget allocation decision."""
    decision: BudgetDecision
    allocated_budget_usd: float
    expected_value: float
    
    # Reasoning
    reason: str = ""
    retry_count: int = 0
    remaining_episode_budget_usd: float = 0.0
    remaining_beat_budget_usd: float = 0.0
    
    # Recommendations
    should_lower_quality: bool = False
    quality_reduction_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "allocated_budget_usd": self.allocated_budget_usd,
            "expected_value": self.expected_value,
            "reason": self.reason,
            "retry_count": self.retry_count,
            "remaining_episode_budget_usd": self.remaining_episode_budget_usd,
            "remaining_beat_budget_usd": self.remaining_beat_budget_usd,
            "should_lower_quality": self.should_lower_quality,
        }


# ============================================================================
# Budget Controller
# ============================================================================

class BudgetController:
    """
    Elastic budget controller for regeneration decisions.
    
    Key features:
    1. Episode-level and beat-level budgets
    2. Value-driven allocation
    3. Diminishing returns tracking
    4. Graceful degradation
    
    Usage:
        controller = BudgetController(episode_id="ep-001")
        
        # Request budget for initial render
        result = controller.request_budget(
            beat_id="beat-001",
            expected_value=1.0,
            is_branch_point=True,
        )
        
        if result.decision == BudgetDecision.PROCEED:
            # Render beat...
            controller.record_attempt(
                beat_id="beat-001",
                success=True,
                cost_usd=0.05,
                quality_score=0.85,
            )
    """
    
    def __init__(
        self,
        episode_id: str,
        config: Optional[BudgetConfig] = None,
    ):
        self.config = config or BudgetConfig()
        self.state = EpisodeBudgetState(
            episode_id=episode_id,
            total_budget_usd=self.config.max_episode_budget_usd,
        )
        
        logger.info(
            f"[budget_controller] initialized for {episode_id}: "
            f"${self.config.max_episode_budget_usd} budget"
        )
    
    def request_budget(
        self,
        beat_id: str,
        expected_value: float = 1.0,
        is_branch_point: bool = False,
        downstream_beats: int = 0,
        is_retry: bool = False,
        previous_quality: float = 0.0,
    ) -> BudgetResult:
        """
        Request budget allocation for a beat.
        
        Args:
            beat_id: ID of beat to render
            expected_value: Base expected value (0-1)
            is_branch_point: Whether this is a story branch point
            downstream_beats: Number of beats that depend on this
            is_retry: Whether this is a retry attempt
            previous_quality: Quality score from previous attempt
            
        Returns:
            BudgetResult with allocation decision
        """
        # Get or create beat state
        if beat_id not in self.state.beat_states:
            initial_budget = self._compute_beat_budget(
                expected_value,
                is_branch_point,
                downstream_beats,
            )
            self.state.beat_states[beat_id] = BeatBudgetState(
                beat_id=beat_id,
                initial_budget_usd=initial_budget,
                initial_expected_value=expected_value,
                current_expected_value=expected_value,
            )
        
        beat_state = self.state.beat_states[beat_id]
        
        # Check episode-level budget
        if self.state.remaining_budget_usd <= 0:
            return BudgetResult(
                decision=BudgetDecision.ABORT,
                allocated_budget_usd=0,
                expected_value=0,
                reason="Episode budget exhausted",
                remaining_episode_budget_usd=0,
            )
        
        # Check beat-level limits
        if beat_state.attempts >= self.config.max_beat_retries:
            if previous_quality > 0.5:
                return BudgetResult(
                    decision=BudgetDecision.ACCEPT_DEGRADED,
                    allocated_budget_usd=0,
                    expected_value=beat_state.current_expected_value,
                    reason=f"Max retries ({self.config.max_beat_retries}) reached, accepting degraded quality",
                    retry_count=beat_state.attempts,
                    should_lower_quality=True,
                    quality_reduction_factor=0.8,
                )
            return BudgetResult(
                decision=BudgetDecision.ABORT,
                allocated_budget_usd=0,
                expected_value=0,
                reason=f"Max retries reached and quality too low ({previous_quality})",
                retry_count=beat_state.attempts,
            )
        
        # Compute expected value with diminishing returns
        if is_retry:
            beat_state.current_expected_value *= self.config.diminishing_returns_factor
            
            # Check if retry is still worthwhile
            if beat_state.current_expected_value < self.config.min_expected_value:
                return BudgetResult(
                    decision=BudgetDecision.ACCEPT_DEGRADED,
                    allocated_budget_usd=0,
                    expected_value=beat_state.current_expected_value,
                    reason="Expected value below minimum threshold",
                    retry_count=beat_state.attempts,
                    should_lower_quality=True,
                )
        
        # Allocate budget
        render_cost = self.config.render_cost_usd
        observation_cost = self.config.observation_cost_usd
        total_cost = render_cost + observation_cost
        
        if total_cost > self.state.remaining_budget_usd:
            return BudgetResult(
                decision=BudgetDecision.ABORT,
                allocated_budget_usd=0,
                expected_value=0,
                reason="Insufficient remaining budget",
                remaining_episode_budget_usd=self.state.remaining_budget_usd,
            )
        
        if total_cost > beat_state.remaining_budget_usd:
            # Try to borrow from episode pool
            if total_cost <= self.state.remaining_budget_usd * 0.2:
                # Allow borrowing up to 20% of remaining
                beat_state.initial_budget_usd += total_cost
            else:
                return BudgetResult(
                    decision=BudgetDecision.ACCEPT_DEGRADED,
                    allocated_budget_usd=0,
                    expected_value=beat_state.current_expected_value,
                    reason="Beat budget exhausted",
                    remaining_beat_budget_usd=beat_state.remaining_budget_usd,
                    should_lower_quality=True,
                )
        
        decision = BudgetDecision.RETRY if is_retry else BudgetDecision.PROCEED
        
        return BudgetResult(
            decision=decision,
            allocated_budget_usd=total_cost,
            expected_value=beat_state.current_expected_value,
            reason="Budget allocated",
            retry_count=beat_state.attempts,
            remaining_episode_budget_usd=self.state.remaining_budget_usd - total_cost,
            remaining_beat_budget_usd=beat_state.remaining_budget_usd - total_cost,
        )
    
    def record_attempt(
        self,
        beat_id: str,
        success: bool,
        cost_usd: float,
        quality_score: float = 0.0,
        error: Optional[str] = None,
    ) -> BeatBudgetState:
        """
        Record an attempt outcome.
        
        Args:
            beat_id: ID of beat
            success: Whether attempt succeeded
            cost_usd: Cost of this attempt
            quality_score: Quality score achieved
            error: Error message if failed
            
        Returns:
            Updated beat budget state
        """
        beat_state = self.state.beat_states.get(beat_id)
        if not beat_state:
            beat_state = BeatBudgetState(
                beat_id=beat_id,
                initial_budget_usd=self.config.base_retry_budget_usd,
            )
            self.state.beat_states[beat_id] = beat_state
        
        # Update beat state
        beat_state.attempts += 1
        beat_state.spent_usd += cost_usd
        beat_state.successful = success
        
        beat_state.attempt_history.append({
            "attempt": beat_state.attempts,
            "success": success,
            "cost_usd": cost_usd,
            "quality_score": quality_score,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Update episode state
        self.state.spent_usd += cost_usd
        self.state.total_attempts += 1
        
        if success:
            self.state.successful_beats += 1
        elif beat_state.attempts >= self.config.max_beat_retries:
            self.state.failed_beats += 1
        
        logger.info(
            f"[budget_controller] recorded attempt: beat={beat_id}, "
            f"success={success}, cost=${cost_usd:.3f}, "
            f"episode_spent=${self.state.spent_usd:.2f}"
        )
        
        return beat_state
    
    def _compute_beat_budget(
        self,
        expected_value: float,
        is_branch_point: bool,
        downstream_beats: int,
    ) -> float:
        """Compute initial budget for a beat based on value."""
        base = self.config.base_retry_budget_usd * self.config.max_beat_retries
        
        # Value multiplier
        value_mult = expected_value
        
        # Branch point multiplier
        if is_branch_point:
            value_mult *= self.config.branch_point_multiplier
        
        # Downstream impact
        downstream_mult = 1.0 + (downstream_beats * self.config.downstream_factor)
        
        budget = base * value_mult * downstream_mult
        
        # Cap at fraction of episode budget
        max_beat_budget = self.state.total_budget_usd * 0.2
        
        return min(budget, max_beat_budget)
    
    def should_retry(
        self,
        beat_id: str,
        quality_score: float,
        quality_threshold: float = 0.7,
    ) -> BudgetResult:
        """
        Convenience method: check if retry is worthwhile.
        
        Args:
            beat_id: ID of beat
            quality_score: Quality score from current attempt
            quality_threshold: Required quality threshold
            
        Returns:
            BudgetResult with retry decision
        """
        if quality_score >= quality_threshold:
            beat_state = self.state.beat_states.get(beat_id)
            if beat_state:
                beat_state.successful = True
            
            return BudgetResult(
                decision=BudgetDecision.PROCEED,
                allocated_budget_usd=0,
                expected_value=1.0,
                reason="Quality threshold met",
            )
        
        return self.request_budget(
            beat_id=beat_id,
            is_retry=True,
            previous_quality=quality_score,
        )
    
    def get_beat_state(self, beat_id: str) -> Optional[BeatBudgetState]:
        """Get budget state for a beat."""
        return self.state.beat_states.get(beat_id)
    
    def get_episode_state(self) -> EpisodeBudgetState:
        """Get episode budget state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get budget statistics."""
        return {
            "episode": self.state.to_dict(),
            "beats": {
                bid: bs.to_dict()
                for bid, bs in self.state.beat_states.items()
            },
            "config": self.config.to_dict(),
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_budget_controller(
    episode_id: str,
    max_budget_usd: Optional[float] = None,
    render_cost_usd: Optional[float] = None,
) -> BudgetController:
    """
    Factory function to create budget controller.
    
    Args:
        episode_id: Episode ID
        max_budget_usd: Override max episode budget
        render_cost_usd: Override render cost
        
    Returns:
        Configured BudgetController
    """
    config = BudgetConfig()
    
    if max_budget_usd:
        config.max_episode_budget_usd = max_budget_usd
    if render_cost_usd:
        config.render_cost_usd = render_cost_usd
    
    return BudgetController(episode_id, config)


def should_allocate_retry(
    controller: BudgetController,
    beat_id: str,
    quality_score: float,
    quality_threshold: float = 0.7,
) -> bool:
    """Quick check if retry should be allocated."""
    result = controller.should_retry(beat_id, quality_score, quality_threshold)
    return result.decision in (BudgetDecision.PROCEED, BudgetDecision.RETRY)
