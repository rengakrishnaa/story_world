"""
Value Estimator

Computes expected value of rendering a beat for budget allocation.
Value is based on downstream impact, story importance, and success probability.

Design Principles:
1. Value is forward-looking (downstream impact)
2. Branch points have higher value
3. Success probability affects expected value
4. Value degrades with quality shortfall
"""

from __future__ import annotations

import os
import json
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Value Components
# ============================================================================

@dataclass
class ValueComponents:
    """Components that contribute to expected value."""
    base_value: float = 1.0
    
    # Story importance
    narrative_weight: float = 1.0  # How important is this beat to the story
    emotional_weight: float = 1.0  # Emotional impact of this beat
    
    # Structural importance
    is_branch_point: bool = False
    downstream_beats: int = 0
    tree_depth: int = 0
    
    # Success factors
    estimated_success_probability: float = 0.8
    previous_attempts: int = 0
    previous_best_quality: float = 0.0
    
    # Risk factors
    is_critical_path: bool = True
    recovery_difficulty: float = 0.5  # How hard to recover if this fails
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValueEstimate:
    """Result of value estimation."""
    expected_value: float
    components: ValueComponents
    
    # Breakdown
    base_value: float = 0.0
    downstream_value: float = 0.0
    probability_adjusted_value: float = 0.0
    
    # Recommendations
    is_high_value: bool = False
    budget_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_value": self.expected_value,
            "base_value": self.base_value,
            "downstream_value": self.downstream_value,
            "probability_adjusted_value": self.probability_adjusted_value,
            "is_high_value": self.is_high_value,
            "budget_multiplier": self.budget_multiplier,
            "components": self.components.to_dict(),
        }


# ============================================================================
# Value Estimator
# ============================================================================

class ValueEstimator:
    """
    Estimates expected value of beat rendering for budget decisions.
    
    Value formula:
        EV = base_value * narrative_weight * downstream_mult * success_prob
        
    Where:
        - base_value: Default value of a beat (1.0)
        - narrative_weight: Story importance (0.5 - 2.0)
        - downstream_mult: 1.0 + 0.1 * downstream_beats
        - success_prob: Estimated probability of success
    
    Usage:
        estimator = ValueEstimator()
        components = ValueComponents(
            narrative_weight=1.5,
            is_branch_point=True,
            downstream_beats=3,
        )
        estimate = estimator.estimate(components)
        
        # Use for budget allocation
        budget_request = estimate.expected_value * base_budget
    """
    
    def __init__(
        self,
        base_value: float = 1.0,
        branch_point_bonus: float = 0.5,
        downstream_factor: float = 0.1,
        max_multiplier: float = 5.0,
    ):
        self.base_value = base_value
        self.branch_point_bonus = branch_point_bonus
        self.downstream_factor = downstream_factor
        self.max_multiplier = max_multiplier
        
        # Historical data for probability estimation
        self._success_history: Dict[str, List[bool]] = {}
    
    def estimate(
        self,
        components: ValueComponents,
    ) -> ValueEstimate:
        """
        Estimate expected value for a beat.
        
        Args:
            components: Value components
            
        Returns:
            ValueEstimate with expected value and breakdown
        """
        # Base value with narrative weight
        base = self.base_value * components.narrative_weight * components.emotional_weight
        
        # Downstream impact
        downstream_mult = 1.0 + (components.downstream_beats * self.downstream_factor)
        
        # Branch point bonus
        if components.is_branch_point:
            downstream_mult += self.branch_point_bonus
        
        # Critical path multiplier
        if components.is_critical_path:
            downstream_mult *= (1.0 + (1.0 - components.recovery_difficulty) * 0.2)
        
        # Cap multiplier
        downstream_mult = min(downstream_mult, self.max_multiplier)
        
        # Success probability adjustment
        success_prob = self._estimate_success_probability(components)
        
        # Compute values
        raw_value = base * downstream_mult
        probability_adjusted = raw_value * success_prob
        
        # Determine if high value
        is_high_value = probability_adjusted > 1.5
        budget_multiplier = min(2.0, 1.0 + (probability_adjusted - 1.0) * 0.5) if probability_adjusted > 1.0 else 1.0
        
        return ValueEstimate(
            expected_value=probability_adjusted,
            components=components,
            base_value=base,
            downstream_value=raw_value,
            probability_adjusted_value=probability_adjusted,
            is_high_value=is_high_value,
            budget_multiplier=budget_multiplier,
        )
    
    def _estimate_success_probability(
        self,
        components: ValueComponents,
    ) -> float:
        """Estimate probability of success on next attempt."""
        base_prob = components.estimated_success_probability
        
        # Adjust based on previous attempts (diminishing returns)
        if components.previous_attempts > 0:
            # Each retry has lower marginal probability of success
            retry_factor = 0.8 ** components.previous_attempts
            base_prob *= retry_factor
            
            # But if previous quality was close, probability is higher
            if components.previous_best_quality > 0.6:
                quality_boost = (components.previous_best_quality - 0.6) * 0.5
                base_prob = min(0.95, base_prob + quality_boost)
        
        return max(0.1, min(0.95, base_prob))
    
    def estimate_from_beat(
        self,
        beat_id: str,
        narrative_weight: float = 1.0,
        is_branch_point: bool = False,
        downstream_beats: int = 0,
        previous_attempts: int = 0,
        previous_best_quality: float = 0.0,
    ) -> ValueEstimate:
        """
        Convenience method to estimate value from beat parameters.
        
        Args:
            beat_id: Beat ID
            narrative_weight: Story importance
            is_branch_point: Whether this is a branch point
            downstream_beats: Number of downstream beats
            previous_attempts: Number of previous attempts
            previous_best_quality: Best quality from previous attempts
            
        Returns:
            ValueEstimate
        """
        components = ValueComponents(
            narrative_weight=narrative_weight,
            is_branch_point=is_branch_point,
            downstream_beats=downstream_beats,
            previous_attempts=previous_attempts,
            previous_best_quality=previous_best_quality,
        )
        
        return self.estimate(components)
    
    def record_outcome(
        self,
        beat_id: str,
        success: bool,
    ) -> None:
        """Record attempt outcome for future probability estimation."""
        if beat_id not in self._success_history:
            self._success_history[beat_id] = []
        self._success_history[beat_id].append(success)
    
    def get_historical_success_rate(
        self,
        beat_id: Optional[str] = None,
    ) -> float:
        """Get historical success rate."""
        if beat_id:
            history = self._success_history.get(beat_id, [])
        else:
            history = [s for hist in self._success_history.values() for s in hist]
        
        if not history:
            return 0.8  # Default assumption
        
        return sum(history) / len(history)


# ============================================================================
# Integration with World Graph
# ============================================================================

def estimate_node_value(
    world_graph,
    node_id: str,
) -> ValueEstimate:
    """
    Estimate value based on position in world state graph.
    
    Args:
        world_graph: WorldStateGraph instance
        node_id: Node to evaluate
        
    Returns:
        ValueEstimate based on graph structure
    """
    estimator = ValueEstimator()
    
    # Get node
    node = world_graph.get_node(node_id)
    if not node:
        return estimator.estimate(ValueComponents())
    
    # Count children (downstream impact)
    children = world_graph.get_children(node_id)
    downstream_count = len(children)
    
    # Check if branch point (multiple children possible)
    is_branch = downstream_count > 1 or node.branch_name != "main"
    
    # Estimate narrative weight from depth
    # Deeper nodes tend to be more critical
    depth_weight = 1.0 + (node.depth * 0.05)
    
    components = ValueComponents(
        narrative_weight=min(2.0, depth_weight),
        is_branch_point=is_branch,
        downstream_beats=downstream_count,
        tree_depth=node.depth,
    )
    
    return estimator.estimate(components)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_value_estimate(
    is_branch_point: bool = False,
    downstream_beats: int = 0,
    previous_attempts: int = 0,
) -> float:
    """Quick value estimation for simple cases."""
    estimator = ValueEstimator()
    estimate = estimator.estimate_from_beat(
        beat_id="quick",
        is_branch_point=is_branch_point,
        downstream_beats=downstream_beats,
        previous_attempts=previous_attempts,
    )
    return estimate.expected_value


def should_invest_in_retry(
    current_quality: float,
    target_quality: float,
    attempt_number: int,
    is_critical: bool = False,
) -> bool:
    """
    Quick decision on whether retry is worthwhile.
    
    Args:
        current_quality: Current quality score
        target_quality: Target quality threshold
        attempt_number: Which attempt this would be
        is_critical: Whether this is a critical beat
        
    Returns:
        True if retry is worthwhile
    """
    quality_gap = target_quality - current_quality
    
    if quality_gap <= 0:
        return False  # Already met threshold
    
    # Diminishing returns
    retry_value = 0.8 ** attempt_number
    
    # Critical beats get more retries
    if is_critical:
        retry_value *= 1.5
    
    # Worth retrying if expected improvement exceeds cost
    expected_improvement = quality_gap * retry_value
    
    return expected_improvement > 0.1 and attempt_number < 5
