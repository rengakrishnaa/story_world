"""
Quality Evaluator

Adaptive, task-dependent quality thresholds for video generation.
Quality is defined as "good enough to make the next decision safely."

Design Principles:
1. Quality is context-dependent, not absolute
2. Different tasks have different quality requirements
3. Thresholds adapt based on downstream impact
4. Quality is multi-dimensional (visual, narrative, continuity)
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Quality Dimensions
# ============================================================================

class QualityDimension(Enum):
    """Dimensions of quality assessment."""
    VISUAL_CLARITY = "visual_clarity"
    MOTION_SMOOTHNESS = "motion_smoothness"
    TEMPORAL_COHERENCE = "temporal_coherence"
    STYLE_CONSISTENCY = "style_consistency"
    ACTION_CLARITY = "action_clarity"
    CHARACTER_RECOGNIZABILITY = "character_recognizability"
    NARRATIVE_COHERENCE = "narrative_coherence"
    CONTINUITY = "continuity"


class TaskType(Enum):
    """Types of tasks with different quality requirements."""
    STORYTELLING = "storytelling"  # Emotional clarity matters most
    SIMULATION = "simulation"      # Physical plausibility critical
    TRAINING_DATA = "training"     # Causal correctness essential
    QA_TESTING = "qa"             # Anomaly detection focus
    PREVIEW = "preview"           # Low quality acceptable
    FINAL_RENDER = "final"        # Highest quality required


# ============================================================================
# Quality Profiles
# ============================================================================

@dataclass
class QualityProfile:
    """
    Task-specific quality requirements.
    
    Each profile defines:
    - Minimum thresholds per dimension
    - Weights for computing overall score
    - Critical dimensions that must be met
    """
    task_type: TaskType
    name: str
    
    # Dimension thresholds (minimum acceptable)
    thresholds: Dict[QualityDimension, float] = field(default_factory=dict)
    
    # Dimension weights for overall score
    weights: Dict[QualityDimension, float] = field(default_factory=dict)
    
    # Critical dimensions (must meet threshold)
    critical_dimensions: List[QualityDimension] = field(default_factory=list)
    
    # Overall threshold
    overall_threshold: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "name": self.name,
            "thresholds": {k.value: v for k, v in self.thresholds.items()},
            "weights": {k.value: v for k, v in self.weights.items()},
            "critical_dimensions": [d.value for d in self.critical_dimensions],
            "overall_threshold": self.overall_threshold,
        }


# Default profiles
QUALITY_PROFILES: Dict[TaskType, QualityProfile] = {
    TaskType.STORYTELLING: QualityProfile(
        task_type=TaskType.STORYTELLING,
        name="Storytelling",
        thresholds={
            QualityDimension.ACTION_CLARITY: 0.7,
            QualityDimension.CHARACTER_RECOGNIZABILITY: 0.75,
            QualityDimension.NARRATIVE_COHERENCE: 0.8,
            QualityDimension.CONTINUITY: 0.65,
        },
        weights={
            QualityDimension.ACTION_CLARITY: 0.25,
            QualityDimension.CHARACTER_RECOGNIZABILITY: 0.2,
            QualityDimension.NARRATIVE_COHERENCE: 0.25,
            QualityDimension.VISUAL_CLARITY: 0.15,
            QualityDimension.MOTION_SMOOTHNESS: 0.15,
        },
        critical_dimensions=[
            QualityDimension.ACTION_CLARITY,
            QualityDimension.CHARACTER_RECOGNIZABILITY,
        ],
        overall_threshold=0.7,
    ),
    
    TaskType.SIMULATION: QualityProfile(
        task_type=TaskType.SIMULATION,
        name="Simulation",
        thresholds={
            QualityDimension.TEMPORAL_COHERENCE: 0.85,
            QualityDimension.MOTION_SMOOTHNESS: 0.8,
            QualityDimension.CONTINUITY: 0.85,
        },
        weights={
            QualityDimension.TEMPORAL_COHERENCE: 0.3,
            QualityDimension.MOTION_SMOOTHNESS: 0.25,
            QualityDimension.CONTINUITY: 0.25,
            QualityDimension.VISUAL_CLARITY: 0.1,
            QualityDimension.STYLE_CONSISTENCY: 0.1,
        },
        critical_dimensions=[
            QualityDimension.TEMPORAL_COHERENCE,
            QualityDimension.CONTINUITY,
        ],
        overall_threshold=0.8,
    ),
    
    TaskType.TRAINING_DATA: QualityProfile(
        task_type=TaskType.TRAINING_DATA,
        name="Training Data",
        thresholds={
            QualityDimension.ACTION_CLARITY: 0.85,
            QualityDimension.TEMPORAL_COHERENCE: 0.85,
            QualityDimension.CONTINUITY: 0.9,
        },
        weights={
            QualityDimension.ACTION_CLARITY: 0.3,
            QualityDimension.TEMPORAL_COHERENCE: 0.3,
            QualityDimension.CONTINUITY: 0.2,
            QualityDimension.CHARACTER_RECOGNIZABILITY: 0.2,
        },
        critical_dimensions=[
            QualityDimension.ACTION_CLARITY,
            QualityDimension.CONTINUITY,
        ],
        overall_threshold=0.85,
    ),
    
    TaskType.QA_TESTING: QualityProfile(
        task_type=TaskType.QA_TESTING,
        name="QA Testing",
        thresholds={
            QualityDimension.VISUAL_CLARITY: 0.6,
            QualityDimension.ACTION_CLARITY: 0.6,
        },
        weights={
            QualityDimension.VISUAL_CLARITY: 0.3,
            QualityDimension.ACTION_CLARITY: 0.3,
            QualityDimension.TEMPORAL_COHERENCE: 0.2,
            QualityDimension.CONTINUITY: 0.2,
        },
        critical_dimensions=[],  # No critical dimensions for QA
        overall_threshold=0.5,
    ),
    
    TaskType.PREVIEW: QualityProfile(
        task_type=TaskType.PREVIEW,
        name="Preview",
        thresholds={
            QualityDimension.ACTION_CLARITY: 0.4,
        },
        weights={
            QualityDimension.ACTION_CLARITY: 0.5,
            QualityDimension.VISUAL_CLARITY: 0.3,
            QualityDimension.MOTION_SMOOTHNESS: 0.2,
        },
        critical_dimensions=[],
        overall_threshold=0.4,
    ),
    
    TaskType.FINAL_RENDER: QualityProfile(
        task_type=TaskType.FINAL_RENDER,
        name="Final Render",
        thresholds={
            QualityDimension.VISUAL_CLARITY: 0.9,
            QualityDimension.MOTION_SMOOTHNESS: 0.85,
            QualityDimension.TEMPORAL_COHERENCE: 0.85,
            QualityDimension.STYLE_CONSISTENCY: 0.9,
            QualityDimension.ACTION_CLARITY: 0.85,
            QualityDimension.CHARACTER_RECOGNIZABILITY: 0.9,
            QualityDimension.NARRATIVE_COHERENCE: 0.85,
            QualityDimension.CONTINUITY: 0.9,
        },
        weights={
            QualityDimension.VISUAL_CLARITY: 0.15,
            QualityDimension.MOTION_SMOOTHNESS: 0.1,
            QualityDimension.TEMPORAL_COHERENCE: 0.15,
            QualityDimension.STYLE_CONSISTENCY: 0.15,
            QualityDimension.ACTION_CLARITY: 0.15,
            QualityDimension.CHARACTER_RECOGNIZABILITY: 0.15,
            QualityDimension.NARRATIVE_COHERENCE: 0.1,
            QualityDimension.CONTINUITY: 0.05,
        },
        critical_dimensions=[
            QualityDimension.VISUAL_CLARITY,
            QualityDimension.STYLE_CONSISTENCY,
        ],
        overall_threshold=0.88,
    ),
}


# ============================================================================
# Quality Context
# ============================================================================

@dataclass
class EvaluationContext:
    """
    Context for quality evaluation.
    Provides information about the task and its downstream impact.
    """
    task_type: TaskType = TaskType.STORYTELLING
    beat_id: Optional[str] = None
    episode_id: Optional[str] = None
    
    # Downstream impact
    is_branch_point: bool = False
    downstream_beats: int = 0
    
    # Previous state for continuity checking
    previous_quality_scores: List[float] = field(default_factory=list)
    
    # Override thresholds
    custom_threshold: Optional[float] = None
    custom_profile: Optional[QualityProfile] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "beat_id": self.beat_id,
            "episode_id": self.episode_id,
            "is_branch_point": self.is_branch_point,
            "downstream_beats": self.downstream_beats,
            "custom_threshold": self.custom_threshold,
        }


@dataclass
class QualityScores:
    """Multi-dimensional quality scores."""
    visual_clarity: float = 0.5
    motion_smoothness: float = 0.5
    temporal_coherence: float = 0.5
    style_consistency: float = 0.5
    action_clarity: float = 0.5
    character_recognizability: float = 0.5
    narrative_coherence: float = 0.5
    continuity: float = 0.5
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def get_dimension(self, dim: QualityDimension) -> float:
        """Get score for a specific dimension."""
        mapping = {
            QualityDimension.VISUAL_CLARITY: self.visual_clarity,
            QualityDimension.MOTION_SMOOTHNESS: self.motion_smoothness,
            QualityDimension.TEMPORAL_COHERENCE: self.temporal_coherence,
            QualityDimension.STYLE_CONSISTENCY: self.style_consistency,
            QualityDimension.ACTION_CLARITY: self.action_clarity,
            QualityDimension.CHARACTER_RECOGNIZABILITY: self.character_recognizability,
            QualityDimension.NARRATIVE_COHERENCE: self.narrative_coherence,
            QualityDimension.CONTINUITY: self.continuity,
        }
        return mapping.get(dim, 0.5)
    
    @classmethod
    def from_observation(cls, obs_quality: Any) -> "QualityScores":
        """Create from observation quality metrics."""
        # obs_quality should be QualityMetrics from observation.py
        if hasattr(obs_quality, 'visual_clarity'):
            return cls(
                visual_clarity=getattr(obs_quality, 'visual_clarity', 0.5),
                motion_smoothness=getattr(obs_quality, 'motion_smoothness', 0.5),
                temporal_coherence=getattr(obs_quality, 'temporal_coherence', 0.5),
                style_consistency=getattr(obs_quality, 'style_consistency', 0.5),
                action_clarity=getattr(obs_quality, 'action_clarity', 0.5),
                character_recognizability=getattr(obs_quality, 'character_recognizability', 0.5),
                narrative_coherence=getattr(obs_quality, 'narrative_coherence', 0.5),
            )
        return cls()


@dataclass
class QualityResult:
    """Result of quality evaluation."""
    is_acceptable: bool
    overall_score: float
    scores: QualityScores
    threshold_used: float
    
    # Dimension-level results
    dimension_results: Dict[str, bool] = field(default_factory=dict)
    failed_dimensions: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    
    # Recommendations
    should_retry: bool = False
    retry_focus: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "overall_score": self.overall_score,
            "scores": self.scores.to_dict(),
            "threshold_used": self.threshold_used,
            "dimension_results": self.dimension_results,
            "failed_dimensions": self.failed_dimensions,
            "critical_failures": self.critical_failures,
            "should_retry": self.should_retry,
            "retry_focus": self.retry_focus,
        }


# ============================================================================
# Quality Evaluator
# ============================================================================

class QualityEvaluator:
    """
    Adaptive quality evaluator for video generation.
    
    Key features:
    1. Task-dependent thresholds
    2. Multi-dimensional quality assessment
    3. Dynamic threshold adjustment based on context
    4. Retry recommendations
    
    Usage:
        evaluator = QualityEvaluator()
        context = EvaluationContext(task_type=TaskType.STORYTELLING)
        result = evaluator.evaluate(quality_scores, context)
        if not result.is_acceptable:
            if result.should_retry:
                # Focus retry on result.retry_focus dimensions
    """
    
    def __init__(self, profiles: Optional[Dict[TaskType, QualityProfile]] = None):
        self.profiles = profiles or QUALITY_PROFILES
        self._custom_thresholds: Dict[str, float] = {}
        
        logger.info(f"[quality_evaluator] initialized with {len(self.profiles)} profiles")
    
    def evaluate(
        self,
        scores: QualityScores,
        context: EvaluationContext,
    ) -> QualityResult:
        """
        Evaluate quality against adaptive thresholds.
        
        Args:
            scores: Multi-dimensional quality scores
            context: Evaluation context
            
        Returns:
            QualityResult with acceptance decision and recommendations
        """
        # Get profile
        profile = context.custom_profile or self.profiles.get(
            context.task_type,
            QUALITY_PROFILES[TaskType.STORYTELLING],
        )
        
        # Compute adaptive threshold
        base_threshold = context.custom_threshold or profile.overall_threshold
        threshold = self._compute_adaptive_threshold(base_threshold, context)
        
        # Compute weighted overall score
        overall_score = self._compute_overall_score(scores, profile)
        
        # Check dimension-level thresholds
        dimension_results = {}
        failed_dimensions = []
        critical_failures = []
        
        for dim, dim_threshold in profile.thresholds.items():
            score = scores.get_dimension(dim)
            passed = score >= dim_threshold
            dimension_results[dim.value] = passed
            
            if not passed:
                failed_dimensions.append(dim.value)
                if dim in profile.critical_dimensions:
                    critical_failures.append(dim.value)
        
        # Determine acceptance
        is_acceptable = (
            overall_score >= threshold
            and len(critical_failures) == 0
        )
        
        # Determine retry recommendation
        should_retry = not is_acceptable and len(failed_dimensions) <= 3
        retry_focus = failed_dimensions[:2] if should_retry else []
        
        result = QualityResult(
            is_acceptable=is_acceptable,
            overall_score=overall_score,
            scores=scores,
            threshold_used=threshold,
            dimension_results=dimension_results,
            failed_dimensions=failed_dimensions,
            critical_failures=critical_failures,
            should_retry=should_retry,
            retry_focus=retry_focus,
        )
        
        logger.debug(
            f"[quality_evaluator] evaluated: "
            f"overall={overall_score:.2f}, threshold={threshold:.2f}, "
            f"acceptable={is_acceptable}"
        )
        
        return result
    
    def _compute_adaptive_threshold(
        self,
        base_threshold: float,
        context: EvaluationContext,
    ) -> float:
        """
        Compute adaptive threshold based on context.
        
        Adjustments:
        - Raise threshold for branch points
        - Raise threshold for high downstream impact
        - Lower threshold if recent history is good
        """
        threshold = base_threshold
        
        # Branch point adjustment (+10%)
        if context.is_branch_point:
            threshold = min(0.95, threshold + 0.1)
        
        # Downstream impact adjustment
        if context.downstream_beats > 5:
            # More downstream beats = higher quality required
            impact_bonus = min(0.1, context.downstream_beats * 0.01)
            threshold = min(0.95, threshold + impact_bonus)
        
        # Historical quality adjustment
        if context.previous_quality_scores:
            avg_recent = sum(context.previous_quality_scores[-5:]) / min(5, len(context.previous_quality_scores))
            if avg_recent > 0.85:
                # Consistently high quality, can be slightly lenient
                threshold = max(0.5, threshold - 0.05)
        
        return threshold
    
    def _compute_overall_score(
        self,
        scores: QualityScores,
        profile: QualityProfile,
    ) -> float:
        """Compute weighted overall quality score."""
        if not profile.weights:
            # Default to simple average
            all_scores = [
                scores.visual_clarity,
                scores.motion_smoothness,
                scores.temporal_coherence,
                scores.style_consistency,
                scores.action_clarity,
                scores.character_recognizability,
                scores.narrative_coherence,
                scores.continuity,
            ]
            return sum(all_scores) / len(all_scores)
        
        total = 0.0
        weight_sum = 0.0
        
        for dim, weight in profile.weights.items():
            score = scores.get_dimension(dim)
            total += score * weight
            weight_sum += weight
        
        if weight_sum == 0:
            return 0.5
        
        return total / weight_sum
    
    def set_custom_threshold(
        self,
        beat_id: str,
        threshold: float,
    ) -> None:
        """Set custom threshold for a specific beat."""
        self._custom_thresholds[beat_id] = threshold
        logger.info(f"[quality_evaluator] custom threshold for {beat_id}: {threshold}")
    
    def get_profile(self, task_type: TaskType) -> QualityProfile:
        """Get profile for a task type."""
        return self.profiles.get(task_type, QUALITY_PROFILES[TaskType.STORYTELLING])
    
    def create_custom_profile(
        self,
        name: str,
        task_type: TaskType,
        overall_threshold: float,
        critical_dimensions: Optional[List[QualityDimension]] = None,
    ) -> QualityProfile:
        """Create a custom quality profile."""
        base = self.profiles.get(task_type, QUALITY_PROFILES[TaskType.STORYTELLING])
        return QualityProfile(
            task_type=task_type,
            name=name,
            thresholds=dict(base.thresholds),
            weights=dict(base.weights),
            critical_dimensions=critical_dimensions or list(base.critical_dimensions),
            overall_threshold=overall_threshold,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def evaluate_observation_quality(
    observation,  # ObservationResult
    task_type: TaskType = TaskType.STORYTELLING,
    is_branch_point: bool = False,
    downstream_beats: int = 0,
) -> QualityResult:
    """
    Convenience function to evaluate observation quality.
    
    Args:
        observation: ObservationResult from video observer
        task_type: Type of task
        is_branch_point: Whether this is a branch point
        downstream_beats: Number of downstream beats
        
    Returns:
        QualityResult
    """
    evaluator = QualityEvaluator()
    
    scores = QualityScores.from_observation(observation.quality)
    context = EvaluationContext(
        task_type=task_type,
        beat_id=observation.beat_id,
        is_branch_point=is_branch_point,
        downstream_beats=downstream_beats,
    )
    
    return evaluator.evaluate(scores, context)


def quick_quality_check(
    overall_score: float,
    task_type: TaskType = TaskType.STORYTELLING,
) -> bool:
    """Quick check if overall score meets task threshold."""
    profile = QUALITY_PROFILES.get(task_type, QUALITY_PROFILES[TaskType.STORYTELLING])
    return overall_score >= profile.overall_threshold
