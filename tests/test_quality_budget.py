"""
Test Suite for Quality & Budget System (Phase 3)

Two-tier testing:
1. Individual component tests - test each class in isolation
2. Integration tests - test with pipeline components
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch


# ===========================================================================
# TIER 1: Individual Component Tests - Quality Evaluator
# ===========================================================================


class TestQualityDimension:
    """Test QualityDimension enum."""
    
    def test_values(self):
        from runtime.quality_evaluator import QualityDimension
        
        assert QualityDimension.VISUAL_CLARITY.value == "visual_clarity"
        assert QualityDimension.CONTINUITY.value == "continuity"


class TestTaskType:
    """Test TaskType enum."""
    
    def test_values(self):
        from runtime.quality_evaluator import TaskType
        
        assert TaskType.STORYTELLING.value == "storytelling"
        assert TaskType.SIMULATION.value == "simulation"
        assert TaskType.TRAINING_DATA.value == "training"


class TestQualityProfile:
    """Test QualityProfile dataclass."""
    
    def test_creation(self):
        from runtime.quality_evaluator import QualityProfile, TaskType, QualityDimension
        
        profile = QualityProfile(
            task_type=TaskType.STORYTELLING,
            name="Test Profile",
            overall_threshold=0.75,
        )
        
        assert profile.task_type == TaskType.STORYTELLING
        assert profile.overall_threshold == 0.75
    
    def test_to_dict(self):
        from runtime.quality_evaluator import QualityProfile, TaskType, QualityDimension
        
        profile = QualityProfile(
            task_type=TaskType.SIMULATION,
            name="Sim Profile",
            thresholds={QualityDimension.CONTINUITY: 0.85},
            critical_dimensions=[QualityDimension.CONTINUITY],
        )
        
        data = profile.to_dict()
        assert data["task_type"] == "simulation"
        assert "continuity" in data["thresholds"]


class TestQualityProfiles:
    """Test default quality profiles."""
    
    def test_all_task_types_have_profiles(self):
        from runtime.quality_evaluator import QUALITY_PROFILES, TaskType
        
        for task_type in TaskType:
            assert task_type in QUALITY_PROFILES
    
    def test_storytelling_profile(self):
        from runtime.quality_evaluator import QUALITY_PROFILES, TaskType, QualityDimension
        
        profile = QUALITY_PROFILES[TaskType.STORYTELLING]
        assert profile.overall_threshold == 0.7
        assert QualityDimension.ACTION_CLARITY in profile.critical_dimensions
    
    def test_final_render_highest_threshold(self):
        from runtime.quality_evaluator import QUALITY_PROFILES, TaskType
        
        final = QUALITY_PROFILES[TaskType.FINAL_RENDER]
        preview = QUALITY_PROFILES[TaskType.PREVIEW]
        
        assert final.overall_threshold > preview.overall_threshold


class TestQualityScores:
    """Test QualityScores dataclass."""
    
    def test_creation_with_defaults(self):
        from runtime.quality_evaluator import QualityScores
        
        scores = QualityScores()
        assert scores.visual_clarity == 0.5
        assert scores.continuity == 0.5
    
    def test_get_dimension(self):
        from runtime.quality_evaluator import QualityScores, QualityDimension
        
        scores = QualityScores(
            visual_clarity=0.9,
            motion_smoothness=0.85,
        )
        
        assert scores.get_dimension(QualityDimension.VISUAL_CLARITY) == 0.9
        assert scores.get_dimension(QualityDimension.MOTION_SMOOTHNESS) == 0.85
    
    def test_from_observation(self):
        from runtime.quality_evaluator import QualityScores
        from models.observation import QualityMetrics
        
        obs_quality = QualityMetrics(
            visual_clarity=0.88,
            motion_smoothness=0.82,
            action_clarity=0.9,
        )
        
        scores = QualityScores.from_observation(obs_quality)
        assert scores.visual_clarity == 0.88
        assert scores.action_clarity == 0.9


class TestEvaluationContext:
    """Test EvaluationContext dataclass."""
    
    def test_creation(self):
        from runtime.quality_evaluator import EvaluationContext, TaskType
        
        context = EvaluationContext(
            task_type=TaskType.SIMULATION,
            beat_id="beat-001",
            is_branch_point=True,
        )
        
        assert context.task_type == TaskType.SIMULATION
        assert context.is_branch_point is True
    
    def test_to_dict(self):
        from runtime.quality_evaluator import EvaluationContext, TaskType
        
        context = EvaluationContext(
            task_type=TaskType.STORYTELLING,
            downstream_beats=5,
        )
        
        data = context.to_dict()
        assert data["task_type"] == "storytelling"
        assert data["downstream_beats"] == 5


class TestQualityEvaluator:
    """Test QualityEvaluator class."""
    
    def test_creation(self):
        from runtime.quality_evaluator import QualityEvaluator
        
        evaluator = QualityEvaluator()
        assert len(evaluator.profiles) > 0
    
    def test_evaluate_acceptable(self):
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        
        evaluator = QualityEvaluator()
        
        scores = QualityScores(
            visual_clarity=0.9,
            motion_smoothness=0.85,
            temporal_coherence=0.8,
            style_consistency=0.85,
            action_clarity=0.9,
            character_recognizability=0.9,
            narrative_coherence=0.85,
            continuity=0.8,
        )
        
        context = EvaluationContext(task_type=TaskType.STORYTELLING)
        result = evaluator.evaluate(scores, context)
        
        assert result.is_acceptable is True
        assert result.overall_score >= 0.7
    
    def test_evaluate_unacceptable(self):
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        
        evaluator = QualityEvaluator()
        
        scores = QualityScores(
            visual_clarity=0.4,
            motion_smoothness=0.3,
            action_clarity=0.4,
        )
        
        context = EvaluationContext(task_type=TaskType.STORYTELLING)
        result = evaluator.evaluate(scores, context)
        
        assert result.is_acceptable is False
        # Many failed dimensions means may not be worth retrying
        assert len(result.failed_dimensions) > 0
    
    def test_critical_dimension_failure(self):
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        
        evaluator = QualityEvaluator()
        
        # Good overall but fails critical dimension
        scores = QualityScores(
            visual_clarity=0.9,
            motion_smoothness=0.9,
            temporal_coherence=0.9,
            style_consistency=0.9,
            action_clarity=0.5,  # Critical dimension fails
            character_recognizability=0.9,
            narrative_coherence=0.9,
            continuity=0.9,
        )
        
        context = EvaluationContext(task_type=TaskType.STORYTELLING)
        result = evaluator.evaluate(scores, context)
        
        # Should fail due to critical dimension
        assert result.is_acceptable is False
        assert "action_clarity" in result.critical_failures
    
    def test_adaptive_threshold_branch_point(self):
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        
        evaluator = QualityEvaluator()
        scores = QualityScores(
            visual_clarity=0.75,
            motion_smoothness=0.75,
            action_clarity=0.75,
            character_recognizability=0.8,
        )
        
        # Normal context - should pass
        normal_context = EvaluationContext(task_type=TaskType.STORYTELLING)
        normal_result = evaluator.evaluate(scores, normal_context)
        
        # Branch point - threshold increases
        branch_context = EvaluationContext(
            task_type=TaskType.STORYTELLING,
            is_branch_point=True,
        )
        branch_result = evaluator.evaluate(scores, branch_context)
        
        assert branch_result.threshold_used > normal_result.threshold_used
    
    def test_preview_lower_threshold(self):
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        
        evaluator = QualityEvaluator()
        
        scores = QualityScores(
            visual_clarity=0.5,
            motion_smoothness=0.5,
            action_clarity=0.5,
        )
        
        context = EvaluationContext(task_type=TaskType.PREVIEW)
        result = evaluator.evaluate(scores, context)
        
        assert result.is_acceptable is True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_evaluate_observation_quality(self):
        from runtime.quality_evaluator import evaluate_observation_quality, TaskType
        from models.observation import ObservationResult, QualityMetrics
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="test.mp4",
            quality=QualityMetrics(
                visual_clarity=0.85,
                motion_smoothness=0.82,
                action_clarity=0.88,
                character_recognizability=0.9,
            ),
        )
        
        result = evaluate_observation_quality(obs, TaskType.STORYTELLING)
        assert result is not None
    
    def test_quick_quality_check(self):
        from runtime.quality_evaluator import quick_quality_check, TaskType
        
        assert quick_quality_check(0.8, TaskType.STORYTELLING) is True
        assert quick_quality_check(0.5, TaskType.STORYTELLING) is False
        assert quick_quality_check(0.5, TaskType.PREVIEW) is True


# ===========================================================================
# TIER 1: Individual Component Tests - Budget Controller
# ===========================================================================


class TestBudgetConfig:
    """Test BudgetConfig dataclass."""
    
    def test_defaults(self):
        from runtime.budget_controller import BudgetConfig
        
        config = BudgetConfig()
        assert config.max_episode_budget_usd == 10.0
        assert config.max_beat_retries == 5
    
    def test_render_cost_from_env(self):
        from runtime.budget_controller import BudgetConfig
        
        # Note: default from field is evaluated at class definition time
        # So env var needs to be set before import or use explicit parameter
        config = BudgetConfig(render_cost_usd=0.10)
        assert config.render_cost_usd == 0.10


class TestBeatBudgetState:
    """Test BeatBudgetState dataclass."""
    
    def test_creation(self):
        from runtime.budget_controller import BeatBudgetState
        
        state = BeatBudgetState(
            beat_id="beat-001",
            initial_budget_usd=0.50,
        )
        
        assert state.remaining_budget_usd == 0.50
        assert state.can_retry is True
    
    def test_can_retry_false_when_successful(self):
        from runtime.budget_controller import BeatBudgetState
        
        state = BeatBudgetState(
            beat_id="beat-001",
            initial_budget_usd=0.50,
            successful=True,
        )
        
        assert state.can_retry is False


class TestEpisodeBudgetState:
    """Test EpisodeBudgetState dataclass."""
    
    def test_creation(self):
        from runtime.budget_controller import EpisodeBudgetState
        
        state = EpisodeBudgetState(
            episode_id="ep-001",
            total_budget_usd=10.0,
        )
        
        assert state.remaining_budget_usd == 10.0
        assert state.utilization_rate == 0.0
    
    def test_utilization_rate(self):
        from runtime.budget_controller import EpisodeBudgetState
        
        state = EpisodeBudgetState(
            episode_id="ep-001",
            total_budget_usd=10.0,
            spent_usd=5.0,
        )
        
        assert state.utilization_rate == 0.5


class TestBudgetDecision:
    """Test BudgetDecision enum."""
    
    def test_values(self):
        from runtime.budget_controller import BudgetDecision
        
        assert BudgetDecision.PROCEED.value == "proceed"
        assert BudgetDecision.RETRY.value == "retry"
        assert BudgetDecision.ABORT.value == "abort"


class TestBudgetController:
    """Test BudgetController class."""
    
    def test_creation(self):
        from runtime.budget_controller import BudgetController
        
        controller = BudgetController(episode_id="ep-001")
        assert controller.state.total_budget_usd == 10.0
    
    def test_request_budget_proceed(self):
        from runtime.budget_controller import BudgetController, BudgetDecision
        
        controller = BudgetController(episode_id="ep-001")
        result = controller.request_budget(
            beat_id="beat-001",
            expected_value=1.0,
        )
        
        assert result.decision == BudgetDecision.PROCEED
        assert result.allocated_budget_usd > 0
    
    def test_record_attempt_success(self):
        from runtime.budget_controller import BudgetController
        
        controller = BudgetController(episode_id="ep-001")
        controller.request_budget(beat_id="beat-001")
        
        state = controller.record_attempt(
            beat_id="beat-001",
            success=True,
            cost_usd=0.05,
            quality_score=0.85,
        )
        
        assert state.successful is True
        assert state.attempts == 1
        assert controller.state.spent_usd == 0.05
    
    def test_record_attempt_failure(self):
        from runtime.budget_controller import BudgetController
        
        controller = BudgetController(episode_id="ep-001")
        controller.request_budget(beat_id="beat-001")
        
        state = controller.record_attempt(
            beat_id="beat-001",
            success=False,
            cost_usd=0.05,
            error="Render failed",
        )
        
        assert state.successful is False
        assert len(state.attempt_history) == 1
        assert state.attempt_history[0]["error"] == "Render failed"
    
    def test_budget_exhausted(self):
        from runtime.budget_controller import BudgetController, BudgetDecision, BudgetConfig
        
        config = BudgetConfig(
            max_episode_budget_usd=0.05,
            render_cost_usd=0.05,
        )
        controller = BudgetController(episode_id="ep-001", config=config)
        
        # First request succeeds
        result1 = controller.request_budget(beat_id="beat-001")
        controller.record_attempt("beat-001", success=True, cost_usd=0.05)
        
        # Second request fails (budget exhausted)
        result2 = controller.request_budget(beat_id="beat-002")
        assert result2.decision == BudgetDecision.ABORT
    
    def test_max_retries_reached(self):
        from runtime.budget_controller import BudgetController, BudgetDecision, BudgetConfig
        
        config = BudgetConfig(max_beat_retries=2)
        controller = BudgetController(episode_id="ep-001", config=config)
        
        # Exhaust retries
        for i in range(2):
            controller.request_budget(beat_id="beat-001", is_retry=(i > 0))
            controller.record_attempt("beat-001", success=False, cost_usd=0.05)
        
        # Next request should be rejected
        result = controller.request_budget(beat_id="beat-001", is_retry=True)
        assert result.decision in (BudgetDecision.ABORT, BudgetDecision.ACCEPT_DEGRADED)
    
    def test_diminishing_returns(self):
        from runtime.budget_controller import BudgetController
        
        controller = BudgetController(episode_id="ep-001")
        
        # First request
        result1 = controller.request_budget(beat_id="beat-001", is_retry=False)
        controller.record_attempt("beat-001", success=False, cost_usd=0.05)
        
        # First retry - lower expected value
        result2 = controller.request_budget(beat_id="beat-001", is_retry=True)
        
        assert result2.expected_value < result1.expected_value
    
    def test_should_retry_method(self):
        from runtime.budget_controller import BudgetController, BudgetDecision
        
        controller = BudgetController(episode_id="ep-001")
        controller.request_budget(beat_id="beat-001")
        
        # Quality meets threshold - no retry needed
        result1 = controller.should_retry("beat-001", quality_score=0.8, quality_threshold=0.7)
        assert result1.decision == BudgetDecision.PROCEED
        
        # Quality below threshold - retry
        controller2 = BudgetController(episode_id="ep-002")
        controller2.request_budget(beat_id="beat-002")
        controller2.record_attempt("beat-002", success=False, cost_usd=0.05)
        
        result2 = controller2.should_retry("beat-002", quality_score=0.5, quality_threshold=0.7)
        assert result2.decision == BudgetDecision.RETRY
    
    def test_branch_point_higher_budget(self):
        from runtime.budget_controller import BudgetController
        
        controller = BudgetController(episode_id="ep-001")
        
        # Normal beat
        result1 = controller.request_budget(beat_id="beat-001", is_branch_point=False)
        
        # Branch point - should get more budget
        result2 = controller.request_budget(beat_id="beat-002", is_branch_point=True)
        
        beat1 = controller.get_beat_state("beat-001")
        beat2 = controller.get_beat_state("beat-002")
        
        assert beat2.initial_budget_usd > beat1.initial_budget_usd


# ===========================================================================
# TIER 1: Individual Component Tests - Value Estimator
# ===========================================================================


class TestValueComponents:
    """Test ValueComponents dataclass."""
    
    def test_creation(self):
        from runtime.value_estimator import ValueComponents
        
        components = ValueComponents(
            narrative_weight=1.5,
            is_branch_point=True,
            downstream_beats=3,
        )
        
        assert components.narrative_weight == 1.5
        assert components.is_branch_point is True


class TestValueEstimate:
    """Test ValueEstimate dataclass."""
    
    def test_to_dict(self):
        from runtime.value_estimator import ValueEstimate, ValueComponents
        
        estimate = ValueEstimate(
            expected_value=1.5,
            components=ValueComponents(),
            base_value=1.0,
            is_high_value=True,
        )
        
        data = estimate.to_dict()
        assert data["expected_value"] == 1.5
        assert data["is_high_value"] is True


class TestValueEstimator:
    """Test ValueEstimator class."""
    
    def test_creation(self):
        from runtime.value_estimator import ValueEstimator
        
        estimator = ValueEstimator()
        assert estimator.base_value == 1.0
    
    def test_basic_estimate(self):
        from runtime.value_estimator import ValueEstimator, ValueComponents
        
        estimator = ValueEstimator()
        components = ValueComponents()
        
        estimate = estimator.estimate(components)
        assert estimate.expected_value > 0
    
    def test_branch_point_increases_value(self):
        from runtime.value_estimator import ValueEstimator, ValueComponents
        
        estimator = ValueEstimator()
        
        normal = estimator.estimate(ValueComponents(is_branch_point=False))
        branch = estimator.estimate(ValueComponents(is_branch_point=True))
        
        assert branch.expected_value > normal.expected_value
    
    def test_downstream_increases_value(self):
        from runtime.value_estimator import ValueEstimator, ValueComponents
        
        estimator = ValueEstimator()
        
        no_downstream = estimator.estimate(ValueComponents(downstream_beats=0))
        with_downstream = estimator.estimate(ValueComponents(downstream_beats=5))
        
        assert with_downstream.expected_value > no_downstream.expected_value
    
    def test_diminishing_returns_on_retries(self):
        from runtime.value_estimator import ValueEstimator, ValueComponents
        
        estimator = ValueEstimator()
        
        first_attempt = estimator.estimate(ValueComponents(previous_attempts=0))
        third_attempt = estimator.estimate(ValueComponents(previous_attempts=2))
        
        assert third_attempt.expected_value < first_attempt.expected_value
    
    def test_estimate_from_beat(self):
        from runtime.value_estimator import ValueEstimator
        
        estimator = ValueEstimator()
        
        estimate = estimator.estimate_from_beat(
            beat_id="beat-001",
            narrative_weight=1.5,
            is_branch_point=True,
            downstream_beats=3,
        )
        
        assert estimate.is_high_value is True


class TestValueEstimatorConvenienceFunctions:
    """Test value estimator convenience functions."""
    
    def test_quick_value_estimate(self):
        from runtime.value_estimator import quick_value_estimate
        
        normal = quick_value_estimate()
        branch = quick_value_estimate(is_branch_point=True)
        
        assert branch > normal
    
    def test_should_invest_in_retry(self):
        from runtime.value_estimator import should_invest_in_retry
        
        # Already meets quality - no retry
        assert should_invest_in_retry(0.8, 0.7, 1) is False
        
        # Below quality, first retry - yes
        assert should_invest_in_retry(0.5, 0.7, 1) is True
        
        # Below quality, many retries - no
        assert should_invest_in_retry(0.5, 0.7, 5) is False


# ===========================================================================
# TIER 2: Integration Tests
# ===========================================================================


class TestQualityBudgetIntegration:
    """Test integration between quality evaluator and budget controller."""
    
    def test_quality_drives_retry_decision(self):
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        from runtime.budget_controller import BudgetController, BudgetDecision
        
        # Create evaluator and controller
        evaluator = QualityEvaluator()
        controller = BudgetController(episode_id="integration-01")
        
        # Initial render
        controller.request_budget(beat_id="beat-001")
        controller.record_attempt("beat-001", success=True, cost_usd=0.05, quality_score=0.5)
        
        # Evaluate quality - marginal failure (should allow retry)
        scores = QualityScores(
            visual_clarity=0.7,
            action_clarity=0.7,
            character_recognizability=0.74,  # Just below threshold
            narrative_coherence=0.7,
            continuity=0.65,
        )
        context = EvaluationContext(task_type=TaskType.STORYTELLING, beat_id="beat-001")
        quality_result = evaluator.evaluate(scores, context)
        
        # Quality not acceptable but close - should retry
        assert quality_result.is_acceptable is False
        
        # Budget controller agrees
        budget_result = controller.should_retry("beat-001", 0.5)
        assert budget_result.decision == BudgetDecision.RETRY
    
    def test_value_affects_budget_allocation(self):
        from runtime.budget_controller import BudgetController
        from runtime.value_estimator import ValueEstimator, ValueComponents
        
        controller = BudgetController(episode_id="integration-02")
        estimator = ValueEstimator()
        
        # Low value beat
        low_value = estimator.estimate(ValueComponents(
            narrative_weight=0.5,
            downstream_beats=0,
        ))
        
        # High value beat
        high_value = estimator.estimate(ValueComponents(
            narrative_weight=2.0,
            is_branch_point=True,
            downstream_beats=5,
        ))
        
        # Request budget with values
        result_low = controller.request_budget(
            beat_id="low-value",
            expected_value=low_value.expected_value,
        )
        
        result_high = controller.request_budget(
            beat_id="high-value",
            expected_value=high_value.expected_value,
            is_branch_point=True,
            downstream_beats=5,
        )
        
        # High value beat should get more budget
        low_state = controller.get_beat_state("low-value")
        high_state = controller.get_beat_state("high-value")
        
        assert high_state.initial_budget_usd > low_state.initial_budget_usd


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    def test_observation_to_quality_to_budget(self):
        from models.observation import ObservationResult, QualityMetrics
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        from runtime.budget_controller import BudgetController, BudgetDecision
        
        # Simulate observation result
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="test.mp4",
            beat_id="beat-001",
            quality=QualityMetrics(
                visual_clarity=0.6,
                motion_smoothness=0.55,
                action_clarity=0.65,
                character_recognizability=0.7,
            ),
        )
        
        # Convert to quality scores
        scores = QualityScores.from_observation(obs.quality)
        
        # Evaluate
        evaluator = QualityEvaluator()
        context = EvaluationContext(
            task_type=TaskType.STORYTELLING,
            beat_id=obs.beat_id,
        )
        quality_result = evaluator.evaluate(scores, context)
        
        # Budget decision based on quality
        controller = BudgetController(episode_id="pipeline-01")
        controller.request_budget(beat_id="beat-001")
        controller.record_attempt(
            "beat-001",
            success=False,
            cost_usd=0.05,
            quality_score=quality_result.overall_score,
        )
        
        budget_result = controller.should_retry(
            "beat-001",
            quality_result.overall_score,
            context.custom_threshold or 0.7,
        )
        
        # Full pipeline result
        assert quality_result.is_acceptable is False  # Quality too low
        assert budget_result.decision == BudgetDecision.RETRY  # Budget allows retry
    
    def test_world_graph_with_quality_budget(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        from runtime.budget_controller import BudgetController, BudgetDecision
        from runtime.value_estimator import ValueEstimator
        
        # Initialize graph
        graph = WorldStateGraph(episode_id="graph-test")
        graph.initialize(WorldState())
        
        # Initialize controllers
        controller = BudgetController(episode_id="graph-test")
        evaluator = QualityEvaluator()
        estimator = ValueEstimator()
        
        # Simulate beat processing
        beat_id = "beat-001"
        
        # Estimate value based on graph position
        estimate = estimator.estimate_from_beat(
            beat_id=beat_id,
            is_branch_point=True,
            downstream_beats=3,
        )
        
        # Request budget
        budget = controller.request_budget(
            beat_id=beat_id,
            expected_value=estimate.expected_value,
            is_branch_point=True,
            downstream_beats=3,
        )
        
        assert budget.decision == BudgetDecision.PROCEED
        
        # Simulate render and observation - high quality to pass branch point threshold
        scores = QualityScores(
            visual_clarity=0.9,
            motion_smoothness=0.85,
            temporal_coherence=0.85,
            style_consistency=0.85,
            action_clarity=0.92,
            character_recognizability=0.9,
            narrative_coherence=0.85,
            continuity=0.85,
        )
        
        context = EvaluationContext(
            task_type=TaskType.STORYTELLING,
            beat_id=beat_id,
            is_branch_point=True,
        )
        
        quality = evaluator.evaluate(scores, context)
        
        # Record and apply transition
        controller.record_attempt(
            beat_id,
            success=quality.is_acceptable,
            cost_usd=0.05,
            quality_score=quality.overall_score,
        )
        
        # Apply transition unconditionally for test
        graph.transition(
            video_uri="test.mp4",
            observation={"narrative_flags": {"beat_001_complete": True}},
            beat_id=beat_id,
            quality_score=quality.overall_score,
        )
        
        # Verify
        assert graph.current.depth == 1
        assert quality.is_acceptable is True


# ===========================================================================
# Run tests
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
