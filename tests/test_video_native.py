"""
Video-Native Compliance Tests (Phase 6)

Tests for:
1. EpisodeOutcome - first-class failure states
2. ObserverVerdict - termination authority
3. State-centric APIs - video as optional debug
4. Episode termination - explicit failure
"""

import pytest
from datetime import datetime
from typing import Dict, Any


# ============================================================================
# Test EpisodeOutcome
# ============================================================================

class TestEpisodeOutcome:
    """Tests for first-class failure states."""
    
    def test_all_outcomes_exist(self):
        """All required outcomes must exist."""
        from models.episode_outcome import EpisodeOutcome
        
        # Success state
        assert hasattr(EpisodeOutcome, "GOAL_ACHIEVED")
        
        # Failure states (NEW - critical for compliance)
        assert hasattr(EpisodeOutcome, "GOAL_IMPOSSIBLE")
        assert hasattr(EpisodeOutcome, "GOAL_ABANDONED")
        assert hasattr(EpisodeOutcome, "DEAD_STATE")
        assert hasattr(EpisodeOutcome, "CONSTRAINT_VIOLATED")
        
        # In-progress states
        assert hasattr(EpisodeOutcome, "IN_PROGRESS")
        assert hasattr(EpisodeOutcome, "PENDING")
    
    def test_is_terminal_success(self):
        """GOAL_ACHIEVED is terminal."""
        from models.episode_outcome import EpisodeOutcome
        assert EpisodeOutcome.GOAL_ACHIEVED.is_terminal
    
    def test_is_terminal_failure(self):
        """All failure states are terminal."""
        from models.episode_outcome import EpisodeOutcome
        
        assert EpisodeOutcome.GOAL_IMPOSSIBLE.is_terminal
        assert EpisodeOutcome.GOAL_ABANDONED.is_terminal
        assert EpisodeOutcome.DEAD_STATE.is_terminal
        assert EpisodeOutcome.CONSTRAINT_VIOLATED.is_terminal
    
    def test_is_not_terminal_in_progress(self):
        """In-progress states are not terminal."""
        from models.episode_outcome import EpisodeOutcome
        
        assert not EpisodeOutcome.IN_PROGRESS.is_terminal
        assert not EpisodeOutcome.PENDING.is_terminal
    
    def test_is_success(self):
        """Only GOAL_ACHIEVED is success."""
        from models.episode_outcome import EpisodeOutcome
        
        assert EpisodeOutcome.GOAL_ACHIEVED.is_success
        assert not EpisodeOutcome.GOAL_IMPOSSIBLE.is_success
        assert not EpisodeOutcome.GOAL_ABANDONED.is_success
    
    def test_is_failure(self):
        """Failure states are failures."""
        from models.episode_outcome import EpisodeOutcome
        
        assert EpisodeOutcome.GOAL_IMPOSSIBLE.is_failure
        assert EpisodeOutcome.GOAL_ABANDONED.is_failure
        assert EpisodeOutcome.DEAD_STATE.is_failure
        assert not EpisodeOutcome.GOAL_ACHIEVED.is_failure


# ============================================================================
# Test ObserverVerdict
# ============================================================================

class TestObserverVerdict:
    """Tests for observer termination authority."""
    
    def test_all_verdicts_exist(self):
        """All required verdicts must exist."""
        from models.episode_outcome import ObserverVerdict
        
        # Normal outcomes
        assert hasattr(ObserverVerdict, "VALID")
        assert hasattr(ObserverVerdict, "DEGRADED")
        assert hasattr(ObserverVerdict, "FAILED")
        
        # Blocking outcomes (NEW - critical for compliance)
        assert hasattr(ObserverVerdict, "IMPOSSIBLE")
        assert hasattr(ObserverVerdict, "CONTRADICTS")
        assert hasattr(ObserverVerdict, "BLOCKS_INTENT")
    
    def test_continuation_allowed(self):
        """Normal verdicts allow continuation."""
        from models.episode_outcome import ObserverVerdict
        
        assert ObserverVerdict.VALID.allows_continuation
        assert ObserverVerdict.DEGRADED.allows_continuation
        assert ObserverVerdict.FAILED.allows_continuation
    
    def test_termination_forced(self):
        """Blocking verdicts force termination."""
        from models.episode_outcome import ObserverVerdict
        
        assert ObserverVerdict.IMPOSSIBLE.forces_termination
        assert ObserverVerdict.CONTRADICTS.forces_termination
        assert ObserverVerdict.BLOCKS_INTENT.forces_termination
    
    def test_continuation_and_termination_exclusive(self):
        """A verdict can't both allow continuation and force termination."""
        from models.episode_outcome import ObserverVerdict
        
        for verdict in ObserverVerdict:
            assert verdict.allows_continuation != verdict.forces_termination


# ============================================================================
# Test TerminationReason
# ============================================================================

class TestTerminationReason:
    """Tests for termination reason details."""
    
    def test_creation(self):
        """Can create termination reason."""
        from models.episode_outcome import (
            TerminationReason, EpisodeOutcome, ObserverVerdict
        )
        
        reason = TerminationReason(
            outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
            verdict=ObserverVerdict.IMPOSSIBLE,
            trigger="observer_veto",
            description="Character cannot fly",
            physics_violation="Gravity prevents this action",
        )
        
        assert reason.outcome == EpisodeOutcome.GOAL_IMPOSSIBLE
        assert reason.verdict == ObserverVerdict.IMPOSSIBLE
        assert reason.physics_violation == "Gravity prevents this action"
    
    def test_to_dict(self):
        """Can serialize to dict."""
        from models.episode_outcome import TerminationReason, EpisodeOutcome
        
        reason = TerminationReason(
            outcome=EpisodeOutcome.DEAD_STATE,
            trigger="no_valid_actions",
            description="No valid actions remain",
        )
        
        d = reason.to_dict()
        assert d["outcome"] == "dead_state"
        assert d["trigger"] == "no_valid_actions"


# ============================================================================
# Test EpisodeResult
# ============================================================================

class TestEpisodeResult:
    """Tests for state-centric episode result."""
    
    def test_creation(self):
        """Can create episode result."""
        from models.episode_outcome import EpisodeResult, EpisodeOutcome
        
        result = EpisodeResult(
            episode_id="ep-001",
            outcome=EpisodeOutcome.GOAL_ACHIEVED,
            state_delta={"characters": {"hero": {"emotion": "happy"}}},
            confidence=0.9,
            total_cost_usd=0.35,
        )
        
        assert result.episode_id == "ep-001"
        assert result.is_success
        assert result.confidence == 0.9
    
    def test_video_is_debug_only(self):
        """Video is added as debug artifact, not primary output."""
        from models.episode_outcome import EpisodeResult, EpisodeOutcome
        
        result = EpisodeResult(
            episode_id="ep-001",
            outcome=EpisodeOutcome.GOAL_ACHIEVED,
        )
        
        result.with_video_debug("s3://bucket/video.mp4", retention_hours=24)
        
        assert result.debug["video_uri"] == "s3://bucket/video.mp4"
        assert result.debug["video_is_debug_only"] is True
        assert result.debug["video_retention_hours"] == 24
    
    def test_to_dict_state_first(self):
        """to_dict returns state-first structure."""
        from models.episode_outcome import EpisodeResult, EpisodeOutcome
        
        result = EpisodeResult(
            episode_id="ep-001",
            outcome=EpisodeOutcome.GOAL_ACHIEVED,
            state_delta={"key": "value"},
            confidence=0.8,
        )
        
        d = result.to_dict()
        
        # State-first fields must be present
        assert "state_delta" in d
        assert "confidence" in d
        assert "outcome" in d
        assert "is_success" in d
        
        # Video is optional (in debug, or None)
        assert "debug" in d or d.get("debug") is None
    
    def test_failure_result(self):
        """Failure results have termination reason."""
        from models.episode_outcome import create_failure_result, EpisodeOutcome
        
        result = create_failure_result(
            episode_id="ep-001",
            outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
            reason="Character cannot fly",
            trigger="observer_veto",
        )
        
        assert not result.is_success
        assert result.is_terminal
        assert result.termination_reason is not None
        assert result.termination_reason.description == "Character cannot fly"


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Tests for episode result factory functions."""
    
    def test_create_success_result(self):
        """create_success_result creates successful result."""
        from models.episode_outcome import create_success_result
        
        result = create_success_result(
            episode_id="ep-001",
            state_delta={"characters": {}},
            cost_usd=0.5,
        )
        
        assert result.is_success
        assert result.total_cost_usd == 0.5
    
    def test_create_impossible_result(self):
        """create_impossible_result creates physics violation."""
        from models.episode_outcome import create_impossible_result
        
        result = create_impossible_result(
            episode_id="ep-001",
            physics_violation="Gravity prevents flying",
            blocked_intent_id="intent-001",
        )
        
        assert result.is_failure
        assert result.termination_reason.physics_violation == "Gravity prevents flying"
    
    def test_create_dead_state_result(self):
        """create_dead_state_result creates dead state."""
        from models.episode_outcome import create_dead_state_result
        
        result = create_dead_state_result(
            episode_id="ep-001",
            reason="No valid actions remain",
        )
        
        assert result.is_failure
        assert result.outcome.value == "dead_state"


# ============================================================================
# Test ObservationResult Verdict Fields
# ============================================================================

class TestObservationResultVerdictFields:
    """Tests for verdict fields in ObservationResult."""
    
    def test_verdict_field_exists(self):
        """ObservationResult has verdict field."""
        from models.observation import ObservationResult
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="s3://bucket/video.mp4",
        )
        
        # Default verdict is valid
        assert obs.verdict == "valid"
    
    def test_forces_termination_field(self):
        """ObservationResult has forces_termination field."""
        from models.observation import ObservationResult
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="s3://bucket/video.mp4",
            forces_termination=True,
        )
        
        assert obs.forces_termination is True
    
    def test_blocks_macro_intent_field(self):
        """ObservationResult has blocks_macro_intent field."""
        from models.observation import ObservationResult
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="s3://bucket/video.mp4",
            blocks_macro_intent="intent-001",
        )
        
        assert obs.blocks_macro_intent == "intent-001"
    
    def test_physics_violation_field(self):
        """ObservationResult has physics_violation field."""
        from models.observation import ObservationResult
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="s3://bucket/video.mp4",
            physics_violation="Character cannot fly",
        )
        
        assert obs.physics_violation == "Character cannot fly"
    
    def test_impossible_verdict(self):
        """ObservationResult can have impossible verdict."""
        from models.observation import ObservationResult
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="s3://bucket/video.mp4",
            verdict="impossible",
            forces_termination=True,
            impossible_reason="Physics contradiction detected",
        )
        
        assert obs.verdict == "impossible"
        assert obs.forces_termination
        assert obs.impossible_reason == "Physics contradiction detected"


# ============================================================================
# Test ConstraintViolation
# ============================================================================

class TestConstraintViolation:
    """Tests for constraint violation."""
    
    def test_creation(self):
        """Can create constraint violation."""
        from models.episode_outcome import ConstraintViolation
        
        violation = ConstraintViolation(
            violation_type="physics",
            severity="fatal",
            description="Character cannot fly",
            forces_episode_termination=True,
        )
        
        assert violation.violation_type == "physics"
        assert violation.forces_episode_termination
    
    def test_to_dict(self):
        """Can serialize to dict."""
        from models.episode_outcome import ConstraintViolation
        
        violation = ConstraintViolation(
            violation_type="continuity",
            severity="error",
            description="Character already dead",
            blocks_macro_intent=True,
        )
        
        d = violation.to_dict()
        assert d["violation_type"] == "continuity"
        assert d["blocks_macro_intent"] is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestVideoNativeIntegration:
    """Integration tests for video-native compliance."""
    
    def test_observer_can_terminate_episode(self):
        """Observer verdict can force episode termination."""
        from models.observation import ObservationResult
        from models.episode_outcome import (
            EpisodeOutcome, EpisodeResult, TerminationReason, ObserverVerdict
        )
        
        # Observer detects impossible action
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="s3://bucket/video.mp4",
            verdict="impossible",
            forces_termination=True,
            physics_violation="Character cannot fly",
        )
        
        # System should terminate episode
        assert obs.forces_termination
        
        # Create termination result
        result = EpisodeResult(
            episode_id="ep-001",
            outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
            termination_reason=TerminationReason(
                outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
                verdict=ObserverVerdict.IMPOSSIBLE,
                trigger="observer_veto",
                description="Action physically impossible",
                physics_violation=obs.physics_violation,
            ),
        )
        
        assert result.outcome == EpisodeOutcome.GOAL_IMPOSSIBLE
        assert result.is_failure
    
    def test_video_as_debug_artifact(self):
        """Video should be optional debug artifact."""
        from models.episode_outcome import EpisodeResult, EpisodeOutcome
        
        # Create result WITHOUT video
        result = EpisodeResult(
            episode_id="ep-001",
            outcome=EpisodeOutcome.GOAL_ACHIEVED,
            state_delta={"characters": {"hero": {"alive": True}}},
            confidence=0.95,
        )
        
        d = result.to_dict()
        
        # Result is valid without video
        assert d["is_success"] is True
        assert d["state_delta"] == {"characters": {"hero": {"alive": True}}}
        
        # Video is optional
        assert d.get("debug") is None or "video_uri" not in d.get("debug", {})
    
    def test_failure_is_first_class(self):
        """Failure is a valid outcome, not an error."""
        from models.episode_outcome import EpisodeOutcome
        
        # All failure types are valid outcomes
        failure_outcomes = [
            EpisodeOutcome.GOAL_IMPOSSIBLE,
            EpisodeOutcome.GOAL_ABANDONED,
            EpisodeOutcome.DEAD_STATE,
            EpisodeOutcome.CONSTRAINT_VIOLATED,
        ]
        
        for outcome in failure_outcomes:
            assert outcome.is_terminal, f"{outcome} should be terminal"
            assert outcome.is_failure, f"{outcome} should be failure"
            # Failure is NOT an error - it's a valid terminal state
            assert outcome.value != "error"


# ============================================================================
# Test Non-Story Use Case
# ============================================================================

class TestNonStoryUseCase:
    """Test that system works for non-story workloads."""
    
    def test_simulation_use_case(self):
        """System can support simulation without story terminology."""
        from models.episode_outcome import EpisodeOutcome, EpisodeResult
        
        # Create a "simulation" result (not story)
        result = EpisodeResult(
            episode_id="sim-001",  # Not "episode"
            outcome=EpisodeOutcome.GOAL_IMPOSSIBLE,
            state_delta={"robot_position": {"x": 0, "y": 0}},
            confidence=0.0,  # Action failed
        )
        
        # Works without story concepts
        assert result.outcome.is_failure
        assert result.state_delta["robot_position"]["x"] == 0
    
    def test_training_data_use_case(self):
        """System can generate training data without watching video."""
        from models.observation import ObservationResult
        from models.episode_outcome import EpisodeResult, EpisodeOutcome
        
        # Observation extracts training data
        obs = ObservationResult(
            observation_id="train-001",
            video_uri="temp://ephemeral/video.mp4",  # Ephemeral
        )
        
        # Result is the training data, not the video
        result = EpisodeResult(
            episode_id="train-001",
            outcome=EpisodeOutcome.GOAL_ACHIEVED,
            state_delta={"training_sample": obs.to_dict()},
        )
        
        # Video can be deleted, training data persists
        assert "training_sample" in result.state_delta
