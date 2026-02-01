"""
Test Suite for Video Observer Agent (Phase 2)

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
from unittest.mock import Mock, patch, MagicMock


# ===========================================================================
# TIER 1: Individual Component Tests - Observation Model
# ===========================================================================


class TestEmotionState:
    """Test EmotionState enum."""
    
    def test_values(self):
        from models.observation import EmotionState
        
        assert EmotionState.NEUTRAL.value == "neutral"
        assert EmotionState.ANGRY.value == "angry"
        assert EmotionState.DETERMINED.value == "determined"
    
    def test_from_string(self):
        from models.observation import EmotionState
        
        emotion = EmotionState("happy")
        assert emotion == EmotionState.HAPPY


class TestContinuityError:
    """Test ContinuityError dataclass."""
    
    def test_creation(self):
        from models.observation import ContinuityError, ContinuityErrorType
        
        error = ContinuityError(
            error_type=ContinuityErrorType.CHARACTER_MISSING,
            description="Saitama not visible",
            severity=0.8,
            affected_entities=["saitama"],
        )
        
        assert error.severity == 0.8
        assert "saitama" in error.affected_entities
    
    def test_serialization(self):
        from models.observation import ContinuityError, ContinuityErrorType
        
        error = ContinuityError(
            error_type=ContinuityErrorType.LOCATION_MISMATCH,
            description="Wrong background",
            severity=0.5,
            frame_range=(10, 20),
        )
        
        data = error.to_dict()
        restored = ContinuityError.from_dict(data)
        
        assert restored.error_type == ContinuityErrorType.LOCATION_MISMATCH
        assert restored.frame_range == (10, 20)


class TestCharacterObservation:
    """Test CharacterObservation dataclass."""
    
    def test_creation(self):
        from models.observation import CharacterObservation, EmotionState
        
        char = CharacterObservation(
            character_id="saitama",
            visible=True,
            position={"x": 0.5, "y": 0.3},
            emotion=EmotionState.NEUTRAL,
            pose="standing",
        )
        
        assert char.character_id == "saitama"
        assert char.emotion == EmotionState.NEUTRAL
    
    def test_serialization(self):
        from models.observation import CharacterObservation, EmotionState
        
        char = CharacterObservation(
            character_id="genos",
            emotion=EmotionState.DETERMINED,
            motion_intensity=0.8,
        )
        
        data = char.to_dict()
        restored = CharacterObservation.from_dict(data)
        
        assert restored.emotion == EmotionState.DETERMINED
        assert restored.motion_intensity == 0.8


class TestEnvironmentObservation:
    """Test EnvironmentObservation dataclass."""
    
    def test_creation(self):
        from models.observation import EnvironmentObservation
        
        env = EnvironmentObservation(
            location_id="city_center",
            time_of_day="noon",
            weather="clear",
            objects_detected=["car", "building"],
        )
        
        assert env.location_id == "city_center"
        assert len(env.objects_detected) == 2
    
    def test_serialization(self):
        from models.observation import EnvironmentObservation
        
        env = EnvironmentObservation(
            location_description="A busy intersection",
            lighting="dramatic",
            mood="tense",
        )
        
        data = env.to_dict()
        restored = EnvironmentObservation.from_dict(data)
        
        assert restored.mood == "tense"


class TestActionObservation:
    """Test ActionObservation dataclass."""
    
    def test_creation(self):
        from models.observation import ActionObservation, ActionOutcome
        
        action = ActionObservation(
            action_description="Saitama defeats monster",
            outcome=ActionOutcome.SUCCESS,
            action_type="attack",
            participants=["saitama", "monster"],
        )
        
        assert action.outcome == ActionOutcome.SUCCESS
        assert "saitama" in action.participants
    
    def test_serialization(self):
        from models.observation import ActionObservation, ActionOutcome
        
        action = ActionObservation(
            action_description="Failed escape attempt",
            outcome=ActionOutcome.FAILED,
            narrative_beat_achieved=False,
            narrative_implications=["tension_increase"],
        )
        
        data = action.to_dict()
        restored = ActionObservation.from_dict(data)
        
        assert restored.outcome == ActionOutcome.FAILED
        assert not restored.narrative_beat_achieved


class TestQualityMetrics:
    """Test QualityMetrics dataclass."""
    
    def test_creation(self):
        from models.observation import QualityMetrics
        
        quality = QualityMetrics(
            visual_clarity=0.9,
            motion_smoothness=0.85,
            temporal_coherence=0.8,
        )
        
        assert quality.visual_clarity == 0.9
    
    def test_compute_overall(self):
        from models.observation import QualityMetrics
        
        quality = QualityMetrics(
            visual_clarity=0.9,
            motion_smoothness=0.9,
            temporal_coherence=0.9,
            style_consistency=0.9,
            action_clarity=0.9,
            character_recognizability=0.9,
            narrative_coherence=0.9,
        )
        
        overall = quality.compute_overall()
        assert 0.85 <= overall <= 0.95
    
    def test_artifact_penalty(self):
        from models.observation import QualityMetrics
        
        quality = QualityMetrics(
            visual_clarity=0.9,
            motion_smoothness=0.9,
            temporal_coherence=0.9,
            style_consistency=0.9,
            action_clarity=0.9,
            character_recognizability=0.9,
            narrative_coherence=0.9,
            artifacts_detected=5,
        )
        
        overall = quality.compute_overall()
        # Should be penalized for artifacts
        assert overall < 0.8


class TestObservationResult:
    """Test ObservationResult dataclass."""
    
    def test_creation(self):
        from models.observation import ObservationResult
        
        result = ObservationResult(
            observation_id="obs-001",
            video_uri="https://example.com/video.mp4",
            beat_id="beat-001",
        )
        
        assert result.observation_id == "obs-001"
        assert result.observer_type == "gemini"
    
    def test_full_observation(self):
        from models.observation import (
            ObservationResult,
            CharacterObservation,
            EnvironmentObservation,
            ActionObservation,
            QualityMetrics,
            EmotionState,
            ActionOutcome,
        )
        
        result = ObservationResult(
            observation_id="obs-002",
            video_uri="https://example.com/video.mp4",
            characters={
                "saitama": CharacterObservation("saitama", emotion=EmotionState.NEUTRAL),
                "genos": CharacterObservation("genos", emotion=EmotionState.DETERMINED),
            },
            environment=EnvironmentObservation(location_id="city"),
            action=ActionObservation(
                action_description="Training session",
                outcome=ActionOutcome.SUCCESS,
            ),
            quality=QualityMetrics(overall_quality=0.85),
        )
        
        assert len(result.characters) == 2
        assert result.environment.location_id == "city"
        assert result.action.outcome == ActionOutcome.SUCCESS
    
    def test_json_serialization(self):
        from models.observation import ObservationResult, CharacterObservation, EmotionState
        
        result = ObservationResult(
            observation_id="obs-003",
            video_uri="test.mp4",
            characters={
                "hero": CharacterObservation("hero", emotion=EmotionState.EXCITED),
            },
        )
        
        json_str = result.to_json()
        restored = ObservationResult.from_json(json_str)
        
        assert restored.observation_id == "obs-003"
        assert "hero" in restored.characters
    
    def test_quality_threshold(self):
        from models.observation import ObservationResult, QualityMetrics
        
        result = ObservationResult(
            observation_id="obs-004",
            video_uri="test.mp4",
            quality=QualityMetrics(overall_quality=0.75),
        )
        
        assert result.is_acceptable(threshold=0.7) is True
        assert result.is_acceptable(threshold=0.8) is False
    
    def test_to_world_state_update(self):
        from models.observation import (
            ObservationResult,
            CharacterObservation,
            EnvironmentObservation,
            ActionObservation,
            EmotionState,
            ActionOutcome,
        )
        
        result = ObservationResult(
            observation_id="obs-005",
            video_uri="test.mp4",
            characters={
                "hero": CharacterObservation(
                    "hero",
                    emotion=EmotionState.ANGRY,
                    position={"x": 0.5, "y": 0.5},
                ),
            },
            environment=EnvironmentObservation(location_id="battlefield"),
            action=ActionObservation(
                action_description="Hero attacks",
                outcome=ActionOutcome.SUCCESS,
                narrative_beat_achieved=True,
            ),
        )
        
        update = result.to_world_state_update()
        
        assert "hero" in update["characters"]
        assert update["characters"]["hero"]["emotion"] == "angry"
        assert update["environment"]["location_id"] == "battlefield"
        assert update["narrative_flags"]["beat_achieved"] is True


class TestTaskContext:
    """Test TaskContext dataclass."""
    
    def test_creation(self):
        from models.observation import TaskContext
        
        context = TaskContext(
            task_type="storytelling",
            beat_id="beat-001",
            expected_characters=["saitama", "genos"],
            expected_action="training",
        )
        
        assert context.task_type == "storytelling"
        assert len(context.expected_characters) == 2
    
    def test_to_dict(self):
        from models.observation import TaskContext
        
        context = TaskContext(
            is_branch_point=True,
            downstream_beats=5,
            required_confidence=0.9,
        )
        
        data = context.to_dict()
        assert data["is_branch_point"] is True
        assert data["downstream_beats"] == 5


# ===========================================================================
# TIER 1: Individual Component Tests - Video Observer
# ===========================================================================


class TestObserverConfig:
    """Test ObserverConfig dataclass."""
    
    def test_defaults(self):
        from agents.video_observer import ObserverConfig
        
        config = ObserverConfig()
        assert config.use_gemini is True
        assert config.frames_to_analyze == 5
        assert config.max_retries == 3
    
    def test_custom(self):
        from agents.video_observer import ObserverConfig
        
        config = ObserverConfig(
            gemini_model="gemini-2.0-flash",
            frames_to_analyze=10,
            use_local=True,
        )
        
        assert config.frames_to_analyze == 10
        assert config.use_local is True


class TestFrameExtractor:
    """Test FrameExtractor class."""
    
    def test_uniform_indices(self):
        from agents.video_observer import FrameExtractor
        
        extractor = FrameExtractor(method="uniform")
        indices = extractor._uniform_indices(100, 5)
        
        assert len(indices) == 5
        assert indices[0] == 0
        assert indices[-1] == 80  # 4 * 20
    
    def test_stub_frames(self):
        from agents.video_observer import FrameExtractor
        
        extractor = FrameExtractor()
        frames = extractor._stub_frames(5)
        
        assert len(frames) == 0  # Empty list for stub


class TestGeminiObserver:
    """Test GeminiObserver class."""
    
    def test_mock_observation(self):
        from agents.video_observer import GeminiObserver, ObserverConfig
        from models.observation import TaskContext
        
        config = ObserverConfig()
        observer = GeminiObserver(config)
        
        context = TaskContext(
            expected_characters=["saitama"],
            expected_action="punch",
        )
        
        result = observer._mock_observation("test-id", context)
        
        assert result.observation_id == "test-id"
        assert "saitama" in result.characters
        assert result.observer_type == "mock"
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": ""})
    def test_no_api_key(self):
        from agents.video_observer import GeminiObserver, ObserverConfig
        
        config = ObserverConfig()
        observer = GeminiObserver(config)
        
        # Should not have client without API key
        assert observer.client is None
    
    def test_parse_response_json(self):
        from agents.video_observer import GeminiObserver, ObserverConfig
        from models.observation import TaskContext
        
        config = ObserverConfig()
        observer = GeminiObserver(config)
        
        # Test JSON parsing
        raw_json = '''```json
{
    "characters": {
        "hero": {
            "visible": true,
            "emotion": "determined",
            "pose": "standing"
        }
    },
    "environment": {
        "location_description": "City street",
        "time_of_day": "noon"
    },
    "action": {
        "action_description": "Hero arrives",
        "outcome": "success"
    },
    "quality": {
        "visual_clarity": 0.9,
        "motion_smoothness": 0.85
    },
    "confidence": 0.88
}
```'''
        
        context = TaskContext()
        result = observer._parse_response(raw_json, "test-id", context, 0.5)
        
        assert "hero" in result.characters
        assert result.confidence == 0.88


class TestVideoObserverAgent:
    """Test VideoObserverAgent class."""
    
    def test_creation(self):
        from agents.video_observer import VideoObserverAgent
        
        agent = VideoObserverAgent()
        
        assert agent.frame_extractor is not None
        assert agent.gemini_observer is not None
        assert agent.config.use_gemini is True
    
    def test_with_custom_config(self):
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        
        config = ObserverConfig(
            use_gemini=False,
            frames_to_analyze=3,
        )
        agent = VideoObserverAgent(config)
        
        assert agent.config.use_gemini is False
        assert agent.config.frames_to_analyze == 3
    
    def test_should_use_local(self):
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        from models.observation import TaskContext
        
        agent = VideoObserverAgent()
        context = TaskContext()
        
        # Should not use local without model
        assert agent._should_use_local(context) is False
        
        # Should not use local at branch points
        context.is_branch_point = True
        assert agent._should_use_local(context) is False
    
    def test_training_buffer(self):
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        
        config = ObserverConfig(record_for_training=True)
        agent = VideoObserverAgent(config)
        
        assert agent.training_buffer == []
        assert agent.get_training_sample_count() == 0


# ===========================================================================
# TIER 1: Individual Component Tests - Internalization
# ===========================================================================


class TestInternalizationConfig:
    """Test InternalizationConfig dataclass."""
    
    def test_defaults(self):
        from agents.observer_internalization import InternalizationConfig
        
        config = InternalizationConfig()
        assert config.batch_size == 16
        assert config.epochs == 50
        assert config.vision_encoder == "resnet18"


class TestTrainingSample:
    """Test TrainingSample dataclass."""
    
    def test_from_dict(self):
        from agents.observer_internalization import TrainingSample
        from models.observation import ObservationResult
        
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="test.mp4",
        )
        
        data = {
            "video_path": "/path/to/video.mp4",
            "frames": [],
            "observation": obs.to_dict(),
        }
        
        sample = TrainingSample.from_dict(data)
        
        assert sample.video_path == "/path/to/video.mp4"
        assert sample.observation.observation_id == "obs-001"


class TestTrainingDataLoader:
    """Test TrainingDataLoader class."""
    
    def test_creation(self):
        from agents.observer_internalization import TrainingDataLoader, InternalizationConfig
        
        config = InternalizationConfig()
        loader = TrainingDataLoader(config)
        
        assert loader.samples == []
    
    def test_stats_empty(self):
        from agents.observer_internalization import TrainingDataLoader, InternalizationConfig
        
        loader = TrainingDataLoader(InternalizationConfig())
        stats = loader.stats()
        
        assert stats["total"] == 0


class TestLocalObserverModel:
    """Test LocalObserverModel class."""
    
    def test_creation(self):
        from agents.observer_internalization import LocalObserverModel, InternalizationConfig
        
        model = LocalObserverModel(InternalizationConfig())
        assert model.is_trained is False
    
    def test_predict_untrained(self):
        from agents.observer_internalization import LocalObserverModel, InternalizationConfig
        from models.observation import TaskContext
        
        model = LocalObserverModel(InternalizationConfig())
        result = model.predict([], TaskContext())
        
        assert result is None  # Can't predict without training


class TestInternalizationTrainer:
    """Test InternalizationTrainer class."""
    
    def test_creation(self):
        from agents.observer_internalization import InternalizationTrainer
        
        trainer = InternalizationTrainer()
        assert trainer.training_metrics == []
    
    def test_load_data_no_dir(self):
        from agents.observer_internalization import InternalizationTrainer, InternalizationConfig
        
        config = InternalizationConfig(training_data_dir="nonexistent_dir")
        trainer = InternalizationTrainer(config)
        
        result = trainer.load_data()
        assert result["loaded"] == 0


class TestHelperFunctions:
    """Test helper/convenience functions."""
    
    def test_check_training_readiness(self):
        from agents.observer_internalization import check_training_readiness
        
        result = check_training_readiness()
        
        assert "ready" in result
        assert "pytorch_available" in result
        assert "training_samples" in result


# ===========================================================================
# TIER 2: Integration Tests
# ===========================================================================


class TestObservationToWorldState:
    """Test integration between observation and world state graph."""
    
    def test_observation_updates_world_state(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        from models.observation import (
            ObservationResult,
            CharacterObservation,
            EnvironmentObservation,
            ActionObservation,
            EmotionState,
            ActionOutcome,
        )
        
        # Create world graph
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState())
        
        # Create observation result
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="test.mp4",
            characters={
                "saitama": CharacterObservation(
                    "saitama",
                    emotion=EmotionState.ANGRY,
                    position={"x": 0.5, "y": 0.5},
                    pose="fighting",
                ),
            },
            environment=EnvironmentObservation(
                location_id="city_center",
                time_of_day="noon",
            ),
            action=ActionObservation(
                action_description="Saitama punches monster",
                outcome=ActionOutcome.SUCCESS,
                narrative_beat_achieved=True,
            ),
        )
        
        # Convert to world state update
        update = obs.to_world_state_update()
        
        # Apply transition
        new_node = graph.transition(
            video_uri="test.mp4",
            observation=update,
            beat_id="beat-001",
        )
        
        # Verify state updated
        assert "saitama" in new_node.world_state.characters
        assert new_node.world_state.characters["saitama"].emotion == "angry"
        assert new_node.world_state.environment.location_id == "city_center"
        assert new_node.world_state.narrative_flags["beat_achieved"] is True


class TestObserverPipeline:
    """Test full observation pipeline."""
    
    def test_observe_with_mock(self):
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        from models.observation import TaskContext
        
        # Use config that will fall back to mock
        config = ObserverConfig(use_gemini=True)
        agent = VideoObserverAgent(config)
        
        context = TaskContext(
            expected_characters=["hero"],
            expected_action="dramatic entrance",
            expected_location="city",
        )
        
        # This should use mock since no real API
        result = agent.observe("nonexistent_video.mp4", context)
        
        assert result is not None
        assert result.observation_id is not None
    
    def test_end_to_end_with_graph(self):
        from models.world_state_graph import WorldStateGraph, WorldState, CharacterState
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        from models.observation import TaskContext
        
        # Initialize graph with character
        graph = WorldStateGraph(episode_id="e2e-test")
        initial_state = WorldState(
            characters={
                "hero": CharacterState("hero", emotion="calm"),
            }
        )
        graph.initialize(initial_state)
        
        # Create observer
        observer = VideoObserverAgent(ObserverConfig())
        
        # Context with expectations
        context = TaskContext(
            beat_id="beat-001",
            expected_characters=["hero"],
            expected_action="training montage",
        )
        
        # Observe (will be mock)
        obs = observer.observe("dummy.mp4", context)
        
        # Convert to update
        update = obs.to_world_state_update()
        
        # Apply to graph
        new_node = graph.transition(
            video_uri="dummy.mp4",
            observation=update,
            beat_id="beat-001",
            quality_score=obs.get_quality_score(),
        )
        
        # Verify
        assert new_node.depth == 1
        assert new_node.transition_beat_id == "beat-001"


class TestTrainingDataCollection:
    """Test training data collection for internalization."""
    
    def test_training_buffer_grows(self):
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        from models.observation import ObservationResult
        
        config = ObserverConfig(record_for_training=True)
        agent = VideoObserverAgent(config)
        
        # Simulate recording observations
        obs = ObservationResult(
            observation_id="obs-001",
            video_uri="test.mp4",
            observer_type="gemini",
        )
        
        agent.training_buffer.append(("test.mp4", obs))
        
        assert len(agent.training_buffer) == 1
    
    def test_flush_training_buffer(self):
        import tempfile
        from agents.video_observer import VideoObserverAgent, ObserverConfig
        from models.observation import ObservationResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ObserverConfig(
                record_for_training=True,
                training_data_dir=tmpdir,
            )
            agent = VideoObserverAgent(config)
            
            # Add 10 samples to trigger flush
            for i in range(10):
                obs = ObservationResult(
                    observation_id=f"obs-{i}",
                    video_uri=f"video{i}.mp4",
                    observer_type="gemini",
                )
                agent.training_buffer.append((f"path{i}.mp4", obs))
            
            # Flush
            agent._flush_training_buffer()
            
            # Buffer should be empty
            assert len(agent.training_buffer) == 0
            
            # File should exist
            files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(files) == 1


# ===========================================================================
# Run tests
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
