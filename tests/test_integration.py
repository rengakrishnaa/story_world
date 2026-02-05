"""
Integration Test Suite (Phase 5)

Two-tier testing:
1. Individual component tests - VideoRenderLoop and EnhancedDecisionLoop
2. Full pipeline integration - all phases working together
"""

import pytest
import json
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


# ===========================================================================
# TIER 1: Individual Component Tests - Video Render Loop
# ===========================================================================


class TestRenderLoopConfig:
    """Test RenderLoopConfig dataclass."""
    
    def test_defaults(self):
        from runtime.video_render_loop import RenderLoopConfig
        
        config = RenderLoopConfig()
        assert config.poll_interval_sec == 0.5
        assert config.quality_task_type == "storytelling"
        assert config.max_retries_per_beat == 3


class TestLoopState:
    """Test LoopState enum."""
    
    def test_values(self):
        from runtime.video_render_loop import LoopState
        
        assert LoopState.IDLE.value == "idle"
        assert LoopState.RUNNING.value == "running"
        assert LoopState.COMPLETE.value == "complete"


class TestLoopStatus:
    """Test LoopStatus dataclass."""
    
    def test_creation(self):
        from runtime.video_render_loop import LoopStatus, LoopState
        
        status = LoopStatus()
        assert status.state == LoopState.IDLE
        assert status.beats_completed == 0
    
    def test_to_dict(self):
        from runtime.video_render_loop import LoopStatus, LoopState
        
        status = LoopStatus(
            state=LoopState.RUNNING,
            beats_completed=5,
            budget_spent_usd=0.25,
        )
        
        data = status.to_dict()
        assert data["state"] == "running"
        assert data["beats_completed"] == 5


class TestVideoRenderLoopCreation:
    """Test VideoRenderLoop creation."""
    
    def test_creation(self):
        from runtime.video_render_loop import VideoRenderLoop
        
        loop = VideoRenderLoop(episode_id="test-001")
        
        assert loop.episode_id == "test-001"
        assert loop.config is not None
    
    def test_creation_with_config(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        
        config = RenderLoopConfig(
            mock_render=True,
            max_retries_per_beat=5,
        )
        loop = VideoRenderLoop(episode_id="test-002", config=config)
        
        assert loop.config.max_retries_per_beat == 5


class TestVideoRenderLoopLazyInit:
    """Test lazy initialization of components."""
    
    def test_story_director_lazy(self):
        from runtime.video_render_loop import VideoRenderLoop
        
        loop = VideoRenderLoop(episode_id="lazy-001")
        
        # Should not be initialized yet
        assert loop._policy_engine is None
        
        # Access triggers initialization
        director = loop.policy_engine
        assert director is not None
        assert loop._policy_engine is not None
    
    def test_world_graph_lazy(self):
        from runtime.video_render_loop import VideoRenderLoop
        
        loop = VideoRenderLoop(episode_id="lazy-002")
        
        # Access triggers initialization
        graph = loop.world_graph
        assert graph is not None
        assert graph.episode_id == "lazy-002"
    
    def test_observer_lazy(self):
        from runtime.video_render_loop import VideoRenderLoop
        
        loop = VideoRenderLoop(episode_id="lazy-003")
        
        observer = loop.observer
        assert observer is not None


class TestVideoRenderLoopRun:
    """Test VideoRenderLoop run cycle."""
    
    def test_run_empty_episode(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        
        config = RenderLoopConfig(mock_render=True)
        loop = VideoRenderLoop(episode_id="empty-001", config=config)
        
        result = loop.run()
        
        assert result["state"] == "complete"
        assert result["beats_completed"] == 0
    
    def test_run_with_beats(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        # Setup intent graph
        intent_graph = GoalGraph(episode_id="beats-001")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Complete",
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="First scene",
            contributes_to=["goal"],
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-002",
            description="Second scene",
            suggested_position=2,
            contributes_to=["goal"],
        ))
        
        config = RenderLoopConfig(mock_render=True, mock_render_delay_sec=0.01)
        loop = VideoRenderLoop(
            episode_id="beats-001",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        assert result["state"] == "complete"
        assert result["beats_completed"] == 2
        assert result["world_graph_depth"] == 2
    
    def test_callbacks(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="callback-001")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Complete",
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Scene",
            contributes_to=["goal"],
        ))
        
        config = RenderLoopConfig(mock_render=True, mock_render_delay_sec=0.01)
        loop = VideoRenderLoop(
            episode_id="callback-001",
            intent_graph=intent_graph,
            config=config,
        )
        
        completed_beats = []
        loop.on_beat_complete(lambda bid: completed_beats.append(bid))
        
        loop.run()
        
        assert "beat-001" in completed_beats


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_render_loop(self):
        from runtime.video_render_loop import create_render_loop
        
        loop = create_render_loop(episode_id="factory-001", mock=True)
        
        assert loop is not None
        assert loop.config.mock_render is True
    
    def test_run_episode(self):
        from runtime.video_render_loop import run_episode
        
        result = run_episode(
            episode_id="run-episode-001",
            beats=[
                {"beat_id": "b1", "description": "Scene 1"},
                {"beat_id": "b2", "description": "Scene 2"},
            ],
            mock=True,
        )
        
        assert result["beats_completed"] == 2
        assert result["state"] == "complete"


# ===========================================================================
# TIER 1: Individual Component Tests - Enhanced Decision Loop
# ===========================================================================


class TestEnhancedDecisionLoop:
    """Test EnhancedDecisionLoop class."""
    
    def test_creation(self):
        from runtime.video_render_loop import EnhancedDecisionLoop
        
        mock_runtime = Mock()
        mock_runtime.episode_id = "enhanced-001"
        mock_redis = Mock()
        
        loop = EnhancedDecisionLoop(
            runtime=mock_runtime,
            gpu_job_queue="test:jobs",
            gpu_result_queue="test:results",
            redis_client=mock_redis,
        )
        
        assert loop.runtime == mock_runtime
        assert loop.gpu_job_queue == "test:jobs"
    
    def test_lazy_components(self):
        from runtime.video_render_loop import EnhancedDecisionLoop
        
        mock_runtime = Mock()
        mock_runtime.episode_id = "enhanced-002"
        mock_redis = Mock()
        
        loop = EnhancedDecisionLoop(
            runtime=mock_runtime,
            gpu_job_queue="test:jobs",
            gpu_result_queue="test:results",
            redis_client=mock_redis,
        )
        
        # Test lazy initialization
        assert loop._world_graph is None
        graph = loop.world_graph
        assert graph is not None


# ===========================================================================
# TIER 2: Full Pipeline Integration Tests
# ===========================================================================


class TestFullPipelineIntegration:
    """Test full pipeline integration with all phases."""
    
    def test_all_phases_integrated(self):
        """Test that all Phase 1-4 components work together."""
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal, MicroAction,
            ActionType
        )
        from models.simulation_goal import GoalType
        
        # Phase 1: World State Graph
        from models.world_state_graph import WorldStateGraph, WorldState
        
        # Phase 2: Observer
        from agents.video_observer import VideoObserverAgent
        
        # Phase 3: Quality & Budget
        from runtime.quality_evaluator import QualityEvaluator
        from runtime.budget_controller import BudgetController
        from runtime.value_estimator import ValueEstimator
        
        # Phase 4: Policy Engine
        from agents.policy_engine import PolicyEngine
        from runtime.action_reactor import ActionReactor
        
        # Build intent graph
        intent_graph = GoalGraph(episode_id="full-integration")
        
        intent_graph.add_goal(SimulationGoal(
            goal_id="main-goal",
            goal_type=GoalType.STATE_TARGET,
            description="Complete the story",
            participants=["hero", "mentor"],
        ))
        
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Introduction",
            participants=["hero"],
            suggested_position=1,
            contributes_to=["main-goal"],
        ))
        
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-002",
            description="Training",
            participants=["hero", "mentor"],
            suggested_position=2,
            contributes_to=["main-goal"],
        ))
        
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-003",
            description="Final challenge",
            participants=["hero"],
            suggested_position=3,
            contributes_to=["main-goal"],
        ))
        
        intent_graph.add_action_template(MicroAction(
            action_id="gesture-wave",
            action_type=ActionType.GESTURE,
            description="Wave greeting",
        ))
        
        # Run the render loop
        config = RenderLoopConfig(
            mock_render=True,
            mock_render_delay_sec=0.01,
            use_mock_observer=True,
        )
        
        loop = VideoRenderLoop(
            episode_id="full-integration",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        # Verify all phases contributed
        assert result["state"] == "complete"
        assert result["beats_completed"] == 3
        assert result["world_graph_depth"] == 3
        assert result["budget_spent_usd"] > 0
        
        # Verify policy engine progress
        progress = result["director_progress"]
        assert progress["goals_completed"] == 1
        assert progress["is_complete"] is True
    
    def test_retry_on_low_quality(self):
        """Test that low quality triggers retry."""
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="retry-test")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Test",
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Single beat",
            contributes_to=["goal"],
        ))
        
        config = RenderLoopConfig(
            mock_render=True,
            mock_render_delay_sec=0.01,
            max_retries_per_beat=3,
        )
        
        loop = VideoRenderLoop(
            episode_id="retry-test",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        # Should complete (mock observer returns acceptable quality)
        assert result["state"] == "complete"
    
    def test_budget_exhaustion(self):
        """Test behavior when budget is exhausted."""
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="budget-test")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Test",
        ))
        
        # Add many beats to exhaust budget
        for i in range(20):
            intent_graph.add_proposal(ActionProposal(
                proposal_id=f"beat-{i+1:03d}",
                description=f"Scene {i+1}",
                suggested_position=i+1,
                contributes_to=["goal"],
            ))
        
        config = RenderLoopConfig(
            mock_render=True,
            mock_render_delay_sec=0.01,
            max_episode_budget_usd=0.50,  # Very low budget
        )
        
        loop = VideoRenderLoop(
            episode_id="budget-test",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        # Should stop when budget exhausted
        assert result["beats_completed"] < 20
        assert result["budget_spent_usd"] <= 0.50 + 0.05  # Slight overage OK


class TestWorldGraphIntegration:
    """Test world graph integration with render loop."""
    
    def test_world_state_updates(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="world-state-test")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Test",
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Scene 1",
            contributes_to=["goal"],
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-002",
            description="Scene 2",
            suggested_position=2,
            contributes_to=["goal"],
        ))
        
        config = RenderLoopConfig(mock_render=True, mock_render_delay_sec=0.01)
        loop = VideoRenderLoop(
            episode_id="world-state-test",
            intent_graph=intent_graph,
            config=config,
        )
        
        # Track state updates
        state_updates = []
        loop.on_state_update(lambda s: state_updates.append(s))
        
        loop.run()
        
        # Should have state updates for each beat
        assert len(state_updates) == 2
        
        # World graph should have 2 transitions
        assert loop.world_graph.current.depth == 2


class TestObserverIntegration:
    """Test observer integration with render loop."""
    
    def test_observer_called_for_each_beat(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="observer-test")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Test",
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Scene",
            contributes_to=["goal"],
        ))
        
        config = RenderLoopConfig(
            mock_render=True,
            mock_render_delay_sec=0.01,
            use_mock_observer=True,
        )
        
        loop = VideoRenderLoop(
            episode_id="observer-test",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        # Observer should have been used
        assert loop.observer is not None
        assert result["beats_completed"] == 1


class TestQualityBudgetIntegration:
    """Test quality and budget integration."""
    
    def test_quality_affects_budget(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="quality-budget-test")
        intent_graph.add_goal(SimulationGoal(
            goal_id="goal",
            goal_type=GoalType.STATE_TARGET,
            description="Test",
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Scene",
            contributes_to=["goal"],
        ))
        
        config = RenderLoopConfig(
            mock_render=True,
            mock_render_delay_sec=0.01,
        )
        
        loop = VideoRenderLoop(
            episode_id="quality-budget-test",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        # Budget should be tracked
        assert result["budget_spent_usd"] > 0
        
        # Budget controller should have state
        assert loop.budget_controller.state.spent_usd > 0


class TestPolicyEngineIntegration:
    """Test policy engine integration."""
    
    def test_director_guides_progression(self):
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal, GoalStatus
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="director-test")
        
        # Add multiple intents
        intent_graph.add_goal(SimulationGoal(
            goal_id="intro",
            goal_type=GoalType.STATE_TARGET,
            description="Introduction",
        ))
        intent_graph.add_goal(SimulationGoal(
            goal_id="climax",
            goal_type=GoalType.STATE_TARGET,
            description="Climax",
        ))
        
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-001",
            description="Opening",
            suggested_position=1,
            contributes_to=["intro"],
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="beat-002",
            description="Conflict",
            suggested_position=2,
            contributes_to=["climax"],
        ))
        
        config = RenderLoopConfig(mock_render=True, mock_render_delay_sec=0.01)
        loop = VideoRenderLoop(
            episode_id="director-test",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        # Both intents should be completed
        assert intent_graph.goals["intro"].status == GoalStatus.COMPLETED
        assert intent_graph.goals["climax"].status == GoalStatus.COMPLETED


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""
    
    def test_simple_story(self):
        """Test a simple 3-beat story."""
        from runtime.video_render_loop import run_episode
        
        result = run_episode(
            episode_id="simple-story",
            beats=[
                {"beat_id": "intro", "description": "Introduction", "characters": ["hero"]},
                {"beat_id": "middle", "description": "Adventure", "characters": ["hero", "friend"]},
                {"beat_id": "end", "description": "Resolution", "characters": ["hero"]},
            ],
            mock=True,
        )
        
        assert result["beats_completed"] == 3
        assert result["state"] == "complete"
    
    def test_complex_story_with_branching(self):
        """Test a more complex story structure."""
        from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig
        from models.goal_graph import (
            GoalGraph, SimulationGoal, ActionProposal,
            ActionType, MicroAction
        )
        from models.simulation_goal import GoalType
        
        intent_graph = GoalGraph(episode_id="complex-story")
        
        # Multiple macro intents
        intent_graph.add_goal(SimulationGoal(
            goal_id="character-growth",
            goal_type=GoalType.STATE_TARGET,
            description="Hero grows stronger",
        ))
        intent_graph.add_goal(SimulationGoal(
            goal_id="defeat-villain",
            goal_type=GoalType.STATE_TARGET,
            description="Defeat the main villain",
        ))
        
        # Act 1: Setup
        intent_graph.add_proposal(ActionProposal(
            proposal_id="act1-intro",
            description="Hero introduced",
            suggested_position=1,
            contributes_to=["character-growth"],
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="act1-call",
            description="Call to adventure",
            suggested_position=2,
            contributes_to=["character-growth"],
        ))
        
        # Act 2: Development
        intent_graph.add_proposal(ActionProposal(
            proposal_id="act2-training",
            description="Training montage",
            suggested_position=3,
            contributes_to=["character-growth"],
        ))
        intent_graph.add_proposal(ActionProposal(
            proposal_id="act2-setback",
            description="Major setback",
            suggested_position=4,
            contributes_to=["character-growth"],
        ))
        
        # Act 3: Resolution
        intent_graph.add_proposal(ActionProposal(
            proposal_id="act3-final",
            description="Final battle",
            suggested_position=5,
            contributes_to=["defeat-villain"],
        ))
        
        config = RenderLoopConfig(mock_render=True, mock_render_delay_sec=0.01)
        loop = VideoRenderLoop(
            episode_id="complex-story",
            intent_graph=intent_graph,
            config=config,
        )
        
        result = loop.run()
        
        assert result["beats_completed"] == 5
        assert result["world_graph_depth"] == 5


# ===========================================================================
# Run tests
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
