"""
Tests for Rational Inaction (Phase 7.5 Gate).
"""
import pytest
from unittest.mock import MagicMock, patch
from runtime.video_render_loop import VideoRenderLoop, RenderLoopConfig, LoopState
from agents.policy_engine import PolicyDecision, DecisionType

class TestRationalInaction:
    
    def test_wait_decision_does_nothing(self):
        """Test that WAIT decision causes inaction."""
        # Setup mocks
        mock_policy = MagicMock()
        mock_policy.next_decision.return_value = PolicyDecision(
            decision_type=DecisionType.WAIT,
            decision_id="wait-decision",
            reasoning="Uncertainty too high, need more info."
        )
        
        real_world = WorldStateGraph(episode_id="wait-test")
        real_world.initialize()
        
        # Initialize loop with corrects args
        loop = VideoRenderLoop(episode_id="wait-test", config=RenderLoopConfig())
        
        # Inject mocks manually since constructor doesn't take them
        loop._policy_engine = mock_policy
        loop._world_graph = real_world
        
        # Mock other components to ensure they are NOT called
        # Mock other components to ensure they are NOT called
        loop._budget_controller = MagicMock()
        loop._value_estimator = MagicMock()
        loop._render_with_retries = MagicMock()
        
        # Execute one cycle
        loop._process_one_cycle()
        
        # Assertions
        # 1. Policy called
        mock_policy.next_decision.assert_called_once()
        
        # 2. Budget NOT requested
        loop.budget_controller.request_budget.assert_not_called()
        
        # 3. Render NOT called
        loop._render_with_retries.assert_not_called()
        
        # 4. Status remains RUNNING (not ERROR)
        assert loop.status.state == LoopState.PENDING # Status init is PENDING
