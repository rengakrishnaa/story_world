"""
Test Suite for World State Graph (Phase 1)

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

# Set up test database
TEST_DB_PATH = None


@pytest.fixture(scope="function")
def test_db():
    """Create temporary test database."""
    global TEST_DB_PATH
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        TEST_DB_PATH = f.name
    os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
    yield TEST_DB_PATH
    # Cleanup - Windows may keep file locked, so ignore errors
    try:
        if os.path.exists(TEST_DB_PATH):
            os.unlink(TEST_DB_PATH)
    except PermissionError:
        pass  # Windows file locking - file will be cleaned up later


# ===========================================================================
# TIER 1: Individual Component Tests
# ===========================================================================


class TestCharacterState:
    """Test CharacterState dataclass."""
    
    def test_creation(self):
        from models.world_state_graph import CharacterState
        
        state = CharacterState(
            character_id="saitama",
            position={"x": 0, "y": 0, "z": 0},
            emotion="calm",
            pose="standing",
        )
        
        assert state.character_id == "saitama"
        assert state.emotion == "calm"
        assert state.visible is True
    
    def test_serialization(self):
        from models.world_state_graph import CharacterState
        
        state = CharacterState(
            character_id="genos",
            emotion="determined",
        )
        
        data = state.to_dict()
        assert data["character_id"] == "genos"
        
        restored = CharacterState.from_dict(data)
        assert restored.character_id == state.character_id
        assert restored.emotion == state.emotion


class TestEnvironmentState:
    """Test EnvironmentState dataclass."""
    
    def test_creation(self):
        from models.world_state_graph import EnvironmentState
        
        env = EnvironmentState(
            location_id="city_rooftop",
            time_of_day="sunset",
            weather="clear",
        )
        
        assert env.location_id == "city_rooftop"
        assert env.time_of_day == "sunset"
    
    def test_serialization(self):
        from models.world_state_graph import EnvironmentState
        
        env = EnvironmentState(
            location_id="training_grounds",
            lighting="dramatic",
            objects=["dummy", "tree"],
        )
        
        data = env.to_dict()
        restored = EnvironmentState.from_dict(data)
        
        assert restored.objects == ["dummy", "tree"]


class TestWorldState:
    """Test WorldState dataclass."""
    
    def test_empty_creation(self):
        from models.world_state_graph import WorldState
        
        state = WorldState()
        assert len(state.characters) == 0
        assert state.environment is None
    
    def test_with_characters(self):
        from models.world_state_graph import WorldState, CharacterState
        
        state = WorldState(
            characters={
                "saitama": CharacterState("saitama", emotion="bored"),
                "genos": CharacterState("genos", emotion="serious"),
            }
        )
        
        assert len(state.characters) == 2
        assert state.characters["saitama"].emotion == "bored"
    
    def test_json_serialization(self):
        from models.world_state_graph import WorldState, CharacterState, EnvironmentState
        
        state = WorldState(
            characters={
                "hero": CharacterState("hero", emotion="determined"),
            },
            environment=EnvironmentState("battlefield"),
            narrative_flags={"boss_appeared": True},
        )
        
        json_str = state.to_json()
        restored = WorldState.from_json(json_str)
        
        assert "hero" in restored.characters
        assert restored.environment.location_id == "battlefield"
        assert restored.narrative_flags["boss_appeared"] is True
    
    def test_merge_observation(self):
        from models.world_state_graph import WorldState, CharacterState
        
        # Initial state
        state = WorldState(
            characters={
                "hero": CharacterState("hero", emotion="calm", position={"x": 0, "y": 0, "z": 0}),
            }
        )
        
        # Observation from video
        observation = {
            "characters": {
                "hero": {"emotion": "angry", "position": {"x": 10, "y": 0, "z": 0}},
                "villain": {"character_id": "villain", "emotion": "smug"},
            },
            "narrative_flags": {"confrontation_started": True},
        }
        
        # Merge
        new_state = state.merge(observation)
        
        # Check updates
        assert new_state.characters["hero"].emotion == "angry"
        assert new_state.characters["hero"].position["x"] == 10
        assert "villain" in new_state.characters
        assert new_state.narrative_flags["confrontation_started"] is True
        
        # Original unchanged
        assert state.characters["hero"].emotion == "calm"


class TestWorldStateNode:
    """Test WorldStateNode dataclass."""
    
    def test_creation(self):
        from models.world_state_graph import WorldStateNode, WorldState, BranchType
        
        node = WorldStateNode(
            node_id="test-node-123",
            episode_id="ep-001",
            parent_id=None,
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            branch_name="main",
            branch_type=BranchType.MAIN,
        )
        
        assert node.node_id == "test-node-123"
        assert node.depth == 0
        assert node.parent_id is None
    
    def test_serialization(self):
        from models.world_state_graph import WorldStateNode, WorldState, BranchType
        
        node = WorldStateNode(
            node_id="node-456",
            episode_id="ep-002",
            parent_id="node-123",
            created_at=datetime.utcnow(),
            world_state=WorldState(narrative_flags={"test": True}),
            transition_video_uri="https://example.com/video.mp4",
            branch_name="main",
            branch_type=BranchType.MAIN,
            depth=5,
            quality_score=0.85,
        )
        
        data = node.to_dict()
        restored = WorldStateNode.from_dict(data)
        
        assert restored.node_id == node.node_id
        assert restored.depth == 5
        assert restored.quality_score == 0.85
        assert restored.world_state.narrative_flags["test"] is True


class TestWorldStateGraph:
    """Test WorldStateGraph class."""
    
    def test_initialize(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        
        graph = WorldStateGraph(episode_id="test-ep")
        root = graph.initialize(WorldState())
        
        assert graph.root is not None
        assert graph.current == root
        assert root.depth == 0
        assert root.parent_id is None
    
    def test_transition(self):
        from models.world_state_graph import WorldStateGraph, WorldState, CharacterState
        
        graph = WorldStateGraph(episode_id="test-ep")
        initial_state = WorldState(
            characters={"hero": CharacterState("hero", emotion="calm")}
        )
        graph.initialize(initial_state)
        
        # Create transition
        observation = {
            "characters": {"hero": {"emotion": "excited"}},
        }
        new_node = graph.transition(
            video_uri="https://example.com/beat1.mp4",
            observation=observation,
            beat_id="beat-1",
            quality_score=0.9,
        )
        
        assert new_node.depth == 1
        assert new_node.parent_id == graph.root.node_id
        assert new_node.world_state.characters["hero"].emotion == "excited"
        assert graph.current == new_node
    
    def test_multiple_transitions(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState())
        
        for i in range(5):
            graph.transition(
                video_uri=f"https://example.com/beat{i}.mp4",
                observation={"narrative_flags": {f"step_{i}": True}},
            )
        
        assert graph.node_count() == 6  # root + 5 transitions
        assert graph.current.depth == 5
    
    def test_fork(self):
        from models.world_state_graph import WorldStateGraph, WorldState, BranchType
        
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState())
        
        # Add some transitions on main
        graph.transition("v1.mp4", {"narrative_flags": {"step1": True}})
        main_node = graph.current
        
        # Fork
        fork_node = graph.fork(branch_name="retry_1", branch_type=BranchType.RETRY)
        
        assert fork_node.branch_name == "retry_1"
        assert fork_node.branch_type == BranchType.RETRY
        assert fork_node.parent_id == main_node.node_id
        assert "retry_1" in graph.list_branches()
    
    def test_switch_branch(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState())
        graph.transition("v1.mp4", {})
        
        main_head = graph.current
        
        # Fork and add transitions
        graph.fork(branch_name="alt")
        graph.transition("alt_v1.mp4", {"narrative_flags": {"alt": True}})
        alt_head = graph.current
        
        # Switch back to main
        graph.switch_branch("main")
        assert graph.current == main_head
        
        # Switch back to alt
        graph.switch_branch("alt")
        assert graph.current == alt_head
    
    def test_replay(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState())
        
        # Create chain of transitions
        for i in range(3):
            graph.transition(f"v{i}.mp4", {})
        
        # Replay from current
        path = graph.replay()
        
        assert len(path) == 4  # root + 3 transitions
        assert path[0].depth == 0  # root
        assert path[-1].depth == 3  # current
    
    def test_counterfactual(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState(narrative_flags={"x": 1}))
        
        root_id = graph.root.node_id
        
        # Create counterfactual - what if x was 100?
        cf_node = graph.counterfactual(
            from_node_id=root_id,
            alt_observation={"narrative_flags": {"x": 100}},
        )
        
        assert cf_node.world_state.narrative_flags["x"] == 100
        assert cf_node.branch_type.value == "cf"
    
    def test_serialization(self):
        from models.world_state_graph import WorldStateGraph, WorldState
        
        graph = WorldStateGraph(episode_id="test-ep")
        graph.initialize(WorldState())
        graph.transition("v1.mp4", {"narrative_flags": {"step": 1}})
        
        # Serialize
        data = graph.to_dict()
        
        # Deserialize
        restored = WorldStateGraph.from_dict(data)
        
        assert restored.episode_id == graph.episode_id
        assert restored.node_count() == 2
        assert restored.current.world_state.narrative_flags["step"] == 1


class TestStateTransition:
    """Test StateTransition dataclass."""
    
    def test_creation(self):
        from models.state_transition import StateTransition, TransitionStatus
        
        trans = StateTransition(
            transition_id="trans-001",
            episode_id="ep-001",
            source_node_id="node-001",
            beat_id="beat-001",
        )
        
        assert trans.status == TransitionStatus.PENDING
        assert trans.is_terminal() is False
    
    def test_status_transitions(self):
        from models.state_transition import (
            StateTransition, TransitionStatus, ActionOutcome
        )
        
        trans = StateTransition(
            transition_id="trans-002",
            episode_id="ep-001",
            source_node_id="node-001",
        )
        
        # Rendering
        trans.mark_rendering()
        assert trans.status == TransitionStatus.RENDERING
        
        # Observing
        trans.mark_observing("https://example.com/video.mp4", 5.0)
        assert trans.status == TransitionStatus.OBSERVING
        assert trans.video_uri == "https://example.com/video.mp4"
        
        # Completed
        trans.mark_completed(
            target_node_id="node-002",
            observation={"test": True},
            action_outcome=ActionOutcome.SUCCESS,
            quality_score=0.88,
        )
        assert trans.status == TransitionStatus.COMPLETED
        assert trans.is_terminal() is True
        assert trans.metrics.quality_score == 0.88
    
    def test_rejection_and_retry(self):
        from models.state_transition import StateTransition, TransitionStatus
        
        trans = StateTransition(
            transition_id="trans-003",
            episode_id="ep-001",
            source_node_id="node-001",
        )
        
        trans.mark_rejected("Quality too low")
        assert trans.status == TransitionStatus.REJECTED
        assert trans.needs_retry() is True
        assert trans.metrics.retry_count == 1
    
    def test_serialization(self):
        from models.state_transition import (
            StateTransition, ActionOutcome, ContinuityError
        )
        
        trans = StateTransition(
            transition_id="trans-004",
            episode_id="ep-001",
            source_node_id="node-001",
            action_outcome=ActionOutcome.PARTIAL,
            continuity_errors=[
                ContinuityError(
                    error_type="character_missing",
                    description="Genos not visible",
                    severity=0.6,
                    affected_entities=["genos"],
                )
            ],
        )
        
        data = trans.to_dict()
        restored = StateTransition.from_dict(data)
        
        assert restored.action_outcome == ActionOutcome.PARTIAL
        assert len(restored.continuity_errors) == 1
        assert restored.continuity_errors[0].error_type == "character_missing"


# ===========================================================================
# TIER 2: Persistence Layer Tests
# ===========================================================================


class TestWorldGraphStore:
    """Test WorldGraphStore persistence layer."""
    
    def test_insert_and_get_node(self, test_db):
        from models.world_state_graph import WorldStateNode, WorldState, BranchType
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        
        node = WorldStateNode(
            node_id="persist-node-001",
            episode_id="ep-001",
            parent_id=None,
            created_at=datetime.utcnow(),
            world_state=WorldState(narrative_flags={"persisted": True}),
            branch_name="main",
            branch_type=BranchType.MAIN,
        )
        
        store.insert_node(node)
        
        retrieved = store.get_node("persist-node-001")
        assert retrieved is not None
        assert retrieved.world_state.narrative_flags["persisted"] is True
    
    def test_get_ancestry(self, test_db):
        from models.world_state_graph import WorldStateNode, WorldState, BranchType
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        
        # Create chain of nodes
        root = WorldStateNode(
            node_id="root",
            episode_id="ep-002",
            parent_id=None,
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            branch_name="main",
            branch_type=BranchType.MAIN,
            depth=0,
        )
        store.insert_node(root)
        
        child1 = WorldStateNode(
            node_id="child1",
            episode_id="ep-002",
            parent_id="root",
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            branch_name="main",
            branch_type=BranchType.MAIN,
            depth=1,
        )
        store.insert_node(child1)
        
        child2 = WorldStateNode(
            node_id="child2",
            episode_id="ep-002",
            parent_id="child1",
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            branch_name="main",
            branch_type=BranchType.MAIN,
            depth=2,
        )
        store.insert_node(child2)
        
        # Get ancestry
        ancestry = store.get_ancestry("child2")
        
        assert len(ancestry) == 3
        assert ancestry[0].node_id == "root"
        assert ancestry[2].node_id == "child2"
    
    def test_insert_and_get_transition(self, test_db):
        from models.state_transition import StateTransition, ActionOutcome
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        
        trans = StateTransition(
            transition_id="trans-persist-001",
            episode_id="ep-003",
            source_node_id="node-001",
            beat_id="beat-001",
            action_outcome=ActionOutcome.SUCCESS,
        )
        
        store.insert_transition(trans)
        
        retrieved = store.get_transition("trans-persist-001")
        assert retrieved is not None
        assert retrieved.beat_id == "beat-001"
        assert retrieved.action_outcome == ActionOutcome.SUCCESS
    
    def test_branch_management(self, test_db):
        from models.world_state_graph import WorldStateNode, WorldState, BranchType
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        
        # Create node on main branch
        main_node = WorldStateNode(
            node_id="main-001",
            episode_id="ep-004",
            parent_id=None,
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            branch_name="main",
            branch_type=BranchType.MAIN,
        )
        store.insert_node(main_node)
        
        # Create node on fork branch
        fork_node = WorldStateNode(
            node_id="fork-001",
            episode_id="ep-004",
            parent_id="main-001",
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            branch_name="retry_1",
            branch_type=BranchType.RETRY,
        )
        store.insert_node(fork_node)
        
        # Check branches
        branches = store.list_branches("ep-004")
        assert "main" in branches
        assert "retry_1" in branches
        
        # Get branch heads
        main_head = store.get_branch_head("ep-004", "main")
        assert main_head.node_id == "main-001"
        
        retry_head = store.get_branch_head("ep-004", "retry_1")
        assert retry_head.node_id == "fork-001"
    
    def test_training_pairs(self, test_db):
        from models.world_state_graph import WorldStateNode, WorldState, BranchType
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        
        # Create nodes with training data
        node1 = WorldStateNode(
            node_id="train-001",
            episode_id="ep-005",
            parent_id=None,
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            transition_video_uri="https://example.com/v1.mp4",
            observation_json='{"test": 1}',
            quality_score=0.9,
            branch_name="main",
            branch_type=BranchType.MAIN,
        )
        store.insert_node(node1)
        
        node2 = WorldStateNode(
            node_id="train-002",
            episode_id="ep-005",
            parent_id="train-001",
            created_at=datetime.utcnow(),
            world_state=WorldState(),
            transition_video_uri="https://example.com/v2.mp4",
            observation_json='{"test": 2}',
            quality_score=0.5,  # Low quality - should be filtered
            branch_name="main",
            branch_type=BranchType.MAIN,
        )
        store.insert_node(node2)
        
        # Get training pairs with threshold
        pairs = store.get_training_pairs(min_quality=0.7)
        
        assert len(pairs) == 1
        assert pairs[0]["quality_score"] == 0.9


# ===========================================================================
# TIER 2: Pipeline Integration Tests
# ===========================================================================


class TestPipelineIntegration:
    """Test world state graph integration with existing pipeline."""
    
    def test_graph_with_store(self, test_db):
        """Test graph operations with persistence layer."""
        from models.world_state_graph import WorldStateGraph, WorldState
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        graph = WorldStateGraph(episode_id="integration-01", store=store)
        
        # Initialize
        graph.initialize(WorldState(narrative_flags={"start": True}))
        
        # Add transitions
        graph.transition(
            video_uri="https://r2.example.com/v1.mp4",
            observation={"narrative_flags": {"step1": True}},
            beat_id="beat-1",
            quality_score=0.85,
        )
        
        graph.transition(
            video_uri="https://r2.example.com/v2.mp4",
            observation={"narrative_flags": {"step2": True}},
            beat_id="beat-2",
            quality_score=0.92,
        )
        
        # Verify persistence
        assert store.count_nodes("integration-01") == 3
        
        # Load from store
        root = store.get_root_node("integration-01")
        assert root is not None
        assert root.world_state.narrative_flags["start"] is True
        
        # Check ancestry
        all_nodes = store.get_episode_nodes("integration-01")
        assert len(all_nodes) == 3
    
    def test_fork_with_persistence(self, test_db):
        """Test branching with persistence."""
        from models.world_state_graph import WorldStateGraph, WorldState, BranchType
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        graph = WorldStateGraph(episode_id="fork-test", store=store)
        
        graph.initialize(WorldState())
        graph.transition("v1.mp4", {"x": 1})
        
        # Fork
        graph.fork(branch_name="retry", branch_type=BranchType.RETRY)
        graph.transition("retry_v1.mp4", {"x": 2})
        
        # Verify branches persisted
        branches = store.list_branches("fork-test")
        assert len(branches) == 2
        
        # Verify head nodes
        main_head = store.get_branch_head("fork-test", "main")
        retry_head = store.get_branch_head("fork-test", "retry")
        
        assert main_head.node_id != retry_head.node_id
    
    def test_replay_from_persistence(self, test_db):
        """Test replay functionality with persisted data."""
        from models.world_state_graph import WorldStateGraph, WorldState
        from runtime.persistence.world_graph_store import WorldGraphStore
        
        store = WorldGraphStore()
        graph = WorldStateGraph(episode_id="replay-test", store=store)
        
        graph.initialize(WorldState())
        for i in range(5):
            graph.transition(f"v{i}.mp4", {"step": i})
        
        current_id = graph.current.node_id
        
        # Replay using store
        path = store.get_ancestry(current_id)
        
        assert len(path) == 6
        assert path[0].parent_id is None
        assert path[-1].node_id == current_id


# ===========================================================================
# Run tests
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
