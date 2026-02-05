"""
Tests for Strict Provenance Enforcement (Phase 7.5 Gate).
"""
import pytest
import uuid
from models.world_state_graph import WorldStateGraph, WorldState, ProvenanceError

class TestStrictProvenance:
    
    def test_transition_requires_observation_id(self):
        """Test that state transition raises error without observation_id."""
        graph = WorldStateGraph(episode_id="prov-test")
        graph.initialize()
        
        # 1. Attempt invalid transition (Planner Injection)
        # Missing observation_id simulates "hallucinated" state update
        invalid_obs = {
            "characters": {"hero": {"visible": True}},
            # No observation_id
        }
        
        with pytest.raises(ProvenanceError) as exc:
            graph.transition(
                video_uri="file:///test.mp4",
                observation=invalid_obs
            )
        
        assert "observation_id" in str(exc.value)
        
        # 2. Attempt invalid transition (Missing Video)
        valid_obs = {
            "observation_id": "obs-001",
            "characters": {"hero": {"visible": True}}
        }
        
        with pytest.raises(ProvenanceError) as exc:
            graph.transition(
                video_uri="", # Missing video
                observation=valid_obs
            )
            
        assert "video_uri" in str(exc.value)
        
    def test_valid_transition_succeeds(self):
        """Test that valid transition with provenance succeeds."""
        graph = WorldStateGraph(episode_id="prov-pass")
        graph.initialize()
        
        valid_obs = {
            "observation_id": "obs-abc-123",
            "characters": {"hero": {"visible": True}}
        }
        
        node = graph.transition(
            video_uri="file:///valid.mp4",
            observation=valid_obs
        )
        
        assert node is not None
        assert node.depth == 1
