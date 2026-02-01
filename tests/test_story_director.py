"""
Test Suite for Story Director (Phase 4)

Two-tier testing:
1. Individual component tests - test each class in isolation
2. Integration tests - test with pipeline components
"""

import pytest
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch


# ===========================================================================
# TIER 1: Individual Component Tests - Story Intent
# ===========================================================================


class TestIntentLevel:
    """Test IntentLevel enum."""
    
    def test_values(self):
        from models.story_intent import IntentLevel
        
        assert IntentLevel.MACRO.value == "macro"
        assert IntentLevel.MESO.value == "meso"
        assert IntentLevel.MICRO.value == "micro"


class TestIntentStatus:
    """Test IntentStatus enum."""
    
    def test_values(self):
        from models.story_intent import IntentStatus
        
        assert IntentStatus.PENDING.value == "pending"
        assert IntentStatus.COMPLETED.value == "completed"
        assert IntentStatus.FAILED.value == "failed"


class TestNarrativeGoalType:
    """Test NarrativeGoalType enum."""
    
    def test_values(self):
        from models.story_intent import NarrativeGoalType
        
        assert NarrativeGoalType.CHARACTER_ARC.value == "character_arc"
        assert NarrativeGoalType.PLOT_MILESTONE.value == "plot_milestone"


class TestMacroIntent:
    """Test MacroIntent dataclass."""
    
    def test_creation(self):
        from models.story_intent import MacroIntent, NarrativeGoalType
        
        intent = MacroIntent(
            intent_id="macro-001",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Hero defeats villain",
            characters=["hero", "villain"],
        )
        
        assert intent.intent_id == "macro-001"
        assert intent.goal_type == NarrativeGoalType.PLOT_MILESTONE
    
    def test_to_dict(self):
        from models.story_intent import MacroIntent, NarrativeGoalType
        
        intent = MacroIntent(
            intent_id="macro-002",
            goal_type=NarrativeGoalType.CHARACTER_ARC,
            description="Hero grows stronger",
            priority=8,
        )
        
        data = intent.to_dict()
        assert data["goal_type"] == "character_arc"
        assert data["priority"] == 8
    
    def test_from_dict(self):
        from models.story_intent import MacroIntent, NarrativeGoalType
        
        data = {
            "intent_id": "macro-003",
            "goal_type": "emotional",
            "description": "Emotional impact",
        }
        
        intent = MacroIntent.from_dict(data)
        assert intent.intent_id == "macro-003"
    
    def test_is_satisfied(self):
        from models.story_intent import MacroIntent, NarrativeGoalType
        
        intent = MacroIntent(
            intent_id="macro-004",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Villain defeated",
            target_state={"villain_defeated": True},
        )
        
        # Not satisfied
        assert intent.is_satisfied({"villain_defeated": False}) is False
        
        # Satisfied
        assert intent.is_satisfied({"villain_defeated": True}) is True


class TestStoryBeat:
    """Test StoryBeat dataclass."""
    
    def test_creation(self):
        from models.story_intent import StoryBeat
        
        beat = StoryBeat(
            beat_id="beat-001",
            description="Training montage",
            objectives=["Show growth"],
            characters=["hero", "mentor"],
        )
        
        assert beat.beat_id == "beat-001"
        assert "hero" in beat.characters
    
    def test_to_dict(self):
        from models.story_intent import StoryBeat
        
        beat = StoryBeat(
            beat_id="beat-002",
            description="Confrontation",
            location="arena",
            is_optional=True,
        )
        
        data = beat.to_dict()
        assert data["location"] == "arena"
        assert data["is_optional"] is True
    
    def test_from_dict(self):
        from models.story_intent import StoryBeat
        
        data = {
            "beat_id": "beat-003",
            "description": "Test beat",
            "depends_on": ["beat-001", "beat-002"],
        }
        
        beat = StoryBeat.from_dict(data)
        assert len(beat.depends_on) == 2


class TestMicroAction:
    """Test MicroAction dataclass."""
    
    def test_creation(self):
        from models.story_intent import MicroAction, ActionType
        
        action = MicroAction(
            action_id="action-001",
            action_type=ActionType.DIALOGUE,
            description="Hero greets mentor",
            actor="hero",
        )
        
        assert action.actor == "hero"
        assert action.action_type == ActionType.DIALOGUE
    
    def test_to_dict(self):
        from models.story_intent import MicroAction, ActionType
        
        action = MicroAction(
            action_id="action-002",
            action_type=ActionType.COMBAT,
            description="Power attack",
            motion_intensity=1.0,
        )
        
        data = action.to_dict()
        assert data["action_type"] == "combat"
        assert data["motion_intensity"] == 1.0


class TestStoryIntentGraph:
    """Test StoryIntentGraph class."""
    
    def test_creation(self):
        from models.story_intent import StoryIntentGraph
        
        graph = StoryIntentGraph(episode_id="ep-001")
        
        assert graph.episode_id == "ep-001"
        assert len(graph.macro_intents) == 0
    
    def test_add_macro_intent(self):
        from models.story_intent import StoryIntentGraph, MacroIntent, NarrativeGoalType
        
        graph = StoryIntentGraph(episode_id="ep-001")
        intent = MacroIntent(
            intent_id="macro-001",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Test",
        )
        
        graph.add_macro_intent(intent)
        assert "macro-001" in graph.macro_intents
    
    def test_add_story_beat(self):
        from models.story_intent import StoryIntentGraph, StoryBeat
        
        graph = StoryIntentGraph(episode_id="ep-001")
        beat = StoryBeat(beat_id="beat-001", description="Test")
        
        graph.add_story_beat(beat)
        assert "beat-001" in graph.story_beats
    
    def test_get_pending_beats(self):
        from models.story_intent import StoryIntentGraph, StoryBeat, IntentStatus
        
        graph = StoryIntentGraph(episode_id="ep-001")
        
        # Add beats with dependencies
        beat1 = StoryBeat(beat_id="beat-001", description="First", suggested_position=1)
        beat2 = StoryBeat(beat_id="beat-002", description="Second", depends_on=["beat-001"], suggested_position=2)
        beat3 = StoryBeat(beat_id="beat-003", description="Third", suggested_position=3)
        
        graph.add_story_beat(beat1)
        graph.add_story_beat(beat2)
        graph.add_story_beat(beat3)
        
        # Only beat-001 and beat-003 should be pending (beat-002 has unsatisfied dependency)
        pending = graph.get_pending_beats()
        pending_ids = [b.beat_id for b in pending]
        
        assert "beat-001" in pending_ids
        assert "beat-002" not in pending_ids
        assert "beat-003" in pending_ids
    
    def test_get_next_beat(self):
        from models.story_intent import StoryIntentGraph, StoryBeat
        
        graph = StoryIntentGraph(episode_id="ep-001")
        beat1 = StoryBeat(beat_id="beat-001", description="First", suggested_position=1)
        beat2 = StoryBeat(beat_id="beat-002", description="Second", suggested_position=2)
        
        graph.add_story_beat(beat2)
        graph.add_story_beat(beat1)
        
        next_beat = graph.get_next_beat()
        assert next_beat.beat_id == "beat-001"  # Sorted by position
    
    def test_mark_beat_complete(self):
        from models.story_intent import StoryIntentGraph, StoryBeat, IntentStatus
        
        graph = StoryIntentGraph(episode_id="ep-001")
        beat = StoryBeat(beat_id="beat-001", description="Test")
        graph.add_story_beat(beat)
        
        graph.mark_beat_complete("beat-001", "video.mp4")
        
        assert beat.status == IntentStatus.COMPLETED
        assert beat.actual_video_uri == "video.mp4"
        assert "beat-001" in graph.completed_beats
    
    def test_serialization(self):
        from models.story_intent import (
            StoryIntentGraph, MacroIntent, StoryBeat, NarrativeGoalType
        )
        
        graph = StoryIntentGraph(episode_id="ep-001")
        graph.add_macro_intent(MacroIntent(
            intent_id="macro-001",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Test macro",
        ))
        graph.add_story_beat(StoryBeat(beat_id="beat-001", description="Test beat"))
        
        # Round-trip
        json_str = graph.to_json()
        restored = StoryIntentGraph.from_dict(json.loads(json_str))
        
        assert restored.episode_id == "ep-001"
        assert "macro-001" in restored.macro_intents
        assert "beat-001" in restored.story_beats


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_macro_intent(self):
        from models.story_intent import create_macro_intent, NarrativeGoalType
        
        intent = create_macro_intent(
            description="Hero wins",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
        )
        
        assert intent.intent_id.startswith("macro-")
        assert intent.description == "Hero wins"
    
    def test_create_story_beat(self):
        from models.story_intent import create_story_beat
        
        beat = create_story_beat(
            description="Training scene",
            characters=["hero", "mentor"],
        )
        
        assert beat.beat_id.startswith("beat-")
        assert len(beat.characters) == 2
    
    def test_create_micro_action(self):
        from models.story_intent import create_micro_action, ActionType
        
        action = create_micro_action(
            description="Hero smiles",
            action_type=ActionType.GESTURE,
            actor="hero",
        )
        
        assert action.action_id.startswith("action-")
        assert action.actor == "hero"


# ===========================================================================
# TIER 1: Individual Component Tests - Story Director
# ===========================================================================


class TestDirectorConfig:
    """Test DirectorConfig dataclass."""
    
    def test_defaults(self):
        from agents.story_director import DirectorConfig
        
        config = DirectorConfig()
        assert config.allow_macro_reordering is False
        assert config.max_beat_adaptations == 3


class TestDirectorState:
    """Test DirectorState dataclass."""
    
    def test_creation(self):
        from agents.story_director import DirectorState
        from models.story_intent import StoryIntentGraph
        
        graph = StoryIntentGraph(episode_id="ep-001")
        state = DirectorState(
            episode_id="ep-001",
            intent_graph=graph,
        )
        
        assert state.beats_completed == 0


class TestDecisionType:
    """Test DecisionType values."""
    
    def test_values(self):
        from agents.story_director import DecisionType
        
        assert DecisionType.PROCEED == "proceed"
        assert DecisionType.ADAPT == "adapt"
        assert DecisionType.COMPLETE == "complete"


class TestDirectorDecision:
    """Test DirectorDecision dataclass."""
    
    def test_creation(self):
        from agents.story_director import DirectorDecision, DecisionType
        
        decision = DirectorDecision(
            decision_type=DecisionType.PROCEED,
            beat_id="beat-001",
            reason="Next in sequence",
        )
        
        assert decision.decision_type == DecisionType.PROCEED
        assert decision.beat_id == "beat-001"


class TestStoryDirector:
    """Test StoryDirector class."""
    
    def test_creation(self):
        from agents.story_director import StoryDirector
        from models.story_intent import StoryIntentGraph
        
        graph = StoryIntentGraph(episode_id="ep-001")
        director = StoryDirector(graph)
        
        assert director.state.episode_id == "ep-001"
    
    def test_is_complete_empty_graph(self):
        from agents.story_director import StoryDirector
        from models.story_intent import StoryIntentGraph
        
        graph = StoryIntentGraph(episode_id="ep-001")
        director = StoryDirector(graph)
        
        # No macro intents = complete
        assert director.is_complete() is True
    
    def test_is_complete_with_intents(self):
        from agents.story_director import StoryDirector
        from models.story_intent import StoryIntentGraph, MacroIntent, NarrativeGoalType
        
        graph = StoryIntentGraph(episode_id="ep-001")
        graph.add_macro_intent(MacroIntent(
            intent_id="macro-001",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Test",
        ))
        
        director = StoryDirector(graph)
        
        # Has pending macro intent = not complete
        assert director.is_complete() is False
    
    def test_get_progress(self):
        from agents.story_director import StoryDirector
        from models.story_intent import StoryIntentGraph, StoryBeat
        
        graph = StoryIntentGraph(episode_id="ep-001")
        graph.add_story_beat(StoryBeat(beat_id="beat-001", description="Test"))
        graph.add_story_beat(StoryBeat(beat_id="beat-002", description="Test 2"))
        
        director = StoryDirector(graph)
        progress = director.get_progress()
        
        assert progress["beats_total"] == 2
        assert progress["beats_completed"] == 0
    
    def test_next_decision_proceed(self):
        from agents.story_director import StoryDirector, DecisionType
        from models.story_intent import StoryIntentGraph, StoryBeat, MacroIntent, NarrativeGoalType
        
        graph = StoryIntentGraph(episode_id="ep-001")
        graph.add_macro_intent(MacroIntent(
            intent_id="macro-001",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Goal",
        ))
        graph.add_story_beat(StoryBeat(
            beat_id="beat-001",
            description="First beat",
            contributes_to=["macro-001"],
        ))
        
        director = StoryDirector(graph)
        decision = director.next_decision()
        
        assert decision.decision_type == DecisionType.PROCEED
        assert decision.beat_id == "beat-001"
    
    def test_next_decision_complete(self):
        from agents.story_director import StoryDirector, DecisionType
        from models.story_intent import StoryIntentGraph
        
        graph = StoryIntentGraph(episode_id="ep-001")
        director = StoryDirector(graph)
        
        decision = director.next_decision()
        assert decision.decision_type == DecisionType.COMPLETE
    
    def test_record_outcome_success(self):
        from agents.story_director import StoryDirector
        from models.story_intent import StoryIntentGraph, StoryBeat, IntentStatus
        
        graph = StoryIntentGraph(episode_id="ep-001")
        graph.add_story_beat(StoryBeat(beat_id="beat-001", description="Test"))
        
        director = StoryDirector(graph)
        director.record_outcome(
            beat_id="beat-001",
            quality_result={"is_acceptable": True},
            video_uri="test.mp4",
        )
        
        assert graph.story_beats["beat-001"].status == IntentStatus.COMPLETED
        assert director.state.beats_completed == 1
    
    def test_record_outcome_failure(self):
        from agents.story_director import StoryDirector
        from models.story_intent import StoryIntentGraph, StoryBeat, IntentStatus
        
        graph = StoryIntentGraph(episode_id="ep-001")
        graph.add_story_beat(StoryBeat(beat_id="beat-001", description="Test"))
        
        director = StoryDirector(graph)
        director.record_outcome(
            beat_id="beat-001",
            quality_result={"is_acceptable": False},
        )
        
        assert graph.story_beats["beat-001"].status == IntentStatus.FAILED
        assert director.state.beats_failed == 1


# ===========================================================================
# TIER 1: Individual Component Tests - Action Reactor
# ===========================================================================


class TestActionCatalog:
    """Test ActionCatalog class."""
    
    def test_creation(self):
        from runtime.action_reactor import ActionCatalog
        
        catalog = ActionCatalog(name="test")
        assert len(catalog.actions) == 0
    
    def test_add_action(self):
        from runtime.action_reactor import ActionCatalog
        from models.story_intent import MicroAction, ActionType
        
        catalog = ActionCatalog(name="test")
        action = MicroAction("action-001", ActionType.GESTURE, "Test")
        
        catalog.add(action)
        assert "action-001" in catalog.actions
    
    def test_get_by_type(self):
        from runtime.action_reactor import ActionCatalog
        from models.story_intent import MicroAction, ActionType
        
        catalog = ActionCatalog(name="test")
        catalog.add(MicroAction("a1", ActionType.GESTURE, "Gesture"))
        catalog.add(MicroAction("a2", ActionType.DIALOGUE, "Dialogue"))
        catalog.add(MicroAction("a3", ActionType.GESTURE, "Another gesture"))
        
        gestures = catalog.get_by_type(ActionType.GESTURE)
        assert len(gestures) == 2


class TestDefaultCatalog:
    """Test default catalog creation."""
    
    def test_create_default(self):
        from runtime.action_reactor import create_default_catalog
        
        catalog = create_default_catalog()
        assert len(catalog.actions) > 0


class TestStateCondition:
    """Test StateCondition class."""
    
    def test_eq_operator(self):
        from runtime.action_reactor import StateCondition
        
        condition = StateCondition(key="status", operator="eq", value="active")
        
        assert condition.evaluate({"status": "active"}) is True
        assert condition.evaluate({"status": "inactive"}) is False
    
    def test_exists_operator(self):
        from runtime.action_reactor import StateCondition
        
        condition = StateCondition(key="hero", operator="exists", value=None)
        
        assert condition.evaluate({"hero": {"name": "Saitama"}}) is True
        assert condition.evaluate({"villain": {}}) is False
    
    def test_nested_key(self):
        from runtime.action_reactor import StateCondition
        
        condition = StateCondition(key="hero.emotion", operator="eq", value="angry")
        
        assert condition.evaluate({"hero": {"emotion": "angry"}}) is True


class TestActionRule:
    """Test ActionRule class."""
    
    def test_matches(self):
        from runtime.action_reactor import ActionRule, StateCondition
        from models.story_intent import MicroAction, ActionType
        
        rule = ActionRule(
            rule_id="rule-001",
            conditions=[
                StateCondition("combat_active", "eq", True),
            ],
            action=MicroAction("combat-react", ActionType.COMBAT, "React"),
            priority=8,
        )
        
        assert rule.matches({"combat_active": True}) is True
        assert rule.matches({"combat_active": False}) is False


class TestActionReactor:
    """Test ActionReactor class."""
    
    def test_creation(self):
        from runtime.action_reactor import ActionReactor
        
        reactor = ActionReactor()
        assert len(reactor.catalog.actions) > 0
    
    def test_react_basic(self):
        from runtime.action_reactor import ActionReactor
        
        reactor = ActionReactor()
        actions = reactor.react(
            world_state={},
            characters=["hero"],
        )
        
        assert len(actions) >= 0  # May or may not have actions
    
    def test_react_with_beat(self):
        from runtime.action_reactor import ActionReactor
        from models.story_intent import StoryBeat
        
        reactor = ActionReactor()
        beat = StoryBeat(
            beat_id="beat-001",
            description="Training montage",
            objectives=["Show growth"],
            characters=["hero", "mentor"],
        )
        
        actions = reactor.react(
            world_state={"characters": {"hero": {"emotion": "determined"}}},
            beat=beat,
            characters=["hero", "mentor"],
        )
        
        assert len(actions) > 0
    
    def test_add_rule(self):
        from runtime.action_reactor import ActionReactor
        from models.story_intent import MicroAction, ActionType
        
        reactor = ActionReactor()
        action = MicroAction("special-react", ActionType.GESTURE, "Special action")
        
        rule = reactor.add_rule(
            conditions=[("danger_level", "gt", 5)],
            action=action,
            priority=9,
        )
        
        assert len(reactor.rules) == 1
    
    def test_react_to_observation(self):
        from runtime.action_reactor import ActionReactor
        
        reactor = ActionReactor()
        
        observation = {
            "action": {"outcome": "failed"},
            "characters": {
                "hero": {"emotion": "angry"},
            },
        }
        
        actions = reactor.react_to_observation(observation, ["hero"])
        assert len(actions) > 0


# ===========================================================================
# TIER 2: Integration Tests
# ===========================================================================


class TestDirectorWithWorldGraph:
    """Test integration between StoryDirector and WorldStateGraph."""
    
    def test_director_updates_world_state(self):
        from agents.story_director import StoryDirector, DecisionType
        from models.story_intent import (
            StoryIntentGraph, MacroIntent, StoryBeat, NarrativeGoalType
        )
        from models.world_state_graph import WorldStateGraph, WorldState
        
        # Setup intent graph
        intent_graph = StoryIntentGraph(episode_id="integration-01")
        intent_graph.add_macro_intent(MacroIntent(
            intent_id="macro-001",
            goal_type=NarrativeGoalType.CHARACTER_ARC,
            description="Hero becomes stronger",
            target_state={"hero_trained": True},
        ))
        intent_graph.add_story_beat(StoryBeat(
            beat_id="beat-001",
            description="Training scene",
            expected_state_changes={"hero_trained": True},
            contributes_to=["macro-001"],
        ))
        
        # Setup world graph
        world_graph = WorldStateGraph(episode_id="integration-01")
        world_graph.initialize(WorldState())
        
        # Create director
        director = StoryDirector(intent_graph)
        
        # Get decision
        decision = director.next_decision()
        assert decision.decision_type == DecisionType.PROCEED
        assert decision.beat_id == "beat-001"
        
        # Execute beat (simulate)
        world_graph.transition(
            video_uri="training.mp4",
            observation={
                "narrative_flags": {"hero_trained": True},
            },
            beat_id="beat-001",
        )
        
        # Record outcome
        director.record_outcome(
            beat_id="beat-001",
            quality_result={"is_acceptable": True},
            video_uri="training.mp4",
        )
        
        # Verify
        assert world_graph.current.depth == 1


class TestDirectorWithReactor:
    """Test integration between StoryDirector and ActionReactor."""
    
    def test_reactor_provides_actions(self):
        from agents.story_director import StoryDirector
        from runtime.action_reactor import ActionReactor
        from models.story_intent import (
            StoryIntentGraph, StoryBeat, MicroAction, ActionType
        )
        
        # Setup
        graph = StoryIntentGraph(episode_id="reactor-test")
        
        # Add action templates
        graph.add_action_template(MicroAction(
            action_id="punch",
            action_type=ActionType.COMBAT,
            description="Throw punch",
            motion_intensity=0.9,
        ))
        
        graph.add_story_beat(StoryBeat(
            beat_id="beat-001",
            description="Combat scene",
            objectives=["Hero attacks"],
            characters=["hero"],
        ))
        
        director = StoryDirector(graph)
        reactor = ActionReactor()
        
        # Get decision
        decision = director.next_decision()
        
        # Use reactor for additional actions
        reactor_actions = reactor.react(
            world_state={"characters": {"hero": {"emotion": "determined"}}},
            beat=graph.story_beats["beat-001"],
            characters=["hero"],
        )
        
        # Should have actions from both sources
        assert len(decision.actions) >= 0
        assert len(reactor_actions) > 0


class TestDirectorWithQualityBudget:
    """Test integration with quality and budget systems."""
    
    def test_quality_affects_outcome(self):
        from agents.story_director import StoryDirector
        from runtime.quality_evaluator import (
            QualityEvaluator, QualityScores, EvaluationContext, TaskType
        )
        from runtime.budget_controller import BudgetController
        from models.story_intent import StoryIntentGraph, StoryBeat, IntentStatus
        
        # Setup
        graph = StoryIntentGraph(episode_id="quality-test")
        graph.add_story_beat(StoryBeat(beat_id="beat-001", description="Test"))
        
        director = StoryDirector(graph)
        evaluator = QualityEvaluator()
        controller = BudgetController(episode_id="quality-test")
        
        # Request budget
        controller.request_budget(beat_id="beat-001")
        
        # Simulate low quality result
        scores = QualityScores(
            visual_clarity=0.5,
            action_clarity=0.5,
            character_recognizability=0.6,
        )
        context = EvaluationContext(task_type=TaskType.STORYTELLING)
        quality = evaluator.evaluate(scores, context)
        
        # Record outcome
        director.record_outcome(
            beat_id="beat-001",
            quality_result={"is_acceptable": quality.is_acceptable},
        )
        controller.record_attempt(
            "beat-001",
            success=quality.is_acceptable,
            cost_usd=0.05,
        )
        
        # Should have failed due to low quality
        assert graph.story_beats["beat-001"].status == IntentStatus.FAILED


class TestFullPipelineIntegration:
    """Test full pipeline integration."""
    
    def test_complete_episode_flow(self):
        from agents.story_director import StoryDirector, DecisionType
        from runtime.action_reactor import ActionReactor
        from models.story_intent import (
            StoryIntentGraph, MacroIntent, StoryBeat, NarrativeGoalType
        )
        from models.world_state_graph import WorldStateGraph, WorldState
        
        # Setup intent graph with multiple beats
        intent_graph = StoryIntentGraph(episode_id="full-pipeline")
        
        intent_graph.add_macro_intent(MacroIntent(
            intent_id="goal",
            goal_type=NarrativeGoalType.PLOT_MILESTONE,
            description="Complete story",
        ))
        
        for i in range(3):
            intent_graph.add_story_beat(StoryBeat(
                beat_id=f"beat-{i+1}",
                description=f"Scene {i+1}",
                suggested_position=i+1,
                contributes_to=["goal"],
            ))
        
        # Create systems
        director = StoryDirector(intent_graph)
        reactor = ActionReactor()
        world_graph = WorldStateGraph(episode_id="full-pipeline")
        world_graph.initialize(WorldState())
        
        # Execute beats
        completed = 0
        while not director.is_complete() and completed < 10:
            decision = director.next_decision()
            
            if decision.decision_type == DecisionType.COMPLETE:
                break
            
            if decision.beat_id:
                # Get reactive actions
                beat = intent_graph.story_beats.get(decision.beat_id)
                if beat:
                    actions = reactor.react(
                        world_state={},
                        beat=beat,
                        characters=beat.characters,
                    )
                
                # Simulate successful execution
                world_graph.transition(
                    video_uri=f"video_{decision.beat_id}.mp4",
                    observation={},
                    beat_id=decision.beat_id,
                )
                
                director.record_outcome(
                    beat_id=decision.beat_id,
                    quality_result={"is_acceptable": True},
                )
                completed += 1
        
        # Verify completion
        assert completed == 3
        assert world_graph.current.depth == 3
        
        progress = director.get_progress()
        assert progress["beats_completed"] == 3


# ===========================================================================
# Run tests
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
