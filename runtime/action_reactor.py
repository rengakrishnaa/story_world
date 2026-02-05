"""
Action Reactor

Fully reactive action selection at the micro level.
Actions are selected based purely on current world state,
not on predetermined plans.

Design Philosophy:
"Moment-to-moment decisions emerge from state, not script."
"""

from __future__ import annotations

import os
import json
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime

from models.goal_graph import (
    MicroAction,
    ActionType,
    ActionProposal, # Replaces StoryBeat
)

logger = logging.getLogger(__name__)


# ============================================================================
# Action Catalogs
# ============================================================================

@dataclass
class ActionCatalog:
    """A collection of action templates organized by type."""
    name: str
    actions: Dict[str, MicroAction] = field(default_factory=dict)
    
    def add(self, action: MicroAction) -> None:
        self.actions[action.action_id] = action
    
    def get_by_type(self, action_type: ActionType) -> List[MicroAction]:
        return [a for a in self.actions.values() if a.action_type == action_type]
    
    def get_by_actor(self, actor: str) -> List[MicroAction]:
        return [a for a in self.actions.values() if a.actor == actor]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "actions": {k: v.to_dict() for k, v in self.actions.items()},
        }


# Default action templates
DEFAULT_DIALOGUE_ACTIONS = [
    MicroAction("dialogue-greeting", ActionType.DIALOGUE, "Character greets warmly"),
    MicroAction("dialogue-confrontation", ActionType.DIALOGUE, "Character confronts directly"),
    MicroAction("dialogue-revelation", ActionType.DIALOGUE, "Character reveals information"),
    MicroAction("dialogue-question", ActionType.DIALOGUE, "Character asks question"),
    MicroAction("dialogue-farewell", ActionType.DIALOGUE, "Character says goodbye"),
]

DEFAULT_MOVEMENT_ACTIONS = [
    MicroAction("move-approach", ActionType.MOVEMENT, "Character approaches", motion_intensity=0.3),
    MicroAction("move-retreat", ActionType.MOVEMENT, "Character steps back", motion_intensity=0.3),
    MicroAction("move-run", ActionType.MOVEMENT, "Character runs", motion_intensity=0.9),
    MicroAction("move-walk", ActionType.MOVEMENT, "Character walks", motion_intensity=0.4),
    MicroAction("move-turn", ActionType.MOVEMENT, "Character turns around", motion_intensity=0.2),
]

DEFAULT_GESTURE_ACTIONS = [
    MicroAction("gesture-point", ActionType.GESTURE, "Character points"),
    MicroAction("gesture-wave", ActionType.GESTURE, "Character waves"),
    MicroAction("gesture-nod", ActionType.GESTURE, "Character nods"),
    MicroAction("gesture-shake-head", ActionType.GESTURE, "Character shakes head"),
    MicroAction("gesture-shrug", ActionType.GESTURE, "Character shrugs"),
]

DEFAULT_EMOTION_ACTIONS = [
    MicroAction("emotion-smile", ActionType.EMOTION_SHIFT, "Character smiles"),
    MicroAction("emotion-frown", ActionType.EMOTION_SHIFT, "Character frowns"),
    MicroAction("emotion-surprise", ActionType.EMOTION_SHIFT, "Character looks surprised"),
    MicroAction("emotion-anger", ActionType.EMOTION_SHIFT, "Character looks angry"),
    MicroAction("emotion-sad", ActionType.EMOTION_SHIFT, "Character looks sad"),
]

DEFAULT_COMBAT_ACTIONS = [
    MicroAction("combat-punch", ActionType.COMBAT, "Character throws punch", motion_intensity=0.9),
    MicroAction("combat-kick", ActionType.COMBAT, "Character delivers kick", motion_intensity=0.9),
    MicroAction("combat-dodge", ActionType.COMBAT, "Character dodges attack", motion_intensity=0.8),
    MicroAction("combat-block", ActionType.COMBAT, "Character blocks attack", motion_intensity=0.7),
    MicroAction("combat-power-attack", ActionType.COMBAT, "Character unleashes power attack", motion_intensity=1.0),
]


def create_default_catalog() -> ActionCatalog:
    """Create catalog with default action templates."""
    catalog = ActionCatalog(name="default")
    
    for action in DEFAULT_DIALOGUE_ACTIONS:
        catalog.add(action)
    for action in DEFAULT_MOVEMENT_ACTIONS:
        catalog.add(action)
    for action in DEFAULT_GESTURE_ACTIONS:
        catalog.add(action)
    for action in DEFAULT_EMOTION_ACTIONS:
        catalog.add(action)
    for action in DEFAULT_COMBAT_ACTIONS:
        catalog.add(action)
    
    return catalog


# ============================================================================
# State Conditions
# ============================================================================

@dataclass  
class StateCondition:
    """A condition that can be evaluated against world state."""
    key: str
    operator: str  # eq, ne, gt, lt, contains, exists
    value: Any
    
    def evaluate(self, world_state: Dict[str, Any]) -> bool:
        """Evaluate condition against world state."""
        actual = self._get_nested(world_state, self.key)
        
        if self.operator == "exists":
            return actual is not None
        elif self.operator == "eq":
            return actual == self.value
        elif self.operator == "ne":
            return actual != self.value
        elif self.operator == "gt":
            return actual is not None and actual > self.value
        elif self.operator == "lt":
            return actual is not None and actual < self.value
        elif self.operator == "contains":
            return self.value in (actual or [])
        
        return False
    
    def _get_nested(self, d: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation."""
        keys = key.split(".")
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return None
        return d


@dataclass
class ActionRule:
    """A rule that maps conditions to actions."""
    rule_id: str
    conditions: List[StateCondition]
    action: MicroAction
    priority: int = 5
    
    def matches(self, world_state: Dict[str, Any]) -> bool:
        """Check if all conditions match."""
        return all(c.evaluate(world_state) for c in self.conditions)


# ============================================================================
# Action Reactor
# ============================================================================

class ActionReactor:
    """
    Reactive action selection engine.
    
    Selects actions based purely on current world state,
    using rules and heuristics rather than predetermined scripts.
    
    Usage:
        reactor = ActionReactor()
        
        # Get actions for current state
        actions = reactor.react(
            world_state=current_state,
            beat=current_beat,
            characters=["hero", "villain"],
        )
        
        # Use actions in rendering
        for action in actions:
            render(action)
    """
    
    def __init__(
        self,
        catalog: Optional[ActionCatalog] = None,
        rules: Optional[List[ActionRule]] = None,
    ):
        self.catalog = catalog or create_default_catalog()
        self.rules = rules or []
        
        # Configuration
        self.max_actions_per_character: int = 2
        self.randomness: float = 0.2
        
        logger.info(
            f"[action_reactor] initialized: "
            f"{len(self.catalog.actions)} actions, {len(self.rules)} rules"
        )
    
    def react(
        self,
        world_state: Dict[str, Any],
        beat: Optional[ActionProposal] = None,
        characters: Optional[List[str]] = None,
        action_types: Optional[List[ActionType]] = None,
    ) -> List[MicroAction]:
        """
        React to current state and select actions.
        
        Args:
            world_state: Current world state
            beat: Optional story beat/proposal context
            characters: Characters to generate actions for
            action_types: Restrict to specific action types
            
        Returns:
            List of selected actions
        """
        characters = characters or []
        selected = []
        
        # 1. Check rules first (highest priority)
        rule_actions = self._evaluate_rules(world_state)
        selected.extend(rule_actions)
        
        # 2. Generate contextual actions
        contextual = self._generate_contextual_actions(
            world_state,
            beat,
            characters,
            action_types,
        )
        selected.extend(contextual)
        
        # 3. Add variety with randomness
        if random.random() < self.randomness:
            random_action = self._select_random_action(characters, action_types)
            if random_action:
                selected.append(random_action)
        
        # Deduplicate and sort by relevance
        selected = self._deduplicate(selected)
        selected.sort(key=lambda a: -a.relevance_score)
        
        logger.debug(f"[action_reactor] selected {len(selected)} actions")
        return selected
    
    def _evaluate_rules(
        self,
        world_state: Dict[str, Any],
    ) -> List[MicroAction]:
        """Evaluate rules and return triggered actions."""
        triggered = []
        
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            if rule.matches(world_state):
                action = MicroAction(
                    action_id=f"{rule.action.action_id}-{id(rule)}",
                    action_type=rule.action.action_type,
                    description=rule.action.description,
                    actor=rule.action.actor,
                    targets=rule.action.targets.copy(),
                    state_effects=rule.action.state_effects.copy(),
                    motion_intensity=rule.action.motion_intensity,
                )
                action.relevance_score = rule.priority / 10
                triggered.append(action)
        
        return triggered
    
    def _generate_contextual_actions(
        self,
        world_state: Dict[str, Any],
        beat: Optional[ActionProposal],
        characters: List[str],
        action_types: Optional[List[ActionType]],
    ) -> List[MicroAction]:
        """Generate actions based on context."""
        actions = []
        
        for char in characters[:5]:  # Limit characters
            char_actions = self._select_for_character(
                char,
                world_state,
                beat,
                action_types,
            )
            actions.extend(char_actions[:self.max_actions_per_character])
        
        return actions
    
    def _select_for_character(
        self,
        character: str,
        world_state: Dict[str, Any],
        beat: Optional[ActionProposal],
        action_types: Optional[List[ActionType]],
    ) -> List[MicroAction]:
        """Select actions for a specific character."""
        candidates = []
        char_state = world_state.get("characters", {}).get(character, {})
        
        # Get character emotion for context
        emotion = char_state.get("emotion", "neutral")
        
        # Filter action templates
        for template in self.catalog.actions.values():
            if action_types and template.action_type not in action_types:
                continue
            
            # Score based on emotion match
            score = self._score_action_for_emotion(template, emotion)
            
            # Boost if beat objectives match
            if beat:
                for obj in beat.objectives:
                    if any(word in template.description.lower() for word in obj.lower().split()):
                        score += 0.3
            
            if score > 0:
                action = MicroAction(
                    action_id=f"{template.action_id}-{character}",
                    action_type=template.action_type,
                    description=template.description.replace("Character", character),
                    actor=character,
                    motion_intensity=template.motion_intensity,
                    duration_hint_sec=template.duration_hint_sec,
                )
                action.relevance_score = score
                candidates.append(action)
        
        # Sort by score and return top
        candidates.sort(key=lambda a: -a.relevance_score)
        return candidates[:3]
    
    def _score_action_for_emotion(
        self,
        action: MicroAction,
        emotion: str,
    ) -> float:
        """Score action relevance to emotion."""
        # Emotion-action affinity matrix
        affinity = {
            "angry": [ActionType.COMBAT, ActionType.MOVEMENT],
            "happy": [ActionType.DIALOGUE, ActionType.GESTURE],
            "sad": [ActionType.EMOTION_SHIFT],
            "fearful": [ActionType.MOVEMENT],
            "neutral": [ActionType.GESTURE, ActionType.DIALOGUE],
            "determined": [ActionType.COMBAT, ActionType.MOVEMENT],
            "excited": [ActionType.GESTURE, ActionType.DIALOGUE],
        }
        
        preferred = affinity.get(emotion, [])
        if action.action_type in preferred:
            return 0.7
        return 0.3
    
    def _select_random_action(
        self,
        characters: List[str],
        action_types: Optional[List[ActionType]],
    ) -> Optional[MicroAction]:
        """Select a random action for variety."""
        candidates = list(self.catalog.actions.values())
        
        if action_types:
            candidates = [a for a in candidates if a.action_type in action_types]
        
        if not candidates:
            return None
        
        template = random.choice(candidates)
        actor = random.choice(characters) if characters else None
        
        action = MicroAction(
            action_id=f"{template.action_id}-random",
            action_type=template.action_type,
            description=template.description,
            actor=actor,
            motion_intensity=template.motion_intensity,
        )
        action.relevance_score = 0.1
        return action
    
    def _deduplicate(
        self,
        actions: List[MicroAction],
    ) -> List[MicroAction]:
        """Remove duplicate actions."""
        seen = set()
        unique = []
        
        for action in actions:
            key = (action.actor, action.action_type, action.description[:30])
            if key not in seen:
                seen.add(key)
                unique.append(action)
        
        return unique
    
    def add_rule(
        self,
        conditions: List[Tuple[str, str, Any]],
        action: MicroAction,
        priority: int = 5,
    ) -> ActionRule:
        """
        Add a reactive rule.
        
        Args:
            conditions: List of (key, operator, value) tuples
            action: Action to trigger
            priority: Rule priority (1-10)
            
        Returns:
            Created ActionRule
        """
        rule = ActionRule(
            rule_id=f"rule-{len(self.rules)}",
            conditions=[
                StateCondition(key=k, operator=op, value=v)
                for k, op, v in conditions
            ],
            action=action,
            priority=priority,
        )
        self.rules.append(rule)
        return rule
    
    def react_to_observation(
        self,
        observation: Dict[str, Any],
        characters: List[str],
    ) -> List[MicroAction]:
        """
        React to a video observation result.
        
        Args:
            observation: Observation from video
            characters: Characters involved
            
        Returns:
            Reactive actions based on observation
        """
        actions = []
        
        # React to action outcomes
        action_outcome = observation.get("action", {}).get("outcome", "unknown")
        
        if action_outcome == "failed":
            # Retry or escalate
            for char in characters[:2]:
                action = MicroAction(
                    action_id=f"react-retry-{char}",
                    action_type=ActionType.GESTURE,
                    description=f"{char} attempts again with more effort",
                    actor=char,
                    motion_intensity=0.7,
                )
                action.relevance_score = 0.8
                actions.append(action)
        
        # React to character states
        for char_id, char_obs in observation.get("characters", {}).items():
            emotion = char_obs.get("emotion")
            if emotion == "angry":
                action = MicroAction(
                    action_id=f"react-anger-{char_id}",
                    action_type=ActionType.EMOTION_SHIFT,
                    description=f"{char_id} expresses anger",
                    actor=char_id,
                    motion_intensity=0.8,
                )
                action.relevance_score = 0.7
                actions.append(action)
        
        return actions


# ============================================================================
# Convenience Functions
# ============================================================================

def create_reactor_with_rules(
    rules_config: Optional[List[Dict[str, Any]]] = None,
) -> ActionReactor:
    """
    Create reactor with custom rules.
    
    Args:
        rules_config: List of rule configurations
        
    Returns:
        Configured ActionReactor
    """
    reactor = ActionReactor()
    
    if rules_config:
        for config in rules_config:
            conditions = [
                (c["key"], c["operator"], c["value"])
                for c in config.get("conditions", [])
            ]
            action_data = config.get("action", {})
            action = MicroAction(
                action_id=action_data.get("action_id", f"rule-action-{len(reactor.rules)}"),
                action_type=ActionType(action_data.get("action_type", "custom")),
                description=action_data.get("description", ""),
            )
            reactor.add_rule(conditions, action, config.get("priority", 5))
    
    return reactor


def quick_react(
    world_state: Dict[str, Any],
    characters: List[str],
) -> List[MicroAction]:
    """Quick reaction without full reactor setup."""
    reactor = ActionReactor()
    return reactor.react(world_state, characters=characters)
