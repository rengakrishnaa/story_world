"""
Versioned World State Graph

Core data structure for video-native world simulation.
Each video render creates a state transition in the graph.
Supports branching, forking, and counterfactual exploration.

Architecture:
    WorldState_t → Video → Observation → WorldState_t+1
                        ↳ fork → alternate timeline
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProvenanceError(Exception):
    """Raised when state update lacks required video provenance."""
    pass


class BranchType(Enum):
    """Type of branch in the world state graph."""
    MAIN = "main"           # Primary timeline
    RETRY = "retry"         # Regeneration attempt
    COUNTERFACTUAL = "cf"   # What-if exploration
    FORK = "fork"           # User-initiated branch


# Schema version for audit/traceability. Bump on breaking changes.
WORLD_STATE_SCHEMA_VERSION = "1.0"

# Explicit field allowlists - no silent fallback from observation model drift
_CHARACTER_STATE_FIELDS = frozenset({"character_id", "position", "emotion", "pose", "facing_direction", "visible"})
_ENVIRONMENT_STATE_FIELDS = frozenset({"location_id", "time_of_day", "weather", "lighting", "objects"})


def _strip_to_schema(data: Dict[str, Any], allowed: frozenset, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Strip dict to only allowed schema fields. Prevents observation model drift from breaking merge."""
    if not data or not isinstance(data, dict):
        return dict(defaults or {})
    stripped = {k: v for k, v in data.items() if k in allowed and v is not None}
    if defaults:
        for k, default in defaults.items():
            if k not in stripped and k in allowed:
                stripped[k] = default
    return stripped


@dataclass
class CharacterState:
    """Observable character state extracted from video."""
    character_id: str
    position: Optional[Dict[str, float]] = None  # x, y, z
    emotion: Optional[str] = None
    pose: Optional[str] = None
    facing_direction: Optional[str] = None
    visible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterState":
        clean = _strip_to_schema(data or {}, _CHARACTER_STATE_FIELDS)
        if not clean.get("character_id") and data:
            clean["character_id"] = data.get("character_id", "unknown")
        return cls(**clean)


@dataclass
class EnvironmentState:
    """
    Observable environment state extracted from video.
    Schema v1.0: location_id, time_of_day, weather, lighting, objects.
    Observation model may have location_description, mood, objects_detected - these are stripped.
    """
    location_id: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    lighting: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentState":
        if not data or not isinstance(data, dict):
            return cls(location_id="unknown")
        # Map observation model fields to schema (objects_detected -> objects)
        normalized = dict(data)
        if "objects_detected" in normalized and "objects" not in normalized:
            normalized["objects"] = normalized.pop("objects_detected", []) or []
        clean = _strip_to_schema(normalized, _ENVIRONMENT_STATE_FIELDS)
        clean.setdefault("location_id", data.get("location_id") or "unknown")
        return cls(**clean)


@dataclass
class WorldState:
    """
    Complete world state at a point in time.
    This is derived from video observation, not static configuration.
    """
    characters: Dict[str, CharacterState] = field(default_factory=dict)
    environment: Optional[EnvironmentState] = None
    relationships: Dict[str, Dict[str, str]] = field(default_factory=dict)
    narrative_flags: Dict[str, Any] = field(default_factory=dict)
    custom_state: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "environment": self.environment.to_dict() if self.environment else None,
            "relationships": self.relationships,
            "narrative_flags": self.narrative_flags,
            "custom_state": self.custom_state,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldState":
        characters = {}
        for k, v in data.get("characters", {}).items():
            characters[k] = CharacterState.from_dict(v)
        
        env_data = data.get("environment")
        environment = EnvironmentState.from_dict(env_data) if env_data else None
        
        return cls(
            characters=characters,
            environment=environment,
            relationships=data.get("relationships", {}),
            narrative_flags=data.get("narrative_flags", {}),
            custom_state=data.get("custom_state", {}),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "WorldState":
        return cls.from_dict(json.loads(json_str))
    
    def merge(self, observation: Dict[str, Any]) -> "WorldState":
        """
        Create new world state by merging observation into current state.
        Used for state transitions after video observation.
        Returns a new WorldState, leaving original unchanged.
        """
        # Deep copy characters to avoid mutating original
        new_characters = {}
        for char_id, char_state in self.characters.items():
            new_characters[char_id] = CharacterState.from_dict(char_state.to_dict())
        
        new_state = WorldState(
            characters=new_characters,
            environment=EnvironmentState.from_dict(self.environment.to_dict()) if self.environment else None,
            relationships={k: dict(v) for k, v in self.relationships.items()},
            narrative_flags=dict(self.narrative_flags),
            custom_state=dict(self.custom_state),
        )
        
        # Update characters from observation (schema-stripped)
        for char_id, char_data in observation.get("characters", {}).items():
            if char_id in new_state.characters:
                existing = new_state.characters[char_id]
                clean = _strip_to_schema(char_data, _CHARACTER_STATE_FIELDS)
                for key, value in clean.items():
                    if value is not None:
                        setattr(existing, key, value)
            else:
                merged = dict(char_data) if isinstance(char_data, dict) else {}
                merged["character_id"] = merged.get("character_id") or char_id
                new_state.characters[char_id] = CharacterState.from_dict(merged)
        
        # Update environment (schema-stripped, no location_description etc.)
        if "environment" in observation:
            env_update = observation["environment"]
            if env_update and isinstance(env_update, dict):
                normalized = dict(env_update)
                if "objects_detected" in normalized and "objects" not in normalized:
                    normalized["objects"] = normalized.get("objects_detected") or []
                if new_state.environment:
                    clean = _strip_to_schema(normalized, _ENVIRONMENT_STATE_FIELDS)
                    for key, value in clean.items():
                        if value is not None and hasattr(new_state.environment, key):
                            setattr(new_state.environment, key, value)
                else:
                    new_state.environment = EnvironmentState.from_dict(normalized)
        
        # Update narrative flags
        new_state.narrative_flags.update(observation.get("narrative_flags", {}))
        
        # Update custom state
        new_state.custom_state.update(observation.get("custom_state", {}))
        
        return new_state


@dataclass
class WorldStateNode:
    """
    A node in the versioned world state graph.
    Each node represents world state after a video transition.
    """
    node_id: str
    episode_id: str
    parent_id: Optional[str]  # None for root node
    created_at: datetime
    world_state: WorldState
    
    # Transition that led to this state
    transition_video_uri: Optional[str] = None
    transition_beat_id: Optional[str] = None
    observation_json: Optional[str] = None  # Raw observation from observer
    
    # Branching metadata
    branch_name: str = "main"
    branch_type: BranchType = BranchType.MAIN
    depth: int = 0  # Distance from root
    
    # Quality metrics at this node
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "episode_id": self.episode_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "world_state": self.world_state.to_dict(),
            "transition_video_uri": self.transition_video_uri,
            "transition_beat_id": self.transition_beat_id,
            "observation_json": self.observation_json,
            "branch_name": self.branch_name,
            "branch_type": self.branch_type.value,
            "depth": self.depth,
            "quality_score": self.quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldStateNode":
        return cls(
            node_id=data["node_id"],
            episode_id=data["episode_id"],
            parent_id=data.get("parent_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            world_state=WorldState.from_dict(data["world_state"]),
            transition_video_uri=data.get("transition_video_uri"),
            transition_beat_id=data.get("transition_beat_id"),
            observation_json=data.get("observation_json"),
            branch_name=data.get("branch_name", "main"),
            branch_type=BranchType(data.get("branch_type", "main")),
            depth=data.get("depth", 0),
            quality_score=data.get("quality_score"),
        )


class WorldStateGraph:
    """
    Versioned, forkable state graph for video-native world simulation.
    
    Key operations:
    - transition(): Create new state from video observation
    - fork(): Create alternate timeline
    - replay(): Get full history to a node
    - counterfactual(): Explore "what if" scenarios
    
    This is the core data structure that makes video a computational primitive:
    - Each video = state transition
    - Each decision = branch
    - History enables replay, debugging, training
    """
    
    def __init__(self, episode_id: str, store=None):
        """
        Initialize world state graph.
        
        Args:
            episode_id: ID of the episode this graph belongs to
            store: Optional persistence store (WorldGraphStore)
        """
        self.episode_id = episode_id
        self.store = store
        
        # In-memory cache
        self._nodes: Dict[str, WorldStateNode] = {}
        self._root_id: Optional[str] = None
        self._current_id: Optional[str] = None
        self._branches: Dict[str, str] = {"main": None}  # branch_name -> head_id
    
    @property
    def root(self) -> Optional[WorldStateNode]:
        """Get root node of the graph."""
        if self._root_id:
            return self._nodes.get(self._root_id)
        return None
    
    @property
    def current(self) -> Optional[WorldStateNode]:
        """Get current node (latest on active branch)."""
        if self._current_id:
            return self._nodes.get(self._current_id)
        return None
    
    def initialize(self, initial_state: Optional[WorldState] = None) -> WorldStateNode:
        """
        Initialize graph with root node.
        
        Args:
            initial_state: Initial world state (empty if not provided)
            
        Returns:
            Root node
        """
        if self._root_id:
            raise ValueError("Graph already initialized")
        
        root = WorldStateNode(
            node_id=str(uuid.uuid4()),
            episode_id=self.episode_id,
            parent_id=None,
            created_at=datetime.utcnow(),
            world_state=initial_state or WorldState(),
            branch_name="main",
            branch_type=BranchType.MAIN,
            depth=0,
        )
        
        self._nodes[root.node_id] = root
        self._root_id = root.node_id
        self._current_id = root.node_id
        self._branches["main"] = root.node_id
        
        if self.store:
            self.store.insert_node(root)
        
        logger.info(f"[world_graph] initialized root node: {root.node_id}")
        return root
    
    def transition(
        self,
        video_uri: str,
        observation: Dict[str, Any],
        beat_id: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> WorldStateNode:
        """
        Create new state node from video observation.
        This is the core operation: video → state transition.
        
        Args:
            video_uri: URI of the video that caused this transition
            observation: Structured observation from VideoObserverAgent
            beat_id: Optional beat ID that generated this video
            quality_score: Optional quality score from evaluator
            
        Returns:
            New state node
            
        Raises:
            ProvenanceError: If observation lacks observation_id or video_uri is missing
        """
        if not self._current_id:
            raise ValueError("Graph not initialized. Call initialize() first.")
        
        # Provenance Check (Hard Gate)
        obs_id = observation.get("observation_id")
        if not obs_id:
            raise ProvenanceError("Strict Provenance Violation: State transition missing 'observation_id'")
            
        if not video_uri:
            raise ProvenanceError("Strict Provenance Violation: State transition missing 'video_uri'")
        
        parent = self._nodes[self._current_id]
        
        # Merge observation into current state
        new_state = parent.world_state.merge(observation)
        
        # Create new node
        new_node = WorldStateNode(
            node_id=str(uuid.uuid4()),
            episode_id=self.episode_id,
            parent_id=parent.node_id,
            created_at=datetime.utcnow(),
            world_state=new_state,
            transition_video_uri=video_uri,
            transition_beat_id=beat_id,
            observation_json=json.dumps(observation),
            branch_name=parent.branch_name,
            branch_type=parent.branch_type,
            depth=parent.depth + 1,
            quality_score=quality_score,
        )
        
        self._nodes[new_node.node_id] = new_node
        self._current_id = new_node.node_id
        self._branches[new_node.branch_name] = new_node.node_id
        
        if self.store:
            self.store.insert_node(new_node)
        
        logger.info(
            f"[world_graph] transition: {parent.node_id[:8]} → {new_node.node_id[:8]} "
            f"(video: {video_uri[:50] if video_uri else 'None'}...)"
        )
        return new_node
    
    def fork(
        self,
        from_node_id: Optional[str] = None,
        branch_name: Optional[str] = None,
        branch_type: BranchType = BranchType.FORK,
    ) -> WorldStateNode:
        """
        Create alternate timeline branch from existing node.
        
        Args:
            from_node_id: Node to branch from (current if not specified)
            branch_name: Name for new branch (auto-generated if not specified)
            branch_type: Type of branch (FORK, RETRY, COUNTERFACTUAL)
            
        Returns:
            New node at head of forked branch
        """
        source_id = from_node_id or self._current_id
        if not source_id:
            raise ValueError("No source node for fork")
        
        source = self._nodes[source_id]
        
        # Generate branch name if not provided
        if not branch_name:
            branch_name = f"{branch_type.value}_{len(self._branches)}"
        
        if branch_name in self._branches:
            raise ValueError(f"Branch '{branch_name}' already exists")
        
        # Create fork node (copy of source with new branch)
        fork_node = WorldStateNode(
            node_id=str(uuid.uuid4()),
            episode_id=self.episode_id,
            parent_id=source.node_id,
            created_at=datetime.utcnow(),
            world_state=WorldState.from_dict(source.world_state.to_dict()),  # Deep copy
            branch_name=branch_name,
            branch_type=branch_type,
            depth=source.depth + 1,
        )
        
        self._nodes[fork_node.node_id] = fork_node
        self._branches[branch_name] = fork_node.node_id
        self._current_id = fork_node.node_id
        
        if self.store:
            self.store.insert_node(fork_node)
        
        logger.info(
            f"[world_graph] fork: {source.node_id[:8]} → {fork_node.node_id[:8]} "
            f"(branch: {branch_name})"
        )
        return fork_node
    
    def switch_branch(self, branch_name: str) -> WorldStateNode:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Name of branch to switch to
            
        Returns:
            Head node of the branch
        """
        if branch_name not in self._branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        head_id = self._branches[branch_name]
        self._current_id = head_id
        
        logger.info(f"[world_graph] switched to branch: {branch_name}")
        return self._nodes[head_id]
    
    def replay(self, to_node_id: Optional[str] = None) -> List[WorldStateNode]:
        """
        Get full history from root to specified node.
        
        Args:
            to_node_id: Target node (current if not specified)
            
        Returns:
            List of nodes from root to target
        """
        target_id = to_node_id or self._current_id
        if not target_id:
            return []
        
        path = []
        current = self._nodes.get(target_id)
        
        while current:
            path.append(current)
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                break
        
        path.reverse()
        return path
    
    def counterfactual(
        self,
        from_node_id: str,
        alt_observation: Dict[str, Any],
    ) -> WorldStateNode:
        """
        Create counterfactual: "What if observation X happened instead?"
        
        Args:
            from_node_id: Node to branch from
            alt_observation: Alternative observation to apply
            
        Returns:
            New node in counterfactual branch
        """
        # First fork
        cf_node = self.fork(
            from_node_id=from_node_id,
            branch_type=BranchType.COUNTERFACTUAL,
        )
        
        # Then apply alternative observation
        parent = self._nodes[from_node_id]
        cf_state = parent.world_state.merge(alt_observation)
        cf_node.world_state = cf_state
        cf_node.observation_json = json.dumps(alt_observation)
        
        if self.store:
            self.store.update_node(cf_node)
        
        logger.info(
            f"[world_graph] counterfactual: {from_node_id[:8]} → {cf_node.node_id[:8]}"
        )
        return cf_node
    
    def get_node(self, node_id: str) -> Optional[WorldStateNode]:
        """Get node by ID."""
        # Try cache first
        if node_id in self._nodes:
            return self._nodes[node_id]
        
        # Try store
        if self.store:
            node = self.store.get_node(node_id)
            if node:
                self._nodes[node_id] = node
                return node
        
        return None
    
    def get_children(self, node_id: str) -> List[WorldStateNode]:
        """Get all child nodes of a node."""
        children = []
        for node in self._nodes.values():
            if node.parent_id == node_id:
                children.append(node)
        
        # Also check store
        if self.store:
            stored_children = self.store.get_children(node_id)
            for child in stored_children:
                if child.node_id not in self._nodes:
                    self._nodes[child.node_id] = child
                    children.append(child)
        
        return children
    
    def list_branches(self) -> Dict[str, str]:
        """List all branches and their head node IDs."""
        return dict(self._branches)
    
    def get_branch_head(self, branch_name: str) -> Optional[WorldStateNode]:
        """Get head node of a branch."""
        head_id = self._branches.get(branch_name)
        if head_id:
            return self._nodes.get(head_id)
        return None
    
    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire graph to dictionary."""
        return {
            "episode_id": self.episode_id,
            "root_id": self._root_id,
            "current_id": self._current_id,
            "branches": self._branches,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], store=None) -> "WorldStateGraph":
        """Deserialize graph from dictionary."""
        graph = cls(episode_id=data["episode_id"], store=store)
        graph._root_id = data.get("root_id")
        graph._current_id = data.get("current_id")
        graph._branches = data.get("branches", {"main": None})
        
        for node_id, node_data in data.get("nodes", {}).items():
            graph._nodes[node_id] = WorldStateNode.from_dict(node_data)
        
        return graph


# Convenience function for common use case
def create_episode_graph(
    episode_id: str,
    initial_characters: Optional[Dict[str, CharacterState]] = None,
    initial_location: Optional[str] = None,
    store=None,
) -> WorldStateGraph:
    """
    Create and initialize a new episode world graph.
    
    Args:
        episode_id: Episode ID
        initial_characters: Optional initial character states
        initial_location: Optional initial location ID
        store: Optional persistence store
        
    Returns:
        Initialized WorldStateGraph
    """
    graph = WorldStateGraph(episode_id=episode_id, store=store)
    
    initial_state = WorldState(
        characters=initial_characters or {},
        environment=EnvironmentState(location_id=initial_location or "unknown") if initial_location else None,
    )
    
    graph.initialize(initial_state)
    return graph
