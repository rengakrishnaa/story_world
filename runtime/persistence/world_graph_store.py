"""
World Graph Store

SQLite persistence layer for the versioned world state graph.
Supports branching, forking, ancestry queries, and efficient replay.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

import uuid

from models.world_state_graph import (
    WorldStateNode,
    WorldState,
    BranchType,
)
from models.state_transition import (
    StateTransition,
    TransitionStatus,
    ActionOutcome,
    TransitionMetrics,
)

logger = logging.getLogger(__name__)


class WorldGraphStore:
    """
    SQLite persistence for world state graph.
    
    Handles:
    - Node storage and retrieval
    - Transition logging
    - Branch management
    - Ancestry queries for replay
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize store.
        
        Args:
            database_url: SQLite URL (default from env DATABASE_URL)
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL not set")
        
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._init_schema()
    
    def _connect(self) -> None:
        """Establish database connection."""
        if self._conn is not None:
            return
        
        path = self.database_url.replace("sqlite:///", "")
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        logger.info(f"[world_graph_store] connected to {path}")
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        return self._conn
    
    def _init_schema(self) -> None:
        """Initialize database schema for world graph."""
        cur = self.conn.cursor()
        try:
            cur.executescript("""
                -- Versioned world state nodes
                CREATE TABLE IF NOT EXISTS world_state_nodes (
                    node_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    parent_id TEXT,
                    created_at TEXT NOT NULL,
                    world_state TEXT NOT NULL,
                    transition_video_uri TEXT,
                    transition_beat_id TEXT,
                    observation_json TEXT,
                    branch_name TEXT DEFAULT 'main',
                    branch_type TEXT DEFAULT 'main',
                    depth INTEGER DEFAULT 0,
                    quality_score REAL,
                    FOREIGN KEY (parent_id) REFERENCES world_state_nodes(node_id)
                );

                -- State transitions (video as computation)
                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT,
                    beat_id TEXT,
                    action_description TEXT,
                    video_uri TEXT,
                    video_duration_sec REAL DEFAULT 0,
                    observation_json TEXT,
                    action_outcome TEXT DEFAULT 'unknown',
                    continuity_errors TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    metrics TEXT,
                    FOREIGN KEY (source_node_id) REFERENCES world_state_nodes(node_id),
                    FOREIGN KEY (target_node_id) REFERENCES world_state_nodes(node_id)
                );

                -- Branch metadata
                CREATE TABLE IF NOT EXISTS world_branches (
                    branch_name TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    head_node_id TEXT,
                    created_at TEXT NOT NULL,
                    description TEXT,
                    PRIMARY KEY (branch_name, episode_id),
                    FOREIGN KEY (head_node_id) REFERENCES world_state_nodes(node_id)
                );

                -- Indexes for efficient queries
                CREATE INDEX IF NOT EXISTS idx_wsn_episode ON world_state_nodes(episode_id);
                CREATE INDEX IF NOT EXISTS idx_wsn_parent ON world_state_nodes(parent_id);
                CREATE INDEX IF NOT EXISTS idx_wsn_branch ON world_state_nodes(branch_name);
                CREATE INDEX IF NOT EXISTS idx_st_episode ON state_transitions(episode_id);
                CREATE INDEX IF NOT EXISTS idx_st_source ON state_transitions(source_node_id);
                CREATE INDEX IF NOT EXISTS idx_st_status ON state_transitions(status);

                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS world_graph_schema (
                    version INTEGER PRIMARY KEY
                );

                INSERT OR IGNORE INTO world_graph_schema (version) VALUES (1);

                -- Constraint memory (reuse discovered constraints as priors)
                CREATE TABLE IF NOT EXISTS constraint_memory (
                    constraint_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT DEFAULT 'generic',
                    intent_pattern TEXT,
                    episode_id TEXT,
                    human_confirmed INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cm_domain ON constraint_memory(domain);
            """)
            self.conn.commit()
            logger.info("[world_graph_store] schema initialized")
        finally:
            cur.close()
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def insert_node(self, node: WorldStateNode) -> None:
        """Insert a new node into the graph."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO world_state_nodes
                (node_id, episode_id, parent_id, created_at, world_state,
                 transition_video_uri, transition_beat_id, observation_json,
                 branch_name, branch_type, depth, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.node_id,
                    node.episode_id,
                    node.parent_id,
                    node.created_at.isoformat(),
                    node.world_state.to_json(),
                    node.transition_video_uri,
                    node.transition_beat_id,
                    node.observation_json,
                    node.branch_name,
                    node.branch_type.value,
                    node.depth,
                    node.quality_score,
                ),
            )
            self.conn.commit()
            
            # Update branch head
            self._update_branch_head(node.episode_id, node.branch_name, node.node_id)
            
        finally:
            cur.close()
    
    def update_node(self, node: WorldStateNode) -> None:
        """Update an existing node."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                UPDATE world_state_nodes
                SET world_state = ?,
                    observation_json = ?,
                    quality_score = ?
                WHERE node_id = ?
                """,
                (
                    node.world_state.to_json(),
                    node.observation_json,
                    node.quality_score,
                    node.node_id,
                ),
            )
            self.conn.commit()
        finally:
            cur.close()
    
    def get_node(self, node_id: str) -> Optional[WorldStateNode]:
        """Get a node by ID."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT node_id, episode_id, parent_id, created_at, world_state,
                       transition_video_uri, transition_beat_id, observation_json,
                       branch_name, branch_type, depth, quality_score
                FROM world_state_nodes
                WHERE node_id = ?
                """,
                (node_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            
            return self._row_to_node(row)
        finally:
            cur.close()
    
    def get_children(self, node_id: str) -> List[WorldStateNode]:
        """Get all child nodes of a node."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT node_id, episode_id, parent_id, created_at, world_state,
                       transition_video_uri, transition_beat_id, observation_json,
                       branch_name, branch_type, depth, quality_score
                FROM world_state_nodes
                WHERE parent_id = ?
                ORDER BY created_at ASC
                """,
                (node_id,),
            )
            return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            cur.close()
    
    def get_ancestry(self, node_id: str) -> List[WorldStateNode]:
        """
        Get full ancestry from root to node.
        Used for replay functionality.
        """
        path = []
        current_id = node_id
        
        while current_id:
            node = self.get_node(current_id)
            if not node:
                break
            path.append(node)
            current_id = node.parent_id
        
        path.reverse()
        return path
    
    def get_episode_nodes(self, episode_id: str) -> List[WorldStateNode]:
        """Get all nodes for an episode."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT node_id, episode_id, parent_id, created_at, world_state,
                       transition_video_uri, transition_beat_id, observation_json,
                       branch_name, branch_type, depth, quality_score
                FROM world_state_nodes
                WHERE episode_id = ?
                ORDER BY depth ASC, created_at ASC
                """,
                (episode_id,),
            )
            return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            cur.close()
    
    def get_root_node(self, episode_id: str) -> Optional[WorldStateNode]:
        """Get root node for an episode."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT node_id, episode_id, parent_id, created_at, world_state,
                       transition_video_uri, transition_beat_id, observation_json,
                       branch_name, branch_type, depth, quality_score
                FROM world_state_nodes
                WHERE episode_id = ? AND parent_id IS NULL
                LIMIT 1
                """,
                (episode_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_node(row)
        finally:
            cur.close()
    
    def _row_to_node(self, row: sqlite3.Row) -> WorldStateNode:
        """Convert database row to WorldStateNode."""
        return WorldStateNode(
            node_id=row["node_id"],
            episode_id=row["episode_id"],
            parent_id=row["parent_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            world_state=WorldState.from_json(row["world_state"]),
            transition_video_uri=row["transition_video_uri"],
            transition_beat_id=row["transition_beat_id"],
            observation_json=row["observation_json"],
            branch_name=row["branch_name"] or "main",
            branch_type=BranchType(row["branch_type"] or "main"),
            depth=row["depth"] or 0,
            quality_score=row["quality_score"],
        )
    
    # =========================================================================
    # Transition Operations
    # =========================================================================
    
    def insert_transition(self, transition: StateTransition) -> None:
        """Insert a new transition."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO state_transitions
                (transition_id, episode_id, source_node_id, target_node_id,
                 beat_id, action_description, video_uri, video_duration_sec,
                 observation_json, action_outcome, continuity_errors, status,
                 error_message, created_at, completed_at, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition.transition_id,
                    transition.episode_id,
                    transition.source_node_id,
                    transition.target_node_id,
                    transition.beat_id,
                    transition.action_description,
                    transition.video_uri,
                    transition.video_duration_sec,
                    transition.observation_json,
                    transition.action_outcome.value,
                    json.dumps([e.to_dict() for e in transition.continuity_errors]),
                    transition.status.value,
                    transition.error_message,
                    transition.created_at.isoformat(),
                    transition.completed_at.isoformat() if transition.completed_at else None,
                    json.dumps(transition.metrics.to_dict()),
                ),
            )
            self.conn.commit()
        finally:
            cur.close()
    
    def update_transition(self, transition: StateTransition) -> None:
        """Update an existing transition."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                UPDATE state_transitions
                SET target_node_id = ?,
                    video_uri = ?,
                    video_duration_sec = ?,
                    observation_json = ?,
                    action_outcome = ?,
                    continuity_errors = ?,
                    status = ?,
                    error_message = ?,
                    completed_at = ?,
                    metrics = ?
                WHERE transition_id = ?
                """,
                (
                    transition.target_node_id,
                    transition.video_uri,
                    transition.video_duration_sec,
                    transition.observation_json,
                    transition.action_outcome.value,
                    json.dumps([e.to_dict() for e in transition.continuity_errors]),
                    transition.status.value,
                    transition.error_message,
                    transition.completed_at.isoformat() if transition.completed_at else None,
                    json.dumps(transition.metrics.to_dict()),
                    transition.transition_id,
                ),
            )
            self.conn.commit()
        finally:
            cur.close()
    
    def get_transition(self, transition_id: str) -> Optional[StateTransition]:
        """Get a transition by ID."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT * FROM state_transitions WHERE transition_id = ?
                """,
                (transition_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_transition(dict(row))
        finally:
            cur.close()

    def record_beat_observation(
        self,
        episode_id: str,
        beat_id: str,
        video_uri: str,
        observation: Dict[str, Any],
        action_description: str = "",
        video_duration_sec: float = 0.0,
        quality_score: float = 0.0,
        transition_status: TransitionStatus = TransitionStatus.COMPLETED,
        action_outcome: Optional[ActionOutcome] = None,
    ) -> None:
        """
        Record an observation from a beat's video into the world graph.
        Creates root node if needed, appends new state node and transition.
        Used by ResultConsumer when observer runs on successful render.
        """
        obs_id = observation.get("observation_id") or str(uuid.uuid4())
        observation.setdefault("observation_id", obs_id)

        # 1. Get or create root node
        root = self.get_root_node(episode_id)
        if not root:
            graph = WorldStateNode(
                node_id=str(uuid.uuid4()),
                episode_id=episode_id,
                parent_id=None,
                created_at=datetime.utcnow(),
                world_state=WorldState(),
                branch_name="main",
                branch_type=BranchType.MAIN,
                depth=0,
            )
            self.insert_node(graph)
            self._update_branch_head(episode_id, "main", graph.node_id)
            parent = graph
        else:
            parent = self.get_branch_head(episode_id, "main") or root

        new_node = None
        if transition_status == TransitionStatus.COMPLETED:
            # 2. Merge observation into parent state
            try:
                merged_state = parent.world_state.merge(observation)
            except Exception as e:
                logger.warning(f"[world_graph_store] merge failed, using minimal state: {e}")
                merged_state = WorldState()

            # 3. Create and insert new node
            new_node = WorldStateNode(
                node_id=str(uuid.uuid4()),
                episode_id=episode_id,
                parent_id=parent.node_id,
                created_at=datetime.utcnow(),
                world_state=merged_state,
                transition_video_uri=video_uri,
                transition_beat_id=beat_id,
                observation_json=json.dumps(observation),
                branch_name="main",
                branch_type=BranchType.MAIN,
                depth=parent.depth + 1,
                quality_score=quality_score,
            )
            self.insert_node(new_node)
            self._update_branch_head(episode_id, "main", new_node.node_id)

        # 4. Create and insert transition
        action = observation.get("action") or {}
        if action_outcome is None:
            outcome_str = action.get("outcome", "success")
            try:
                action_outcome = ActionOutcome(outcome_str)
            except ValueError:
                action_outcome = ActionOutcome.SUCCESS

        trans = StateTransition(
            transition_id=str(uuid.uuid4()),
            episode_id=episode_id,
            source_node_id=parent.node_id,
            target_node_id=new_node.node_id if new_node else None,
            beat_id=beat_id,
            action_description=action_description or action.get("action_description", ""),
            video_uri=video_uri,
            video_duration_sec=video_duration_sec,
            observation_json=json.dumps(observation),
            action_outcome=action_outcome,
            status=transition_status,
            completed_at=datetime.utcnow(),
            metrics=TransitionMetrics(quality_score=quality_score),
        )
        self.insert_transition(trans)
        logger.info(f"[world_graph_store] recorded observation for {episode_id}/{beat_id}")

    def get_episode_transitions(
        self,
        episode_id: str,
        status: Optional[TransitionStatus] = None,
    ) -> List[StateTransition]:
        """Get transitions for an episode, optionally filtered by status."""
        cur = self.conn.cursor()
        try:
            if status:
                cur.execute(
                    """
                    SELECT * FROM state_transitions
                    WHERE episode_id = ? AND status = ?
                    ORDER BY created_at ASC
                    """,
                    (episode_id, status.value),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM state_transitions
                    WHERE episode_id = ?
                    ORDER BY created_at ASC
                    """,
                    (episode_id,),
                )
            return [self._row_to_transition(dict(row)) for row in cur.fetchall()]
        finally:
            cur.close()
    
    def _row_to_transition(self, row: Dict[str, Any]) -> StateTransition:
        """Convert database row to StateTransition."""
        from models.state_transition import (
            StateTransition,
            TransitionStatus,
            ActionOutcome,
            TransitionMetrics,
            ContinuityError,
        )
        
        continuity_errors = []
        if row.get("continuity_errors"):
            for e in json.loads(row["continuity_errors"]):
                continuity_errors.append(ContinuityError.from_dict(e))
        
        metrics = TransitionMetrics()
        if row.get("metrics"):
            metrics = TransitionMetrics.from_dict(json.loads(row["metrics"]))
        
        return StateTransition(
            transition_id=row["transition_id"],
            episode_id=row["episode_id"],
            source_node_id=row["source_node_id"],
            target_node_id=row.get("target_node_id"),
            beat_id=row.get("beat_id", ""),
            action_description=row.get("action_description", ""),
            video_uri=row.get("video_uri"),
            video_duration_sec=row.get("video_duration_sec", 0.0),
            observation_json=row.get("observation_json"),
            action_outcome=ActionOutcome(row.get("action_outcome", "unknown")),
            continuity_errors=continuity_errors,
            status=TransitionStatus(row.get("status", "pending")),
            error_message=row.get("error_message"),
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row.get("completed_at") else None
            ),
            metrics=metrics,
        )
    
    # =========================================================================
    # Branch Operations
    # =========================================================================
    
    def _update_branch_head(
        self,
        episode_id: str,
        branch_name: str,
        head_node_id: str,
    ) -> None:
        """Update or create branch head pointer."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO world_branches (branch_name, episode_id, head_node_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(branch_name, episode_id) DO UPDATE SET head_node_id = ?
                """,
                (
                    branch_name,
                    episode_id,
                    head_node_id,
                    datetime.utcnow().isoformat(),
                    head_node_id,
                ),
            )
            self.conn.commit()
        finally:
            cur.close()
    
    def get_branch_head(
        self,
        episode_id: str,
        branch_name: str,
    ) -> Optional[WorldStateNode]:
        """Get head node of a branch."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT head_node_id FROM world_branches
                WHERE episode_id = ? AND branch_name = ?
                """,
                (episode_id, branch_name),
            )
            row = cur.fetchone()
            if not row or not row["head_node_id"]:
                return None
            return self.get_node(row["head_node_id"])
        finally:
            cur.close()
    
    def list_branches(self, episode_id: str) -> Dict[str, str]:
        """List all branches for an episode."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT branch_name, head_node_id FROM world_branches
                WHERE episode_id = ?
                """,
                (episode_id,),
            )
            return {row["branch_name"]: row["head_node_id"] for row in cur.fetchall()}
        finally:
            cur.close()
    
    # =========================================================================
    # Analytics / Training Data
    # =========================================================================
    
    def get_training_pairs(
        self,
        episode_id: Optional[str] = None,
        min_quality: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Get (video, observation) pairs for training internalization model.
        
        Args:
            episode_id: Optional filter by episode
            min_quality: Minimum quality score to include
            
        Returns:
            List of dicts with video_uri, observation_json, quality_score
        """
        cur = self.conn.cursor()
        try:
            if episode_id:
                cur.execute(
                    """
                    SELECT transition_video_uri, observation_json, quality_score
                    FROM world_state_nodes
                    WHERE episode_id = ?
                      AND transition_video_uri IS NOT NULL
                      AND observation_json IS NOT NULL
                      AND quality_score >= ?
                    ORDER BY created_at ASC
                    """,
                    (episode_id, min_quality),
                )
            else:
                cur.execute(
                    """
                    SELECT transition_video_uri, observation_json, quality_score
                    FROM world_state_nodes
                    WHERE transition_video_uri IS NOT NULL
                      AND observation_json IS NOT NULL
                      AND quality_score >= ?
                    ORDER BY created_at ASC
                    """,
                    (min_quality,),
                )
            
            return [
                {
                    "video_uri": row["transition_video_uri"],
                    "observation_json": row["observation_json"],
                    "quality_score": row["quality_score"],
                }
                for row in cur.fetchall()
            ]
        finally:
            cur.close()
    
    def count_nodes(self, episode_id: Optional[str] = None) -> int:
        """Count total nodes, optionally filtered by episode."""
        cur = self.conn.cursor()
        try:
            if episode_id:
                cur.execute(
                    "SELECT COUNT(*) FROM world_state_nodes WHERE episode_id = ?",
                    (episode_id,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM world_state_nodes")
            return cur.fetchone()[0]
        finally:
            cur.close()
    
    def count_transitions(self, episode_id: Optional[str] = None) -> int:
        """Count total transitions, optionally filtered by episode."""
        cur = self.conn.cursor()
        try:
            if episode_id:
                cur.execute(
                    "SELECT COUNT(*) FROM state_transitions WHERE episode_id = ?",
                    (episode_id,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM state_transitions")
            return cur.fetchone()[0]
        finally:
            cur.close()
