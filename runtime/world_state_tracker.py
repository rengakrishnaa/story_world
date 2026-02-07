"""
World State Tracker

Tracks entities, positions, and velocities over time across beats.
Feeds from ObservationResult and WorldState transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class EntitySnapshot:
    """Single point-in-time state for an entity."""
    entity_id: str
    beat_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    position: Optional[Dict[str, float]] = None  # x, y or x, y, z
    velocity: Optional[Dict[str, float]] = None  # dx, dy or dx, dy, dz
    pose: Optional[str] = None
    visible: bool = True
    source: str = "observer"  # observer, inferred, numeric_sim


@dataclass
class WorldStateTracker:
    """
    Tracks entities and their state over time for causality/consistency checks.
    """

    episode_id: str
    history: Dict[str, List[EntitySnapshot]] = field(default_factory=dict)  # entity_id -> [snapshots]
    beat_order: List[str] = field(default_factory=list)  # ordered beat_ids

    def ingest_observation(
        self,
        beat_id: str,
        observation: Any,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Ingest ObservationResult or world state update.
        Extracts character positions and appends to history.
        """
        ts = timestamp or datetime.utcnow()
        if beat_id not in self.beat_order:
            self.beat_order.append(beat_id)

        # From ObservationResult
        if hasattr(observation, "characters"):
            for char_id, char_obs in observation.characters.items():
                pos = None
                if hasattr(char_obs, "position") and char_obs.position:
                    pos = dict(char_obs.position)
                snap = EntitySnapshot(
                    entity_id=char_id,
                    beat_id=beat_id,
                    timestamp=ts,
                    position=pos,
                    pose=getattr(char_obs, "pose", None),
                    visible=getattr(char_obs, "visible", True),
                    source="observer",
                )
                self._append_snapshot(char_id, snap)

        # From dict (e.g. to_world_state_update)
        elif isinstance(observation, dict):
            for char_id, char_data in observation.get("characters", {}).items():
                pos = None
                if isinstance(char_data, dict) and char_data.get("position"):
                    pos = dict(char_data["position"])
                snap = EntitySnapshot(
                    entity_id=char_id,
                    beat_id=beat_id,
                    timestamp=ts,
                    position=pos,
                    pose=char_data.get("pose") if isinstance(char_data, dict) else None,
                    visible=char_data.get("visible", True) if isinstance(char_data, dict) else True,
                    source="observer",
                )
                self._append_snapshot(char_id, snap)

    def _append_snapshot(self, entity_id: str, snap: EntitySnapshot) -> None:
        if entity_id not in self.history:
            self.history[entity_id] = []
        self.history[entity_id].append(snap)

    def get_entity_history(
        self,
        entity_id: str,
        max_snapshots: Optional[int] = None,
    ) -> List[EntitySnapshot]:
        """Return ordered snapshots for entity, newest last."""
        snaps = self.history.get(entity_id, [])
        if max_snapshots:
            snaps = snaps[-max_snapshots:]
        return snaps

    def get_velocity_estimate(
        self,
        entity_id: str,
        dt: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Estimate velocity from last two position snapshots.
        dt: optional time delta in seconds; if None, uses beat spacing as proxy.
        """
        snaps = self.get_entity_history(entity_id, max_snapshots=2)
        if len(snaps) < 2 or snaps[-1].position is None or snaps[-2].position is None:
            return None
        p0 = snaps[-2].position
        p1 = snaps[-1].position
        if None in p0.values() or None in p1.values():
            return None

        if dt is None:
            if snaps[-1].timestamp and snaps[-2].timestamp:
                dt = (snaps[-1].timestamp - snaps[-2].timestamp).total_seconds()
            else:
                dt = 1.0  # assume 1 beat = 1 second proxy

        if dt <= 0:
            return None

        vel = {}
        for k in set(p0.keys()) & set(p1.keys()):
            vel[k] = (p1[k] - p0[k]) / dt
        return vel if vel else None

    def get_last_position(self, entity_id: str) -> Optional[Dict[str, float]]:
        snaps = self.get_entity_history(entity_id, max_snapshots=1)
        return snaps[-1].position if snaps else None
