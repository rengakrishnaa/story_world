"""
Constraint Reuse as Memory

Persists discovered constraints and injects them as priors for planner/observer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class StoredConstraint:
    constraint_id: str
    name: str
    domain: str
    intent_pattern: str
    human_confirmed: Optional[bool] = None
    created_at: str = ""
    episode_id: str = ""


def _get_store():
    """Lazy import to avoid circular deps."""
    try:
        from runtime.persistence.world_graph_store import WorldGraphStore
        return WorldGraphStore()
    except Exception as e:
        logger.debug(f"[constraint_memory] WorldGraphStore unavailable: {e}")
        return None


def record_constraints(
    episode_id: str,
    intent: str,
    constraints: List[str],
    domain: str = "generic",
) -> None:
    """Persist discovered constraints for reuse."""
    store = _get_store()
    if not store:
        return
    try:
        cur = store.conn.cursor()
        for c in constraints:
            if not c or not c.strip():
                continue
            name = c.strip().lower().replace(" ", "_")
            cid = f"{domain}:{name}"[:200]
            cur.execute(
                """
                INSERT OR REPLACE INTO constraint_memory
                (constraint_id, name, domain, intent_pattern, episode_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    name[:100],
                    domain[:50],
                    (intent or "")[:500],
                    episode_id,
                    datetime.utcnow().isoformat(),
                ),
            )
        store.conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"[constraint_memory] record failed: {e}")


def get_prior_constraints(
    intent: str,
    domain: str = "generic",
    limit: int = 20,
) -> List[str]:
    """Fetch prior constraints matching intent/domain for injection as priors."""
    store = _get_store()
    if not store:
        return []
    try:
        cur = store.conn.cursor()
        cur.execute(
            """
            SELECT name FROM constraint_memory
            WHERE (domain = ? OR domain = 'generic')
            AND (human_confirmed IS NULL OR human_confirmed = 1)
            ORDER BY created_at DESC LIMIT ?
            """,
            (domain, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows] if rows else []
    except Exception as e:
        logger.debug(f"[constraint_memory] get_prior failed: {e}")
        return []
