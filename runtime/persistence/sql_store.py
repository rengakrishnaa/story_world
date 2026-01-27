import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Set
import sqlite3
import psycopg2
from dotenv import load_dotenv



class SQLStore:
    def __init__(self, lazy=False):
        self._conn = None
        self.lazy = lazy

        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL not set")

        if not lazy:
            self._connect()

   # -------------------------------------------------

    def _cursor(self):
        return self.conn.cursor()

    def _ph(self):
        return "?" if self.backend == "sqlite" else "%s"


    def _init_sqlite_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            world_id TEXT,
            intent TEXT,
            policies TEXT,
            state TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS beats (
            beat_id TEXT PRIMARY KEY,
            episode_id TEXT,
            spec TEXT,
            state TEXT,
            last_error TEXT,
            cost_spent REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY,
            episode_id TEXT,
            beat_id TEXT,
            model TEXT,
            prompt TEXT,
            success INTEGER,
            metrics TEXT,
            started_at TEXT,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            beat_id TEXT,
            type TEXT,
            uri TEXT,
            version INTEGER,
            attempt_id TEXT
        );
        """)
        self.conn.commit()
        cur.close()

    # -------------------------------------------------
    # Episodes
    # -------------------------------------------------

    def create_episode(self, episode_id, world_id, intent, policies, state):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO episodes
                (episode_id, world_id, intent, policies, state)
                VALUES ({self._ph()}, {self._ph()}, {self._ph()}, {self._ph()}, {self._ph()})
                """,
                (
                    episode_id,
                    world_id,
                    intent,
                    json.dumps(policies or {}),
                    state,
                ),
            )
            if self.backend == "sqlite":
                self.conn.commit()
        finally:
            cur.close()

    def get_episode(self, episode_id):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT episode_id, world_id, intent, policies, state
                FROM episodes
                WHERE episode_id = {self._ph()}
                """,
                (episode_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return {
                "episode_id": row[0],
                "world_id": row[1],
                "intent": row[2],
                "policies": json.loads(row[3]) if isinstance(row[3], str) else row[3],
                "state": row[4],
            }
        finally:
            cur.close()

    def update_episode_state(self, episode_id, state, ts=None):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                UPDATE episodes
                SET state = {self._ph()}, updated_at = {self._ph()}
                WHERE episode_id = {self._ph()}
                """,
                (
                    state,
                    ts or datetime.utcnow().isoformat(),
                    episode_id,
                ),
            )
            if self.backend == "sqlite":
                self.conn.commit()
        finally:
            cur.close()

    # -------------------------------------------------
    # Beats
    # -------------------------------------------------

    def create_beat(self, episode_id, beat_spec: Dict):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO beats
                (beat_id, episode_id, spec, state)
                VALUES ({self._ph()}, {self._ph()}, {self._ph()}, {self._ph()})
                """,
                (
                    beat_spec["id"],
                    episode_id,
                    json.dumps(beat_spec),
                    "PENDING",
                ),
            )
            if self.backend == "sqlite":
                self.conn.commit()
        finally:
            cur.close()

    def get_beat(self, beat_id):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT beat_id, episode_id, spec, state
                FROM beats
                WHERE beat_id = {self._ph()}
                """,
                (beat_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "beat_id": row[0],
                "episode_id": row[1],
                "spec": json.loads(row[2]),
                "state": row[3],
            }
        finally:
            cur.close()

    def get_beats(self, episode_id: str):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT beat_id, spec, state, last_error
                FROM beats
                WHERE episode_id = {self._ph()}
                ORDER BY beat_id
                """,
                (episode_id,),
            )

            rows = cur.fetchall()
            beats = []

            for r in rows:
                beats.append({
                    "beat_id": r[0],
                    **json.loads(r[1]),
                    "state": r[2],
                    "last_error": r[3],
                })

            return beats
        finally:
            cur.close()


    def count_attempts(self, beat_id: str) -> int:
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM attempts
                WHERE beat_id = {self._ph()}
                """,
                (beat_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            cur.close()


    def get_beats_by_state(self, episode_id, states: Set[str]):
        cur = self._cursor()
        try:
            placeholders = ",".join([self._ph()] * len(states))
            cur.execute(
                f"""
                SELECT beat_id, spec
                FROM beats
                WHERE episode_id = {self._ph()}
                AND state IN ({placeholders})
                """,
                (episode_id, *states),
            )
            return [{"beat_id": r[0], **json.loads(r[1])} for r in cur.fetchall()]
        finally:
            cur.close()

    def mark_beat_state(self, beat_id, state, error=None):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                UPDATE beats
                SET state = {self._ph()}, last_error = {self._ph()}
                WHERE beat_id = {self._ph()}
                """,
                (state, error, beat_id),
            )
            if self.backend == "sqlite":
                self.conn.commit()
        finally:
            cur.close()

    # -------------------------------------------------
    # Attempts
    # -------------------------------------------------

    def record_attempt(self, episode_id, beat_id, success, metrics):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO attempts
                (attempt_id, episode_id, beat_id, success, metrics, started_at, completed_at)
                VALUES ({self._ph()}, {self._ph()}, {self._ph()}, {self._ph()}, {self._ph()}, {self._ph()}, {self._ph()})
                """,
                (
                    str(uuid.uuid4()),
                    episode_id,
                    beat_id,
                    int(success),
                    json.dumps(metrics or {}),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            if self.backend == "sqlite":
                self.conn.commit()
        finally:
            cur.close()

    def get_attempts(self, episode_id: str):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT
                    attempt_id,
                    episode_id,
                    beat_id,
                    success,
                    metrics,
                    started_at,
                    completed_at
                FROM attempts
                WHERE episode_id = {self._ph()}
                ORDER BY started_at ASC
                """,
                (episode_id,),
            )

            rows = cur.fetchall()
            attempts = []

            for r in rows:
                attempts.append({
                    "attempt_id": r[0],
                    "episode_id": r[1],
                    "beat_id": r[2],
                    "success": bool(r[3]),
                    "metrics": json.loads(r[4]) if r[4] else {},
                    "started_at": r[5],
                    "completed_at": r[6],
                })

            return attempts
        finally:
            cur.close()


    def get_artifacts(self, episode_id: str):
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT
                    a.artifact_id,
                    a.beat_id,
                    a.type,
                    a.uri,
                    a.version,
                    a.attempt_id
                FROM artifacts a
                JOIN beats b ON a.beat_id = b.beat_id
                WHERE b.episode_id = {self._ph()}
                ORDER BY a.version ASC
                """,
                (episode_id,),
            )

            rows = cur.fetchall()
            artifacts = []

            for r in rows:
                artifacts.append({
                    "artifact_id": r[0],
                    "beat_id": r[1],
                    "type": r[2],
                    "uri": r[3],
                    "version": r[4],
                    "attempt_id": r[5],
                })

            return artifacts
        finally:
            cur.close()

    def record_artifact(
        self,
        beat_id: str,
        artifact_type: str,
        uri: str,
        attempt_id: str = None
    ):
        """
        Record an artifact (video, keyframe, etc.) for a beat.
        
        Args:
            beat_id: ID of the beat this artifact belongs to
            artifact_type: Type of artifact (e.g., 'video', 'keyframe', 'thumbnail')
            uri: URI/URL where artifact is stored
            attempt_id: Optional attempt ID that generated this artifact
        """
        cur = self._cursor()
        try:
            # Get current version for this beat + artifact type
            cur.execute(
                f"""
                SELECT COALESCE(MAX(version), 0)
                FROM artifacts
                WHERE beat_id = {self._ph()} AND type = {self._ph()}
                """,
                (beat_id, artifact_type)
            )
            row = cur.fetchone()
            version = (row[0] if row else 0) + 1
            
            # Insert new artifact
            cur.execute(
                f"""
                INSERT INTO artifacts
                (artifact_id, beat_id, type, uri, version, attempt_id)
                VALUES ({self._ph()}, {self._ph()}, {self._ph()}, {self._ph()}, {self._ph()}, {self._ph()})
                """,
                (
                    str(uuid.uuid4()),
                    beat_id,
                    artifact_type,
                    uri,
                    version,
                    attempt_id
                )
            )
            
            if self.backend == "sqlite":
                self.conn.commit()
                
        finally:
            cur.close()

    def all_beats_completed(self, episode_id: str) -> bool:
        """
        Check if all beats for an episode are in ACCEPTED state.
        
        Args:
            episode_id: Episode ID to check
            
        Returns:
            True if all beats are completed, False otherwise
        """
        cur = self._cursor()
        try:
            # Count total beats
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM beats
                WHERE episode_id = {self._ph()}
                """,
                (episode_id,)
            )
            row = cur.fetchone()
            total_beats = row[0] if row else 0
            
            if total_beats == 0:
                return False
            
            # Count accepted beats
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM beats
                WHERE episode_id = {self._ph()} AND state = 'ACCEPTED'
                """,
                (episode_id,)
            )
            row = cur.fetchone()
            accepted_beats = row[0] if row else 0
            
            return accepted_beats == total_beats
            
        finally:
            cur.close()

    def any_beats_failed(self, episode_id: str) -> bool:
        """
        Check if any beats for an episode are in ABORTED state.
        
        Args:
            episode_id: Episode ID to check
            
        Returns:
            True if any beats failed, False otherwise
        """
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM beats
                WHERE episode_id = {self._ph()} AND state = 'ABORTED'
                """,
                (episode_id,)
            )
            row = cur.fetchone()
            failed_count = row[0] if row else 0
            
            return failed_count > 0
            
        finally:
            cur.close()

    def _connect(self):
        if self._conn is not None:
            return

        if self.database_url.startswith("sqlite"):
            path = self.database_url.replace("sqlite:///", "")
            self._conn = sqlite3.connect(
                path,
                check_same_thread=False
            )
            self.backend = "sqlite"
            print(f"[sql] using sqlite ({path})")

            self._init_sqlite_schema()

        elif self.database_url.startswith("postgres"):
            self._conn = psycopg2.connect(self.database_url)
            self.backend = "postgres"
            print("[sql] using postgres")

        else:
            raise RuntimeError(
                f"Unsupported DATABASE_URL: {self.database_url}"
            )

    @property
    def conn(self):
        if self._conn is None:
            self._connect()
        return self._conn


