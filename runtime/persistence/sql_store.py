import json
import psycopg2
from datetime import datetime
from typing import Dict, List
from runtime.episode_state import EpisodeState
from runtime.beat_state import BeatState
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


class SQLStore:
    def __init__(self):
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set")

        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True

    # ---------- EPISODES ----------

    def create_episode(self, episode_id, world_id, intent, policies, state):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodes
                (episode_id, world_id, intent, policies, state)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (episode_id, world_id, intent, json.dumps(policies), state),
            )

    def update_episode_state(self, episode_id, state, ts=None):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE episodes
                SET state = %s, updated_at = %s
                WHERE episode_id = %s
                """,
                (state, ts or datetime.utcnow(), episode_id),
            )

    def get_attempts(self, beat_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT attempt_id, success
                FROM attempts
                WHERE beat_id = %s
                """,
                (beat_id,),
            )
            return cur.fetchall()

    # ---------- BEATS ----------

    def create_beat(self, episode_id, beat_spec: Dict):
        beat_id = beat_spec["id"]

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO beats
                (beat_id, episode_id, spec, state)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    beat_id,
                    episode_id,
                    json.dumps(beat_spec),
                    BeatState.PENDING,
                ),
            )

    def get_pending_beats(self, episode_id) -> List[Dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT beat_id, spec
                FROM beats
                WHERE episode_id = %s AND state = %s
                """,
                (episode_id, BeatState.PENDING),
            )
            return [{"id": r[0], **r[1]} for r in cur.fetchall()]

    def mark_beat_state(self, beat_id, state, error=None):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE beats
                SET state = %s, last_error = %s
                WHERE beat_id = %s
                """,
                (state, error, beat_id),
            )

    def all_beats_completed(self, episode_id) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM beats
                WHERE episode_id = %s
                AND state NOT IN (%s, %s)
                """,
                (episode_id, BeatState.ACCEPTED, BeatState.ABORTED),
            )
            return cur.fetchone()[0] == 0

    def any_beats_failed(self, episode_id) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM beats
                WHERE episode_id = %s
                AND state = %s
                """,
                (episode_id, BeatState.ABORTED),
            )
            return cur.fetchone()[0] > 0

    # ---------- ATTEMPTS ----------

    def record_attempt(self, episode_id, beat_id, model, prompt, success, metrics):
        attempt_id = str(uuid.uuid4())

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attempts
                (attempt_id,episode_id, beat_id, model, prompt, success, metrics, started_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    attempt_id,
                    episode_id,
                    beat_id,
                    model,
                    prompt,
                    success,
                    json.dumps(metrics),
                    datetime.utcnow(),
                    datetime.utcnow(),
                ),
            )
        return attempt_id

    # ---------- ARTIFACTS ----------

    def record_artifact(self, beat_id, attempt_id, type_, uri, version=1):
        artifact_id = str(uuid.uuid4())

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts
                (artifact_id, beat_id, type, uri, version, attempt_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (artifact_id, beat_id, type_, uri, version, attempt_id),
            )

    def get_episode(self, episode_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT episode_id, world_id, intent, policies, state
                FROM episodes
                WHERE episode_id = %s
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
                "policies": row[3],
                "state": row[4],
            }


    def get_beats(self, episode_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT beat_id, state, cost_spent, last_error
                FROM beats
                WHERE episode_id = %s
                """,
                (episode_id,),
            )
            return [
                {
                    "beat_id": r[0],
                    "state": r[1],
                    "cost_spent": r[2],
                    "last_error": r[3],
                }
                for r in cur.fetchall()
            ]

    def get_artifacts(self, episode_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.artifact_id, a.beat_id, a.type, a.uri, a.version
                FROM artifacts a
                JOIN beats b ON a.beat_id = b.beat_id
                WHERE b.episode_id = %s
                """,
                (episode_id,),
            )
            return [
                {
                    "artifact_id": r[0],
                    "beat_id": r[1],
                    "type": r[2],
                    "uri": r[3],
                    "version": r[4],
                }
                for r in cur.fetchall()
            ]