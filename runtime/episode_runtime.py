from datetime import datetime
import uuid

from runtime.episode_state import EpisodeState
from runtime.beat_state import BeatState
from runtime.snapshot import EpisodeSnapshot
from runtime.persistence.sql_store import SQLStore

from runtime.policies.retry_policy import RetryPolicy
from runtime.policies.quality_policy import QualityPolicy
from runtime.policies.cost_policy import CostPolicy
import os
from datetime import datetime



class EpisodeRuntime:
    def __init__(self, episode_id, world_id, intent, policies, sql: SQLStore):
        self.episode_id = episode_id
        self.world_id = world_id
        self.intent = intent
        self.policies = policies or {}

        self.sql = sql

        self.retry_policy = RetryPolicy(self.policies.get("retry", {}))
        self.quality_policy = QualityPolicy(self.policies.get("quality", {}))
        self.cost_policy = CostPolicy(self.policies.get("cost", {}))

        self.state = EpisodeState.CREATED

    # =========================================================
    # Lifecycle
    # =========================================================

    @classmethod
    def create(cls, world_id, intent, policies, sql):
        episode_id = str(uuid.uuid4())

        sql.create_episode(
            episode_id=episode_id,
            world_id=world_id,
            intent=intent,
            policies=policies,
            state=EpisodeState.CREATED,
        )

        return cls(episode_id, world_id, intent, policies, sql)

    @classmethod
    def load(cls, episode_id, sql):
        episode = sql.get_episode(episode_id)
        if not episode:
            raise RuntimeError(f"Episode {episode_id} not found")

        runtime = cls(
            episode_id=episode_id,
            world_id=episode["world_id"],
            intent=episode["intent"],
            policies=episode["policies"],
            sql=sql,
        )

        runtime.state = episode["state"]
        return runtime

    # =========================================================
    # Planning
    # =========================================================

    def plan(self, planner):
        if self.state != EpisodeState.CREATED:
            return

        beats = planner.generate_beats(self.intent)

        for beat in beats:
            beat["id"] = f"{self.episode_id}:{beat['id']}"
            self.sql.create_beat(self.episode_id, beat)

        self._advance(EpisodeState.PLANNED)

    # =========================================================
    # Scheduling
    # =========================================================

    def schedule(self):
        if self.state == EpisodeState.EXECUTING:
            return

        if self.state not in {
            EpisodeState.PLANNED,
            EpisodeState.PARTIALLY_COMPLETED,
        }:
            raise RuntimeError(f"Cannot execute episode in state {self.state}")

        self._advance(EpisodeState.EXECUTING)

    # =========================================================
    # Decision Loop API
    # =========================================================

    def get_executable_beats(self):
        return self.sql.get_beats_by_state(
            self.episode_id,
            states={BeatState.PENDING},
        )


    def build_gpu_job(self, beat_id: str, job_id: str) -> dict:
        beat = self.sql.get_beat(beat_id)
        if not beat:
            raise RuntimeError(f"Beat {beat_id} not found")

        spec = beat["spec"]

        backend = spec.get("backend")
        if not backend:
            raise RuntimeError("Beat spec must define backend")

        duration_sec = float(beat.get("duration_sec", 4.0))

        motion = {
            "engine": "sparse",
            "params": {
                "reuse_poses": True,
                "temporal_smoothing": True,
                "strength": float(spec.get("motion_strength", 0.85)),
            },
        }

        input_payload = {
            "prompt": spec.get("description"),
            "duration_sec": duration_sec,
            "motion": motion,
            "style": spec.get("style", "cinematic"),
        }

        # ---------------------------------------
        # Backend-specific inputs
        # ---------------------------------------

        if backend == "animatediff":
            # Required for real runs; optional only in PIPELINE_VALIDATE mode
            if "start_frame_path" in spec:
                input_payload["start_frame_path"] = spec["start_frame_path"]

            if "end_frame_path" in spec:
                input_payload["end_frame_path"] = spec["end_frame_path"]

        return {
            "job_id": job_id,
            "backend": backend,

            "input": input_payload,

            "output": {
                "path": f"episodes/{self.episode_id}/beats/{beat_id}",
            },

            "meta": {
                "episode_id": self.episode_id,
                "beat_id": beat_id,
                "attempt": self.sql.count_attempts(beat_id),
                "created_at": datetime.utcnow().isoformat(),
            },
        }


    # =========================================================
    # Result Handling
    # =========================================================

    def mark_beat_success(self, beat_id, artifacts, metrics):
        self.sql.record_attempt(
            episode_id=self.episode_id,
            beat_id=beat_id,
            success=True,
            metrics=metrics,
        )

        for artifact_type, uri in artifacts.items():
            self.sql.record_artifact(
                beat_id=beat_id,
                artifact_type=artifact_type,
                uri=uri,
            )

        self.sql.mark_beat_state(beat_id, BeatState.ACCEPTED)
        self._recompute_episode_state()

    def mark_beat_failure(self, beat_id, error, metrics):
        self.sql.record_attempt(
            episode_id=self.episode_id,
            beat_id=beat_id,
            success=False,
            metrics={"error": error, **(metrics or {})},
        )

        decision = self.retry_policy.decide(
            beat={"id": beat_id},
            attempts=self.sql.get_attempts(beat_id),
            observation=None,
            error=error,
        )

        if decision.action == "RETRY":
            self.sql.mark_beat_state(beat_id, BeatState.PENDING)
        else:
            self.sql.mark_beat_state(beat_id, BeatState.ABORTED, decision.reason)

        self._recompute_episode_state()

    # =========================================================
    # State & Snapshot
    # =========================================================

    def snapshot(self):
        return EpisodeSnapshot.from_runtime(self).to_dict()

    def is_terminal(self):
        return self.state in {
            EpisodeState.COMPLETED,
            EpisodeState.FAILED,
        }

    # =========================================================
    # Internal
    # =========================================================

    def _advance(self, new_state):
        self.state = new_state
        self.sql.update_episode_state(self.episode_id, new_state, datetime.utcnow())

    def _recompute_episode_state(self):
        if self.sql.all_beats_completed(self.episode_id):
            self._advance(EpisodeState.COMPLETED)
        elif self.sql.any_beats_failed(self.episode_id):
            self._advance(EpisodeState.PARTIALLY_COMPLETED)
