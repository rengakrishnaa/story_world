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

    def submit_pending_beats(self, redis_store):
        """
        Submits all PENDING beats to the GPU queue.
        """
        pending_beats = self.get_executable_beats()
        
        if not pending_beats:
            print(f"[EpisodeRuntime] No pending beats to submit for episode {self.episode_id}")
            return
        
        print(f"[EpisodeRuntime] Submitting {len(pending_beats)} pending beats for episode {self.episode_id}")
        
        for beat in pending_beats:
            # Use beat_id from DB row (get_beats_by_state returns beat_id + spec)
            beat_id = beat.get("beat_id") or beat.get("id")
            if not beat_id:
                print(f"[EpisodeRuntime] Skipping beat with no ID: {beat}")
                continue
            
            try:
                # 1. Build Payload
                job_id = str(uuid.uuid4())
                job = self.build_gpu_job(beat_id, job_id)
                
                # 2. Push to Redis
                queue_name = redis_store.push_gpu_job(job)
                print(f"[EpisodeRuntime] Submitted job {job_id} for beat {beat_id} to queue '{queue_name}'")
                
                # 3. Update State
                self.sql.mark_beat_state(beat_id, BeatState.EXECUTING)
                
            except Exception as e:
                # Fallback: Mark as aborted if submission fails (e.g. invalid spec)
                import traceback
                print(f"[EpisodeRuntime] Failed to submit beat {beat_id}: {e}")
                traceback.print_exc()
                self.sql.mark_beat_state(beat_id, BeatState.ABORTED, str(e))

    def augment_beat_for_observability(self, beat_id: str) -> bool:
        """
        Re-render with different camera/scale when observer could not extract physics.
        Do not abort — try alternate framing. Returns False when observability cap reached.
        """
        beat = self.sql.get_beat(beat_id)
        if not beat:
            return False
        spec = dict(beat.get("spec") or {})
        attempt = int(spec.get("observability_attempt", 0))
        if attempt >= 2:
            return False
        spec["observability_attempt"] = attempt + 1
        self.sql.update_beat_spec(beat_id, spec)
        return True

    def abort_beat_observability_cap(self, beat_id: str) -> None:
        """Force abort when observability cap reached to stop re-render loop."""
        self.sql.mark_beat_state(beat_id, BeatState.ABORTED, "Max observability re-renders exceeded")
        self._recompute_episode_state()

    def refine_pending_beats(self, prior_constraints: list) -> int:
        """
        Progressive constraint tightening: augment pending beat descriptions
        with prior observer output. Returns number of beats refined.
        """
        from runtime.beat_refinement import refine_beat_description
        from models.episode_outcome import has_physics_constraint

        if not has_physics_constraint(prior_constraints):
            return 0

        pending = self.sql.get_beats_by_state(self.episode_id, {"PENDING"})
        refined = 0
        for beat in pending:
            beat_id = beat.get("beat_id") or beat.get("id")
            if not beat_id:
                continue
            # Build spec from beat (exclude beat_id/state from persisted spec)
            spec = {k: v for k, v in beat.items() if k not in ("beat_id", "state")}
            if "id" not in spec and "id" in beat:
                spec["id"] = beat["id"]
            updated = refine_beat_description(spec, prior_constraints)
            if updated.get("description") != spec.get("description"):
                self.sql.update_beat_spec(beat_id, updated)
                refined += 1
        return refined

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

        # If animatediff is selected but no frames were provided, fall back to svd.
        # This matches the runtime evidence from Runpod:
        # "Animatediff backend requires start_frame_path and end_frame_path unless PIPELINE_VALIDATE=true"
        original_backend = backend
        if backend == "animatediff":
            has_start = "start_frame_path" in spec and bool(spec.get("start_frame_path"))
            has_end = "end_frame_path" in spec and bool(spec.get("end_frame_path"))
            if not (has_start and has_end):
                backend = "svd"

        duration_sec = float(spec.get("duration_sec", beat.get("duration_sec", 4.0)))
        desc = spec.get("description") or ""
        observability_attempt = int(spec.get("observability_attempt", 0))

        # Physics observability contract: dynamics beats must expose motion
        from runtime.physics_observability import get_render_hints
        hint = get_render_hints(desc, self.intent or "", observability_attempt)
        prompt = (hint + desc).strip() if hint else desc

        motion = {
            "engine": "sparse",
            "params": {
                "reuse_poses": True,
                "temporal_smoothing": True,
                "strength": float(spec.get("motion_strength", 0.85)),
            },
        }

        input_payload = {
            "prompt": prompt,
            "duration_sec": duration_sec,
            "motion": motion,
            "style": spec.get("style", "neutral"),
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

        # Path-safe beat id: no ':' in object keys (R2/S3 400 on colons)
        beat_path_id = beat_id.split(":")[-1] if ":" in beat_id else beat_id.replace(":", "_")

        return {
            "job_id": job_id,
            "backend": backend,

            "input": input_payload,

            "output": {
                "path": f"episodes/{self.episode_id}/beats/{beat_path_id}",
            },

            "meta": {
                "episode_id": self.episode_id,
                "beat_id": beat_id,
                "attempt": self.sql.count_attempts(beat_id),
                "created_at": datetime.utcnow().isoformat(),
                "original_backend": original_backend,
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

    def mark_beat_failure(self, beat_id, error, metrics, observer_verdict: str = None, observation=None):
        self.sql.record_attempt(
            episode_id=self.episode_id,
            beat_id=beat_id,
            success=False,
            metrics={"error": error, **(metrics or {})},
        )

        attempts = self.sql.get_attempts(self.episode_id)
        beat_attempts = [a for a in (attempts or []) if a.get("beat_id") == beat_id]
        decision = self.retry_policy.decide(
            beat={"id": beat_id},
            attempts=beat_attempts,
            observation=observation,
            error=error,
            observer_verdict=observer_verdict,
        )

        if decision.action == "RETRY":
            self.sql.mark_beat_state(beat_id, BeatState.PENDING)
        else:
            self.sql.mark_beat_state(beat_id, BeatState.ABORTED, decision.reason)

        self._recompute_episode_state()

    def mark_episode_impossible(self, reason: str = ""):
        """Set episode to IMPOSSIBLE (observer veto)."""
        self._advance(EpisodeState.IMPOSSIBLE)
        if reason:
            # Best-effort: persist as last_error on all pending beats
            try:
                beats = self.sql.get_beats(self.episode_id)
                for b in beats:
                    if (b.get("state") or "").upper() in ("PENDING", "EXECUTING"):
                        self.sql.mark_beat_state(b["beat_id"], BeatState.ABORTED, reason)
            except Exception:
                pass

    # =========================================================
    # State & Snapshot
    # =========================================================

    def snapshot(self):
        return EpisodeSnapshot.from_runtime(self).to_dict()

    def is_terminal(self):
        return self.state in {
            EpisodeState.COMPLETED,
            EpisodeState.FAILED,
            EpisodeState.IMPOSSIBLE,
            EpisodeState.DEAD_STATE,
            EpisodeState.ABANDONED,
            EpisodeState.EPISTEMICALLY_BLOCKED,
        }

    # =========================================================
    # Internal
    # =========================================================

    def _advance(self, new_state):
        self.state = new_state
        self.sql.update_episode_state(self.episode_id, new_state, datetime.utcnow())

    def _recompute_episode_state(self):
        """
        Advance episode state based on beat execution.
        
        INVARIANT: No EXECUTING when constraints cannot be evaluated.
        If any beat is epistemically incomplete, episode is EPISTEMICALLY_BLOCKED.
        """
        if self.state in {
            EpisodeState.IMPOSSIBLE,
            EpisodeState.DEAD_STATE,
            EpisodeState.ABANDONED,
            EpisodeState.EPISTEMICALLY_BLOCKED,
        }:
            return
        # Epistemic halt takes precedence: execution impossible
        if self.sql.any_beats_epistemically_incomplete(self.episode_id):
            print(f"[EpisodeRuntime] Epistemic halt -> EPISTEMICALLY_BLOCKED for {self.episode_id}")
            self._advance(EpisodeState.EPISTEMICALLY_BLOCKED)
            return
        if self.sql.all_beats_completed(self.episode_id):
            print(f"[EpisodeRuntime] All beats completed -> COMPLETED for {self.episode_id}")
            self._advance(EpisodeState.COMPLETED)
        elif self.sql.any_beats_failed(self.episode_id):
            print(f"[EpisodeRuntime] Some beats failed -> PARTIALLY_COMPLETED for {self.episode_id}")
            self._advance(EpisodeState.PARTIALLY_COMPLETED)
