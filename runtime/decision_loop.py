import time
import json
import uuid
from typing import Dict


class RuntimeDecisionLoop:
    def __init__(
        self,
        runtime,
        gpu_job_queue: str,
        gpu_result_queue: str,
        redis_client,
        poll_interval: float = 0.5,
    ):
        self.runtime = runtime
        self.redis = redis_client              # ✅ FIX
        self.gpu_job_queue = gpu_job_queue     # ✅ FIX
        self.gpu_result_queue = gpu_result_queue
        self.poll_interval = poll_interval

        # job_id -> beat_id
        self.active_jobs: Dict[str, str] = {}

    # -------------------------------------------------
    # Core Loop
    # -------------------------------------------------

    def run(self):
        print(f"[runtime] decision loop started for {self.runtime.episode_id}")

        self._submit_ready_beats()

        while not self.runtime.is_terminal():
            self._consume_gpu_results()
            self._submit_ready_beats()
            time.sleep(self.poll_interval)

        print(f"[runtime] episode {self.runtime.episode_id} completed")

    # -------------------------------------------------
    # Beat → GPU Job submission
    # -------------------------------------------------

    def _submit_ready_beats(self):
        ready_beats = self.runtime.get_executable_beats()

        for beat in ready_beats:
            beat_id = beat.get("beat_id") or beat.get("id")
            if not beat_id:
                continue

            if beat_id in self.active_jobs.values():
                continue

            job_id = str(uuid.uuid4())

            gpu_job = self.runtime.build_gpu_job(
                beat_id=beat_id,
                job_id=job_id,
            )

            # Route results to a per-episode queue to avoid consuming stale results
            # from previous runs / other episodes on a shared RESULT_QUEUE.
            meta = gpu_job.get("meta") or {}
            meta["result_queue"] = self.gpu_result_queue
            gpu_job["meta"] = meta

            # Mark beat as in-flight so it won't be resubmitted every loop tick.
            try:
                from runtime.beat_state import BeatState
                self.runtime.sql.mark_beat_state(beat_id, BeatState.EXECUTING)
            except Exception as e:
                # Best-effort; if we can't mark executing, still submit the job
                print(f"[runtime] failed to mark beat {beat_id} EXECUTING: {e}")
            self.redis.rpush(self.gpu_job_queue, json.dumps(gpu_job))
            self.active_jobs[job_id] = beat_id

            print(f"[runtime] submitted beat {beat_id} as job {job_id}")

    # -------------------------------------------------
    # GPU Result Consumption
    # -------------------------------------------------

    def _consume_gpu_results(self):
        result = self.redis.blpop(self.gpu_result_queue, timeout=1)
        if not result:
            return

        _, payload = result
        result_data = json.loads(payload)

        job_id = result_data.get("job_id")
        if job_id not in self.active_jobs:
            print(f"[runtime] unknown job_id {job_id}")
            return

        beat_id = self.active_jobs.pop(job_id)
        self._handle_result(beat_id, result_data)

    # -------------------------------------------------
    # Result Handling
    # -------------------------------------------------

    def _handle_result(self, beat_id: str, result: dict):
        status = result.get("status")
        artifacts = result.get("artifacts", {})
        error = result.get("error")
        runtime_metrics = result.get("runtime", {})

        if status == "success":
            self.runtime.mark_beat_success(
                beat_id=beat_id,
                artifacts=artifacts,
                metrics=runtime_metrics,
            )
            
            # Also store in Redis for episode_composer to find
            # Key format: render_results:{world_id} -> beat_id -> result
            render_result = {
                "status": "completed",
                "video_url": artifacts.get("video"),
                "confidence": runtime_metrics.get("confidence", 0.8),
                "duration_sec": runtime_metrics.get("duration_sec", 8.0),
                "beat_id": beat_id,
            }
            self.redis.hset(
                f"render_results:{self.runtime.world_id}",
                beat_id,
                json.dumps(render_result)
            )
            
            print(f"[runtime] beat {beat_id} succeeded")
        else:
            self.runtime.mark_beat_failure(
                beat_id=beat_id,
                error=error,
                metrics=runtime_metrics,
            )
            print(f"[runtime] beat {beat_id} failed: {error}")
