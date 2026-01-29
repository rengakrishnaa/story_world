import time
import json
import uuid
from typing import Dict

# #region agent log
import json as _json
import os as _os
from pathlib import Path as _Path
def _dbg(hypothesisId: str, location: str, message: str, data: dict, runId: str = "run1"):
    try:
        _default = _Path(__file__).resolve().parents[1] / ".cursor" / "debug.log"
        _p = _Path(_os.getenv("CURSOR_DEBUG_LOG_PATH", str(_default)))
        _p.parent.mkdir(parents=True, exist_ok=True)
        with _p.open("a", encoding="utf-8") as f:
            f.write(_json.dumps({
                "sessionId": "debug-session",
                "runId": runId,
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion


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
        _dbg("C", "runtime/decision_loop.py:run", "loop started", {
            "episode_id": self.runtime.episode_id,
            "job_queue": self.gpu_job_queue,
            "result_queue": self.gpu_result_queue,
        })

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
        _dbg("C", "runtime/decision_loop.py:_submit_ready_beats", "ready beats fetched", {
            "episode_id": self.runtime.episode_id,
            "ready_beats_count": len(ready_beats) if ready_beats is not None else None,
            "job_queue": self.gpu_job_queue,
        })

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

            self.redis.rpush(self.gpu_job_queue, json.dumps(gpu_job))
            self.active_jobs[job_id] = beat_id
            _dbg("A", "runtime/decision_loop.py:_submit_ready_beats", "pushed gpu job", {
                "episode_id": self.runtime.episode_id,
                "beat_id": beat_id,
                "job_id": job_id,
                "job_queue": self.gpu_job_queue,
                "backend": gpu_job.get("backend"),
            })

            print(f"[runtime] submitted beat {beat_id} as job {job_id}")

    # -------------------------------------------------
    # GPU Result Consumption
    # -------------------------------------------------

    def _consume_gpu_results(self):
        result = self.redis.blpop(self.gpu_result_queue, timeout=1)
        if not result:
            _dbg("C", "runtime/decision_loop.py:_consume_gpu_results", "no result (timeout)", {
                "episode_id": self.runtime.episode_id,
                "result_queue": self.gpu_result_queue,
                "active_jobs_count": len(self.active_jobs),
            })
            return

        _, payload = result
        result_data = json.loads(payload)

        job_id = result_data.get("job_id")
        if job_id not in self.active_jobs:
            print(f"[runtime] unknown job_id {job_id}")
            _dbg("D", "runtime/decision_loop.py:_consume_gpu_results", "unknown job_id result", {
                "episode_id": self.runtime.episode_id,
                "job_id": job_id,
                "result_queue": self.gpu_result_queue,
            })
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
            print(f"[runtime] beat {beat_id} succeeded")
        else:
            self.runtime.mark_beat_failure(
                beat_id=beat_id,
                error=error,
                metrics=runtime_metrics,
            )
            print(f"[runtime] beat {beat_id} failed")
