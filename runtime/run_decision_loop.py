# runtime/run_decision_loop.py

import os
import sys
import redis

from runtime.persistence.sql_store import SQLStore
from runtime.episode_runtime import EpisodeRuntime
from runtime.decision_loop import RuntimeDecisionLoop
from dotenv import load_dotenv
load_dotenv()

# #region agent log
import json as _json
import time as _time
from pathlib import Path as _Path
def _dbg(hypothesisId: str, location: str, message: str, data: dict, runId: str = "run1"):
    try:
        _default = _Path(__file__).resolve().parents[1] / ".cursor" / "debug.log"
        _p = _Path(os.getenv("CURSOR_DEBUG_LOG_PATH", str(_default)))
        _p.parent.mkdir(parents=True, exist_ok=True)
        with _p.open("a", encoding="utf-8") as f:
            f.write(_json.dumps({
                "sessionId": "debug-session",
                "runId": runId,
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(_time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m runtime.run_decision_loop <episode_id>")
        return

    episode_id = sys.argv[1]
    print(f"\n[run_decision_loop] starting for episode {episode_id}")

    # SQL (local sqlite is fine)
    try:
        sql = SQLStore()
        print(f"[run_decision_loop] connected to SQL: {os.getenv('DATABASE_URL')}")
    except Exception as e:
        print(f"[run_decision_loop] SQL error: {e}")
        return

    # 🔑 Redis client (REAL Redis URL from env)
    redis_url = os.getenv("REDIS_URL")
    print(f"[run_decision_loop] connecting to Redis: {redis_url[:15]}...")
    _dbg("B", "runtime/run_decision_loop.py:redis", "redis url present", {
        "episode_id": episode_id,
        "REDIS_URL_set": bool(redis_url),
        "REDIS_URL_prefix": (redis_url[:12] if redis_url else None),
    })
    
    try:
        redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=10,
        )
        redis_client.ping()
        print("[run_decision_loop] Redis ping successful")
        _dbg("B", "runtime/run_decision_loop.py:redis", "redis ping ok", {"episode_id": episode_id})
    except Exception as e:
        print(f"[run_decision_loop] Redis error: {e}")
        _dbg("B", "runtime/run_decision_loop.py:redis", "redis ping failed", {"episode_id": episode_id, "error": str(e)})
        return

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    # ✅ THIS is where your snippet belongs
    gpu_job_queue = os.getenv("GPU_JOB_QUEUE", "storyworld:gpu:jobs")
    gpu_result_queue = os.getenv("GPU_RESULT_QUEUE", "storyworld:gpu:results")
    
    print(f"[run_decision_loop] job_queue: {gpu_job_queue}")
    print(f"[run_decision_loop] result_queue: {gpu_result_queue}")
    _dbg("A", "runtime/run_decision_loop.py:queues", "queue names chosen", {
        "episode_id": episode_id,
        "GPU_JOB_QUEUE": os.getenv("GPU_JOB_QUEUE"),
        "GPU_RESULT_QUEUE": os.getenv("GPU_RESULT_QUEUE"),
        "JOB_QUEUE": os.getenv("JOB_QUEUE"),
        "RESULT_QUEUE": os.getenv("RESULT_QUEUE"),
        "chosen_job_queue": gpu_job_queue,
        "chosen_result_queue": gpu_result_queue,
    })

    loop = RuntimeDecisionLoop(
        runtime=runtime,
        gpu_job_queue=gpu_job_queue,
        gpu_result_queue=gpu_result_queue,
        redis_client=redis_client,
    )

    loop.run()


if __name__ == "__main__":
    main()
