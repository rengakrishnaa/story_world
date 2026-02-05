#!/usr/bin/env python3
"""
StoryWorld Bridge: Redis -> RunPod Serverless

Polls Redis for jobs, invokes RunPod Serverless endpoint, pushes result to Redis.
Runs on free/low-cost tier (Render, Fly.io, Railway). No GPU needed—RunPod does the work.

Set env vars:
  REDIS_URL
  JOB_QUEUE (default: storyworld:gpu:jobs)
  RESULT_QUEUE (default: storyworld:gpu:results)
  RUNPOD_API_KEY
  RUNPOD_ENDPOINT_ID (your RunPod Serverless endpoint ID)
"""
import os
import json
import time
import ssl

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import redis

REDIS_URL = os.getenv("REDIS_URL")
JOB_QUEUE = os.getenv("JOB_QUEUE", "storyworld:gpu:jobs")
RESULT_QUEUE = os.getenv("RESULT_QUEUE", "storyworld:gpu:results")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL required")
if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
    raise RuntimeError("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID required for bridge")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_timeout=300,  # RunPod can take 60-180s per job
    socket_connect_timeout=10,
    retry_on_timeout=True,
    ssl_cert_reqs=ssl.CERT_REQUIRED,
)


def run_once():
    """Process one job: blpop -> RunPod -> rpush result."""
    job_data = redis_client.blpop(JOB_QUEUE, timeout=10)
    if not job_data:
        return False

    _, payload = job_data
    try:
        job = json.loads(payload)
    except Exception:
        print("[bridge] invalid job payload")
        return True

    job_id = job.get("job_id", "unknown")
    job_meta = job.get("meta") or {}
    target_queue = job_meta.get("result_queue") or RESULT_QUEUE

    print(f"[bridge] job={job_id} -> RunPod endpoint {RUNPOD_ENDPOINT_ID}")

    try:
        import runpod
        runpod.api_key = RUNPOD_API_KEY
        endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
        result = endpoint.run_sync(input=job, timeout=300)
    except Exception as e:
        print(f"[bridge] RunPod error: {e}")
        result = {
            "job_id": job_id,
            "status": "failure",
            "artifacts": {},
            "runtime": {"backend": job.get("backend", "unknown")},
            "error": {"message": str(e)},
            "meta": {"episode_id": job_meta.get("episode_id"), "beat_id": job_meta.get("beat_id")},
        }

    redis_client.rpush(target_queue, json.dumps(result))
    print(f"[bridge] job={job_id} status={result.get('status', 'unknown')} -> {target_queue}")
    return True


def main():
    cron_mode = os.getenv("BRIDGE_CRON_MODE", "false").lower() == "true"
    print("[bridge] Started. Polling Redis, invoking RunPod Serverless.")
    print(f"[bridge] JOB_QUEUE={JOB_QUEUE} RESULT_QUEUE={RESULT_QUEUE} ENDPOINT={RUNPOD_ENDPOINT_ID} cron_mode={cron_mode}")

    if cron_mode:
        # For Render/Fly cron: process one job and exit. Cron runs every 1-2 min.
        try:
            run_once()
        except Exception as e:
            print(f"[bridge] error: {e}")
        return

    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[bridge] error: {e}")
        time.sleep(1)


if __name__ == "__main__":
    main()
