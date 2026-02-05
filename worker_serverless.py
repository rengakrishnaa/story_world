"""
StoryWorld RunPod Serverless Worker Handler

Deploy to RunPod Serverless. Set ALL env vars in RunPod Console > Endpoint > Settings > Environment Variables.
They persist—no need to set each time. Pay only when jobs run.

Env vars (set once in RunPod Console):
  REDIS_URL, JOB_QUEUE, RESULT_QUEUE (not used by handler—bridge uses these)
  S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION
  DEFAULT_BACKEND, VEO_FALLBACK_BACKEND, USE_DIFFUSION
  GEMINI_API_KEY (for Veo backend)
"""
import os
import json

# RunPod Serverless: handler receives event, returns result
try:
    import runpod
except ImportError:
    runpod = None


def handler(event):
    """
    RunPod Serverless handler. event["input"] = job payload from Redis.
    Returns result dict (job_id, status, artifacts, runtime, error, meta).
    """
    job = event.get("input") or event
    if isinstance(job, str):
        job = json.loads(job)

    # Import here so worker logic loads inside RunPod container
    from worker import execute_job

    result = execute_job(job)
    job_meta = job.get("meta") or {}
    result["meta"] = result.get("meta") or {}
    result["meta"]["episode_id"] = job_meta.get("episode_id")
    result["meta"]["beat_id"] = job_meta.get("beat_id")
    return result


if runpod:
    runpod.serverless.start({"handler": handler})
else:
    # Local test: python worker_serverless.py '{"input": {...}}'
    import sys
    payload = sys.argv[1] if len(sys.argv) > 1 else "{}"
    ev = json.loads(payload)
    out = handler(ev)
    print(json.dumps(out, indent=2))
