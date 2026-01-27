import os
import json
import time
import uuid
import asyncio
import traceback
import importlib
from typing import Dict, Any
import ssl
import redis
import boto3
from dotenv import load_dotenv

# =========================================================
# ENV
# =========================================================

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
JOB_QUEUE = os.getenv("JOB_QUEUE")
RESULT_QUEUE = os.getenv("RESULT_QUEUE")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

if not all([S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY]):
    raise RuntimeError("S3 configuration incomplete")

# =========================================================
# CLIENTS
# =========================================================

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_timeout=30,
    socket_connect_timeout=10,
    retry_on_timeout=True,
    health_check_interval=30,
    ssl_cert_reqs=ssl.CERT_REQUIRED,
)



s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
)

# =========================================================
# BACKENDS
# =========================================================

BACKEND_MAP = {
    "stub": "agents.backends.stub_backend",
    "veo": "agents.backends.veo_backend",
    "animatediff": "agents.backends.animatediff_backend",
    "svd": "agents.backends.svd_backend",
}

# =========================================================
# HARD MOTION VALIDATION
# =========================================================

def validate_sparse_motion(job: Dict[str, Any]):
    input_spec = job.get("input")
    if not input_spec:
        raise RuntimeError("GPU job missing input spec")

    motion = input_spec.get("motion")
    if not motion:
        raise RuntimeError("Sparse motion is mandatory")

    if motion.get("engine") != "sparse":
        raise RuntimeError("Only sparse motion is allowed")

    if not isinstance(motion.get("params"), dict):
        raise RuntimeError("motion.params must be a dict")

    return motion

# =========================================================
# UTILS
# =========================================================

def load_backend(name: str):
    if name not in BACKEND_MAP:
        raise RuntimeError(f"Unknown backend '{name}'")

    module = importlib.import_module(BACKEND_MAP[name])
    if not hasattr(module, "render"):
        raise RuntimeError(f"Backend '{name}' missing render()")

    return module.render


def upload_artifact(local_path: str, remote_path: str) -> str:
    s3.upload_file(local_path, S3_BUCKET, remote_path)
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{remote_path}"

# =========================================================
# CORE EXECUTION
# =========================================================

def execute_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("job_id", str(uuid.uuid4()))
    backend_name = job.get("backend")

    start = time.time()

    result = {
        "job_id": job_id,
        "status": "failure",
        "artifacts": {},
        "runtime": {},
        "error": None,
    }

    try:
        motion = validate_sparse_motion(job)

        print(
            f"[gpu-worker] job={job_id} "
            f"backend={backend_name} "
            f"motion_engine=sparse "
            f"params={motion.get('params')}"
        )

        render_fn = load_backend(backend_name)
        outputs = render_fn(job["input"])

        for artifact_type, local_path in outputs.items():
            remote_path = f"{job['output']['path']}/{artifact_type}"
            uri = upload_artifact(local_path, remote_path)
            result["artifacts"][artifact_type] = uri

        result["status"] = "success"

    except Exception as e:
        print("\n[gpu-worker] EXECUTION ERROR")
        print(traceback.format_exc())
        result["error"] = {
            "message": str(e),
            "trace": traceback.format_exc(),
        }

    finally:
        result["runtime"] = {
            "backend": backend_name,
            "latency_sec": round(time.time() - start, 3),
            "gpu": os.getenv("NVIDIA_VISIBLE_DEVICES", "unknown"),
            "motion_engine": "sparse",
        }

    return result

# =========================================================
# WORKER LOOP
# =========================================================

async def worker_loop():
    print("[gpu-worker] started")

    while True:
        job_data = redis_client.blpop(JOB_QUEUE, timeout=5)
        if not job_data:
            await asyncio.sleep(1)
            continue

        _, payload = job_data

        try:
            job = json.loads(payload)
        except Exception:
            print("[gpu-worker] invalid job payload")
            continue

        result = execute_job(job)
        redis_client.rpush(RESULT_QUEUE, json.dumps(result))

        print(
            f"[gpu-worker] job={result['job_id']} "
            f"status={result['status']}"
        )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    asyncio.run(worker_loop())
