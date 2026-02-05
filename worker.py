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
# Must match main app (RedisStore) so ResultConsumer receives results
JOB_QUEUE = os.getenv("JOB_QUEUE", "storyworld:gpu:jobs")
RESULT_QUEUE = os.getenv("RESULT_QUEUE", "storyworld:gpu:results")

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


def upload_artifact(local_path: str, remote_path: str) -> tuple:
    """
    Upload to R2/S3. Returns (public_uri, presigned_uri).
    Use presigned_uri for observer: private buckets require signed GET.
    """
    s3.upload_file(local_path, S3_BUCKET, remote_path)
    public_uri = f"{S3_ENDPOINT}/{S3_BUCKET}/{remote_path}"
    try:
        presigned = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": remote_path},
            ExpiresIn=3600,
        )
        if presigned and "X-Amz-" in presigned:
            return (public_uri, presigned)
        raise ValueError("Presigned URL missing auth params")
    except Exception as e:
        print(f"[gpu-worker] presign failed: {e}, observer may get 400 on private R2")
        return (public_uri, public_uri)

# =========================================================
# VEO CREDIT EXHAUSTION FALLBACK
# =========================================================

def _is_credit_exhausted(exc: BaseException) -> bool:
    """Detect API credit/quota exhaustion for fallback to free open-source backend."""
    msg = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 503):
        return True
    indicators = ("429", "resource_exhausted", "quota", "rate limit", "credit", "too many requests")
    return any(ind in msg for ind in indicators)


# =========================================================
# CORE EXECUTION
# =========================================================

def execute_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("job_id", str(uuid.uuid4()))
    backend_name = job.get("backend")
    fallback_backend = os.getenv("VEO_FALLBACK_BACKEND", "svd")  # Best free open-source fallback

    start = time.time()
    effective_backend = backend_name

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
        try:
            outputs = render_fn(job["input"])
        except Exception as e:
            if backend_name == "veo" and _is_credit_exhausted(e):
                print(f"[gpu-worker] Veo credit exhausted, falling back to {fallback_backend}: {e}")
                if fallback_backend in BACKEND_MAP and fallback_backend != "veo":
                    render_fn = load_backend(fallback_backend)
                    input_spec = dict(job.get("input") or {})
                    input_spec["_credit_exhausted"] = True  # Skip API; use local SDXL only
                    outputs = render_fn(input_spec)
                    effective_backend = fallback_backend
                else:
                    raise
            else:
                raise

        for artifact_type, local_path in outputs.items():
            # Ensure video artifacts have .mp4 extension
            if artifact_type == "video" and not local_path.endswith(".mp4"):
                # Backend should return .mp4, but ensure it
                import shutil
                new_path = local_path + ".mp4" if not local_path.endswith(".mp4") else local_path
                if new_path != local_path:
                    shutil.copy2(local_path, new_path)
                    local_path = new_path
            
            # Add file extension to remote path if missing
            remote_path = f"{job['output']['path']}/{artifact_type}"
            if artifact_type == "video" and not remote_path.endswith(".mp4"):
                remote_path = f"{remote_path}.mp4"
            
            public_uri, presigned_uri = upload_artifact(local_path, remote_path)
            result["artifacts"][artifact_type] = presigned_uri
            result["artifacts"][f"{artifact_type}_public"] = public_uri
            # OPTION A: copy to shared dir when worker+observer share volume
            artifacts_dir = os.getenv("ARTIFACTS_DIR")
            if artifacts_dir and artifact_type == "video":
                import shutil
                out_path = job.get("output", {}).get("path", "episodes/unknown/beats/unknown")
                dest_dir = os.path.join(artifacts_dir, os.path.dirname(out_path))
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, "video.mp4")
                shutil.copy2(local_path, dest)
                result["artifacts"]["video_local_path"] = os.path.abspath(dest)

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
            "backend": effective_backend,
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
        job_meta = job.get("meta") or {}
        target_result_queue = job_meta.get("result_queue") or RESULT_QUEUE
        result["meta"] = result.get("meta") or {}
        result["meta"]["episode_id"] = job_meta.get("episode_id")
        result["meta"]["beat_id"] = job_meta.get("beat_id")
        redis_client.rpush(target_result_queue, json.dumps(result))

        print(
            f"[gpu-worker] job={result['job_id']} "
            f"status={result['status']}"
        )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    asyncio.run(worker_loop())
