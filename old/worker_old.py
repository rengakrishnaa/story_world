import asyncio
import json
import os
import time
import redis
from dotenv import load_dotenv
from agents.beat_observer import BeatObserver
import logging
logger = logging.getLogger(__name__)
from agents.shot_renderer import ProductionShotRenderer
from agents.episode_assembler import get_assembler
from agents.retry_controller import RetryController
from agents.observability_store import ObservabilityStore
from models.beat_attempt import BeatAttempt
from models.beat_observation_record import BeatObservationRecord
from datetime import datetime
from runtime.persistence.sql_store import SQLStore
from runtime.persistence.redis_store import RedisStore

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WORLD_ID = os.getenv("WORLD_ID", "demo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from PIL import Image

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

r = redis.Redis(host="localhost", port=6379, db=0)
renderer = ProductionShotRenderer(GEMINI_API_KEY)
observer = BeatObserver(WORLD_ID)

sql = SQLStore()
redis_store = RedisStore(r)

retry_controller = RetryController()
observability = ObservabilityStore(REDIS_URL)

QUEUE = f"episode:*:queue"

async def worker_loop():
    print("[worker] waiting for tasks")

    while True:
        job = r.blpop(QUEUE, timeout=5)
        if not job:
            await asyncio.sleep(1)
            continue

        _, payload = job
        task = json.loads(payload)

        episode_id = task["episode_id"]
        beat_id = task["beat_id"]
        spec = task["execution_spec"]

        print(f"🎬 Executing beat {beat_id}")

        try:
            # 1️⃣ Render
            keyframe = renderer.render_beat_keyframe(None, spec)
            video_url = renderer.render_veo_video(None, spec, keyframe)

            # 2️⃣ Observe
            observation = observer.observe(
                spec,
                {
                    "keyframe_path": keyframe["keyframe_path"],
                    "video_path": video_url,
                    "model": spec.get("force_model", "default"),
                },
            )

            # 3️⃣ Persist attempt + artifacts
            attempt_id = sql.record_attempt(
                beat_id=beat_id,
                model=spec.get("force_model", "default"),
                prompt=spec["description"],
                success=observation.success,
                metrics={"confidence": observation.confidence},
            )

            sql.record_artifact(beat_id, attempt_id, "keyframe", keyframe["keyframe_url"])
            sql.record_artifact(beat_id, attempt_id, "video", video_url)

            # 4️⃣ Emit observation to runtime
            redis_store.emit_observation(
                episode_id,
                {
                    "beat_id": beat_id,
                    "attempt_id": attempt_id,
                    "observation": observation.to_dict(),
                },
            )

            print(f"✅ Beat {beat_id} executed")

        except Exception as e:
            redis_store.emit_observation(
                episode_id,
                {
                    "beat_id": beat_id,
                    "error": str(e),
                },
            )
            print(f"❌ Beat {beat_id} failed: {e}")

if __name__ == "__main__":
    asyncio.run(worker_loop())
