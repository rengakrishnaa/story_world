import asyncio
import json
import os
import redis
from dotenv import load_dotenv

from agents.shot_renderer import ProductionShotRenderer
from agents.beat_observer import BeatObserver

from runtime.persistence.sql_store import SQLStore
from runtime.persistence.redis_store import RedisStore

from runtime.control.contracts import ExecutionObservation
#from runtime.control.control_runtime import ControlRuntime

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

r = redis.from_url(REDIS_URL)

renderer = ProductionShotRenderer(GEMINI_API_KEY)
observer = BeatObserver(world_id=None)

sql = SQLStore()
redis_store = RedisStore(r)

'''control = ControlRuntime(
    episode=None,          # already created upstream
    beats=[],
    retry_limit=2,
    event_log=sql,
    state_store=sql,
    queue=redis_store,
)'''

QUEUE = "episode:queue"  # ✅ single global queue


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

        attempt_id = None

        print(f"🎬 Executing beat {beat_id}")

        success = False
        error_msg = None
        artifacts = []
        metrics = {}

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

            success = observation.success
            metrics["confidence"] = observation.confidence

            # 3️⃣ Persist (legacy, allowed)
            attempt_id = sql.record_attempt(
                episode_id=episode_id,
                beat_id=beat_id,
                model=spec.get("force_model", "default"),
                prompt=spec["description"],
                success=success,
                metrics=metrics,
            )

            sql.record_artifact(beat_id, attempt_id, "keyframe", keyframe["keyframe_url"])
            sql.record_artifact(beat_id, attempt_id, "video", video_url)

            artifacts = [
                {"type": "keyframe", "uri": keyframe["keyframe_url"]},
                {"type": "video", "uri": video_url},
            ]

            print(f"✅ Beat {beat_id} executed")

        except Exception as e:
            error_msg = str(e)

            attempt_id = sql.record_attempt(
                episode_id=episode_id,
                beat_id=beat_id,
                model=spec.get("force_model", "default"),
                prompt=spec.get("description"),
                success=False,
                metrics={"error": error_msg},
            )

            print(f"❌ Beat {beat_id} failed: {error_msg}")

        # 🔥 SINGLE, CANONICAL EMISSION (always runs)
        result_payload = {
            "beat_id": beat_id,
            "success": success,
            "error": error_msg,
            "metrics": metrics,
            "artifacts": artifacts,
        }

        '''control.handle_worker_result(
            ExecutionObservation(
                episode_id=episode_id,
                beat_id=beat_id,
                attempt_id=attempt_id,
                success=success,
                confidence=metrics.get("confidence", 0.0),
                failure_type=None if success else "execution_error",
                explanation=error_msg,
                artifacts={a["type"]: a["uri"] for a in artifacts},
                metrics=metrics,
            )
        )'''





if __name__ == "__main__":
    asyncio.run(worker_loop())
