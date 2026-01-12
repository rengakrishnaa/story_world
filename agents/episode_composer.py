import json
import redis
import os
from pathlib import Path
from typing import List

from models.composed_shot import ComposedShot
from models.episode_plan import EpisodePlan

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

class EpisodeComposer:
    def __init__(self, world_id: str):
        self.world_id = world_id

    def compose(self, plan: EpisodePlan) -> List[ComposedShot]:
        shots = []

        for beat_id in plan.beats:
            raw = r.hget(f"render_results:{self.world_id}", beat_id)
            if not raw:
                if not plan.allow_gaps:
                    raise RuntimeError(f"Missing beat {beat_id}")
                continue

            result = json.loads(raw)

            if result["status"] != "completed":
                continue

            if result.get("confidence", 0) < plan.required_confidence:
                continue

            shots.append(
                ComposedShot(
                    beat_id=beat_id,
                    video_path=Path(
                        result["video_url"].replace(
                            "http://localhost:8000/static/", ""
                        )
                    ),
                    duration=result.get("duration_sec", 5.0),
                    confidence=result.get("confidence", 0.5)
                )
            )

        if not shots:
            raise RuntimeError("No valid shots to compose")

        return shots
