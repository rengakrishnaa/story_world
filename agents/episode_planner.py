import redis
import os
from models.episode_plan import EpisodePlan

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

class EpisodePlanner:
    def build_plan(self, world_id: str) -> EpisodePlan:
        beats = r.lrange(f"episode_beats:{world_id}", 0, -1)
        beats = [b.decode() for b in beats]

        return EpisodePlan(
            world_id=world_id,
            beats=beats,
            required_confidence=0.6,
            allow_gaps=False
        )
