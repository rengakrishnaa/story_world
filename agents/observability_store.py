import json
import redis
from datetime import datetime
from typing import Dict

from models.beat_attempt import BeatAttempt
from models.beat_observation_record import BeatObservationRecord

class ObservabilityStore:
    def __init__(self, redis_url: str):
        self.r = redis.from_url(redis_url)

    # ---------- Beat Attempts ----------
    def record_attempt(self, attempt: BeatAttempt):
        key = f"beat_attempts:{attempt.world_id}:{attempt.beat_id}"
        self.r.rpush(key, json.dumps(attempt.__dict__))

    # ---------- Beat Observations ----------
    def record_observation(self, obs: BeatObservationRecord):
        key = f"beat_observations:{obs.world_id}:{obs.beat_id}"
        self.r.rpush(key, json.dumps(obs.__dict__))

    # ---------- Episode Telemetry ----------
    def record_episode_telemetry(self, world_id: str, telemetry: Dict):
        key = f"episode_telemetry:{world_id}"
        self.r.set(key, json.dumps(telemetry))

    # ---------- Learning Queries ----------
    def get_attempts(self, world_id: str, beat_id: str):
        key = f"beat_attempts:{world_id}:{beat_id}"
        return [json.loads(x) for x in self.r.lrange(key, 0, -1)]
