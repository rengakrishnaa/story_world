import json

class RedisStore:
    def __init__(self, redis_client):
        self.redis = redis_client

    # ---- QUEUE ----

    def enqueue_beat(self, episode_id, beat):
        """
        Enqueue a beat for execution.
        """
        payload = {
            "episode_id": episode_id,
            "beat_id": beat["id"] if "id" in beat else beat["beat_id"],
            "execution_spec": beat,
        }

        # Single global queue (intentionally simple)
        self.redis.lpush("episode:queue", json.dumps(payload))

    def enqueue_retry(self, episode_id, payload):
        """
        Re-enqueue a beat after retry decision.
        """
        payload["episode_id"] = episode_id
        self.redis.lpush("episode:queue", json.dumps(payload))

    # ---- OBSERVATIONS ----

    def emit_observation(self, episode_id, payload):
        self.redis.lpush(
            f"episode:{episode_id}:observations",
            json.dumps(payload),
        )

    def pop_observation(self, episode_id, timeout=5):
        res = self.redis.brpop(
            f"episode:{episode_id}:observations",
            timeout=timeout,
        )
        if not res:
            return None
        _, data = res
        return json.loads(data)
