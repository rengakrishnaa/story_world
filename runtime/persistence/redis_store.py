import json
import redis

class RedisStore:
    def __init__(self, redis_client=None, url=None, lazy=False):
        self._redis = redis_client
        self.url = url
        self.lazy = lazy

        if not lazy and self._redis is None:
            self._connect()

    # -----------------------------
    # Internal
    # -----------------------------

    def _connect(self):
        if self._redis is None:
            if not self.url:
                raise RuntimeError("Redis URL not provided")

            self._redis = redis.from_url(
                self.url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

    @property
    def redis(self):
        if self._redis is None:
            self._connect()
        return self._redis

    # -----------------------------
    # GPU QUEUE
    # -----------------------------

    def push_gpu_job(self, payload: dict):
        self.redis.rpush(
            "storyworld:gpu:jobs",
            json.dumps(payload),
        )

    def pop_gpu_job(self, timeout=5):
        res = self.redis.blpop(
            "storyworld:gpu:jobs",
            timeout=timeout,
        )
        if not res:
            return None
        _, data = res
        return json.loads(data)

    def push_gpu_result(self, payload: dict):
        self.redis.rpush(
            "storyworld:gpu:results",
            json.dumps(payload),
        )

    def pop_gpu_result(self, timeout=5):
        res = self.redis.blpop(
            "storyworld:gpu:results",
            timeout=timeout,
        )
        if not res:
            return None
        _, data = res
        return json.loads(data)

    # -----------------------------
    # OBSERVATIONS
    # -----------------------------

    def emit_observation(self, episode_id, payload):
        self.redis.rpush(
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
