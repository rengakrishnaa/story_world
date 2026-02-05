import json
import os
import redis

# Queue names from config/env for production
def _job_queue():
    return os.getenv("GPU_JOB_QUEUE", os.getenv("JOB_QUEUE", "storyworld:gpu:jobs"))


def _result_queue():
    return os.getenv("GPU_RESULT_QUEUE", os.getenv("RESULT_QUEUE", "storyworld:gpu:results"))


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

            try:
                self._redis = redis.from_url(
                    self.url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                # Test connection
                self._redis.ping()
                print(f"[RedisStore] Connected to Redis (queue: {_job_queue()})")
            except Exception as e:
                print(f"[RedisStore] Failed to connect to Redis: {e}")
                raise

    @property
    def redis(self):
        if self._redis is None:
            self._connect()
        return self._redis

    # -----------------------------
    # GPU QUEUE
    # -----------------------------

    def push_gpu_job(self, payload: dict):
        """
        Push a GPU job to the Redis queue.
        Returns the queue name for logging.
        """
        queue_name = _job_queue()
        try:
            self.redis.rpush(
                queue_name,
                json.dumps(payload),
            )
            return queue_name
        except Exception as e:
            print(f"[RedisStore] Failed to push job to queue '{queue_name}': {e}")
            raise

    def pop_gpu_job(self, timeout=5):
        res = self.redis.blpop(
            _job_queue(),
            timeout=timeout,
        )
        if not res:
            return None
        _, data = res
        return json.loads(data)

    def push_gpu_result(self, payload: dict):
        self.redis.rpush(
            _result_queue(),
            json.dumps(payload),
        )

    def pop_gpu_result(self, timeout=5):
        try:
            res = self.redis.blpop(
                _result_queue(),
                timeout=timeout,
            )
        except Exception as e:
            if "Timeout" in type(e).__name__ or "timeout" in str(e).lower():
                return None
            raise
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
