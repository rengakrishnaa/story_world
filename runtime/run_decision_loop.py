# runtime/run_decision_loop.py

import os
import sys
import redis

from runtime.persistence.sql_store import SQLStore
from runtime.episode_runtime import EpisodeRuntime
from runtime.decision_loop import RuntimeDecisionLoop
from dotenv import load_dotenv
load_dotenv()

def main():
    episode_id = sys.argv[1]

    # SQL (local sqlite is fine)
    sql = SQLStore()

    # 🔑 Redis client (REAL Redis URL from env)
    redis_client = redis.from_url(
        os.getenv("REDIS_URL"),
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=10,
    )

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    # ✅ THIS is where your snippet belongs
    loop = RuntimeDecisionLoop(
        runtime=runtime,
        gpu_job_queue=os.getenv("GPU_JOB_QUEUE", "storyworld:gpu:jobs"),
        gpu_result_queue=os.getenv("GPU_RESULT_QUEUE", "storyworld:gpu:results"),
        redis_client=redis_client,
    )

    loop.run()


if __name__ == "__main__":
    main()
