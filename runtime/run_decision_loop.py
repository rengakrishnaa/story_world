# runtime/run_decision_loop.py

import os
import sys
import redis
from dotenv import load_dotenv

from config import GPU_JOB_QUEUE, GPU_RESULT_QUEUE, GPU_RESULT_QUEUE_PREFIX
from runtime.persistence.sql_store import SQLStore
from runtime.episode_runtime import EpisodeRuntime
from runtime.decision_loop import RuntimeDecisionLoop

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m runtime.run_decision_loop <episode_id>")
        return

    episode_id = sys.argv[1]
    print(f"\n[run_decision_loop] starting for episode {episode_id}")

    # SQL (local sqlite is fine)
    try:
        sql = SQLStore()
        print(f"[run_decision_loop] connected to SQL: {os.getenv('DATABASE_URL')}")
    except Exception as e:
        print(f"[run_decision_loop] SQL error: {e}")
        return

    # 🔑 Redis client (REAL Redis URL from env)
    redis_url = os.getenv("REDIS_URL")
    print(f"[run_decision_loop] connecting to Redis: {redis_url[:15]}...")
    
    try:
        redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=10,
        )
        redis_client.ping()
        print("[run_decision_loop] Redis ping successful")
    except Exception as e:
        print(f"[run_decision_loop] Redis error: {e}")
        return

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    gpu_job_queue = GPU_JOB_QUEUE
    base_result_queue = GPU_RESULT_QUEUE_PREFIX or GPU_RESULT_QUEUE
    gpu_result_queue = f"{base_result_queue}:{episode_id}"
    
    print(f"[run_decision_loop] job_queue: {gpu_job_queue}")
    print(f"[run_decision_loop] result_queue: {gpu_result_queue}")

    loop = RuntimeDecisionLoop(
        runtime=runtime,
        gpu_job_queue=gpu_job_queue,
        gpu_result_queue=gpu_result_queue,
        redis_client=redis_client,
    )

    loop.run()


if __name__ == "__main__":
    main()
