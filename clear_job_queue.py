#!/usr/bin/env python3
"""
Clear the Redis GPU job queue. Run this if you have old jobs (e.g. backend=stub)
stuck in the queue from before changing DEFAULT_BACKEND.
"""
import os
import redis
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("REDIS_URL")
if not url:
    print("REDIS_URL not set")
    exit(1)

job_q = os.getenv("GPU_JOB_QUEUE", os.getenv("JOB_QUEUE", "storyworld:gpu:jobs"))
print(f"Clearing queue: {job_q}")

r = redis.Redis.from_url(url, decode_responses=True)
count = r.llen(job_q)
r.delete(job_q)
print(f"Cleared {count} old job(s). Run a new simulation to get jobs with the correct backend.")
