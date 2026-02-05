
import sqlite3
import json
import os
import redis
import uuid
import time
from dotenv import load_dotenv

def diagnose_and_fix():
    load_dotenv()
    
    # 1. Connect to SQL
    print("🔍 Scanning Local Database for stuck jobs...")
    conn = sqlite3.connect("local.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 2. Find Executing Episodes
    cursor.execute("SELECT episode_id, state FROM episodes WHERE state = 'EXECUTING'")
    stuck_episodes = cursor.fetchall()
    
    if not stuck_episodes:
        print("✅ No stuck episodes found in local DB.")
        return

    print(f"⚠️  Found {len(stuck_episodes)} stuck episodes:")
    
    # 3. For each stuck episode, find the executing beat
    for ep in stuck_episodes:
        ep_id = ep["episode_id"]
        print(f"\n   Episode: {ep_id}")
        
        cursor.execute("SELECT beat_id, state FROM beats WHERE episode_id = ? AND state = 'EXECUTING'", (ep_id,))
        beats = cursor.fetchall()
        
        if not beats:
            print("      (No executing beats found - Episode state might be desynced)")
            continue
            
        for beat in beats:
            beat_id = beat["beat_id"]
            short_id = beat_id.split(':')[-1] if ':' in beat_id else beat_id
            print(f"      - Beat: {short_id} (State: EXECUTING)")
            
            # 4. Inject Result into Redis
            print(f"      💡 ACTION: Sending fake SUCCESS result to Redis for {short_id}...")
            
            fake_result = {
                "job_id": f"debug-{uuid.uuid4()}",
                "status": "success",
                "artifacts": {
                    "video": "https://www.w3schools.com/html/mov_bbb.mp4" # Placeholder video
                },
                "runtime": {
                    "backend": "debug_injector",
                    "latency_sec": 0.1,
                    "cost": 0.0,
                    "gpu": "virtual"
                },
                "meta": {
                    "episode_id": ep_id,
                    "beat_id": beat_id, # Full ID required
                    "result_queue": os.getenv("GPU_RESULT_QUEUE", os.getenv("RESULT_QUEUE", "storyworld:gpu:results"))
                }
            }
            
            _push_to_redis(fake_result)
            print("         -> Sent! Check Dashboard in 5 seconds.")

def _push_to_redis(payload):
    url = os.getenv("REDIS_URL")
    if not url:
        print("❌ REDIS_URL missing")
        return
        
    r = redis.Redis.from_url(url, decode_responses=True, ssl_cert_reqs=None)
    queue = payload["meta"]["result_queue"]
    print(f"         (Pushing to queue: {queue})")
    r.rpush(queue, json.dumps(payload))

if __name__ == "__main__":
    diagnose_and_fix()
