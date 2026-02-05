
import sqlite3
import redis
import json
import os
import uuid
import time
from dotenv import load_dotenv

def verify_fix():
    load_dotenv()
    
    # 1. Setup
    ep_id = f"test-fix-{uuid.uuid4()}"
    beat_id = f"{ep_id}:beat-1"
    
    print(f"🧪 Starting Verification Test for Episode: {ep_id}")
    
    # 2. Insert Fake Episode into SQL (Directly)
    conn = sqlite3.connect("local.db")
    cursor = conn.cursor()
    
    try:
        # Create Episode
        cursor.execute(
            "INSERT INTO episodes (episode_id, world_id, intent, policies, state, updated_at) VALUES (?, ?, ?, '{}', 'EXECUTING', ?)",
            (ep_id, "default", "test", time.time())
        )
        # Create Beat
        beat_spec = {"id": beat_id, "description": "Test Beat"}
        cursor.execute(
            "INSERT INTO beats (beat_id, episode_id, spec, state) VALUES (?, ?, ?, 'EXECUTING')",
            (beat_id, ep_id, json.dumps(beat_spec))
        )
        conn.commit()
        print("   ✅ Inserted dummy episode into SQLite")
        
        # 3. Push Fake Result to Redis
        r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True, ssl_cert_reqs=None)
        q = os.getenv("RESULT_QUEUE", "storyworld:gpu:results")
        
        payload = {
            "job_id": f"job-{uuid.uuid4()}",
            "status": "success",
            "runtime": {"duration": 0.1},
            "artifacts": {"video": "http://example.com/video.mp4"},
            "meta": {
                "episode_id": ep_id,
                "beat_id": beat_id
            }
        }
        
        r.rpush(q, json.dumps(payload))
        print(f"   ✅ Pushed success result to Redis Queue: {q}")
        
        # 4. Poll for Update
        print("   ⏳ Looping for 10s to see if API picks it up...")
        for i in range(10):
            time.sleep(1)
            cursor.execute("SELECT state FROM beats WHERE beat_id = ?", (beat_id,))
            row = cursor.fetchone()
            if row and row[0] == "ACCEPTED":
                print("\n🎉 SUCCESS! Database updated to ACCEPTED.")
                print("   The ResultConsumer is working correctly.")
                return True
            print(".", end="", flush=True)
            
        print("\n❌ FAILURE. Database state remained EXECUTING.")
        print("   Check: Is 'uvicorn' running? Is ResultConsumer started?")
        return False
        
    finally:
        # Cleanup
        cursor.execute("DELETE FROM episodes WHERE episode_id = ?", (ep_id,))
        cursor.execute("DELETE FROM beats WHERE episode_id = ?", (ep_id,))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    verify_fix()
