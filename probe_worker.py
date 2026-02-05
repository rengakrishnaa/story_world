
import os
import redis
import json
import uuid
import time
from dotenv import load_dotenv

def probe_worker():
    load_dotenv()
    
    # 1. Setup Redis
    url = os.getenv("REDIS_URL")
    if not url:
        print("❌ REDIS_URL missing")
        return
        
    r = redis.Redis.from_url(url, decode_responses=True, ssl_cert_reqs=None)
    
    job_q = os.getenv("GPU_JOB_QUEUE", os.getenv("JOB_QUEUE", "storyworld:gpu:jobs"))
    res_q = f"probe-return-{uuid.uuid4()}" # Use a unique temporary queue
    
    job_id = f"probe-{uuid.uuid4()}"
    
    # 2. Build Probe Job (Veo Backend - Expected Failure)
    payload = {
        "job_id": job_id,
        "backend": "veo", # TEST REAL BACKEND
        "input": {
            "prompt": "", # Trigger "Requires Prompt" error immediately
            "motion": {
                "engine": "sparse",
                "params": {"test": "probe"}
            }
        },
        "output": {"path": "probes/test"},
        "meta": {
            "episode_id": "probe-episode",
            "beat_id": "probe-beat",
            "result_queue": res_q  # Force reply to our temp queue
        }
    }
    
    print(f"📡 Sending Probe Job to: {job_q}")
    print(f"👂 Listening on temp queue: {res_q}")
    
    r.rpush(job_q, json.dumps(payload))
    
    # 3. Wait for reply
    print("⏳ Waiting for worker reply (Timeout: 10s)...")
    start = time.time()
    while time.time() - start < 10:
        res = r.blpop(res_q, timeout=1)
        if res:
            _, val = res
            data = json.loads(val)
            print(f"\n✅ PROBE SUCCESS!")
            print(f"   Received reply from worker: {data.get('job_id')}")
            print(f"   Status: {data.get('status')}")
            
            # Cleanup
            r.delete(res_q)
            return True
        print(".", end="", flush=True)
        
    print("\n\n❌ PROBE TIMEOUT")
    print("   The worker did not reply within 10 seconds.")
    print("   Possible causes:")
    print("   1. Worker is dead/crashed.")
    print("   2. Worker ignores 'meta.result_queue'.")
    print("   3. Worker crashed processing the job (check Runpod logs!).")
    
    # Cleanup
    r.delete(res_q)
    return False

if __name__ == "__main__":
    probe_worker()
