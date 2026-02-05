
import os
import redis
import json
from dotenv import load_dotenv

def inspect_queues():
    load_dotenv()
    
    url = os.getenv("REDIS_URL")
    if not url:
        print("❌ REDIS_URL missing")
        return
        
    r = redis.Redis.from_url(url, decode_responses=True, ssl_cert_reqs=None)
    
    # Identify queues
    job_q = os.getenv("GPU_JOB_QUEUE", os.getenv("JOB_QUEUE", "storyworld:gpu:jobs"))
    res_q = os.getenv("GPU_RESULT_QUEUE", os.getenv("RESULT_QUEUE", "storyworld:gpu:results"))
    
    print(f"--- Queue Inspection ---")
    print(f"Target Job Queue:    {job_q}")
    print(f"Target Result Queue: {res_q}")
    
    # Check lengths
    try:
        j_len = r.llen(job_q)
        r_len = r.llen(res_q)
        print(f"\n[Status]")
        print(f"Jobs Pending:    {j_len}")
        print(f"Results Pending: {r_len}")
        
        # Peek at Results if any
        if r_len > 0:
            print(f"\n[Result Peek (Last 5 items)]")
            items = r.lrange(res_q, -5, -1)
            for i, item in enumerate(items):
                try:
                    data = json.loads(item)
                    print(f"  {i+1}. Job: {data.get('job_id')} | Status: {data.get('status')} | Ep: {data.get('meta', {}).get('episode_id')}")
                    # print(f"     Full: {item[:100]}...") 
                except:
                    print(f"  {i+1}. [Invalid JSON] {item}")
        else:
            print("\n(Result queue is empty, meaning either Consumer ate them OR Worker never wrote them)")
            
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")

if __name__ == "__main__":
    inspect_queues()
