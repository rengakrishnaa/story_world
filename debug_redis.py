
import os
import redis
import ssl
from dotenv import load_dotenv

def check_queues():
    load_dotenv()
    url = os.getenv("REDIS_URL")
    if not url:
        print("❌ REDIS_URL not found in .env")
        return

    print(f"🔌 Connecting to Redis: {url.split('@')[-1]}") # Hide credentials
    
    try:
        r = redis.Redis.from_url(
            url, 
            decode_responses=True,
            ssl_cert_reqs=ssl.CERT_NONE # Relax SSL for debugging
        )
        r.ping()
        print("✅ Connection Successful")
        
        # Check Queues
        queues = [
            "storyworld:gpu:jobs",
            "storyworld:gpu:results"
        ]
        
        print("\n📊 Queue Status:")
        for q in queues:
            length = r.llen(q)
            print(f"   - {q}: {length} items")
            
            if length > 0:
                print(f"     ⚠️  Next item: {r.lrange(q, 0, 0)}")

    except Exception as e:
        print(f"❌ Redis Connection Failed: {e}")

if __name__ == "__main__":
    check_queues()
