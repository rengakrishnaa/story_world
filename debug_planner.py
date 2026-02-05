
import os
import sys
import asyncio
from dotenv import load_dotenv

# Mock FastAPI dependencies
from runtime.episode_runtime import EpisodeRuntime
from runtime.persistence.sql_store import SQLStore
from runtime.persistence.redis_store import RedisStore
from agents.narrative_planner import ProductionNarrativePlanner
from config import get_simulation_policies

def debug_plan():
    load_dotenv(override=True)
    print("🔧 Starting Planner Debug...")
    print(f"   CWD: {os.getcwd()}")
    
    url = os.getenv("REDIS_URL")
    if not url:
        print("❌ CRITICAL: REDIS_URL not found in environment!")
        # Try explicit load
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            print(f"   Found .env at {env_path}, loading...")
            load_dotenv(env_path, override=True)
            url = os.getenv("REDIS_URL")
    
    print(f"   REDIS_URL: {url[:10]}... (Len: {len(url) if url else 0})")
    
    # 1. Setup Stores
    sql = SQLStore()
    redis_store = RedisStore(url=url)
    
    # 2. Create Runtime
    print("   Creating Episode...")
    runtime = EpisodeRuntime.create(
        world_id="default",
        intent="Robot stacking boxes",
        policies=get_simulation_policies(),
        sql=sql,
    )
    print(f"   Episode ID: {runtime.episode_id}")
    
    # 3. Plan
    print("   Planning...")
    try:
        planner = ProductionNarrativePlanner(
            world_id="default",
            redis_client=redis_store.redis,
            use_mock=False,
        )
        runtime.plan(planner)
        print("   ✅ Planning Complete")
    except Exception as e:
        print(f"   ❌ Planning Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Schedule
    print("   Scheduling...")
    try:
        runtime.schedule()
        print("   ✅ Scheduling Complete")
    except Exception as e:
        print(f"   ❌ Scheduling Failed: {e}")
        traceback.print_exc()
        return

    # 5. Submit
    print("   Submitting to Redis...")
    try:
        runtime.submit_pending_beats(redis_store)
        print("   ✅ Submission Complete")
    except Exception as e:
        print(f"   ❌ Submission Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 Success! The pipeline ran locally without error.")
    print("   If this worked, the API should also work.")

if __name__ == "__main__":
    debug_plan()
