import os
from fastapi import FastAPI
from dotenv import load_dotenv
import importlib
from runtime.decision_loop import RuntimeDecisionLoop

load_dotenv()

redis = importlib.import_module("redis")

# ⚠️ DO NOT initialize heavy objects at import time
sql = None
redis_store = None

app = FastAPI(title="StoryWorld Runtime")

# =====================================================
# LIFESPAN (runs once, safely)
# =====================================================

@app.on_event("startup")
def startup_event():
    print(">>> startup reached")

    global sql, redis_store

    from runtime.persistence.sql_store import SQLStore
    from runtime.persistence.redis_store import RedisStore

    # Lazy SQL
    sql = SQLStore(lazy=True)

    # Lazy Redis
    redis_store = RedisStore(
        url=os.getenv("REDIS_URL"),
        lazy=True,
    )

    print(">>> startup: infrastructure wired")

# =====================================================
# API
# =====================================================

@app.post("/episodes")
def create_episode(world_id: str, intent: str):
    from runtime.episode_runtime import EpisodeRuntime

    runtime = EpisodeRuntime.create(
        world_id=world_id,
        intent=intent,
        policies={
            "retry": {"max_attempts": 2},
            "quality": {"min_confidence": 0.7},
            "cost": {"max_cost": 5.0},
        },
        sql=sql,
    )

    return {
        "episode_id": runtime.episode_id,
        "state": runtime.state,
    }


@app.post("/episodes/{episode_id}/plan")
def plan_episode(episode_id: str):
    from runtime.episode_runtime import EpisodeRuntime
    from agents.narrative_planner import ProductionNarrativePlanner
    from runtime.planner_adapter import PlannerAdapter

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    planner = ProductionNarrativePlanner(
        world_id=runtime.world_id,
        redis_client=redis_store.redis,   
        use_mock=True,
    )

    adapter = PlannerAdapter(planner)     

    runtime.plan(adapter)


    return {"state": runtime.state}


import subprocess
import sys
import os

@app.post("/episodes/{episode_id}/execute")
def execute_episode(episode_id: str):
    from runtime.episode_runtime import EpisodeRuntime

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    runtime.schedule()

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "runtime.run_decision_loop",
            episode_id,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return {"state": runtime.state}



@app.get("/episodes/{episode_id}")
def episode_status(episode_id: str):
    from runtime.episode_runtime import EpisodeRuntime

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    return runtime.snapshot()
