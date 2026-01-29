import os
from fastapi import FastAPI
from dotenv import load_dotenv
import importlib
from runtime.decision_loop import RuntimeDecisionLoop

load_dotenv()

redis = importlib.import_module("redis")

# #region agent log
import json as _json
import time as _time
from pathlib import Path as _Path
def _dbg(hypothesisId: str, location: str, message: str, data: dict, runId: str = "run1"):
    try:
        _default = _Path(__file__).resolve().parent / ".cursor" / "debug.log"
        _p = _Path(os.getenv("CURSOR_DEBUG_LOG_PATH", str(_default)))
        _p.parent.mkdir(parents=True, exist_ok=True)
        with _p.open("a", encoding="utf-8") as f:
            f.write(_json.dumps({
                "sessionId": "debug-session",
                "runId": runId,
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(_time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion

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

    # Use environment variable to control mock vs real planner
    use_mock = os.getenv("USE_MOCK_PLANNER", "false").lower() == "true"

    planner = ProductionNarrativePlanner(
        world_id=runtime.world_id,
        redis_client=redis_store.redis,   
        use_mock=use_mock,  # ✅ FIXED: Now respects env variable
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

    _dbg("A", "main.py:execute_episode", "execute called", {
        "episode_id": episode_id,
        "REDIS_URL_set": bool(os.getenv("REDIS_URL")),
        "JOB_QUEUE": os.getenv("JOB_QUEUE"),
        "RESULT_QUEUE": os.getenv("RESULT_QUEUE"),
        "GPU_JOB_QUEUE": os.getenv("GPU_JOB_QUEUE"),
        "GPU_RESULT_QUEUE": os.getenv("GPU_RESULT_QUEUE"),
        "LOCAL_MODE": os.getenv("LOCAL_MODE"),
    })

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    runtime.schedule()

    log_file = open("decision_loop.log", "a")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "runtime.run_decision_loop",
            episode_id,
        ],
        stdout=log_file,
        stderr=log_file,
        bufsize=1,
    )

    _dbg("A", "main.py:execute_episode", "spawned run_decision_loop", {
        "pid": getattr(proc, "pid", None),
        "episode_id": episode_id,
    })

    return {"state": runtime.state}



@app.get("/episodes/{episode_id}")
def episode_status(episode_id: str):
    from runtime.episode_runtime import EpisodeRuntime

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    return runtime.snapshot()
