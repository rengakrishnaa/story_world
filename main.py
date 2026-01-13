import os
import threading
from fastapi import FastAPI
from dotenv import load_dotenv
import importlib
redis = importlib.import_module("redis")

from runtime.episode_runtime import EpisodeRuntime
from runtime.persistence.sql_store import SQLStore
from runtime.persistence.redis_store import RedisStore
from runtime.decision_loop import RuntimeDecisionLoop
from runtime.stub_planner import StubPlanner
from runtime.policies.retry_policy import RetryPolicy
from runtime.policies.quality_policy import QualityPolicy
from runtime.policies.cost_policy import CostPolicy
from agents.narrative_planner import ProductionNarrativePlanner
from runtime.planner_adapter import PlannerAdapter

load_dotenv()

app = FastAPI(title="StoryWorld Runtime")

# ---------------- Policies ----------------

DEFAULT_POLICIES = {
    "retry": {"max_attempts": 2},
    "quality": {"min_confidence": 0.7},
    "cost": {"max_cost": 5.0},
}


# ---------------- Infrastructure ----------------

sql = SQLStore()
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
redis_store = RedisStore(redis_client)

# ---------------- API ----------------

@app.post("/episodes")
def create_episode(world_id: str, intent: str):
    runtime = EpisodeRuntime.create(
        world_id=world_id,
        intent=intent,
        policies=DEFAULT_POLICIES,
        sql=sql,
        redis=redis_store,
    )

    return {
        "episode_id": runtime.episode_id,
        "state": runtime.state,
    }


@app.post("/episodes/{episode_id}/plan")
def plan_episode(episode_id: str):
    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
        redis=redis_store,
        policies=None,
    )

    narrative_planner = ProductionNarrativePlanner(
        world_id=runtime.world_id,
        use_mock=True,   # keep mock until stable
    )

    planner = PlannerAdapter(narrative_planner)
    runtime.plan(planner)

    return {"state": runtime.state}



@app.post("/episodes/{episode_id}/execute")
def execute_episode(episode_id: str):
    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
        redis=redis_store,
        policies=None,
    )

    runtime.schedule()

    loop = RuntimeDecisionLoop(runtime)
    threading.Thread(target=loop.run, daemon=True).start()

    return {"state": runtime.state}


@app.get("/episodes/{episode_id}")
def episode_status(episode_id: str):
    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
        redis=redis_store,
        policies=None,
    )

    return runtime.snapshot()
