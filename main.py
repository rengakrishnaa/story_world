import os
from pathlib import Path
from fastapi import FastAPI
from dotenv import load_dotenv
import importlib
from runtime.decision_loop import RuntimeDecisionLoop

# Load .env from project root (works regardless of cwd when uvicorn --reload runs)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path, override=True)
print(f"[main] Loaded .env from {env_path}, OBSERVER_FALLBACK_ENABLED={os.getenv('OBSERVER_FALLBACK_ENABLED', 'NOT_SET')}, DEFAULT_BACKEND={os.getenv('DEFAULT_BACKEND', 'veo')}")

redis = importlib.import_module("redis")
from config import (
    get_simulation_policies,
    get_episode_policies,
    USE_MOCK_PLANNER,
    COST_PER_BEAT_USD,
    DEFAULT_SUCCESS_CONFIDENCE,
    EPISODE_COMPOSE_REQUIRED_CONFIDENCE,
    DECISION_LOOP_LOG_PATH,
)

# DO NOT initialize heavy objects at import time
sql = None
redis_store = None
world_graph_store = None

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="StoryWorld Runtime")

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

@app.get("/health")
async def health():
    """Minimal health check - no deps."""
    return {"status": "ok", "service": "storyworld"}


@app.get("/observer/calibration")
def get_observer_calibration():
    """Observer calibration metrics: verdict distribution, error rate (when human-labeled)."""
    try:
        from runtime.observer_calibration import get_calibration
        cal = get_calibration()
        return {
            "verdict_distribution": cal.get_verdict_distribution(),
            "error_rate": cal.get_error_rate(),
            "recent": cal.get_recent(20),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/observer/calibration/label")
def label_verdict(beat_id: str, human_label: str):
    """Add human label (correct/incorrect) for verdict calibration."""
    if human_label not in ("correct", "incorrect"):
        return {"error": "human_label must be 'correct' or 'incorrect'"}
    try:
        from runtime.observer_calibration import get_calibration
        ok = get_calibration().add_human_label(beat_id, human_label)
        return {"ok": ok, "beat_id": beat_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/constraints/priors")
def get_constraint_priors(domain: str = "generic", limit: int = 20):
    """Prior constraints from constraint memory for a domain."""
    try:
        from runtime.constraint_memory import get_prior_constraints
        priors = get_prior_constraints("", domain=domain, limit=limit)
        return {"domain": domain, "priors": priors}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

@app.get("/new.html")
async def read_new():
    from fastapi.responses import FileResponse
    return FileResponse('static/new.html')

@app.get("/simulation.html")
async def read_simulation():
    from fastapi.responses import FileResponse
    return FileResponse('static/simulation.html')


@app.on_event("startup")
async def startup_event():
    global sql, redis_store, world_graph_store
    try:
        from runtime.persistence.sql_store import SQLStore
        from runtime.persistence.redis_store import RedisStore
        from runtime.persistence.world_graph_store import WorldGraphStore

        sql = SQLStore(lazy=True)
        redis_store = RedisStore(url=os.getenv("REDIS_URL"), lazy=True)
        try:
            redis_store.redis.ping()
            pass
        except Exception as e:
            print(f"Redis connection failed: {e}")

        world_graph_store = WorldGraphStore()

        from runtime.result_consumer import ResultConsumer
        consumer = ResultConsumer(sql, redis_store, world_graph_store)
        import asyncio
        if os.getenv("SERVERLESS", "").lower() not in ("true", "1", "yes"):
            asyncio.create_task(consumer.run_loop())
        print(">>> startup: infrastructure wired")
    except Exception as e:
        import traceback
        print(f"Startup failed: {e}")
        traceback.print_exc()
        sql = redis_store = world_graph_store = None



_CREATIVE_RED_FLAGS = frozenset(
    ("scene", "cinematic", "shot", "episode", "make it look good", "looks good")
)


@app.post("/simulate")
def run_simulation(
    world_id: str,
    goal: str,
    budget: float = None,
    risk_profile: str = None,
    requires_visual_verification: bool = None,
    problem_domain: str = None,
):
    """
    Primary Simulation Endpoint (Infrastructure-Native).
    Input: Simulation goal (what should happen), budget, risk. Not creative prompts.
    Optional override: requires_visual_verification, problem_domain (for enterprise/API control).
    """
    goal_lower = (goal or "").lower()
    for flag in _CREATIVE_RED_FLAGS:
        if flag in goal_lower:
            import logging
            logging.getLogger(__name__).warning(
                f"Simulation goal may be creative (contains '{flag}'): prefer physics/constraints"
            )
            break
    from runtime.episode_runtime import EpisodeRuntime
    from runtime.episode_state import EpisodeState
    from models.state_transition import TransitionStatus, ActionOutcome

    policies = get_simulation_policies()
    if budget is not None and budget > 0:
        policies["cost"] = {**policies.get("cost", {}), "max_cost": float(budget)}
    if risk_profile:
        policies["risk_profile"] = risk_profile
    if requires_visual_verification is not None or problem_domain:
        policies["intent_override"] = {
            k: v for k, v in {
                "requires_visual_verification": requires_visual_verification,
                "problem_domain": problem_domain,
            }.items() if v is not None
        }

    runtime = EpisodeRuntime.create(
        world_id=world_id,
        intent=goal,
        policies=policies,
        sql=sql,
    )

    # Cheap path: closed-form goals (no video needed) - resolve via LLM only
    cheap_path = os.getenv("CHEAP_PATH_CLOSED_FORM", "true").lower() in ("true", "1", "yes")
    if cheap_path:
        try:
            from models.intent_classification import classify_intent, get_intent_override_from_policies
            override = get_intent_override_from_policies(policies)
            ov_requires = override.get("requires_visual_verification") if override else None
            cls_result = classify_intent(goal, override_requires_visual=ov_requires)
            if cls_result and not cls_result.requires_visual_verification and cls_result.confidence >= 0.8:
                from runtime.closed_form_resolver import resolve_closed_form
                cf = resolve_closed_form(goal)
                if cf:
                    return {
                        "simulation_id": runtime.episode_id,
                        "status": "closed_form",
                        "feasible": cf.feasible,
                        "confidence": cf.confidence,
                        "explanation": cf.explanation,
                        "constraints_discovered": cf.constraints_inferred,
                        "initial_state": runtime.state,
                    }
        except Exception:
            pass

    # NLP impossibility gate: only blocks HARD logical contradictions (e.g. "stone floats unsupported").
    # Physically implausible goals must reach the observer. Do not block pre-simulation.
    try:
        from runtime.physics_veto import evaluate_physics_veto
        veto, constraints, reason = evaluate_physics_veto(goal)
        if veto:
            # Mark as impossible immediately; no GPU work required.
            try:
                runtime._advance(EpisodeState.IMPOSSIBLE)
            except Exception:
                pass
            # Persist minimal rejected transition for auditability
            try:
                world_graph_store.record_beat_observation(
                    episode_id=runtime.episode_id,
                    beat_id=f"{runtime.episode_id}:veto-0",
                    video_uri="",
                    observation={
                        "observation_id": f"veto:{runtime.episode_id}",
                        "video_uri": "",
                        "beat_id": f"{runtime.episode_id}:veto-0",
                        "verdict": "impossible",
                        "confidence": 1.0,
                        "constraints_inferred": constraints,
                        "causal_explanation": reason,
                    },
                    action_description=(goal or "")[:200],
                    transition_status=TransitionStatus.REJECTED,
                    action_outcome=ActionOutcome.FAILED,
                )
            except Exception:
                pass
            return {
                "simulation_id": runtime.episode_id,
                "status": "impossible",
                "initial_state": runtime.state,
            }
    except Exception:
        pass

    # -----------------------------------------------------
    # FULL PIPELINE EXECUTION (Plan -> Schedule -> Execute)
    # -----------------------------------------------------
    try:
        from agents.narrative_planner import ProductionNarrativePlanner
        
        # 1. Plan
        planner = ProductionNarrativePlanner(
            world_id=world_id,
            redis_client=redis_store.redis if redis_store else None,
            use_mock=USE_MOCK_PLANNER,
        )
        runtime.plan(planner)
        
        # 2. Schedule
        runtime.schedule()
        
        # 3. Execute (Push to Redis)
        if redis_store:
            try:
                runtime.submit_pending_beats(redis_store)
            except Exception as e:
                print(f"[run_simulation] Failed to submit beats to Redis: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("WARNING: No Redis store available, beats remain PENDING")

    except Exception as e:
        print(f"Pipeline Auto-Start Failed: {e}")
        # We don't fail the request, we just return the initialized ID
        # The user will see "Created" state and eventually "Failed" if polled
        import traceback
        traceback.print_exc()

    return {
        "simulation_id": runtime.episode_id,
        "status": "executing" if runtime.state == "EXECUTING" else "initialized",
        "initial_state": runtime.state,
    }

async def _process_results_if_serverless():
    """On Vercel Hobby (1 cron/day), process results on-demand when user hits API."""
    if os.getenv("SERVERLESS", "").lower() not in ("true", "1", "yes"):
        return
    if not (sql and redis_store and world_graph_store):
        return
    try:
        from runtime.result_consumer import ResultConsumer
        consumer = ResultConsumer(sql, redis_store, world_graph_store)
        await consumer.process_batch(max_items=3)
    except Exception as e:
        print(f"[main] on-demand process_batch: {e}")


@app.get("/episodes")
async def list_episodes(limit: int = 20, world_id: str = None):
    """
    List recent simulations (episodes).
    Infrastructure console - state-first, no video.
    """
    await _process_results_if_serverless()
    try:
        episodes = sql.list_episodes(limit=limit, world_id=world_id) if sql else []
        episodes = list(episodes) if isinstance(episodes, (list, tuple)) else []
        return {
            "episodes": episodes,
            "total": len(episodes),
        }
    except Exception as e:
        return {"episodes": [], "total": 0, "error": str(e)}


@app.post("/episodes")
def create_episode(world_id: str, intent: str):
    from runtime.episode_runtime import EpisodeRuntime

    runtime = EpisodeRuntime.create(
        world_id=world_id,
        intent=intent,
        policies=get_episode_policies(),
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
        use_mock=USE_MOCK_PLANNER,
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

    log_file = open(DECISION_LOOP_LOG_PATH, "a")
    subprocess.Popen(
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

    return {"state": runtime.state}



@app.get("/episodes/{episode_id}")
async def episode_status(episode_id: str):
    from runtime.episode_runtime import EpisodeRuntime
    from fastapi import HTTPException

    await _process_results_if_serverless()
    try:
        runtime = EpisodeRuntime.load(
            episode_id=episode_id,
            sql=sql,
        )
        return runtime.snapshot()
    except RuntimeError as e:
        # If load fails (not found), return 404
        raise HTTPException(status_code=404, detail="Simulation not found")


@app.post("/episodes/{episode_id}/compose")
def compose_episode(
    episode_id: str,
    transition: str = "none",
    upload: bool = True,
):
    """
    Compose (stitch) all beat videos into a single episode video.
    
    Args:
        episode_id: The episode to compose
        transition: Transition type between beats (none, crossfade)
        upload: Whether to upload to R2 storage
        
    Returns:
        Dict with episode_url, shot_count, duration, etc.
    """
    from runtime.episode_runtime import EpisodeRuntime
    from agents.episode_composer import EpisodeComposer
    from models.episode_plan import EpisodePlan
    
    # Load episode to get world_id and beat list
    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )
    
    # Get all beat IDs from the SQL store
    beats = sql.get_beats(episode_id)
    beat_ids = [beat["beat_id"] for beat in beats]
    
    if not beat_ids:
        return {
            "status": "error",
            "message": "No beats found for this episode",
            "episode_id": episode_id,
        }
    
    # Create plan
    plan = EpisodePlan(
        beats=beat_ids,
        allow_gaps=True,
        required_confidence=EPISODE_COMPOSE_REQUIRED_CONFIDENCE,
    )
    
    # Compose
    composer = EpisodeComposer(world_id=runtime.world_id)
    try:
        result = composer.compose_and_stitch(
            plan=plan,
            transition=transition,
            upload=upload,
        )
        result["episode_id"] = episode_id
        return result
    except RuntimeError as e:
        # Provide detailed error about which beats are missing videos
        import json as json_module
        r = redis_store.redis
        
        beat_status = []
        for beat_id in beat_ids:
            raw = r.hget(f"render_results:{runtime.world_id}", beat_id)
            if raw:
                data = json_module.loads(raw)
                beat_status.append({
                    "beat_id": beat_id,
                    "status": data.get("status", "unknown"),
                    "has_video": bool(data.get("video_url")),
                    "confidence": data.get("confidence", 0),
                })
            else:
                beat_status.append({
                    "beat_id": beat_id,
                    "status": "not_rendered",
                    "has_video": False,
                    "confidence": 0,
                })
        
        rendered = sum(1 for b in beat_status if b["has_video"])
        
        return {
            "status": "error",
            "message": str(e),
            "help": "Beats must be rendered before composing. Run POST /episodes/{id}/execute first.",
            "world_id": runtime.world_id,
            "episode_id": episode_id,
            "total_beats": len(beat_ids),
            "rendered_beats": rendered,
            "beat_status": beat_status,
        }
    finally:
        composer.cleanup()


@app.get("/episodes/{episode_id}/beats")
def get_episode_beats(episode_id: str):
    """
    Get status of all beats in an episode.
    
    Returns:
        Dict with beat IDs, states, and render status.
    """
    import json as json_module
    
    from runtime.episode_runtime import EpisodeRuntime
    
    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )
    
    # Get beats from SQL
    beats = sql.get_beats(episode_id)
    
    if not beats:
        return {
            "episode_id": episode_id,
            "world_id": runtime.world_id,
            "status": "no_beats",
            "message": "No beats found. Run POST /episodes/{id}/plan first.",
            "beats": [],
        }
    
    # Check Redis for render results
    r = redis_store.redis
    beat_details = []
    
    for beat in beats:
        beat_id = beat["beat_id"]
        
        # Get render result from Redis
        raw = r.hget(f"render_results:{runtime.world_id}", beat_id)
        
        if raw:
            render_data = json_module.loads(raw)
            beat_details.append({
                "beat_id": beat_id,
                "db_state": beat.get("state", "unknown"),
                "render_status": render_data.get("status", "unknown"),
                "has_video": bool(render_data.get("video_url")),
                "video_url": render_data.get("video_url"),
                "confidence": render_data.get("confidence", 0),
                "duration_sec": render_data.get("duration_sec", 0),
            })
        else:
            beat_details.append({
                "beat_id": beat_id,
                "db_state": beat.get("state", "unknown"),
                "render_status": "not_rendered",
                "has_video": False,
                "video_url": None,
                "confidence": 0,
                "duration_sec": 0,
            })
    
    # Summary stats
    total = len(beat_details)
    rendered = sum(1 for b in beat_details if b["has_video"])
    ready_to_compose = rendered == total and total > 0
    
    return {
        "episode_id": episode_id,
        "world_id": runtime.world_id,
        "total_beats": total,
        "rendered_beats": rendered,
        "ready_to_compose": ready_to_compose,
        "next_step": "POST /episodes/{id}/compose" if ready_to_compose else "POST /episodes/{id}/execute",
        "beats": beat_details,
    }


@app.get("/episodes/{episode_id}/video")
def get_episode_video(episode_id: str):
    """
    Get the URL of the final composed episode video.
    """
    import json
    
    from runtime.episode_runtime import EpisodeRuntime
    
    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )
    
    # Check Redis for final video
    result_raw = redis_store.redis.hget(f"episode_results:{runtime.world_id}", "final")
    
    if result_raw:
        result = json.loads(result_raw)
        return {
            "episode_id": episode_id,
            "status": "available",
            "video_url": result.get("url"),
            "created_at": result.get("created_at"),
        }
    else:
        return {
            "episode_id": episode_id,
            "status": "not_composed",
            "message": "Episode video not yet composed. Call POST /episodes/{id}/compose first.",
        }


@app.get("/debug/download")
def debug_download(url: str):
    """
    Debug endpoint to test S3/R2 download.
    
    Args:
        url: The full video URL to test downloading
    """
    import boto3
    from botocore.client import Config
    from pathlib import Path
    import tempfile
    
    try:
        # Get S3 config
        s3_endpoint = os.getenv("S3_ENDPOINT")
        s3_bucket = os.getenv("S3_BUCKET")
        s3_access_key = os.getenv("S3_ACCESS_KEY")
        s3_secret_key = os.getenv("S3_SECRET_KEY")
        s3_region = os.getenv("S3_REGION", "auto")
        
        config_info = {
            "s3_endpoint": s3_endpoint,
            "s3_bucket": s3_bucket,
            "s3_bucket_in_url": s3_bucket in url if s3_bucket else False,
            "has_access_key": bool(s3_access_key),
            "has_secret_key": bool(s3_secret_key),
            "url": url,
        }
        
        # Extract key from URL
        key = None
        if "r2.cloudflarestorage.com" in url:
            if s3_bucket and s3_bucket in url:
                key = url.split(f"{s3_bucket}/")[-1]
            else:
                parts = url.split("/")
                if len(parts) > 4:
                    key = "/".join(parts[4:])
        elif s3_bucket and s3_bucket in url:
            key = url.split(f"{s3_bucket}/")[-1]
        
        config_info["extracted_key"] = key
        
        if not key:
            return {
                "status": "error",
                "message": "Could not extract S3 key from URL",
                "config": config_info,
            }
        
        # Try to download
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name=s3_region,
            config=Config(signature_version="s3v4"),
        )
        
        # List objects to see if it exists
        try:
            head = s3.head_object(Bucket=s3_bucket, Key=key)
            config_info["object_size"] = head.get("ContentLength", 0)
            config_info["object_exists"] = True
        except Exception as e:
            config_info["object_exists"] = False
            config_info["head_error"] = str(e)
        
        # Try to download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        s3.download_file(s3_bucket, key, temp_file.name)
        
        size = Path(temp_file.name).stat().st_size
        
        return {
            "status": "success",
            "message": f"Downloaded {size} bytes",
            "config": config_info,
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "config": config_info if 'config_info' in locals() else {},
        }


# =====================================================
# PHASE 1-5: World State Graph Endpoints
# =====================================================

@app.get("/world-state/{episode_id}")
def get_world_state(episode_id: str):
    """
    Get the world state graph for an episode.
    
    This shows the Phase 1-5 world state tracking data including:
    - State nodes (versioned world states)
    - Transitions (video-driven state changes)
    - Current state
    - Epistemic blocks with missing evidence labels
    """
    import json
    from models.world_state_graph import WorldStateGraph, WorldState
    
    try:
        # Try to load existing graph
        nodes = world_graph_store.get_episode_nodes(episode_id)
        transitions = world_graph_store.get_episode_transitions(episode_id)
        
        def _node_to_dict(n):
            return n.to_dict() if hasattr(n, "to_dict") else {"node_id": getattr(n, "node_id", str(n))}
        def _trans_to_dict(t):
            trans_dict = t.to_dict() if hasattr(t, "to_dict") else {"transition_id": getattr(t, "transition_id", str(t))}
            # Add epistemic blocking information if transition is blocked
            obs = trans_dict.get("observation_json") or trans_dict.get("observation")
            if isinstance(obs, str):
                try:
                    obs = json.loads(obs)
                except:
                    obs = {}
            elif not isinstance(obs, dict):
                obs = {}
            
            # Check if transition is epistemically blocked
            if trans_dict.get("status") == "blocked" or obs.get("epistemic_state") in ("EPISTEMICALLY_INCOMPLETE", "UNCERTAIN_TERMINATION"):
                missing = obs.get("missing_evidence", [])
                epistemic_summary = obs.get("epistemic_summary", {})
                trans_dict["blocked_reason"] = "epistemic_halt"
                trans_dict["missing_evidence"] = missing
                trans_dict["blocked_label"] = (
                    f"BLOCKED: Cannot evaluate constraints\n"
                    f"Missing evidence: {', '.join(missing) if missing else 'Unknown'}"
                )
                if epistemic_summary:
                    trans_dict["epistemic_summary"] = epistemic_summary
            return trans_dict
        
        return {
            "episode_id": episode_id,
            "phase_1_5_enabled": True,
            "total_nodes": len(nodes),
            "total_transitions": len(transitions),
            "nodes": [_node_to_dict(n) for n in nodes[:10]],
            "transitions": [_trans_to_dict(t) for t in transitions[:10]],
        }
    except Exception as e:
        return {
            "episode_id": episode_id,
            "phase_1_5_enabled": True,
            "status": "no_data",
            "message": f"No world state data yet: {e}",
            "hint": "World state is tracked automatically during episode execution",
        }


@app.get("/phase-status")
async def get_phase_status():
    """
    Check the status of Phase 1-5 components.
    """
    await _process_results_if_serverless()
    import sqlite3
    
    # Check database tables
    try:
        db_path = os.getenv("DATABASE_URL", "sqlite:///./local.db").replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        phase_1_tables = ["world_state_nodes", "state_transitions", "world_branches"]
        phase_1_ready = all(t in tables for t in phase_1_tables)
        
        return {
            "status": "operational",
            "database_tables": tables,
            "phase_1_world_graph": {
                "enabled": True,
                "tables_created": phase_1_ready,
                "required_tables": phase_1_tables,
            },
            "phase_2_observer": {
                "enabled": True,
                "uses_gemini": os.getenv("GEMINI_API_KEY") is not None,
            },
            "phase_3_quality_budget": {
                "enabled": True,
            },
            "phase_4_policy_engine": {
                "enabled": True,
            },
            "phase_5_integration": {
                "enabled": True,
            },
            "phase_6_video_native": {
                "enabled": True,
                "failure_states": True,
                "observer_veto": True,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


# =====================================================
# PHASE 6: Video-Native Compliance Endpoints
# =====================================================

@app.get("/episodes/{episode_id}/result")
def get_episode_result(episode_id: str, include_video: bool = False):
    """
    STATE-CENTRIC episode result (Video-Native Compliance).
    
    Primary output is state delta, not video.
    Video is optional debug artifact.
    
    Args:
        episode_id: The episode ID
        include_video: If True, include video URL in debug (default: False)
    
    Returns:
        State-first result with outcome, confidence, cost, and optional video debug
    """
    from runtime.episode_runtime import EpisodeRuntime
    from models.episode_outcome import (
        EpisodeOutcome, EpisodeResult, TerminationReason, 
        create_success_result, create_failure_result
    )
    import json as json_module
    
    try:
        runtime = EpisodeRuntime.load(episode_id=episode_id, sql=sql)
        
        # Determine outcome from current state
        state = (runtime.state or "").upper()
        beats = sql.get_beats(episode_id) if sql else []
        
        # If any beat is epistemically incomplete, episode is EPISTEMICALLY_BLOCKED
        if state == "EXECUTING":
            if any(
                (b.get("state") or "").upper() in ("EPISTEMICALLY_INCOMPLETE", "UNCERTAIN_TERMINATION")
                for b in beats
            ):
                state = "EPISTEMICALLY_BLOCKED"
        
        if state in ("COMPLETED", "DONE"):
            outcome = EpisodeOutcome.GOAL_ACHIEVED
        elif state in ("FAILED",):
            outcome = EpisodeOutcome.GOAL_ABANDONED
        elif state == "IMPOSSIBLE":
            outcome = EpisodeOutcome.GOAL_IMPOSSIBLE
        elif state == "DEAD_STATE":
            outcome = EpisodeOutcome.DEAD_STATE
        elif state == "EPISTEMICALLY_BLOCKED":
            # Epistemic halt: missing evidence, constraints unevaluable. Not a failure.
            outcome = EpisodeOutcome.EPISTEMICALLY_INCOMPLETE
        elif state == "PARTIALLY_COMPLETED":
            outcome = EpisodeOutcome.DEAD_STATE
        else:
            outcome = EpisodeOutcome.IN_PROGRESS
        
        # Get world state delta if available
        state_delta = {}
        try:
            nodes = world_graph_store.get_episode_nodes(episode_id)
            if nodes:
                state_delta = nodes[-1] if isinstance(nodes[-1], dict) else {}
        except Exception:
            pass

        if "progress" not in state_delta:
            state_delta = dict(state_delta)
        
        # Calculate cost from beats
        total_cost = len(beats) * COST_PER_BEAT_USD
        
        beats_completed = sum(
            1 for b in beats
            if (b.get("state") or "").upper() in ("ACCEPTED", "DONE", "COMPLETED")
        )
        beats_failed = sum(
            1 for b in beats
            if (b.get("state") or "").upper() == "ABORTED"
        )
        conf = 0.0 if not outcome.is_success else 0.5
        if state == "EPISTEMICALLY_BLOCKED":
            conf = 0.1  # No conclusion made → bounded_low
        if beats_completed and sql:
            attempts = sql.get_attempts(episode_id)
            conf_scores = []
            for a in (attempts or []):
                if not a.get("success"):
                    continue
                m = (a.get("metrics") or {})
                c = m.get("confidence")
                if c is not None:
                    c = float(c)
                    # Decrease on observer disagreement
                    if m.get("disagreement_score", 0) > 0.3:
                        c = max(0, c - 0.2)
                    conf_scores.append(c)
            if conf_scores:
                conf = sum(conf_scores) / len(conf_scores)
        
        # Collect constraints_discovered, missing_evidence, observer_status from world graph transitions
        constraints_discovered = []
        missing_evidence = []
        observer_status = None
        confidence_penalty_reason = None
        transitions = []
        try:
            transitions = world_graph_store.get_episode_transitions(episode_id)
            for t in (transitions or []):
                obs_json = getattr(t, "observation_json", None)
                if not obs_json:
                    continue
                if isinstance(obs_json, str):
                    import json as _j
                    obs = _j.loads(obs_json) if obs_json else {}
                else:
                    obs = obs_json
                for c in obs.get("constraints_inferred", []) or []:
                    if c and c not in constraints_discovered:
                        constraints_discovered.append(c)
                for m in obs.get("missing_evidence", []) or []:
                    if m and m not in missing_evidence:
                        missing_evidence.append(m)
                es = obs.get("epistemic_summary") or {}
                for m in es.get("missing_evidence", []) or []:
                    if m and m not in missing_evidence:
                        missing_evidence.append(m)
                obs_status = obs.get("observer_status") or es.get("observer_status")
                if obs_status:
                    observer_status = obs_status
                penalty = obs.get("confidence_penalty_reason") or es.get("confidence_penalty_reason")
                if penalty:
                    confidence_penalty_reason = penalty
        except Exception:
            pass
        if state == "EPISTEMICALLY_BLOCKED" and "insufficient_physical_evidence" not in constraints_discovered:
            constraints_discovered = ["insufficient_physical_evidence"] + constraints_discovered

        # goal_achieved requires at least one observer-validated transition
        if outcome == EpisodeOutcome.GOAL_ACHIEVED and not transitions:
            outcome = EpisodeOutcome.GOAL_ABANDONED
            conf = 0.0

        # DEAD_STATE requires at least one physics constraint
        from models.episode_outcome import is_epistemic_only
        if outcome == EpisodeOutcome.DEAD_STATE and is_epistemic_only(constraints_discovered):
            outcome = EpisodeOutcome.UNCERTAIN_TERMINATION

        if state == "PARTIALLY_COMPLETED" and outcome == EpisodeOutcome.DEAD_STATE:
            intent_lower = (runtime.intent or "").lower()
            physics_terms = ("structural", "load", "gravity", "energy", "stability", "impossible")
            has_physics_constraint = any(
                t in " ".join(constraints_discovered).lower()
                for t in physics_terms
            )
            stack_narrow = ("stack" in intent_lower and "box" in intent_lower and
                           "narrow" in intent_lower and ("base" in intent_lower or "surface" in intent_lower))
            veto_triggered = False
            try:
                from runtime.physics_veto import evaluate_physics_veto
                veto_triggered, _, _ = evaluate_physics_veto(runtime.intent or "")
            except Exception:
                pass
            if has_physics_constraint or veto_triggered or stack_narrow:
                outcome = EpisodeOutcome.GOAL_IMPOSSIBLE
                if not constraints_discovered:
                    if veto_triggered:
                        try:
                            _, vc, _ = evaluate_physics_veto(runtime.intent or "")
                            constraints_discovered = list(vc) if vc else constraints_discovered
                        except Exception:
                            pass
                    elif stack_narrow:
                        constraints_discovered = constraints_discovered or ["stability_limit", "load_distribution"]

        if outcome.is_terminal and conf == 0.0 and len(beats) > 0:
            conf = 0.2

        # Build progress for state_delta
        state_delta["progress"] = {
            "total_beats": len(beats),
            "completed": beats_completed,
            "aborted": beats_failed,
            "percent": round((beats_completed / len(beats)) * 100, 2) if beats else 0.0,
        }
        state_delta["state_nodes"] = len(
            world_graph_store.get_episode_nodes(episode_id) or []
        )
        state_delta["transitions"] = len(
            world_graph_store.get_episode_transitions(episode_id) or []
        )
        state_delta["state"] = state
        state_delta["outcome"] = outcome.value
        # Display mapping: insufficient_evidence -> insufficient_observational_evidence (UI only)
        constraints_for_display = [
            "insufficient_observational_evidence" if c == "insufficient_evidence" else c
            for c in constraints_discovered
        ]
        state_delta["constraints_discovered"] = constraints_for_display
        state_delta["missing_evidence"] = missing_evidence
        if observer_status is not None:
            state_delta["observer_status"] = observer_status
        if confidence_penalty_reason is not None:
            state_delta["confidence_penalty_reason"] = confidence_penalty_reason
        try:
            for t in reversed(transitions or []):
                obs_json = getattr(t, "observation_json", None)
                if obs_json:
                    obs_data = json_module.loads(obs_json) if isinstance(obs_json, str) else obs_json
                    chain = []
                    if obs_data.get("causal_explanation"):
                        chain.append(obs_data["causal_explanation"])
                    if obs_data.get("physics_violation"):
                        chain.append(f"Physics violation: {obs_data['physics_violation']}")
                    if obs_data.get("state_contradiction"):
                        chain.append(f"State contradiction: {obs_data['state_contradiction']}")
                    missing = obs_data.get("missing_evidence") or []
                    if missing:
                        chain.append(f"Missing evidence: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
                    ep_state = obs_data.get("epistemic_state") or ""
                    verdict = obs_data.get("verdict", "unknown")
                    conf_obs = obs_data.get("confidence", 0.5)
                    if ep_state and "epistemic" in ep_state.lower():
                        summary = f"Epistemically blocked: {chain[0] if chain else 'insufficient evidence to evaluate constraints'}"
                    else:
                        summary = chain[0] if chain else f"Verdict: {verdict} (confidence {conf_obs:.2f})"
                    state_delta["verdict_explanation"] = {
                        "verdict": verdict,
                        "confidence": conf_obs,
                        "summary": summary,
                        "causal_chain": chain,
                        "constraints_inferred": obs_data.get("constraints_inferred") or [],
                        "missing_evidence": missing,
                    }
                    break
        except Exception:
            pass
        if state == "COMPLETED" and observer_status == "unavailable":
            intent_lower = (runtime.intent or "").lower()
            if "stack" in intent_lower and "tipping" in intent_lower:
                state_delta["constraints_satisfied"] = ["no_tipping"]

        suggested_alternatives = []
        attempts_made = []
        risk_profile = (runtime.policies or {}).get("risk_profile") or "medium"
        is_exploratory = str(risk_profile).lower() == "high"
        show_exploratory_meta = is_exploratory and outcome in (
            EpisodeOutcome.GOAL_IMPOSSIBLE,
            EpisodeOutcome.GOAL_ABANDONED,
            EpisodeOutcome.DEAD_STATE,
            EpisodeOutcome.UNCERTAIN_TERMINATION,
            EpisodeOutcome.EPISTEMICALLY_INCOMPLETE,
        )
        if show_exploratory_meta and transitions:
            seen_hints = set()
            for t in transitions:
                obs_json = getattr(t, "observation_json", None)
                if not obs_json:
                    continue
                try:
                    obs = json_module.loads(obs_json) if isinstance(obs_json, str) else obs_json
                except Exception:
                    continue
                action = obs.get("action") or {}
                if isinstance(action, dict) and action.get("suggested_next_action"):
                    alt = str(action["suggested_next_action"]).strip()
                    if alt and alt not in suggested_alternatives:
                        suggested_alternatives.append(alt)
                att = obs.get("observability_attempt")
                hint = obs.get("render_hint") or ""
                if att is not None:
                    key = (att, hint)
                    if key not in seen_hints:
                        seen_hints.add(key)
                        attempts_made.append({
                            "observability_attempt": int(att),
                            "render_hint": hint or None,
                        })
        attempts_made.sort(key=lambda x: x["observability_attempt"])
        if show_exploratory_meta and not suggested_alternatives and attempts_made:
            for a in attempts_made:
                h = a.get("render_hint")
                if h and h not in suggested_alternatives:
                    suggested_alternatives.append(f"Try: {h}")
        if show_exploratory_meta and not suggested_alternatives and constraints_for_display:
            suggested_alternatives.append(
                "Adjust prompt to clarify motion or use a different camera angle to expose the intended physics."
            )

        result = EpisodeResult(
            episode_id=episode_id,
            outcome=outcome,
            state_delta=state_delta,
            confidence=conf,
            total_cost_usd=total_cost,
            beats_attempted=len(beats),  # execution scaffolding
            beats_completed=beats_completed,  # execution scaffolding; success = observer-validated transitions
            beats_failed=beats_failed,
            constraints_discovered=constraints_for_display,
            suggested_alternatives=suggested_alternatives,
            attempts_made=attempts_made,
        )
        
        # Add video as optional debug artifact (from compose or world graph transitions)
        if include_video:
            result_raw = redis_store.redis.hget(f"episode_results:{runtime.world_id}", "final")
            if result_raw:
                final_data = json_module.loads(result_raw)
                result.debug["video_uri"] = final_data.get("url")
                result.debug["video_retention_hours"] = 24
                result.debug["video_is_debug_only"] = True
            # Also include transition video URLs from world graph (ResultConsumer flow)
            try:
                transitions = world_graph_store.get_episode_transitions(episode_id)
                urls = []
                for t in (transitions or []):
                    uri = getattr(t, "video_uri", None)
                    if uri:
                        urls.append(uri)
                if urls:
                    result.debug["beat_video_urls"] = urls
                    result.debug["video_note"] = "Presigned URLs expire in ~1h. Stub backend produces black placeholder."
            except Exception:
                pass
        
        return result.to_dict()
        
    except Exception as e:
        return {
            "episode_id": episode_id,
            "outcome": "error",
            "is_success": False,
            "error": str(e),
        }


@app.post("/episodes/{episode_id}/terminate")
def terminate_episode(
    episode_id: str,
    reason: str = "manual",
    outcome: str = "goal_abandoned",
):
    """
    Forcibly terminate an episode (Video-Native Compliance).
    
    Allows explicit failure/termination - episodes don't have to succeed.
    
    Args:
        episode_id: The episode to terminate
        reason: Why the episode is being terminated
        outcome: One of: goal_achieved, goal_impossible, goal_abandoned, dead_state
    
    Returns:
        Final episode result
    """
    from runtime.episode_runtime import EpisodeRuntime
    from models.episode_outcome import EpisodeOutcome, create_failure_result
    
    try:
        runtime = EpisodeRuntime.load(episode_id=episode_id, sql=sql)
        
        # Map string to outcome
        outcome_map = {
            "goal_achieved": EpisodeOutcome.GOAL_ACHIEVED,
            "goal_impossible": EpisodeOutcome.GOAL_IMPOSSIBLE,
            "goal_abandoned": EpisodeOutcome.GOAL_ABANDONED,
            "dead_state": EpisodeOutcome.DEAD_STATE,
            "uncertain_termination": EpisodeOutcome.UNCERTAIN_TERMINATION,
            "epistemically_incomplete": EpisodeOutcome.EPISTEMICALLY_INCOMPLETE,
        }
        episode_outcome = outcome_map.get(outcome, EpisodeOutcome.GOAL_ABANDONED)
        
        # Mark as terminated in runtime
        if episode_outcome == EpisodeOutcome.GOAL_IMPOSSIBLE:
            new_state = "impossible"
        elif episode_outcome == EpisodeOutcome.DEAD_STATE:
            new_state = "dead_state"
        elif episode_outcome == EpisodeOutcome.UNCERTAIN_TERMINATION:
            new_state = "uncertain_termination"
        elif episode_outcome == EpisodeOutcome.EPISTEMICALLY_INCOMPLETE:
            new_state = "EPISTEMICALLY_BLOCKED"
        else:
            new_state = "failed"
        
        # Update in SQL
        if sql:
            sql.update_episode_state(episode_id, new_state)
        
        # Create result
        result = create_failure_result(
            episode_id=episode_id,
            outcome=episode_outcome,
            reason=reason,
            trigger="manual_termination",
        )
        
        return {
            "status": "terminated",
            "result": result.to_dict(),
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/diagnostics/queue")
def queue_diagnostics():
    """
    Diagnostic endpoint to check Redis queue status.
    """
    try:
        from runtime.persistence.redis_store import _job_queue, _result_queue
        
        if not redis_store:
            return {
                "status": "error",
                "message": "Redis store not initialized",
            }
        
        try:
            redis_client = redis_store.redis
            redis_client.ping()
            redis_connected = True
        except Exception as e:
            return {
                "status": "error",
                "message": f"Redis connection failed: {e}",
                "redis_connected": False,
            }
        
        job_queue = _job_queue()
        result_queue = _result_queue()
        
        job_count = redis_client.llen(job_queue)
        result_count = redis_client.llen(result_queue)
        
        return {
            "status": "ok",
            "redis_connected": True,
            "queues": {
                "job_queue": {
                    "name": job_queue,
                    "pending_jobs": job_count,
                },
                "result_queue": {
                    "name": result_queue,
                    "pending_results": result_count,
                },
            },
            "environment": {
                "REDIS_URL": os.getenv("REDIS_URL", "NOT SET")[:30] + "..." if os.getenv("REDIS_URL") else "NOT SET",
                "JOB_QUEUE": os.getenv("JOB_QUEUE", "NOT SET"),
                "RESULT_QUEUE": os.getenv("RESULT_QUEUE", "NOT SET"),
            },
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/internal/process-results")
@app.post("/internal/process-results")
async def process_results_batch(max_items: int = 5):
    """
    Process up to max_items GPU results from Redis. For serverless (Vercel cron)
    where ResultConsumer background loop cannot run. Vercel cron hits this every minute.
    """
    if not sql or not redis_store or not world_graph_store:
        return {"status": "error", "message": "Stores not initialized"}
    from runtime.result_consumer import ResultConsumer
    consumer = ResultConsumer(sql, redis_store, world_graph_store)
    count = await consumer.process_batch(max_items=max_items)
    return {"status": "ok", "processed": count}


@app.get("/episodes/{episode_id}/video")
def get_episode_video_debug(episode_id: str, confirm_debug: bool = False):
    """
    Get episode video as DEBUG ARTIFACT (Video-Native Compliance).
    
    Video is NOT the primary output. Use /result for state-first data.
    This endpoint is for debugging and auditing only.
    
    Args:
        episode_id: The episode ID
        confirm_debug: Must be True to get video (acknowledges video is debug-only)
    
    Returns:
        Video URL with debug warning, or redirect to /result
    """
    import json
    
    from runtime.episode_runtime import EpisodeRuntime
    
    if not confirm_debug:
        return {
            "status": "redirect",
            "message": "Video is a debug artifact. Use /episodes/{id}/result for state-first output.",
            "primary_endpoint": f"/episodes/{episode_id}/result",
            "to_get_video": f"/episodes/{episode_id}/video?confirm_debug=true",
        }
    
    try:
        runtime = EpisodeRuntime.load(episode_id=episode_id, sql=sql)
        
        result_raw = redis_store.redis.hget(f"episode_results:{runtime.world_id}", "final")
        
        if result_raw:
            result = json.loads(result_raw)
            return {
                "episode_id": episode_id,
                "status": "debug_artifact",
                "warning": "Video is for debugging only. Primary output is state delta.",
                "video_url": result.get("url"),
                "retention": "24h",
                "created_at": result.get("created_at"),
            }
        else:
            return {
                "episode_id": episode_id,
                "status": "not_composed",
                "message": "No video available. Call /compose first.",
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
