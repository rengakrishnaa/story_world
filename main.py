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

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    runtime.schedule()

    log_file = open("decision_loop.log", "a")
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
def episode_status(episode_id: str):
    from runtime.episode_runtime import EpisodeRuntime

    runtime = EpisodeRuntime.load(
        episode_id=episode_id,
        sql=sql,
    )

    return runtime.snapshot()


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
        required_confidence=0.3,
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
