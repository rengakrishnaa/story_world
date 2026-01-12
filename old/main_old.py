import os
import asyncio
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import redis
from runtime.registry import RuntimeRegistry
from runtime.persistence.sql_store import SQLStore
from runtime.persistence.redis_store import RedisStore

# Load YOUR .env
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = RedisStore(REDIS_URL)

# Import ALL your cleaned modules
from agents.episode_assembler import get_assembler  # [file:28]
from agents.narrative_planner import ProductionNarrativePlanner  # [file:29]
from agents.shot_renderer import ProductionShotRenderer  # [file:30]

app = FastAPI(title="StoryWorld Engine v2 - Production")

sql = SQLStore()

registry = RuntimeRegistry()

# Production directories
for dir_path in ["outputs", "keyframes", "thumbnails", "videos", "uploads"]:
    Path(dir_path).mkdir(exist_ok=True)

# Static file serving
app.mount("/static", StaticFiles(directory="."), name="static")
app.mount("/keyframes", StaticFiles(directory="keyframes"), name="keyframes")
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

logger = lambda msg: print(f"🚀 {msg}")

# Global instances (production registry)
_world_registry = {}

def get_world(world_id: str) -> Dict:  # ✅ SYNC - NO async!
    """Production world registry - SYNC version."""
    if world_id not in _world_registry:
        from agents.episode_assembler import get_assembler
        from agents.narrative_planner import ProductionNarrativePlanner  
        from agents.shot_renderer import ProductionShotRenderer
        
        _world_registry[world_id] = {
            "assembler": get_assembler(world_id),
            "planner": ProductionNarrativePlanner(world_id),
            "renderer": ProductionShotRenderer(world_id)  # ✅ SYNC constructors
        }
    return _world_registry[world_id]

## 🚀 PRODUCTION API ENDPOINTS

@app.get("/")
async def root():
    return {"message": "StoryWorld Engine v2 - Upload script → Auto anime episode"}

@app.post("/create-episode")
async def create_episode(world_id: str = Form("demo"), script: str = Form("Saitama fights monster")):
    """FULL END-TO-END: Script → Plan → Queue → Render."""
    world = get_world(world_id)
    
    # 1. Plan episode
    demo_world = {
        "characters": [{"name": "Saitama", "description": "bald hero"}, 
                      {"name": "Genos", "description": "cyborg disciple"}],
        "locations": [{"name": "rooftop", "description": "ruined city skyline"}]
    }
    
    plan = world["planner"].plan_episode(demo_world, script)
    logger(f"📖 Planned: {plan.title}")
    
    # 2. Queue ALL shots
    assembler = world["assembler"]
    for act in plan.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                beat_data = {
                    "id": beat.id,
                    "description": beat.description,
                    "characters": beat.characters,
                    "location": beat.location,
                    "estimated_duration_sec": beat.estimated_duration_sec
                }
                assembler.queue_shot_render(world_id, beat_data)
    
    preview = assembler.build_preview()
    logger(f"🎬 Queued {preview['shot_count']} shots")
    
    return JSONResponse({
        "world_id": world_id,
        "plan": plan.to_dict(),
        "preview": preview,
        "status": f"{preview['shot_count']} shots queued! Run workers.",
        "next_step": f"/progress/{world_id}"
    })

## ADD THIS ENDPOINT after the /create-episode endpoint:

@app.post("/render/beat")
async def render_beat(world_json: str = Form(...), beat_json: str = Form(...)):
    """Manual single beat render (for testing)."""
    world = get_world("demo")  # Use demo world
    
    # Parse inputs
    world_data = json.loads(world_json)
    beat_data = json.loads(beat_json)
    
    # Render keyframe + video
    renderer = world["renderer"]
    keyframe_url = renderer.render_beat_keyframe(None,beat_data)
    
    return JSONResponse({
        "status": "rendered",
        "keyframe_url": keyframe_url,
        "video_url": None,  # Add video_url when ready
        "next_step": "Check /static/keyframe.png"
    })

@app.get("/progress/{world_id}")
async def progress(world_id: str):
    """Real-time render progress."""
    if world_id not in _world_registry:
        return JSONResponse({"error": "World not found"}, status_code=404)
    
    assembler = get_assembler(world_id)
    preview = assembler.build_preview()
    
    # Count completed renders
    results = {}
    for beat_id in [shot.shot_id for shot in assembler.shots]:
        result = r.hget(f"render_results:{world_id}", beat_id)
        if result:
            results[beat_id] = json.loads(result)
    
    return JSONResponse({
        "world_id": world_id,
        "queued": preview["queued_shots"],
        "completed": len(results),
        "total": preview["shot_count"],
        "progress": f"{len(results)/preview['shot_count']*100:.1f}%",
        "sample_video": list(results.values())[-1].get("video_url") if results else None,
        "cost": preview["cost_breakdown"]
    })

@app.get("/preview/{world_id}")
async def episode_preview(world_id: str):
    """Episode preview with sample shots."""
    assembler = get_assembler(world_id)
    preview = assembler.build_preview()
    return JSONResponse(preview)

## 🔥 PRODUCTION CLI

def run_pipeline(world_id: str = "demo", script: str = "Saitama vs monster"):
    """CLI: End-to-end pipeline."""
    print(f"🎬 StoryWorld Pipeline: {world_id}")
    
    # Step 1: Plan + Queue
    world = asyncio.run(asyncio.wait_for(get_world(world_id), timeout=10))
    demo_world = {"characters": ["Saitama"], "locations": ["rooftop"]}
    
    plan = world["planner"].plan_episode(demo_world, script)
    print(f"📖 Plan: {plan.title} ({plan.total_duration_min}min)")
    
    # Queue shots
    for act in plan.acts:
        print(f"  Act: {act.name}")
        for scene in act.scenes[:2]:  # First 2 scenes for demo
            for beat in scene.beats:
                world["assembler"].queue_shot_render(world_id, {
                    "id": beat.id, "description": beat.description,
                    "estimated_duration_sec": beat.estimated_duration_sec
                })
    
    print(f"✅ Queued {len(plan.acts[0].scenes[0].beats)} shots")
    print("Run: python shot_renderer.py worker --world", world_id)

def start_workers(world_id: str = "demo", count: int = 2):
    """Spawn worker processes."""
    import multiprocessing
    for i in range(count):
        p = multiprocessing.Process(target=lambda: subprocess.run([
            "python", "shot_renderer.py", "worker", "--world-id", world_id
        ]))
        p.start()
    print(f"🚀 Started {count} workers for {world_id}")

def serve_api(host: str = "0.0.0.0", port: int = 8000):
    """Production FastAPI server."""
    uvicorn.run(app, host=host, port=port, log_level="info")

## MAIN ENTRYPOINT
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py [run|worker|api|demo] [world_id]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "run":
        world_id = sys.argv[2] if len(sys.argv) > 2 else "demo"
        run_pipeline(world_id)
    
    elif cmd == "demo":
        run_pipeline("anime_demo", "Saitama one-punches monster into space")
    
    elif cmd == "worker":
        from agents.shot_renderer import worker_main
        worker_main()
    
    elif cmd == "api":
        serve_api()
    
    elif cmd == "workers":
        world_id = sys.argv[2] if len(sys.argv) > 2 else "demo"
        start_workers(world_id)
