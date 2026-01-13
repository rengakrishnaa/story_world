import os
import importlib
redis = importlib.import_module("redis")

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import uuid
import json
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Redis connection (localhost:6379 for dev, env var for prod)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

@dataclass
class CostTracker:
    gemini_calls: float = 0.0
    nanobanana_calls: float = 0.0
    total_cost: float = 0.0
    
    def add_gemini_call(self, tokens: float = 1000) -> float:
        cost = tokens / 1_000_000 * 0.25
        self.gemini_calls += tokens
        self.total_cost += cost
        return cost
    
    def add_nanobanana_call(self) -> float:
        cost = 0.05
        self.nanobanana_calls += 1
        self.total_cost += cost
        return cost
    
    def add_veo_cost(self, duration_sec: float) -> float:
        cost = duration_sec * 0.008
        self.total_cost += cost
        return cost

@dataclass
class ShotData:
    shot_id: str
    beat: Dict[str, Any]
    estimated_duration_sec: float = 5.0
    status: str = "pending"
    created: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

class ProductionEpisodeAssembler:
    def __init__(self, world_id: str, output_dir: Path = Path("outputs")):
        self.world_id = world_id
        self.shots: List[ShotData] = []
        self.cost_tracker = CostTracker()
        self.output_dir = output_dir / world_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.queue_key = f"render_queue:{world_id}"
        
    def add_shot(self, shot_data: Dict[str, Any]) -> str:
        shot_id = shot_data.get("shot_id", str(uuid.uuid4()))
        beat = shot_data.get("beat", {})
        duration = beat.get("estimated_duration_sec", 5.0)
        
        shot = ShotData(shot_id=shot_id, beat=beat, estimated_duration_sec=duration)
        self.shots.append(shot)
        logger.info(f"Added shot {shot_id}: {beat.get('description', 'no desc')}")
        return shot_id
    
    def queue_shot_render(
        self,
        world_id: str,
        beat: Dict[str, Any],
        priority: str = "medium"
    ) -> str:
        job_id = str(uuid.uuid4())
        duration = beat.get("estimated_duration_sec", 5.0)

        # 1️⃣ Register locally
        shot = ShotData(
            shot_id=beat.get("id", job_id),
            beat=beat,
            estimated_duration_sec=duration,
            status="queued"
        )
        self.shots.append(shot)

        # ✅ NEW: register beat order for episode composition
        r.rpush(
            f"episode_beats:{world_id}",
            beat["id"]
        )

        # 2️⃣ Push to Redis render queue
        job = {
            "job_id": job_id,
            "world_id": world_id,
            "beat": beat,
            "priority": priority,
            "status": "queued",
            "created": pd.Timestamp.now().isoformat()
        }

        r.lpush(self.queue_key, json.dumps(job))

        logger.info(
            f"Queued shot {shot.shot_id} | Redis job {job_id}"
        )

        return job_id


    
    def get_queue_status(self) -> Dict[str, Any]:
        """Real Redis queue length."""
        queued_count = r.llen(self.queue_key)
        return {
            "queued_jobs": queued_count,
            "world_id": self.world_id,
            "queue_key": self.queue_key
        }
    
    def build_preview(self) -> Dict[str, Any]:
        total_duration = sum(shot.estimated_duration_sec for shot in self.shots)
        gemini_cost = self.cost_tracker.gemini_calls / 1_000_000 * 0.25
        nanobanana_cost = self.cost_tracker.nanobanana_calls * 0.05
        veo_cost = total_duration * 0.008
        total_cost = gemini_cost + nanobanana_cost + veo_cost
        
        queued_count = r.llen(self.queue_key)
        
        preview = {
            "total_duration_sec": total_duration,
            "shot_count": len(self.shots),
            "queued_shots": queued_count,
            "preview_shots": [
                {"shot_id": s.shot_id, "description": s.beat.get("description", ""), "duration": s.estimated_duration_sec}
                for s in self.shots[:8]
            ],
            "cost_breakdown": {
                "gemini": f"${gemini_cost:.3f}",
                "nanobanana": f"${nanobanana_cost:.3f}",
                "veo_estimated": f"${veo_cost:.3f}",
                "total": f"${total_cost:.2f}"
            },
            "status": f"Storyboard ready - {queued_count} shots in Redis queue",
            "world_id": self.world_id,
            "redis_queue": self.queue_key
        }
        
        preview_path = self.output_dir / "episode_preview.json"
        with open(preview_path, "w") as f:
            json.dump(preview, f, indent=2, default=str)
        
        logger.info(f"Preview saved: {preview_path} | Queue: {queued_count}")
        return preview

# Global registry
_world_assemblers: Dict[str, ProductionEpisodeAssembler] = {}

def get_assembler(world_id: str) -> ProductionEpisodeAssembler:
    if world_id not in _world_assemblers:
        _world_assemblers[world_id] = ProductionEpisodeAssembler(world_id)
    return _world_assemblers[world_id]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-id", default="demo")
    parser.add_argument("--add-shot", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    
    assembler = get_assembler(args.world_id)
    
    if args.add_shot:
        beat = {"description": "Hero walks through futuristic city", "camera": "wide shot", "estimated_duration_sec": 8.0}
        assembler.add_shot({"beat": beat})
        assembler.queue_shot_render(args.world_id, beat)
    
    if args.preview:
        preview = assembler.build_preview()
        print(json.dumps(preview, indent=2))
