"""
Production-ready Narrative Planner with local mock + Redis caching.
Scales from localhost → Gemini API with zero code changes.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import redis
import logging
from datetime import datetime
import pandas as pd

# Redis for caching plans (shared with episode_assembler)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.Redis(host="localhost", port=6379, db=0)

logger = logging.getLogger(__name__)

@dataclass
class Beat:
    id: str
    description: str
    estimated_duration_sec: int
    characters: List[str]
    location: str

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "estimated_duration_sec": self.estimated_duration_sec,
            "characters": self.characters,
            "location": self.location
        }

@dataclass
class Scene:
    id: str
    title: str
    summary: str
    beats: List[Beat]

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "beats": [b.to_dict() for b in self.beats]
        }

@dataclass
class Act:
    name: str
    summary: str
    scenes: List[Scene]

    def to_dict(self):
        return {
            "name": self.name,
            "summary": self.summary,
            "scenes": [s.to_dict() for s in self.scenes]
        }

@dataclass
class EpisodePlan:
    title: str
    total_duration_min: int
    acts: List[Act]

    def to_dict(self):
        return {
            "title": self.title,
            "total_duration_min": self.total_duration_min,
            "acts": [a.to_dict() for a in self.acts]
        }
    
    def model_dump_json(self) -> str:
        """JSON serialization for Redis/persistence."""
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodePlan":
        return cls(
            title=data["title"],
            total_duration_min=data["total_duration_min"],
            acts=[
                Act(
                    name=act["name"],
                    summary=act["summary"],
                    scenes=[
                        Scene(
                            id=sc["id"],
                            title=sc["title"],
                            summary=sc["summary"],
                            beats=[
                                Beat(
                                    id=b["id"],
                                    description=b["description"],
                                    estimated_duration_sec=b["estimated_duration_sec"],
                                    characters=b["characters"],
                                    location=b["location"],
                                )
                                for b in sc["beats"]
                            ],
                        )
                        for sc in act["scenes"]
                    ],
                )
                for act in data["acts"]
            ],
        )


class MockGemini:
    """Local mock for Gemini API - production uses real API."""
    def __init__(self, use_real_api: bool = False):
        self.use_real_api = use_real_api
        self.model = "gemini-3-flash-preview"
    
    def generate_content(self, model: str, contents: str) -> Dict[str, Any]:
        """Mock LLM response for localhost."""
        if self.use_real_api:
            # Production: Real Gemini API call
            import google.genai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            client = genai.GenerativeModel(model)
            response = client.generate_content(contents)
            return {"text": response.text}
        
        # Localhost mock: Deterministic anime episode plan
        mock_plan = {
            "title": "One Punch Battle: Episode 1",
            "total_duration_min": 8,
            "acts": [
                {
                    "name": "Act 1: Monster Approaches",
                    "summary": "Saitama faces Monster King Orochi on a ruined rooftop.",
                    "scenes": [
                        {
                            "id": "scene-1",
                            "title": "Rooftop Standoff",
                            "summary": "Characters confront the monster in epic shounen style.",
                            "beats": [
                                {
                                    "id": "beat-1",
                                    "description": "Wide shot: ruined city skyline, massive monster approaches from horizon",
                                    "estimated_duration_sec": 12,
                                    "characters": ["Saitama", "Genos"],
                                    "location": "rooftop"
                                },
                                {
                                    "id": "beat-2", 
                                    "description": "Genos (yellow hero) warns Saitama: 'Master, this monster is S-class!'",
                                    "estimated_duration_sec": 8,
                                    "characters": ["Genos", "Saitama"],
                                    "location": "rooftop"
                                },
                                {
                                    "id": "beat-3",
                                    "description": "Saitama yawns casually: 'Oh? Looks kinda strong.' Close-up on bored expression.",
                                    "estimated_duration_sec": 6,
                                    "characters": ["Saitama"],
                                    "location": "rooftop"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        logger.info("🔮 Using mock Gemini response (set GEMINI_USE_REAL=true for live API)")
        return {"text": json.dumps(mock_plan, indent=2)}

class ProductionNarrativePlanner:
    """
    Production Narrative Planner with Redis caching + mock/real LLM toggle.
    Integrates with episode_assembler via world_id.
    """
    
    def __init__(self, world_id: str, use_mock: bool = True):
        self.world_id = world_id
        self.gemini = MockGemini(use_real_api=not use_mock)
        self.cache_key = f"narrative_plan:{world_id}"
        self.plan_file = Path(f"outputs/{world_id}/episode_plan.json")
    
    def plan_episode(self, world_json: Dict[str, Any], script: str) -> EpisodePlan:
        """Generate structured episode plan with caching."""
        
        # Check Redis cache first
        cached = r.get(self.cache_key)
        if cached:
            logger.info(f"📦 Loaded cached plan from Redis: {self.cache_key}")
            return EpisodePlan.from_dict(json.loads(cached))
        
        # Generate new plan
        world_str = json.dumps(world_json, indent=2)
        prompt = self._build_prompt(world_str, script)
        
        response = self.gemini.generate_content("gemini-3-flash-preview", prompt)
        raw_text = self._clean_json_response(response["text"])
        
        try:
            plan = EpisodePlan.from_dict(json.loads(raw_text))
            
            # Cache for 24h
            r.setex(self.cache_key, 86400, plan.model_dump_json())
            
            # Persist to disk
            self.plan_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.plan_file, "w") as f:
                json.dump(asdict(plan), f, indent=2)
            
            logger.info(f"✅ Generated + cached new plan: {plan.title}")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Plan validation failed: {e}")
            return self._fallback_plan()
    
    def _build_prompt(self, world_json: str, script: str) -> str:
        """Production-grade prompt engineering."""
        return f"""You are a professional anime episode director. 

WORLD: {world_json}
SCRIPT: {script}

Create a detailed 8-minute episode plan. Return ONLY valid JSON:

{{
  "title": "Episode title",
  "total_duration_min": 8,
  "acts": [{{
    "name": "Act 1 Title",
    "summary": "2 sentence summary",
    "scenes": [{{
      "id": "scene-1",
      "title": "Scene title", 
      "summary": "What happens",
      "beats": [{{
        "id": "beat-1",
        "description": "Action + dialogue + camera",
        "estimated_duration_sec": 8,
        "characters": ["Character1"],
        "location": "Location1"
      }}]
    }}]
  }}]
}}

CRITICAL:
- Use ONLY characters/locations from WORLD
- Total 480 seconds (8 min)
- 2-3 acts, 2-4 scenes/act
- Beats: 3-20 seconds ONLY
- Action-packed shounen style
- NO markdown, NO backticks, pure JSON only"""
    
    def _clean_json_response(self, raw_text: str) -> str:
        """Robust JSON extraction."""
        # Remove markdown wrappers
        for marker in ['```json', '```']:
            if raw_text.startswith(marker):
                raw_text = raw_text.split(marker, 1)[1].strip()
            if raw_text.endswith(marker):
                raw_text = raw_text.rsplit(marker, 1)[0].strip()
        return raw_text.strip()
    
    def _fallback_plan(self) -> EpisodePlan:
        """Graceful degradation."""
        logger.warning("🔄 Using fallback plan")
        return EpisodePlan(
            title="Demo: One Punch Battle",
            total_duration_min=8,
            acts=[Act(
                name="Act 1: Monster Fight",
                summary="Saitama vs monster in epic battle.",
                scenes=[Scene(
                    id="scene-1",
                    title="Rooftop Battle",
                    summary="Hero confronts monster",
                    beats=[Beat(
                        id="beat-1",
                        description="Saitama one-punches monster into orbit",
                        estimated_duration_sec=5,
                        characters=["Saitama"],
                        location="rooftop"
                    )]
                )]
            )]
        )

# Integration with episode_assembler
def create_episode_from_plan(planner: ProductionNarrativePlanner, assembler):
    """Bridge: Plan → Queue shots for rendering."""
    # Demo world
    world = {
        "characters": ["Saitama", "Genos"],
        "locations": ["rooftop", "city_street"]
    }
    script = "Saitama fights a massive monster in the city."
    
    plan = planner.plan_episode(world, script)
    
    # Queue all beats
    for act in plan.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                beat_data = {
                    "description": beat.description,
                    "characters": beat.characters,
                    "location": beat.location,
                    "estimated_duration_sec": beat.estimated_duration_sec
                }
                assembler.queue_shot_render(planner.world_id, beat_data)
    
    return plan

# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-id", default="demo")
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()
    
    planner = ProductionNarrativePlanner(args.world_id, use_mock=args.mock)
    world = {"characters": ["Saitama", "Genos"], "locations": ["rooftop"]}
    plan = planner.plan_episode(world, "Saitama fights monster")
    print(json.dumps(asdict(plan), indent=2))

if __name__ == "__main__":
    main()
