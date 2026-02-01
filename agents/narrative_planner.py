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
import hashlib
import pandas as pd

try:
    from models.world import WorldGraph
except ImportError:
    WorldGraph = None



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
        """Return episode plan text from real Gemini or a local mock.

        In mock mode or when Gemini is unavailable, we generate a simple
        plan that actually reflects the user's script instead of using
        a hardcoded One Punch demo.
        """
        # Try to extract the SCRIPT: ... section from the prompt
        script_text = ""
        marker = "SCRIPT:"
        idx = contents.find(marker)
        if idx != -1:
            script_part = contents[idx + len(marker) :]
            sep = "\n\n"
            end_idx = script_part.find(sep)
            if end_idx != -1:
                script_text = script_part[:end_idx].strip()
            else:
                script_text = script_part.strip()
        else:
            script_text = contents.strip()

        if not self.use_real_api:
            logger.warning("🧪 Narrative planner running in MOCK mode")
            return {"text": json.dumps(self._get_mock_payload(script_text))}

        if self.use_real_api:
            # Production: Real Gemini API call with retry and fallback logic
            from google import genai
            import time
            import random
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("❌ GEMINI_API_KEY missing, using mock data")
                return {"text": json.dumps(self._get_mock_payload(script_text))}

            client = genai.Client(api_key=api_key)
            
            # Use a more available model by default
            target_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔮 Calling Gemini API ({target_model}) - attempt {attempt + 1}")
                    response = client.models.generate_content(
                        model=target_model,
                        contents=contents
                    )
                    return {"text": response.text}
                except Exception as e:
                    if "503" in str(e) or "overloaded" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) + random.random()
                            logger.warning(f"⚠️ Gemini overloaded (503). Retrying in {wait_time:.2f}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error("❌ Gemini failed after retries. Falling back to mock data.")
                    else:
                        logger.error(f"❌ Gemini API error: {e}")
                        break
        
        # Fallback to mock data if real API failed or is disabled
        return {"text": json.dumps(self._get_mock_payload(script_text))}

    def _get_mock_payload(self, script: str) -> Dict[str, Any]:
        """Provides a simple, intent-driven mock episode plan for fallback."""
        base_description = script or "Generic anime scene"
        title = (base_description[:40] + "...") if len(base_description) > 40 else base_description

        return {
            "title": f"{title} - Episode 1",
            "total_duration_min": 2,
            "acts": [
                {
                    "name": "Act 1",
                    "summary": f"Visualizes the prompt: {base_description}",
                    "scenes": [
                        {
                            "id": "scene-1",
                            "title": "Main Sequence",
                            "summary": base_description,
                            "beats": [
                                {
                                    "id": "beat-1",
                                    "description": f"Establishing shot: {base_description}",
                                    "estimated_duration_sec": 8,
                                    "characters": [],
                                    "location": "unspecified",
                                },
                                {
                                    "id": "beat-2",
                                    "description": f"Character/world reacts: {base_description}",
                                    "estimated_duration_sec": 6,
                                    "characters": [],
                                    "location": "unspecified",
                                },
                                {
                                    "id": "beat-3",
                                    "description": f"Climactic visual moment: {base_description}",
                                    "estimated_duration_sec": 6,
                                    "characters": [],
                                    "location": "unspecified",
                                },
                            ],
                        }
                    ],
                }
            ],
        }

class ProductionNarrativePlanner:
    """
    Production Narrative Planner with Redis caching + mock/real LLM toggle.
    Integrates with episode_assembler via world_id.
    """
    
    def __init__(
        self,
        world_id: str,
        redis_client=None,
        use_mock: Optional[bool] = None
    ):
        self.redis = redis_client

        if use_mock is None:
            use_mock = os.getenv("USE_MOCK_PLANNER", "false").lower() == "true"

        env = os.getenv("ENV", "local").lower()

        if env == "production" and use_mock:
            raise RuntimeError(
                "USE_MOCK_PLANNER=true is forbidden in production"
            )


        self.world_id = world_id
        self.gemini = MockGemini(use_real_api=not use_mock)

        self.plan_file = Path(f"outputs/{world_id}/episode_plan.json")

    def _cache_key(self, script: str) -> str:
        normalized = script.strip().lower().encode("utf-8")
        digest = hashlib.sha256(normalized).hexdigest()
        return f"narrative_plan:{self.world_id}:{digest}"


    
    def plan_episode(self, world_json: Dict[str, Any], script: str) -> EpisodePlan:
        """Generate structured episode plan with caching."""
        cache_key = self._cache_key(script)
        
        cached = self.redis.get(cache_key) if self.redis else None
        if cached:
            logger.info(f"📦 Loaded cached plan from Redis: {cache_key}")
            return EpisodePlan.from_dict(json.loads(cached))
        
        world_str = json.dumps(world_json, indent=2)
        prompt = self._build_prompt(world_str, script)
        
        response = self.gemini.generate_content("gemini-3-flash-preview", prompt)
        raw_text = self._clean_json_response(response["text"])
        
        try:
            plan = EpisodePlan.from_dict(json.loads(raw_text))
            
            if self.redis:
                self.redis.setex(cache_key, 86400, plan.model_dump_json())
            
            self.plan_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.plan_file, "w") as f:
                json.dump(asdict(plan), f, indent=2)
            
            logger.info(f"✅ Generated + cached new plan: {plan.title}")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Plan validation failed: {e}")
            return self._fallback_plan(script)
    
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
    
    def _fallback_plan(self, script: str) -> EpisodePlan:
        """Graceful degradation when JSON parsing fails."""
        logger.warning("🔄 Using fallback plan")
        base_description = script or "Generic anime scene"

        return EpisodePlan(
            title=f"Fallback: {base_description[:40]}",
            total_duration_min=2,
            acts=[
                Act(
                    name="Act 1",
                    summary=f"Fallback visualization of: {base_description}",
                    scenes=[
                        Scene(
                            id="scene-1",
                            title="Fallback Scene",
                            summary=base_description,
                            beats=[
                                Beat(
                                    id="beat-1",
                                    description=base_description,
                                    estimated_duration_sec=6,
                                    characters=[],
                                    location="unspecified",
                                )
                            ],
                        )
                    ],
                )
            ],
        )

    def generate_beats(self, intent: str) -> List[Dict[str, Any]]:
        """
        Generate beats from user intent. This is the main entry point
        called by episode_runtime.plan().
        
        Args:
            intent: User's story intent/description
            
        Returns:
            List of beat dictionaries with id, description, duration, etc.
        """
        # Detect visual style from intent
        try:
            from agents.style_detector import detect_style
            style_profile = detect_style(intent)
            detected_style = style_profile.style.value
            style_confidence = style_profile.confidence
            logger.info(f"Auto-detected style: {detected_style} ({style_confidence:.2f})")
        except Exception as e:
            logger.warning(f"Style detection failed, defaulting to cinematic: {e}")
            detected_style = "cinematic"
            style_profile = None
        
        # Load world data
        world_json = self._load_world_data()
        
        # Initialize character consistency engine
        try:
            from agents.character_consistency import get_consistency_engine
            char_engine = get_consistency_engine(self.world_id)
            char_engine.register_from_world(world_json)
            logger.info(f"Character consistency engine initialized with {len(char_engine.characters)} characters")
        except Exception as e:
            logger.warning(f"Character consistency init failed: {e}")
            char_engine = None
        
        # Generate episode plan
        episode_plan = self.plan_episode(world_json, intent)
        
        # Flatten plan into beats
        beats = []
        beat_counter = 1
        
        for act in episode_plan.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    # Select backend based on style profile
                    backend = self._select_backend(beat)
                    if style_profile and style_profile.recommended_model:
                        backend = style_profile.recommended_model
                    
                    # Convert Beat dataclass to dict format expected by runtime
                    beat_dict = {
                        "id": f"beat-{beat_counter}",
                        "description": beat.description,
                        "duration_sec": beat.estimated_duration_sec,
                        "characters": beat.characters,
                        "location": beat.location,
                        "backend": backend,
                        "style": detected_style,
                        "motion_strength": 0.85,
                    }
                    
                    # Add style profile data if available
                    if style_profile:
                        beat_dict["style_profile"] = {
                            "style_prefix": style_profile.style_prefix,
                            "style_suffix": style_profile.style_suffix,
                            "negative_prompt": style_profile.negative_prompt,
                            "lighting_style": style_profile.lighting_style,
                            "color_temperature": style_profile.color_temperature,
                        }
                    
                    # Add character consistency data
                    if char_engine and beat.characters:
                        char_conditioning = char_engine.build_character_conditioning(
                            beat.characters,
                            style=detected_style
                        )
                        beat_dict["character_conditioning"] = char_conditioning
                        
                        # Enhance description with character details
                        beat_dict["enhanced_description"] = char_engine.enhance_prompt_with_characters(
                            beat.description,
                            beat.characters,
                            style=detected_style
                        )
                    
                    # Add motion config based on beat description
                    try:
                        from agents.motion.enhanced_motion_engine import get_motion_engine
                        motion_engine = get_motion_engine()
                        motion_config = motion_engine.detect_motion_type(beat.description)
                        beat_dict["motion_config"] = motion_config.to_dict()
                        logger.debug(f"Beat {beat_counter}: motion={motion_config.motion_type.value}")
                    except Exception as e:
                        logger.warning(f"Motion detection failed: {e}")
                    
                    # Add cinematic specification (camera, shot, lighting, color)
                    try:
                        from agents.cinematic import direct_beat
                        cinematic_spec = direct_beat(beat.description, detected_style)
                        beat_dict["cinematic_spec"] = cinematic_spec.to_dict()
                        beat_dict["cinematic_prompt"] = cinematic_spec.get_full_prompt(beat.description)
                        logger.debug(f"Beat {beat_counter}: {cinematic_spec.get_summary()}")
                    except Exception as e:
                        logger.warning(f"Cinematic direction failed: {e}")
                    
                    beats.append(beat_dict)
                    beat_counter += 1
        
        logger.info(f"Generated {len(beats)} beats from intent (style: {detected_style})")
        return beats
    
    def _load_world_data(self) -> Dict[str, Any]:
        """
        Load world data for the given world_id.
        Tries multiple sources in order:
        1. Cached world JSON file
        2. World extractor from reference images
        3. Empty world fallback
        """
        # Try loading cached world
        world_file = Path(f"outputs/{self.world_id}/world.json")
        if world_file.exists():
            logger.info(f"Loading cached world from {world_file}")
            with open(world_file, 'r') as f:
                return json.load(f)
        
        # Try extracting from images
        try:
            from agents.world_extractor import WorldExtractor
            
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                extractor = WorldExtractor(api_key)
                
                # Look for reference images
                uploads_dir = Path("uploads")
                char_images = list(uploads_dir.glob("img_*.webp")) + list(uploads_dir.glob("img_*.png"))
                loc_images = list(uploads_dir.glob("location*.jpg")) + list(uploads_dir.glob("location*.png"))
                
                if char_images or loc_images:
                    logger.info(f"Extracting world from {len(char_images)} characters, {len(loc_images)} locations")
                    world_graph = extractor.extract_from_images(
                        [str(img) for img in char_images],
                        [str(img) for img in loc_images]
                    )
                    
                    # Cache the extracted world
                    world_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(world_file, 'w') as f:
                        f.write(world_graph.to_json())
                    
                    return world_graph.to_dict()
        except Exception as e:
            logger.warning(f"World extraction failed: {e}")
        
        # Fallback to empty world
        logger.warning(f"No world data found for {self.world_id}, using empty world")
        return {
            "characters": [],
            "locations": []
        }
    
    def _select_backend(self, beat: Beat) -> str:
        """
        Select appropriate rendering backend based on beat characteristics.
        
        Args:
            beat: Beat to analyze
            
        Returns:
            Backend name (animatediff, veo, svd, stub)
        """
        # For production, use animatediff as default
        # Can be overridden via environment variable
        default_backend = os.getenv("DEFAULT_BACKEND", "animatediff")
        
        # Logic to select backend based on beat content
        # For now, just use default
        return default_backend



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    planner = ProductionNarrativePlanner(
        args.world_id,
        use_mock=args.mock,
    )

    # Minimal safe demo world
    world = {
        "characters": [],
        "locations": []
    }

    plan = planner.plan_episode(world, args.intent)
    print(json.dumps(asdict(plan), indent=2))


if __name__ == "__main__":
    main()
