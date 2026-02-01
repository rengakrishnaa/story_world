"""
Cinematic Director - Main Orchestrator

The central orchestrator that combines camera system, shot composer,
lighting engine, and color grading into a unified cinematic specification.

This is the main entry point for the cinematic system. It takes a beat
description and genre, and produces a complete CinematicSpec that
contains all visual parameters for video generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from .camera_system import CameraSystem, CameraSpec, select_camera, get_camera_prompt
from .shot_composer import ShotComposer, ShotSpec, compose_shot, get_shot_prompt
from .lighting_engine import LightingEngine, LightingSpec, create_lighting, get_lighting_prompt
from .color_grading import ColorGradingEngine, ColorGrade, PostProcessSpec, create_grade, get_color_prompt
from .genre_profiles import GenreProfile, get_genre_profile, list_genres

logger = logging.getLogger(__name__)


# ============================================
# CINEMATIC SPECIFICATION
# ============================================

@dataclass
class CinematicSpec:
    """
    Complete cinematic specification for a beat.
    
    Contains all visual parameters needed for video generation:
    - Camera configuration
    - Shot framing
    - Lighting setup
    - Color grading
    - Post-processing
    - Prompt modifiers
    """
    # Genre
    genre: str = "cinematic"
    genre_profile: Optional[GenreProfile] = None
    
    # Camera
    camera: Optional[CameraSpec] = None
    
    # Shot
    shot: Optional[ShotSpec] = None
    
    # Lighting
    lighting: Optional[LightingSpec] = None
    
    # Color
    color_grade: Optional[ColorGrade] = None
    post_process: Optional[PostProcessSpec] = None
    
    # Combined prompts
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_prompt: str = ""
    
    # Full technical prompt (all modifiers combined)
    technical_prompt: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "genre": self.genre,
            "camera": self.camera.to_dict() if self.camera else None,
            "shot": self.shot.to_dict() if self.shot else None,
            "lighting": self.lighting.to_dict() if self.lighting else None,
            "color_grade": self.color_grade.to_dict() if self.color_grade else None,
            "post_process": self.post_process.to_dict() if self.post_process else None,
            "prompt_prefix": self.prompt_prefix,
            "prompt_suffix": self.prompt_suffix,
            "negative_prompt": self.negative_prompt,
            "technical_prompt": self.technical_prompt,
        }
    
    def get_full_prompt(self, base_description: str) -> str:
        """
        Combine base description with all cinematic modifiers.
        
        Args:
            base_description: The beat's core description
            
        Returns:
            Full prompt with all technical specifications
        """
        parts = []
        
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)
        
        parts.append(base_description)
        
        if self.technical_prompt:
            parts.append(self.technical_prompt)
        
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)
        
        return ", ".join(parts)
    
    def get_summary(self) -> str:
        """Get a brief summary of the cinematic spec."""
        summary_parts = []
        
        if self.camera:
            summary_parts.append(f"Camera: {self.camera.movement.value}, {self.camera.lens.focal_length}mm")
        
        if self.shot:
            summary_parts.append(f"Shot: {self.shot.shot_type.value}")
        
        if self.lighting:
            summary_parts.append(f"Light: {self.lighting.setup.value}")
        
        if self.color_grade:
            summary_parts.append(f"LUT: {self.color_grade.lut.value}")
        
        return " | ".join(summary_parts)


# ============================================
# CINEMATIC DIRECTOR
# ============================================

class CinematicDirector:
    """
    The main orchestrator that creates complete cinematic specifications.
    
    Takes a beat description and genre, and produces a CinematicSpec
    with all visual parameters for video generation.
    """
    
    def __init__(self):
        self.camera_system = CameraSystem()
        self.shot_composer = ShotComposer()
        self.lighting_engine = LightingEngine()
        self.grading_engine = ColorGradingEngine()
    
    def direct_beat(
        self,
        description: str,
        genre: str = "cinematic",
        time_of_day: Optional[str] = None,
        override_camera: Optional[Dict] = None,
        override_lighting: Optional[Dict] = None,
    ) -> CinematicSpec:
        """
        Create complete cinematic specification for a beat.
        
        Args:
            description: Text description of the beat
            genre: Visual style (anime, pixar, noir, etc.)
            time_of_day: Optional override for time of day
            override_camera: Optional camera overrides
            override_lighting: Optional lighting overrides
            
        Returns:
            CinematicSpec with complete visual configuration
        """
        # Get genre profile
        profile = get_genre_profile(genre)
        
        # Create camera spec
        camera = self.camera_system.select_camera_for_beat(description, genre)
        
        # Apply genre camera preferences
        if profile.camera.get("preferred_lens"):
            camera.lens = camera.lens.from_mm(profile.camera["preferred_lens"])
        if profile.camera.get("preferred_rig"):
            camera.rig = profile.camera["preferred_rig"]
        
        # Create shot spec
        shot = self.shot_composer.compose_shot(description, genre)
        
        # Determine time of day
        if time_of_day is None:
            time_of_day = self._detect_time(description, profile)
        
        # Create lighting spec
        lighting = self.lighting_engine.create_lighting(description, genre, time_of_day)
        
        # Create color grading
        color_grade, post_process = self.grading_engine.create_grade(
            description, genre, time_of_day
        )
        
        # Apply genre post-processing preferences
        if profile.post:
            if "bloom" in profile.post:
                post_process.bloom = profile.post["bloom"]
            if "chromatic_aberration" in profile.post:
                post_process.chromatic_aberration = profile.post["chromatic_aberration"]
            if "vignette" in profile.post:
                post_process.vignette = profile.post["vignette"]
            if "grain" in profile.post:
                post_process.film_grain = profile.post["grain"]
        
        # Build technical prompt
        technical_prompt = self._build_technical_prompt(camera, shot, lighting, color_grade, post_process)
        
        # Create spec
        spec = CinematicSpec(
            genre=genre,
            genre_profile=profile,
            camera=camera,
            shot=shot,
            lighting=lighting,
            color_grade=color_grade,
            post_process=post_process,
            prompt_prefix=profile.prompt_prefix,
            prompt_suffix=profile.prompt_suffix,
            negative_prompt=profile.negative_prompt,
            technical_prompt=technical_prompt,
        )
        
        return spec
    
    def _detect_time(self, description: str, profile: GenreProfile) -> str:
        """Detect time of day from description or genre."""
        desc_lower = description.lower()
        
        # Check explicit time mentions
        time_keywords = {
            "dawn": ["dawn", "sunrise", "early morning"],
            "golden_am": ["golden hour", "morning glow"],
            "morning": ["morning", "breakfast"],
            "midday": ["noon", "midday"],
            "afternoon": ["afternoon", "daytime"],
            "golden_pm": ["sunset", "golden hour", "evening glow"],
            "dusk": ["dusk", "twilight"],
            "blue_hour": ["blue hour"],
            "night": ["night", "midnight", "dark"],
            "neon_night": ["neon", "club", "city night"],
            "moonlit": ["moonlit", "moonlight"],
        }
        
        for time, keywords in time_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return time
        
        # Check genre preferences
        preferred_times = profile.lighting.get("preferred_times", [])
        if preferred_times:
            return preferred_times[0].value if hasattr(preferred_times[0], 'value') else str(preferred_times[0])
        
        return "afternoon"
    
    def _build_technical_prompt(
        self,
        camera: CameraSpec,
        shot: ShotSpec,
        lighting: LightingSpec,
        color: ColorGrade,
        post: PostProcessSpec,
    ) -> str:
        """Build combined technical prompt from all specs."""
        all_modifiers = []
        
        # Camera modifiers
        all_modifiers.extend(camera.to_prompt_modifiers())
        
        # Shot modifiers
        all_modifiers.extend(shot.to_prompt_modifiers())
        
        # Lighting modifiers
        all_modifiers.extend(lighting.to_prompt_modifiers())
        
        # Color modifiers
        all_modifiers.extend(color.to_prompt_modifiers())
        all_modifiers.extend(post.to_prompt_modifiers())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_modifiers = []
        for m in all_modifiers:
            if m.lower() not in seen:
                seen.add(m.lower())
                unique_modifiers.append(m)
        
        return ", ".join(unique_modifiers)
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genre names."""
        return list_genres()


# Singleton instance
_cinematic_director: Optional[CinematicDirector] = None


def get_cinematic_director() -> CinematicDirector:
    """Get or create the global cinematic director."""
    global _cinematic_director
    if _cinematic_director is None:
        _cinematic_director = CinematicDirector()
    return _cinematic_director


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def direct_beat(
    description: str,
    genre: str = "cinematic",
    time_of_day: Optional[str] = None,
) -> CinematicSpec:
    """
    Convenience function to create cinematic spec for a beat.
    
    Args:
        description: Beat description
        genre: Visual style
        time_of_day: Optional time of day
        
        Returns:
        CinematicSpec with complete configuration
    """
    director = get_cinematic_director()
    return director.direct_beat(description, genre, time_of_day)


def get_cinematic_prompt(description: str, genre: str = "cinematic") -> str:
    """
    Get the full cinematic prompt for a beat.
    
    Args:
        description: Beat description
        genre: Visual style
        
    Returns:
        Full prompt with all modifiers
    """
    spec = direct_beat(description, genre)
    return spec.get_full_prompt(description)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CINEMATIC DIRECTOR TEST")
    print("="*60)
    
    director = CinematicDirector()
    
    # Test various beats with different genres
    test_beats = [
        ("Two warriors clash in an epic battle at sunset", "anime"),
        ("Close-up on Maya's tearful expression as she reads the letter", "romance"),
        ("Wide establishing shot of the futuristic neon city at night", "cyberpunk"),
        ("The detective examines the mysterious letter in his dim office", "noir"),
        ("Friendly robots explore a colorful alien planet", "pixar"),
        ("The hero walks through a magical glowing forest", "fantasy"),
        ("Creepy figure emerges from the shadows", "horror"),
        ("Soft meadow scene with wind blowing through the grass", "ghibli"),
        ("High-speed car chase through narrow streets", "action"),
        ("Interview with the expert in their natural environment", "documentary"),
    ]
    
    print("\n--- Cinematic Direction Test ---\n")
    
    for description, genre in test_beats:
        spec = director.direct_beat(description, genre)
        
        print(f"Beat: \"{description[:45]}...\"")
        print(f"  Genre: {genre}")
        print(f"  {spec.get_summary()}")
        print(f"  Technical prompt preview: {spec.technical_prompt[:60]}...")
        print()
    
    # Test full prompt generation
    print("--- Full Prompt Generation Test ---\n")
    
    test_desc = "The hero realizes the truth with growing horror"
    for genre in ["anime", "noir", "horror"]:
        spec = director.direct_beat(test_desc, genre)
        full_prompt = spec.get_full_prompt(test_desc)
        print(f"{genre.upper()}:")
        print(f"  {full_prompt[:100]}...")
        print()
    
    # List all genres
    print(f"--- Available Genres: {len(director.get_available_genres())} ---")
    for g in director.get_available_genres():
        print(f"  {g}")
    
    print("\n✅ Cinematic director working!")
