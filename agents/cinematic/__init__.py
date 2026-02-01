"""
Professional Cinematic System

Production-level cinematic techniques for video generation.
Implements industry-grade camera work, lighting, color grading,
and genre profiles used in film, anime, Pixar, and broadcast.

Usage:
    from agents.cinematic import direct_beat, get_cinematic_prompt
    
    # Get complete cinematic specification
    spec = direct_beat("Epic battle at sunset", genre="anime")
    
    # Get full prompt for AI generation
    prompt = get_cinematic_prompt("Hero realizes the truth", genre="noir")
    
Components:
    - CameraSystem: 40+ camera movements, lens simulation
    - ShotComposer: 28 shot types, composition rules
    - LightingEngine: 26 lighting setups, time-of-day
    - ColorGradingEngine: 21 LUTs, post-processing
    - GenreProfiles: 12 complete genre presets
    - CinematicDirector: Main orchestrator
"""

# Camera System
from .camera_system import (
    CameraMovement,
    CameraLens,
    CameraRig,
    CameraSpec,
    CameraSystem,
    LensType,
    RigCharacteristics,
    get_camera_system,
    select_camera,
    get_camera_prompt,
)

# Shot Composer
from .shot_composer import (
    ShotType,
    CompositionType,
    ShotSpec,
    ShotComposer,
    get_shot_composer,
    compose_shot,
    get_shot_prompt,
)

# Lighting Engine
from .lighting_engine import (
    LightingSetup,
    TimeOfDay,
    LightQuality,
    LightSource,
    LightingSpec,
    LightingEngine,
    GENRE_LIGHTING,
    get_lighting_engine,
    create_lighting,
    get_lighting_prompt,
)

# Color Grading
from .color_grading import (
    ColorLUT,
    ColorCurve,
    ColorBalance,
    SplitTone,
    ColorGrade,
    PostProcessEffect,
    PostProcessSpec,
    ColorGradingEngine,
    GENRE_COLOR_PRESETS,
    get_grading_engine,
    create_grade,
    get_color_prompt,
)

# Genre Profiles
from .genre_profiles import (
    GenreProfile,
    GENRE_PROFILES,
    get_genre_profile,
    list_genres,
    get_genre_prompt,
)

# Cinematic Director (Main Interface)
from .cinematic_director import (
    CinematicSpec,
    CinematicDirector,
    get_cinematic_director,
    direct_beat,
    get_cinematic_prompt,
)

__all__ = [
    # Camera
    "CameraMovement",
    "CameraLens",
    "CameraRig",
    "CameraSpec",
    "CameraSystem",
    "LensType",
    "select_camera",
    "get_camera_prompt",
    
    # Shot
    "ShotType",
    "CompositionType",
    "ShotSpec",
    "ShotComposer",
    "compose_shot",
    "get_shot_prompt",
    
    # Lighting
    "LightingSetup",
    "TimeOfDay",
    "LightingSpec",
    "LightingEngine",
    "create_lighting",
    "get_lighting_prompt",
    
    # Color
    "ColorLUT",
    "ColorGrade",
    "PostProcessSpec",
    "ColorGradingEngine",
    "create_grade",
    "get_color_prompt",
    
    # Genre
    "GenreProfile",
    "GENRE_PROFILES",
    "get_genre_profile",
    "list_genres",
    
    # Director (Main Interface)
    "CinematicSpec",
    "CinematicDirector",
    "direct_beat",
    "get_cinematic_prompt",
]
