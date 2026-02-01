"""
Genre Profiles - Complete Visual Style Presets

Production-level genre profiles that combine camera, shot, lighting,
and color grading into cohesive visual styles. Each profile represents
the complete technical specification for a genre as used in professional
film, anime, Pixar, and broadcast production.

This module provides:
1. GenreProfile - Complete technical preset for each genre
2. Get ready-to-use configurations for any genre
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from .camera_system import CameraMovement, CameraRig, LensType
from .shot_composer import ShotType, CompositionType
from .lighting_engine import LightingSetup, TimeOfDay
from .color_grading import ColorLUT

logger = logging.getLogger(__name__)


# ============================================
# GENRE PROFILE DATACLASS
# ============================================

@dataclass
class GenreProfile:
    """
    Complete technical profile for a visual genre.
    
    Combines all cinematic parameters into a cohesive style
    that can be applied to video generation.
    """
    name: str
    description: str
    
    # Camera preferences
    camera: Dict[str, Any] = field(default_factory=dict)
    
    # Shot preferences
    shots: Dict[str, Any] = field(default_factory=dict)
    
    # Lighting preferences
    lighting: Dict[str, Any] = field(default_factory=dict)
    
    # Color grading
    color: Dict[str, Any] = field(default_factory=dict)
    
    # Post-processing
    post: Dict[str, Any] = field(default_factory=dict)
    
    # Prompt modifiers
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_prompt: str = ""
    
    def get_prompt_modifiers(self) -> str:
        """Get all prompt modifiers as a string."""
        modifiers = []
        if self.prompt_prefix:
            modifiers.append(self.prompt_prefix)
        if self.prompt_suffix:
            modifiers.append(self.prompt_suffix)
        return ", ".join(modifiers)


# ============================================
# COMPLETE GENRE PROFILES
# ============================================

GENRE_PROFILES: Dict[str, GenreProfile] = {
    
    # ========================================
    # ANIME
    # ========================================
    "anime": GenreProfile(
        name="Anime",
        description="Japanese animation style with cel-shading, vibrant colors, and dynamic action",
        
        camera={
            "preferred_lens": 50,
            "movement_style": "static_with_snap_zooms",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.SNAP_ZOOM,
                CameraMovement.WHIP_PAN,
                CameraMovement.PUSH_IN,
            ],
            "preferred_rig": CameraRig.TRIPOD,
            "speed_lines": True,
            "impact_frames": True,
            "dutch_angle_frequency": 0.2,
        },
        
        shots={
            "close_up_ratio": 0.4,
            "preferred_shots": [
                ShotType.CLOSE_UP,
                ShotType.MEDIUM_CLOSE_UP,
                ShotType.IMPACT_SHOT,
                ShotType.BEAUTY_SHOT,
            ],
            "dynamic_angles": True,
            "composition": CompositionType.DYNAMIC_DIAGONAL,
            "dramatic_poses": True,
        },
        
        lighting={
            "setup": LightingSetup.CEL_SHADED,
            "specular_highlights": "sharp",
            "shadow_style": "hard_edge",
            "ambient_occlusion": 0.2,
            "rim_light_intensity": 0.7,
            "preferred_times": [TimeOfDay.AFTERNOON, TimeOfDay.GOLDEN_HOUR_PM],
        },
        
        color={
            "lut": ColorLUT.ANIME_POP,
            "saturation": 1.25,
            "vibrance": 1.3,
            "contrast": 1.15,
            "skin_tone": "warm_bright",
            "shadow_tint": (0.15, 0.12, 0.2),  # Purple tint
        },
        
        post={
            "chromatic_aberration": 0.25,
            "bloom": 0.35,
            "sharpening": 0.4,
            "grain": 0.05,
        },
        
        prompt_prefix="anime style, Japanese animation, cel-shaded, 2D animation",
        prompt_suffix="detailed linework, vibrant colors, studio quality animation",
        negative_prompt="realistic, photorealistic, 3D render, western cartoon",
    ),
    
    # ========================================
    # PIXAR / 3D ANIMATION
    # ========================================
    "pixar": GenreProfile(
        name="Pixar / 3D Animation",
        description="High-quality 3D animation with expressive characters and rich lighting",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "smooth_dolly",
            "preferred_movements": [
                CameraMovement.DOLLY_IN,
                CameraMovement.DOLLY_OUT,
                CameraMovement.CRANE_SHOT,
                CameraMovement.ARC_LEFT,
            ],
            "preferred_rig": CameraRig.DOLLY,
            "depth_of_field": "cinematic_shallow",
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.MEDIUM_CLOSE_UP,
                ShotType.TWO_SHOT,
                ShotType.ESTABLISHING,
            ],
            "character_framing": "generous_headroom",
            "composition": CompositionType.RULE_OF_THIRDS,
            "squash_stretch_timing": True,
        },
        
        lighting={
            "setup": LightingSetup.AMBIENT_OCCLUSION,
            "subsurface_scattering": True,
            "ambient_occlusion": 0.6,
            "rim_light_intensity": 0.4,
            "rim_light_color": "warm",
            "soft_shadows": True,
        },
        
        color={
            "lut": ColorLUT.PIXAR_BRIGHT,
            "saturation": 1.1,
            "vibrance": 1.2,
            "contrast": 1.0,
            "shadows_lifted": 0.1,
            "color_palette": "complementary_harmonious",
        },
        
        post={
            "bloom": 0.2,
            "depth_blur": 0.5,
            "vignette": 0.15,
            "softening": 0.1,
        },
        
        prompt_prefix="Pixar style, 3D animated, Disney quality, CGI",
        prompt_suffix="expressive characters, rich textures, professional CG animation",
        negative_prompt="2D, flat, anime, realistic, photorealistic",
    ),
    
    # ========================================
    # STUDIO GHIBLI
    # ========================================
    "ghibli": GenreProfile(
        name="Studio Ghibli",
        description="Soft, pastoral anime style with watercolor aesthetics and emotional depth",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "gentle_slow",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.FLOATING,
                CameraMovement.DRIFT,
                CameraMovement.PAN_RIGHT,
            ],
            "preferred_rig": CameraRig.TRIPOD,
            "contemplative_pacing": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.LONG_SHOT,
                ShotType.ESTABLISHING,
                ShotType.MEDIUM_SHOT,
                ShotType.BEAUTY_SHOT,
            ],
            "landscape_focus": True,
            "composition": CompositionType.RULE_OF_THIRDS,
            "nature_elements": True,
        },
        
        lighting={
            "setup": LightingSetup.FLAT,
            "soft_diffused": True,
            "ambient": 0.5,
            "gentle_shadows": True,
            "preferred_times": [TimeOfDay.GOLDEN_HOUR_AM, TimeOfDay.AFTERNOON],
        },
        
        color={
            "lut": ColorLUT.GHIBLI_SOFT,
            "saturation": 1.05,
            "vibrance": 1.1,
            "contrast": 0.9,
            "pastel_palette": True,
            "shadow_tint": (0.3, 0.25, 0.35),  # Soft purple
        },
        
        post={
            "bloom": 0.25,
            "softening": 0.2,
            "grain": 0.1,
        },
        
        prompt_prefix="Studio Ghibli style, Miyazaki inspired, watercolor aesthetic",
        prompt_suffix="soft colors, pastoral, emotional depth, hand-drawn feel",
        negative_prompt="harsh, dark, violent, 3D, CGI",
    ),
    
    # ========================================
    # FILM NOIR
    # ========================================
    "noir": GenreProfile(
        name="Film Noir",
        description="Classic black-and-white crime drama with high contrast and dramatic shadows",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "static_or_slow_dolly",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.DOLLY_IN,
                CameraMovement.TRACKING_SHOT,
            ],
            "preferred_rig": CameraRig.TRIPOD,
            "dutch_angle_frequency": 0.35,
            "low_angle_frequency": 0.4,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.OVER_THE_SHOULDER,
            ],
            "silhouettes": True,
            "composition": CompositionType.DYNAMIC_DIAGONAL,
            "window_shadows": True,
        },
        
        lighting={
            "setup": LightingSetup.LOW_KEY,
            "contrast_ratio": "8:1",
            "venetian_blind_shadows": True,
            "single_source_motivated": True,
            "harsh_shadows": True,
        },
        
        color={
            "lut": ColorLUT.NOIR,
            "saturation": 0.3,
            "contrast": 1.5,
            "crush_blacks": True,
            "highlight_protection": False,
        },
        
        post={
            "grain": 0.5,
            "vignette": 0.4,
            "halation": 0.15,
            "film_type": "35mm",
        },
        
        prompt_prefix="film noir, black and white, high contrast, 1940s style",
        prompt_suffix="dramatic shadows, venetian blinds, moody atmosphere, detective story",
        negative_prompt="colorful, bright, cheerful, modern",
    ),
    
    # ========================================
    # CYBERPUNK
    # ========================================
    "cyberpunk": GenreProfile(
        name="Cyberpunk",
        description="Futuristic neon-lit dystopia with high-tech aesthetics and gritty atmosphere",
        
        camera={
            "preferred_lens": 24,
            "movement_style": "dynamic_handheld",
            "preferred_movements": [
                CameraMovement.TRACKING_SHOT,
                CameraMovement.STEADICAM,
                CameraMovement.PUSH_IN,
                CameraMovement.CRASH_ZOOM,
            ],
            "preferred_rig": CameraRig.GIMBAL,
            "anamorphic": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.ESTABLISHING,
                ShotType.MEDIUM_LONG_SHOT,
                ShotType.CLOSE_UP,
                ShotType.POV,
            ],
            "composition": CompositionType.LEADING_LINES,
            "reflection_shots": True,
            "neon_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.PRACTICAL,
            "neon_colors": True,
            "color_gels": [(180, 1.0, 0.8), (300, 1.0, 0.8)],  # Cyan and Magenta
            "wet_reflections": True,
            "fog_density": 0.3,
            "preferred_times": [TimeOfDay.NEON_NIGHT, TimeOfDay.NIGHT],
        },
        
        color={
            "lut": ColorLUT.CYBERPUNK_NEON,
            "saturation": 1.4,
            "vibrance": 1.5,
            "contrast": 1.3,
            "split_tone_shadows": (270, 0.4),  # Purple
            "split_tone_highlights": (180, 0.3),  # Teal
        },
        
        post={
            "bloom": 0.5,
            "chromatic_aberration": 0.4,
            "lens_flare": 0.3,
            "anamorphic_flare": True,
            "grain": 0.2,
        },
        
        prompt_prefix="cyberpunk, futuristic, neon lights, dystopian",
        prompt_suffix="rain-slicked streets, holographic advertisements, high-tech low-life",
        negative_prompt="natural, pastoral, historical, clean, bright daylight",
    ),
    
    # ========================================
    # HORROR
    # ========================================
    "horror": GenreProfile(
        name="Horror",
        description="Dark, unsettling atmosphere designed to create fear and tension",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "steadicam_stalking",
            "preferred_movements": [
                CameraMovement.STEADICAM,
                CameraMovement.DOLLY_IN,
                CameraMovement.STATIC,
                CameraMovement.HANDHELD,
            ],
            "preferred_rig": CameraRig.STEADICAM,
            "dutch_angle_frequency": 0.25,
        },
        
        shots={
            "preferred_shots": [
                ShotType.CLOSE_UP,
                ShotType.POV,
                ShotType.LONG_SHOT,
                ShotType.REACTION_SHOT,
            ],
            "composition": CompositionType.NEGATIVE_SPACE,
            "off_center_framing": True,
            "hidden_threats": True,
        },
        
        lighting={
            "setup": LightingSetup.UNDER_LIGHTING,
            "low_key": True,
            "motivated_practicals": True,
            "flickering": True,
            "shadow_play": True,
        },
        
        color={
            "lut": ColorLUT.HORROR_GREEN,
            "saturation": 0.6,
            "contrast": 1.25,
            "temperature": -20,
            "shadow_tint": (0.05, 0.08, 0.05),  # Sickly green
        },
        
        post={
            "vignette": 0.5,
            "grain": 0.35,
            "chromatic_aberration": 0.15,
        },
        
        prompt_prefix="horror, dark, unsettling, scary",
        prompt_suffix="atmospheric tension, creepy shadows, fear-inducing",
        negative_prompt="bright, cheerful, colorful, safe",
    ),
    
    # ========================================
    # ROMANCE
    # ========================================
    "romance": GenreProfile(
        name="Romance",
        description="Warm, soft aesthetic designed to enhance emotional intimacy",
        
        camera={
            "preferred_lens": 85,
            "movement_style": "gentle_slow",
            "preferred_movements": [
                CameraMovement.DOLLY_IN,
                CameraMovement.ARC_RIGHT,
                CameraMovement.FLOATING,
                CameraMovement.STATIC,
            ],
            "preferred_rig": CameraRig.DOLLY,
            "shallow_dof": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.CLOSE_UP,
                ShotType.TWO_SHOT,
                ShotType.MEDIUM_CLOSE_UP,
                ShotType.OVER_THE_SHOULDER,
            ],
            "composition": CompositionType.RULE_OF_THIRDS,
            "eye_contact_emphasis": True,
        },
        
        lighting={
            "setup": LightingSetup.THREE_POINT,
            "soft_key": True,
            "warm_fill": True,
            "rim_light": True,
            "preferred_times": [TimeOfDay.GOLDEN_HOUR_PM, TimeOfDay.DUSK],
        },
        
        color={
            "lut": ColorLUT.WARM_SUNSET,
            "saturation": 1.05,
            "vibrance": 1.1,
            "contrast": 0.95,
            "temperature": 25,
            "skin_tones_protected": True,
        },
        
        post={
            "bloom": 0.35,
            "softening": 0.25,
            "vignette": 0.15,
            "grain": 0.1,
        },
        
        prompt_prefix="romantic, soft lighting, warm tones",
        prompt_suffix="intimate atmosphere, beautiful, emotional depth",
        negative_prompt="harsh, cold, violent, scary",
    ),
    
    # ========================================
    # ACTION
    # ========================================
    "action": GenreProfile(
        name="Action",
        description="High-energy visuals with dynamic camera work and intense styling",
        
        camera={
            "preferred_lens": 24,
            "movement_style": "dynamic_aggressive",
            "preferred_movements": [
                CameraMovement.TRACKING_SHOT,
                CameraMovement.WHIP_PAN,
                CameraMovement.CRASH_ZOOM,
                CameraMovement.HANDHELD,
            ],
            "preferred_rig": CameraRig.GIMBAL,
            "speed_ramp": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_LONG_SHOT,
                ShotType.FULL_SHOT,
                ShotType.CLOSE_UP,
                ShotType.IMPACT_SHOT,
            ],
            "composition": CompositionType.DYNAMIC_DIAGONAL,
            "wide_for_action": True,
        },
        
        lighting={
            "setup": LightingSetup.RIM,
            "high_contrast": True,
            "dramatic_shadows": True,
            "rim_intensity": 0.8,
        },
        
        color={
            "lut": ColorLUT.TEAL_ORANGE,
            "saturation": 1.1,
            "vibrance": 1.15,
            "contrast": 1.25,
        },
        
        post={
            "bloom": 0.2,
            "sharpening": 0.3,
            "motion_blur": 0.3,
        },
        
        prompt_prefix="action scene, dynamic, intense",
        prompt_suffix="high energy, dramatic, cinematic action",
        negative_prompt="static, calm, peaceful, slow",
    ),
    
    # ========================================
    # FANTASY
    # ========================================
    "fantasy": GenreProfile(
        name="Fantasy",
        description="Magical, ethereal atmosphere with rich color and mystical lighting",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "epic_sweeping",
            "preferred_movements": [
                CameraMovement.CRANE_SHOT,
                CameraMovement.DOLLY_OUT,
                CameraMovement.ARC_LEFT,
                CameraMovement.FLOATING,
            ],
            "preferred_rig": CameraRig.CRANE,
        },
        
        shots={
            "preferred_shots": [
                ShotType.ESTABLISHING,
                ShotType.LONG_SHOT,
                ShotType.BEAUTY_SHOT,
                ShotType.MEDIUM_SHOT,
            ],
            "composition": CompositionType.DEPTH_LAYERING,
            "epic_scale": True,
        },
        
        lighting={
            "setup": LightingSetup.MOTIVATED,
            "god_rays": True,
            "magical_rim": True,
            "ethereal_glow": True,
            "preferred_times": [TimeOfDay.GOLDEN_HOUR_AM, TimeOfDay.BLUE_HOUR],
        },
        
        color={
            "lut": ColorLUT.FANTASY_ETHEREAL,
            "saturation": 1.15,
            "vibrance": 1.2,
            "contrast": 1.0,
            "shadow_tint": (0.15, 0.12, 0.2),  # Magical purple
        },
        
        post={
            "bloom": 0.4,
            "glow": 0.35,
            "halation": 0.2,
            "vignette": 0.1,
        },
        
        prompt_prefix="fantasy, magical, ethereal",
        prompt_suffix="mystical atmosphere, enchanted, epic fantasy world",
        negative_prompt="realistic, modern, urban, mundane",
    ),
    
    # ========================================
    # CINEMATIC / REALISTIC
    # ========================================
    "cinematic": GenreProfile(
        name="Cinematic",
        description="Standard Hollywood-style visual treatment",
        
        camera={
            "preferred_lens": 50,
            "movement_style": "professional_varied",
            "preferred_movements": [
                CameraMovement.DOLLY_IN,
                CameraMovement.TRACKING_SHOT,
                CameraMovement.CRANE_SHOT,
                CameraMovement.STATIC,
            ],
            "preferred_rig": CameraRig.DOLLY,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.ESTABLISHING,
                ShotType.OVER_THE_SHOULDER,
            ],
            "composition": CompositionType.RULE_OF_THIRDS,
        },
        
        lighting={
            "setup": LightingSetup.THREE_POINT,
            "professional_balance": True,
        },
        
        color={
            "lut": ColorLUT.TEAL_ORANGE,
            "saturation": 1.05,
            "vibrance": 1.1,
            "contrast": 1.1,
        },
        
        post={
            "bloom": 0.15,
            "vignette": 0.2,
            "grain": 0.15,
        },
        
        prompt_prefix="cinematic, film quality, professional",
        prompt_suffix="movie-like, high production value",
        negative_prompt="amateur, low quality, webcam",
    ),
    
    # ========================================
    # DOCUMENTARY
    # ========================================
    "documentary": GenreProfile(
        name="Documentary",
        description="Natural, realistic style emphasizing authenticity",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "handheld_natural",
            "preferred_movements": [
                CameraMovement.HANDHELD,
                CameraMovement.STATIC,
                CameraMovement.TRACKING_SHOT,
            ],
            "preferred_rig": CameraRig.SHOULDER,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.LONG_SHOT,
            ],
            "composition": CompositionType.RULE_OF_THIRDS,
            "natural_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.AVAILABLE,
            "natural_light": True,
            "minimal_modification": True,
        },
        
        color={
            "lut": ColorLUT.NATURAL,
            "saturation": 1.0,
            "vibrance": 1.0,
            "contrast": 1.05,
        },
        
        post={
            "grain": 0.1,
        },
        
        prompt_prefix="documentary style, realistic, natural",
        prompt_suffix="authentic, real-world, unscripted feel",
        negative_prompt="stylized, fantastical, artificial",
    ),
    
    # ========================================
    # CARTOON (WESTERN ANIMATION)
    # ========================================
    "cartoon": GenreProfile(
        name="Cartoon",
        description="Western animation style with bold colors and exaggerated features",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "dynamic_comedic",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.WHIP_PAN,
                CameraMovement.SNAP_ZOOM,
            ],
            "preferred_rig": CameraRig.TRIPOD,
            "smear_frames": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.FULL_SHOT,
            ],
            "composition": CompositionType.CENTER_FRAME,
            "exaggerated_poses": True,
        },
        
        lighting={
            "setup": LightingSetup.FLAT,
            "simple_shadows": True,
            "bold_contrast": True,
        },
        
        color={
            "lut": ColorLUT.CARTOON_VIBRANT,
            "saturation": 1.4,
            "vibrance": 1.4,
            "contrast": 1.1,
        },
        
        post={
            "sharpening": 0.3,
        },
        
        prompt_prefix="cartoon style, western animation, animated",
        prompt_suffix="bold colors, expressive, animated show quality",
        negative_prompt="realistic, photorealistic, anime, 3D CGI",
    ),
    
    # ========================================
    # REALISTIC / PHOTOREALISTIC
    # ========================================
    "realistic": GenreProfile(
        name="Realistic",
        description="Photorealistic style mimicking real-world cinematography",
        
        camera={
            "preferred_lens": 50,
            "movement_style": "natural_subtle",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.DOLLY_IN,
                CameraMovement.HANDHELD,
            ],
            "preferred_rig": CameraRig.TRIPOD,
            "realistic_motion": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.ESTABLISHING,
            ],
            "composition": CompositionType.RULE_OF_THIRDS,
            "natural_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.NATURAL,
            "realistic_falloff": True,
            "physically_accurate": True,
        },
        
        color={
            "lut": ColorLUT.NATURAL,
            "saturation": 1.0,
            "vibrance": 1.0,
            "contrast": 1.0,
        },
        
        post={
            "grain": 0.05,
        },
        
        prompt_prefix="photorealistic, realistic, lifelike",
        prompt_suffix="natural lighting, real-world, authentic",
        negative_prompt="cartoon, anime, stylized, artificial",
    ),
    
    # ========================================
    # THRILLER / SUSPENSE
    # ========================================
    "thriller": GenreProfile(
        name="Thriller",
        description="Tense, suspenseful atmosphere with unsettling visual style",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "slow_creeping",
            "preferred_movements": [
                CameraMovement.DOLLY_IN,
                CameraMovement.STEADICAM,
                CameraMovement.STATIC,
                CameraMovement.PUSH_IN,
            ],
            "preferred_rig": CameraRig.STEADICAM,
            "dutch_angle_frequency": 0.2,
        },
        
        shots={
            "preferred_shots": [
                ShotType.CLOSE_UP,
                ShotType.OVER_THE_SHOULDER,
                ShotType.POV,
                ShotType.MEDIUM_SHOT,
            ],
            "composition": CompositionType.NEGATIVE_SPACE,
            "claustrophobic_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.LOW_KEY,
            "contrast_ratio": "4:1",
            "motivated_shadows": True,
            "pools_of_light": True,
        },
        
        color={
            "lut": ColorLUT.COOL_STEEL,
            "saturation": 0.85,
            "contrast": 1.2,
            "temperature": -15,
        },
        
        post={
            "vignette": 0.35,
            "grain": 0.2,
            "chromatic_aberration": 0.1,
        },
        
        prompt_prefix="thriller, suspenseful, tense",
        prompt_suffix="psychological tension, atmospheric, gripping",
        negative_prompt="bright, cheerful, comedic, relaxed",
    ),
    
    # ========================================
    # WAR / MILITARY
    # ========================================
    "war": GenreProfile(
        name="War / Military",
        description="Gritty, intense war cinematography with documentary influence",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "handheld_chaotic",
            "preferred_movements": [
                CameraMovement.HANDHELD,
                CameraMovement.TRACKING_SHOT,
                CameraMovement.WHIP_PAN,
            ],
            "preferred_rig": CameraRig.SHOULDER,
            "shake_intensity": 0.4,
        },
        
        shots={
            "preferred_shots": [
                ShotType.CLOSE_UP,
                ShotType.MEDIUM_LONG_SHOT,
                ShotType.POV,
                ShotType.ESTABLISHING,
            ],
            "composition": CompositionType.DYNAMIC_DIAGONAL,
            "chaos_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.NATURAL,
            "harsh_daylight": True,
            "smoke_atmosphere": True,
            "practical_fire": True,
        },
        
        color={
            "lut": ColorLUT.BLEACH_BYPASS,
            "saturation": 0.7,
            "contrast": 1.3,
            "temperature": -10,
        },
        
        post={
            "grain": 0.5,
            "sharpening": 0.4,
            "vignette": 0.2,
            "motion_blur": 0.3,
        },
        
        prompt_prefix="war film, military, combat",
        prompt_suffix="gritty, intense, battlefield chaos, soldier perspective",
        negative_prompt="peaceful, clean, stylized, colorful",
    ),
    
    # ========================================
    # MUSICAL
    # ========================================
    "musical": GenreProfile(
        name="Musical",
        description="Vibrant, theatrical style with dynamic choreography emphasis",
        
        camera={
            "preferred_lens": 24,
            "movement_style": "sweeping_choreographed",
            "preferred_movements": [
                CameraMovement.CRANE_SHOT,
                CameraMovement.TRACKING_SHOT,
                CameraMovement.ARC_LEFT,
                CameraMovement.DOLLY_OUT,
            ],
            "preferred_rig": CameraRig.CRANE,
            "follow_choreography": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.FULL_SHOT,
                ShotType.LONG_SHOT,
                ShotType.MEDIUM_SHOT,
                ShotType.GROUP_SHOT,
            ],
            "composition": CompositionType.CENTER_FRAME,
            "stage_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.HIGH_KEY,
            "theatrical_spots": True,
            "colored_gels": True,
            "rim_emphasis": True,
        },
        
        color={
            "lut": ColorLUT.PIXAR_BRIGHT,
            "saturation": 1.25,
            "vibrance": 1.3,
            "contrast": 1.05,
        },
        
        post={
            "bloom": 0.3,
            "glow": 0.2,
            "lens_flare": 0.15,
        },
        
        prompt_prefix="musical, theatrical, Broadway style",
        prompt_suffix="vibrant performance, choreographed, stage lighting",
        negative_prompt="dark, dull, static, realistic",
    ),
    
    # ========================================
    # SCI-FI
    # ========================================
    "scifi": GenreProfile(
        name="Science Fiction",
        description="Futuristic clean aesthetic with advanced technology emphasis",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "smooth_precise",
            "preferred_movements": [
                CameraMovement.DOLLY_IN,
                CameraMovement.TRACKING_SHOT,
                CameraMovement.CRANE_SHOT,
                CameraMovement.STEADICAM,
            ],
            "preferred_rig": CameraRig.GIMBAL,
            "anamorphic": True,
        },
        
        shots={
            "preferred_shots": [
                ShotType.ESTABLISHING,
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.POV,
            ],
            "composition": CompositionType.LEADING_LINES,
            "symmetry": True,
        },
        
        lighting={
            "setup": LightingSetup.PRACTICAL,
            "holographic_accents": True,
            "panel_lighting": True,
            "clean_shadows": True,
        },
        
        color={
            "lut": ColorLUT.COOL_STEEL,
            "saturation": 1.0,
            "contrast": 1.15,
            "temperature": -20,
            "accent_color": (0.0, 0.8, 1.0),  # Cyan
        },
        
        post={
            "bloom": 0.25,
            "lens_flare": 0.2,
            "chromatic_aberration": 0.15,
            "anamorphic_flare": True,
        },
        
        prompt_prefix="science fiction, futuristic, advanced technology",
        prompt_suffix="sleek design, space age, high-tech environment",
        negative_prompt="old-fashioned, rustic, fantasy, medieval",
    ),
    
    # ========================================
    # WESTERN
    # ========================================
    "western": GenreProfile(
        name="Western",
        description="Classic American Western with dusty, sun-baked aesthetic",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "static_with_slow_dolly",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.DOLLY_IN,
                CameraMovement.PAN_RIGHT,
                CameraMovement.TRACKING_SHOT,
            ],
            "preferred_rig": CameraRig.TRIPOD,
            "low_angle_frequency": 0.3,
        },
        
        shots={
            "preferred_shots": [
                ShotType.EXTREME_LONG_SHOT,
                ShotType.CLOSE_UP,
                ShotType.MEDIUM_SHOT,
                ShotType.ESTABLISHING,
            ],
            "composition": CompositionType.RULE_OF_THIRDS,
            "wide_landscapes": True,
            "standoff_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.NATURAL,
            "harsh_sun": True,
            "dust_particles": True,
            "preferred_times": [TimeOfDay.MIDDAY, TimeOfDay.GOLDEN_HOUR_PM],
        },
        
        color={
            "lut": ColorLUT.GOLDEN_VINTAGE,
            "saturation": 1.0,
            "contrast": 1.2,
            "temperature": 30,
        },
        
        post={
            "grain": 0.3,
            "vignette": 0.25,
            "halation": 0.1,
        },
        
        prompt_prefix="western, cowboy, American frontier",
        prompt_suffix="dusty, sun-baked, old west atmosphere",
        negative_prompt="modern, urban, futuristic, green",
    ),
    
    # ========================================
    # VINTAGE / RETRO
    # ========================================
    "vintage": GenreProfile(
        name="Vintage / Retro",
        description="Old film aesthetic with aged, nostalgic quality",
        
        camera={
            "preferred_lens": 50,
            "movement_style": "classic_smooth",
            "preferred_movements": [
                CameraMovement.STATIC,
                CameraMovement.DOLLY_IN,
                CameraMovement.PAN_LEFT,
            ],
            "preferred_rig": CameraRig.TRIPOD,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.FULL_SHOT,
                ShotType.TWO_SHOT,
            ],
            "composition": CompositionType.CENTER_FRAME,
            "classic_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.THREE_POINT,
            "soft_key": True,
            "warm_fill": True,
        },
        
        color={
            "lut": ColorLUT.VINTAGE_FILM,
            "saturation": 0.9,
            "contrast": 0.95,
            "temperature": 15,
            "lifted_blacks": True,
        },
        
        post={
            "grain": 0.6,
            "vignette": 0.3,
            "halation": 0.25,
            "scratches": 0.15,
            "light_leak": 0.1,
            "film_type": "35mm",
        },
        
        prompt_prefix="vintage, retro, old film",
        prompt_suffix="nostalgic, aged film stock, classic cinema",
        negative_prompt="modern, digital, clean, sharp",
    ),
    
    # ========================================
    # MODERN / CONTEMPORARY
    # ========================================
    "modern": GenreProfile(
        name="Modern / Contemporary",
        description="Clean, contemporary visual style with modern aesthetics",
        
        camera={
            "preferred_lens": 35,
            "movement_style": "smooth_controlled",
            "preferred_movements": [
                CameraMovement.STEADICAM,
                CameraMovement.DOLLY_IN,
                CameraMovement.TRACKING_SHOT,
            ],
            "preferred_rig": CameraRig.GIMBAL,
        },
        
        shots={
            "preferred_shots": [
                ShotType.MEDIUM_SHOT,
                ShotType.CLOSE_UP,
                ShotType.ESTABLISHING,
            ],
            "composition": CompositionType.RULE_OF_THIRDS,
            "clean_framing": True,
        },
        
        lighting={
            "setup": LightingSetup.NATURAL,
            "soft_natural": True,
            "window_light": True,
        },
        
        color={
            "lut": ColorLUT.NATURAL,
            "saturation": 1.05,
            "contrast": 1.05,
            "temperature": 0,
        },
        
        post={
            "bloom": 0.1,
            "vignette": 0.1,
        },
        
        prompt_prefix="modern, contemporary, current day",
        prompt_suffix="clean aesthetic, modern setting, present day",
        negative_prompt="vintage, retro, old, historical",
    ),
}



# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def get_genre_profile(genre: str) -> GenreProfile:
    """
    Get a genre profile by name.
    
    Args:
        genre: Genre name (case-insensitive)
        
    Returns:
        GenreProfile for the specified genre
    """
    genre_lower = genre.lower()
    
    if genre_lower in GENRE_PROFILES:
        return GENRE_PROFILES[genre_lower]
    
    # Try to find partial match
    for key in GENRE_PROFILES:
        if genre_lower in key or key in genre_lower:
            return GENRE_PROFILES[key]
    
    # Default to cinematic
    return GENRE_PROFILES["cinematic"]


def list_genres() -> List[str]:
    """Get list of available genre names."""
    return list(GENRE_PROFILES.keys())


def get_genre_prompt(genre: str) -> Tuple[str, str, str]:
    """
    Get prompt modifiers for a genre.
    
    Args:
        genre: Genre name
        
    Returns:
        Tuple of (prefix, suffix, negative)
    """
    profile = get_genre_profile(genre)
    return (profile.prompt_prefix, profile.prompt_suffix, profile.negative_prompt)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENRE PROFILES TEST")
    print("="*60)
    
    print(f"\n--- Available Genres: {len(GENRE_PROFILES)} ---\n")
    
    for name, profile in GENRE_PROFILES.items():
        print(f"Genre: {profile.name}")
        print(f"  Description: {profile.description[:50]}...")
        print(f"  Camera Lens: {profile.camera.get('preferred_lens', 50)}mm")
        print(f"  Lighting: {profile.lighting.get('setup', 'N/A')}")
        print(f"  LUT: {profile.color.get('lut', 'N/A')}")
        print(f"  Prompt Prefix: {profile.prompt_prefix[:40]}...")
        print()
    
    # Test retrieval
    print("--- Profile Retrieval Test ---\n")
    
    test_genres = ["anime", "PIXAR", "film_noir", "unknown_genre", "cyber"]
    for g in test_genres:
        profile = get_genre_profile(g)
        print(f"'{g}' -> {profile.name}")
    
    print("\n✅ Genre profiles working!")
