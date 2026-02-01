"""
Professional Color Grading System

Industry-grade color science and grading for cinematic video generation.
Implements LUTs, color curves, and post-processing effects used in
film, anime, Pixar, and broadcast production.

This module provides:
1. ColorLUT - Industry standard Look-Up Tables
2. ColorGrade - Color grading parameters
3. PostProcessing - Visual effects (bloom, grain, etc.)
4. ColorGradingEngine - Orchestrates all color decisions
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# ============================================
# COLOR LUTS (14+ types)
# ============================================

class ColorLUT(Enum):
    """
    Industry-standard Look-Up Tables for color grading.
    35+ LUTs covering film stocks, cinematic looks, and animation styles.
    """
    # Modern Film Looks (6)
    TEAL_ORANGE = "teal_orange"          # Hollywood blockbuster standard
    BLEACH_BYPASS = "bleach_bypass"      # Desaturated, high contrast (war films)
    CROSS_PROCESS = "cross_process"      # Inverted curves (retro, music video)
    VINTAGE_FILM = "vintage_film"        # Faded blacks, warm tint
    COOL_STEEL = "cool_steel"            # Blue-gray (sci-fi, thriller)
    WARM_SUNSET = "warm_sunset"          # Orange-red push (romance)
    
    # Classic Looks (3)
    NOIR = "noir"                        # High contrast, desaturated
    SEPIA = "sepia"                      # Brown-toned vintage
    MONOCHROME = "monochrome"            # Black and white
    
    # Animation Styles (4)
    ANIME_POP = "anime_pop"              # Vibrant saturation, sharp
    GHIBLI_SOFT = "ghibli_soft"          # Soft pastels, warm shadows
    PIXAR_BRIGHT = "pixar_bright"        # Clean, saturated, balanced
    CARTOON_VIBRANT = "cartoon_vibrant"  # High saturation, bold
    
    # Genre Specific (3)
    CYBERPUNK_NEON = "cyberpunk_neon"    # High saturation, split toning
    HORROR_GREEN = "horror_green"        # Sickly desaturated green tint
    FANTASY_ETHEREAL = "fantasy_ethereal" # Soft glow, magical
    
    # Time Effects (3)
    DAY_FOR_NIGHT = "day_for_night"      # Blue tint, low brightness
    FADED_MEMORY = "faded_memory"        # Low contrast, lifted blacks
    GOLDEN_VINTAGE = "golden_vintage"    # Warm, aged look
    
    # Natural (2)
    NATURAL = "natural"                  # Minimal grading
    LOG_TO_REC709 = "log_rec709"         # Standard conversion
    
    # Film Stocks - NEW (8)
    KODAK_5219 = "kodak_5219"            # Kodak Vision3 500T (Tungsten)
    KODAK_5207 = "kodak_5207"            # Kodak Vision3 250D (Daylight)
    FUJI_ETERNA = "fuji_eterna"          # Fuji Eterna 500T
    KODAK_PORTRA = "kodak_portra"        # Kodak Portra (portrait)
    KODAK_EKTAR = "kodak_ektar"          # Kodak Ektar (vibrant)
    FUJI_400H = "fuji_400h"              # Fuji Pro 400H (soft pastel)
    KODAK_EKTACHROME = "kodak_ektachrome"  # E6 slide film (saturated)
    CINESTILL_800T = "cinestill_800t"    # Cinestill tungsten (halation)
    
    # Broadcast/HDR - NEW (4)
    REC2020_HDR = "rec2020_hdr"          # Wide gamut HDR
    ACES_FILMIC = "aces_filmic"          # ACES workflow
    BROADCAST_709 = "broadcast_709"      # Standard broadcast
    SDR_PUNCHY = "sdr_punchy"            # Enhanced SDR
    
    # Specialized - NEW (2)
    MICHAEL_BAY = "michael_bay"          # Orange/teal extreme
    DAVID_FINCHER = "david_fincher"      # Dark, desaturated green




# ============================================
# COLOR PARAMETERS
# ============================================

@dataclass
class ColorCurve:
    """RGB curve adjustment."""
    shadows: float = 0.0                 # -1 to 1, lift shadows
    midtones: float = 0.0                # -1 to 1, gamma adjustment
    highlights: float = 0.0              # -1 to 1, pulldown highlights
    
    def to_prompt_description(self) -> str:
        """Get description for prompt."""
        parts = []
        if self.shadows > 0.1:
            parts.append("lifted shadows")
        elif self.shadows < -0.1:
            parts.append("crushed blacks")
        if self.highlights > 0.1:
            parts.append("bright highlights")
        elif self.highlights < -0.1:
            parts.append("rolled off highlights")
        return ", ".join(parts) if parts else ""


@dataclass
class ColorBalance:
    """Color balance for shadows/midtones/highlights."""
    shadow_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB offset
    midtone_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SplitTone:
    """Split toning for shadows and highlights."""
    shadow_hue: float = 0.0              # 0-360 degrees
    shadow_saturation: float = 0.0       # 0-1
    highlight_hue: float = 0.0           # 0-360 degrees
    highlight_saturation: float = 0.0    # 0-1
    balance: float = 0.0                 # -1 to 1, shift toward shadow/highlight


# ============================================
# COLOR GRADE SPECIFICATION
# ============================================

@dataclass
class ColorGrade:
    """
    Complete color grading specification.
    """
    # LUT
    lut: ColorLUT = ColorLUT.NATURAL
    lut_intensity: float = 1.0           # 0-1, blend with original
    
    # Basic adjustments
    saturation: float = 1.0              # 0-2
    vibrance: float = 1.0                # 0-2, protects skin tones
    contrast: float = 1.0                # 0-2
    brightness: float = 0.0              # -1 to 1
    exposure: float = 0.0                # -3 to 3 stops
    
    # Curves
    curve: ColorCurve = field(default_factory=ColorCurve)
    
    # Color balance
    balance: ColorBalance = field(default_factory=ColorBalance)
    
    # Split toning
    split_tone: SplitTone = field(default_factory=SplitTone)
    
    # Temperature and tint
    temperature: float = 0.0             # -100 to 100 (cool to warm)
    tint: float = 0.0                    # -100 to 100 (green to magenta)
    
    # Skin tone protection
    preserve_skin_tones: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lut": self.lut.value,
            "lut_intensity": self.lut_intensity,
            "saturation": self.saturation,
            "vibrance": self.vibrance,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "exposure": self.exposure,
            "temperature": self.temperature,
            "tint": self.tint,
            "curve": {
                "shadows": self.curve.shadows,
                "midtones": self.curve.midtones,
                "highlights": self.curve.highlights,
            },
        }
    
    def to_prompt_modifiers(self) -> List[str]:
        """Generate prompt modifiers for this color grade."""
        modifiers = []
        
        # LUT descriptions
        lut_prompts = {
            ColorLUT.TEAL_ORANGE: ["teal and orange color grading", "Hollywood color grade"],
            ColorLUT.BLEACH_BYPASS: ["bleach bypass", "desaturated high contrast"],
            ColorLUT.CROSS_PROCESS: ["cross-processed colors", "vintage color shift"],
            ColorLUT.VINTAGE_FILM: ["vintage film colors", "faded retro look"],
            ColorLUT.COOL_STEEL: ["cool blue-gray tones", "steel blue grade"],
            ColorLUT.WARM_SUNSET: ["warm sunset tones", "golden color grade"],
            ColorLUT.NOIR: ["noir black and white", "high contrast monochrome"],
            ColorLUT.SEPIA: ["sepia toned", "brown vintage"],
            ColorLUT.MONOCHROME: ["black and white", "monochrome"],
            ColorLUT.ANIME_POP: ["vibrant anime colors", "saturated bold colors"],
            ColorLUT.GHIBLI_SOFT: ["soft pastel colors", "Ghibli color palette"],
            ColorLUT.PIXAR_BRIGHT: ["bright clean colors", "Pixar color grade"],
            ColorLUT.CARTOON_VIBRANT: ["vibrant cartoon colors", "bold saturated"],
            ColorLUT.CYBERPUNK_NEON: ["neon cyberpunk colors", "split toning"],
            ColorLUT.HORROR_GREEN: ["sickly green tint", "horror color grade"],
            ColorLUT.FANTASY_ETHEREAL: ["ethereal magical colors", "soft glow"],
            ColorLUT.DAY_FOR_NIGHT: ["day for night", "blue night simulation"],
            ColorLUT.FADED_MEMORY: ["faded memory colors", "low contrast vintage"],
            ColorLUT.GOLDEN_VINTAGE: ["golden vintage tones", "warm aged"],
            ColorLUT.NATURAL: ["natural colors", "realistic color"],
        }
        modifiers.extend(lut_prompts.get(self.lut, ["cinematic color grading"]))
        
        # Saturation
        if self.saturation > 1.2:
            modifiers.append("highly saturated colors")
        elif self.saturation < 0.7:
            modifiers.append("desaturated muted colors")
        
        # Contrast
        if self.contrast > 1.3:
            modifiers.append("high contrast")
        elif self.contrast < 0.7:
            modifiers.append("low contrast")
        
        # Temperature
        if self.temperature > 30:
            modifiers.append("warm color temperature")
        elif self.temperature < -30:
            modifiers.append("cool color temperature")
        
        # Curve description
        curve_desc = self.curve.to_prompt_description()
        if curve_desc:
            modifiers.append(curve_desc)
        
        return modifiers


# ============================================
# POST-PROCESSING EFFECTS
# ============================================

class PostProcessEffect(Enum):
    """
    Post-processing visual effects - 25 types.
    Industry-standard effects for film, broadcast, and animation.
    """
    # Optical Effects (7)
    BLOOM = "bloom"                      # Highlight glow
    CHROMATIC_ABERRATION = "ca"          # Color fringing
    LENS_FLARE = "lens_flare"            # Light artifact
    ANAMORPHIC_FLARE = "anamorphic"      # Horizontal blue flares
    HALATION = "halation"                # Red light bleed (film)
    DEPTH_OF_FIELD = "dof"               # Background blur
    BARREL_DISTORTION = "barrel"         # Lens distortion
    
    # Film Simulation (6)
    FILM_GRAIN = "grain"                 # Organic texture
    LIGHT_LEAK = "light_leak"            # Film light leak effect
    SCRATCHES = "scratches"              # Old film damage
    DUST_PARTICLES = "dust"              # Floating dust/dirt
    GATE_WEAVE = "gate_weave"            # Film projector wobble
    FLICKER = "flicker"                  # Old projection flicker
    
    # Diffusion/Focus (4)
    VIGNETTE = "vignette"                # Edge darkening
    GLOW = "glow"                        # Overall soft glow
    SOFTENING = "soften"                 # Diffusion
    SHARPENING = "sharpen"               # Detail enhancement
    
    # Motion (2)
    MOTION_BLUR = "motion_blur"          # Movement trails
    SPEED_LINES = "speed_lines"          # Anime motion effect
    
    # Retro/VFX (6)
    SCAN_LINES = "scan_lines"            # CRT effect
    VHS_GLITCH = "vhs_glitch"            # Analog distortion
    DIGITAL_GLITCH = "digital_glitch"    # Datamosh effect
    PIXELATE = "pixelate"                # Low resolution
    POSTERIZE = "posterize"              # Reduced colors
    DITHER = "dither"                    # Retro dithering




@dataclass
class PostProcessSpec:
    """
    Post-processing effects specification.
    """
    # Individual effects (0-1 intensity)
    bloom: float = 0.0
    chromatic_aberration: float = 0.0
    vignette: float = 0.0
    film_grain: float = 0.0
    halation: float = 0.0
    lens_flare: float = 0.0
    motion_blur: float = 0.0
    sharpening: float = 0.0
    softening: float = 0.0
    glow: float = 0.0
    
    # Film simulation
    film_type: str = "none"              # none, 35mm, 16mm, super8
    
    # Aspect ratio
    letterbox: float = 0.0               # 0-1, cinematic bars
    aspect_ratio: str = "16:9"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bloom": self.bloom,
            "chromatic_aberration": self.chromatic_aberration,
            "vignette": self.vignette,
            "film_grain": self.film_grain,
            "halation": self.halation,
            "lens_flare": self.lens_flare,
            "motion_blur": self.motion_blur,
            "sharpening": self.sharpening,
            "softening": self.softening,
            "glow": self.glow,
            "film_type": self.film_type,
            "aspect_ratio": self.aspect_ratio,
        }
    
    def to_prompt_modifiers(self) -> List[str]:
        """Generate prompt modifiers for post-processing."""
        modifiers = []
        
        if self.bloom > 0.3:
            modifiers.append("soft bloom effect")
        if self.chromatic_aberration > 0.2:
            modifiers.append("chromatic aberration")
        if self.vignette > 0.3:
            modifiers.append("vignette")
        if self.film_grain > 0.3:
            modifiers.append("film grain texture")
        if self.halation > 0.2:
            modifiers.append("halation light bleed")
        if self.lens_flare > 0.3:
            modifiers.append("lens flare")
        if self.softening > 0.3:
            modifiers.append("soft diffused look")
        if self.glow > 0.3:
            modifiers.append("soft glow")
        
        if self.film_type != "none":
            modifiers.append(f"{self.film_type} film")
        
        if self.aspect_ratio == "2.39:1":
            modifiers.append("anamorphic widescreen")
        elif self.aspect_ratio == "4:3":
            modifiers.append("4:3 aspect ratio")
        
        return modifiers


# ============================================
# GENRE COLOR PRESETS
# ============================================

GENRE_COLOR_PRESETS = {
    "anime": {
        "lut": ColorLUT.ANIME_POP,
        "saturation": 1.25,
        "vibrance": 1.3,
        "contrast": 1.15,
        "post": {
            "chromatic_aberration": 0.2,
            "bloom": 0.35,
            "sharpening": 0.4,
        },
    },
    "pixar": {
        "lut": ColorLUT.PIXAR_BRIGHT,
        "saturation": 1.1,
        "vibrance": 1.2,
        "contrast": 1.0,
        "post": {
            "bloom": 0.2,
            "vignette": 0.1,
            "softening": 0.15,
        },
    },
    "noir": {
        "lut": ColorLUT.NOIR,
        "saturation": 0.3,
        "vibrance": 0.4,
        "contrast": 1.5,
        "curve": {"shadows": -0.2, "highlights": -0.1},
        "post": {
            "film_grain": 0.5,
            "vignette": 0.4,
        },
    },
    "cyberpunk": {
        "lut": ColorLUT.CYBERPUNK_NEON,
        "saturation": 1.4,
        "vibrance": 1.5,
        "contrast": 1.3,
        "split_tone": {"shadow_hue": 270, "shadow_sat": 0.4, "highlight_hue": 180, "highlight_sat": 0.3},
        "post": {
            "bloom": 0.5,
            "chromatic_aberration": 0.4,
            "lens_flare": 0.3,
        },
    },
    "horror": {
        "lut": ColorLUT.HORROR_GREEN,
        "saturation": 0.6,
        "vibrance": 0.7,
        "contrast": 1.25,
        "temperature": -20,
        "post": {
            "vignette": 0.5,
            "film_grain": 0.3,
        },
    },
    "romance": {
        "lut": ColorLUT.WARM_SUNSET,
        "saturation": 1.05,
        "vibrance": 1.1,
        "contrast": 0.95,
        "temperature": 25,
        "post": {
            "bloom": 0.35,
            "softening": 0.25,
            "vignette": 0.15,
        },
    },
    "action": {
        "lut": ColorLUT.TEAL_ORANGE,
        "saturation": 1.1,
        "vibrance": 1.15,
        "contrast": 1.2,
        "post": {
            "bloom": 0.2,
            "sharpening": 0.3,
        },
    },
    "fantasy": {
        "lut": ColorLUT.FANTASY_ETHEREAL,
        "saturation": 1.15,
        "vibrance": 1.2,
        "contrast": 1.0,
        "post": {
            "bloom": 0.4,
            "glow": 0.35,
            "halation": 0.2,
        },
    },
    "ghibli": {
        "lut": ColorLUT.GHIBLI_SOFT,
        "saturation": 1.05,
        "vibrance": 1.1,
        "contrast": 0.9,
        "curve": {"shadows": 0.1},
        "post": {
            "bloom": 0.25,
            "softening": 0.2,
        },
    },
    "documentary": {
        "lut": ColorLUT.NATURAL,
        "saturation": 1.0,
        "vibrance": 1.0,
        "contrast": 1.05,
        "post": {
            "film_grain": 0.1,
        },
    },
    "cinematic": {
        "lut": ColorLUT.TEAL_ORANGE,
        "saturation": 1.05,
        "vibrance": 1.1,
        "contrast": 1.1,
        "post": {
            "bloom": 0.15,
            "vignette": 0.2,
            "film_grain": 0.15,
        },
    },
    "realistic": {
        "lut": ColorLUT.NATURAL,
        "saturation": 1.0,
        "vibrance": 1.0,
        "contrast": 1.0,
        "post": {},
    },
}


# ============================================
# COLOR GRADING ENGINE
# ============================================

class ColorGradingEngine:
    """
    Professional color grading engine that creates
    color specifications based on genre and scene content.
    """
    
    def __init__(self):
        self.default_lut = ColorLUT.NATURAL
    
    def create_grade(
        self,
        description: str,
        genre: str = "cinematic",
        time_of_day: str = "afternoon",
    ) -> Tuple[ColorGrade, PostProcessSpec]:
        """
        Create color grading specification for a scene.
        
        Args:
            description: Scene description
            genre: Visual style
            time_of_day: Time of day for temperature adjustment
            
        Returns:
            Tuple of (ColorGrade, PostProcessSpec)
        """
        desc_lower = description.lower()
        
        # Get genre preset
        preset = GENRE_COLOR_PRESETS.get(genre, GENRE_COLOR_PRESETS["cinematic"])
        
        # Build color grade
        grade = ColorGrade(
            lut=preset.get("lut", self.default_lut),
            saturation=preset.get("saturation", 1.0),
            vibrance=preset.get("vibrance", 1.0),
            contrast=preset.get("contrast", 1.0),
            temperature=preset.get("temperature", 0.0),
        )
        
        # Apply curve if specified
        if "curve" in preset:
            grade.curve = ColorCurve(
                shadows=preset["curve"].get("shadows", 0.0),
                midtones=preset["curve"].get("midtones", 0.0),
                highlights=preset["curve"].get("highlights", 0.0),
            )
        
        # Apply split tone if specified
        if "split_tone" in preset:
            st = preset["split_tone"]
            grade.split_tone = SplitTone(
                shadow_hue=st.get("shadow_hue", 0.0),
                shadow_saturation=st.get("shadow_sat", 0.0),
                highlight_hue=st.get("highlight_hue", 0.0),
                highlight_saturation=st.get("highlight_sat", 0.0),
            )
        
        # Adjust for time of day
        grade = self._adjust_for_time(grade, time_of_day)
        
        # Detect emotional adjustments
        grade = self._apply_emotional_adjustments(grade, desc_lower)
        
        # Build post-processing
        post_preset = preset.get("post", {})
        post = PostProcessSpec(
            bloom=post_preset.get("bloom", 0.0),
            chromatic_aberration=post_preset.get("chromatic_aberration", 0.0),
            vignette=post_preset.get("vignette", 0.0),
            film_grain=post_preset.get("film_grain", 0.0),
            halation=post_preset.get("halation", 0.0),
            lens_flare=post_preset.get("lens_flare", 0.0),
            sharpening=post_preset.get("sharpening", 0.0),
            softening=post_preset.get("softening", 0.0),
            glow=post_preset.get("glow", 0.0),
        )
        
        return grade, post
    
    def _adjust_for_time(self, grade: ColorGrade, time: str) -> ColorGrade:
        """Adjust color grade for time of day."""
        time_adjustments = {
            "dawn": {"temperature": -15, "saturation_mod": 0.95},
            "golden_am": {"temperature": 30, "saturation_mod": 1.1},
            "morning": {"temperature": 5, "saturation_mod": 1.0},
            "midday": {"temperature": 0, "contrast_mod": 1.1},
            "afternoon": {"temperature": 10, "saturation_mod": 1.02},
            "golden_pm": {"temperature": 35, "saturation_mod": 1.15},
            "dusk": {"temperature": 15, "saturation_mod": 0.95},
            "blue_hour": {"temperature": -25, "saturation_mod": 1.0},
            "night": {"temperature": -10, "brightness_mod": -0.1},
            "neon_night": {"temperature": -5, "saturation_mod": 1.3},
            "moonlit": {"temperature": -30, "saturation_mod": 0.8},
        }
        
        adj = time_adjustments.get(time, {})
        
        if "temperature" in adj:
            grade.temperature += adj["temperature"]
        if "saturation_mod" in adj:
            grade.saturation *= adj["saturation_mod"]
        if "contrast_mod" in adj:
            grade.contrast *= adj["contrast_mod"]
        if "brightness_mod" in adj:
            grade.brightness += adj["brightness_mod"]
        
        return grade
    
    def _apply_emotional_adjustments(self, grade: ColorGrade, desc: str) -> ColorGrade:
        """Apply emotional color adjustments based on scene."""
        # Sad/melancholy
        if any(w in desc for w in ["sad", "melancholy", "grief", "mourning", "tears"]):
            grade.saturation *= 0.85
            grade.temperature -= 10
        
        # Happy/joyful
        elif any(w in desc for w in ["happy", "joy", "celebration", "laughter", "cheerful"]):
            grade.saturation *= 1.1
            grade.temperature += 10
        
        # Intense/dramatic
        elif any(w in desc for w in ["intense", "dramatic", "confrontation", "rage"]):
            grade.contrast *= 1.15
        
        # Peaceful/calm
        elif any(w in desc for w in ["peaceful", "calm", "serene", "tranquil"]):
            grade.contrast *= 0.95
        
        # Mysterious
        elif any(w in desc for w in ["mysterious", "enigma", "secret", "hidden"]):
            grade.curve.shadows -= 0.1
        
        return grade
    
    def get_lut_description(self, lut: ColorLUT) -> str:
        """Get description of LUT effect."""
        descriptions = {
            ColorLUT.TEAL_ORANGE: "Hollywood blockbuster look with complementary teal shadows and orange midtones",
            ColorLUT.BLEACH_BYPASS: "High contrast, desaturated look inspired by skipping bleach in film development",
            ColorLUT.VINTAGE_FILM: "Faded film look with lifted blacks and warm color shift",
            ColorLUT.ANIME_POP: "Vibrant, highly saturated colors with sharp contrast typical of anime",
            ColorLUT.GHIBLI_SOFT: "Soft pastel tones with warm shadows inspired by Studio Ghibli",
            ColorLUT.CYBERPUNK_NEON: "High saturation with electric pinks, teals, and purples",
            ColorLUT.NOIR: "Classic black and white with deep contrast and minimal mid-grays",
        }
        return descriptions.get(lut, "Cinematic color grading")


# Singleton instance
_grading_engine: Optional[ColorGradingEngine] = None


def get_grading_engine() -> ColorGradingEngine:
    """Get or create the global color grading engine."""
    global _grading_engine
    if _grading_engine is None:
        _grading_engine = ColorGradingEngine()
    return _grading_engine


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def create_grade(
    description: str,
    genre: str = "cinematic",
    time: str = "afternoon"
) -> Tuple[ColorGrade, PostProcessSpec]:
    """
    Convenience function to create color grading.
    
    Args:
        description: Scene description
        genre: Visual style
        time: Time of day
        
    Returns:
        Tuple of (ColorGrade, PostProcessSpec)
    """
    engine = get_grading_engine()
    return engine.create_grade(description, genre, time)


def get_color_prompt(grade: ColorGrade, post: PostProcessSpec) -> str:
    """
    Generate prompt modifiers from color specs.
    
    Args:
        grade: ColorGrade configuration
        post: PostProcessSpec configuration
        
    Returns:
        String of prompt modifiers
    """
    modifiers = grade.to_prompt_modifiers() + post.to_prompt_modifiers()
    return ", ".join(modifiers)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROFESSIONAL COLOR GRADING ENGINE TEST")
    print("="*60)
    
    engine = ColorGradingEngine()
    
    # Test various scenes
    test_scenes = [
        ("Two warriors clash in epic battle", "action", "afternoon"),
        ("Romantic sunset scene on the beach", "romance", "golden_pm"),
        ("Dark mysterious alley at night", "noir", "night"),
        ("Bright cheerful anime school scene", "anime", "morning"),
        ("Cyberpunk city with neon lights", "cyberpunk", "neon_night"),
        ("Magical forest with ethereal glow", "fantasy", "golden_am"),
        ("Soft meadow scene with gentle breeze", "ghibli", "afternoon"),
        ("Creepy abandoned hospital", "horror", "night"),
        ("Pixar-style family kitchen", "pixar", "morning"),
        ("Sad farewell at the train station", "cinematic", "dusk"),
    ]
    
    print("\n--- Color Grading Test ---\n")
    
    for desc, genre, time in test_scenes:
        grade, post = engine.create_grade(desc, genre, time)
        print(f"Scene: \"{desc[:35]}...\"")
        print(f"  Genre: {genre}, Time: {time}")
        print(f"  LUT: {grade.lut.value}")
        print(f"  Saturation: {grade.saturation:.2f}")
        print(f"  Contrast: {grade.contrast:.2f}")
        print(f"  Temperature: {grade.temperature:.0f}K offset")
        effects = [k for k, v in post.to_dict().items() if isinstance(v, (int, float)) and v > 0.1]
        if effects:
            print(f"  Post FX: {', '.join(effects)}")
        print()
    
    # Test all LUTs
    print(f"\n--- Available LUTs: {len(ColorLUT)} ---")
    for lut in ColorLUT:
        print(f"  {lut.value}: {lut.name}")
    
    print(f"\n--- Genre Presets: {len(GENRE_COLOR_PRESETS)} ---")
    for genre in GENRE_COLOR_PRESETS.keys():
        preset = GENRE_COLOR_PRESETS[genre]
        print(f"  {genre}: {preset['lut'].value}")
    
    print("\n✅ Color grading engine working!")
