"""
Professional Lighting Engine

Industry-grade lighting techniques for cinematic video generation.
Implements 15+ lighting setups, time-of-day systems, and genre-specific
lighting used in film, anime, Pixar, and broadcast production.

This module provides:
1. LightingSetup - Core lighting configurations (three-point, Rembrandt, etc.)
2. TimeOfDay - Time-based lighting (golden hour, blue hour, etc.)
3. GenreLighting - Genre-specific lighting presets
4. LightingEngine - Orchestrates all lighting decisions
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# ============================================
# LIGHTING SETUPS (15+ types)
# ============================================

class LightingSetup(Enum):
    """
    Professional lighting setups used in film and photography.
    40+ setups covering portrait, mood, accent, natural, special, and animated styles.
    """
    # Classic portrait lighting (7)
    THREE_POINT = "three_point"          # Key + Fill + Back (standard)
    REMBRANDT = "rembrandt"              # Triangle shadow on cheek
    LOOP = "loop"                        # Small shadow from nose
    SPLIT = "split"                      # Half/half lighting
    BUTTERFLY = "butterfly"              # Beauty lighting, shadow under nose
    BROAD = "broad"                      # Lit side toward camera
    SHORT = "short"                      # Lit side away from camera
    
    # Mood lighting (4)
    HIGH_KEY = "high_key"                # Bright, minimal shadows (comedy)
    LOW_KEY = "low_key"                  # Dark, heavy shadows (drama/noir)
    SILHOUETTE = "silhouette"            # Backlit, no detail
    CHIAROSCURO = "chiaroscuro"          # Extreme contrast (Caravaggio style)
    
    # Accent lighting (4)
    RIM = "rim"                          # Edge/back light for separation
    EDGE = "edge"                        # Similar to rim, more stylized
    HAIR = "hair"                        # Top back light for hair highlight
    KICK = "kick"                        # Side accent light
    
    # Practical and natural (4)
    PRACTICAL = "practical"              # Visible in-scene light sources
    MOTIVATED = "motivated"              # Light from apparent source
    NATURAL = "natural"                  # Simulating natural light
    AVAILABLE = "available"              # Using existing light
    
    # Special effects (4)
    UNDER_LIGHTING = "under"             # Light from below (horror)
    TOP_LIGHTING = "top"                 # Light from above
    SIDE_LIGHTING = "side"               # Strong side light
    CATCH_LIGHT = "catch"                # Reflected light in eyes
    
    # Animated styles (3)
    FLAT = "flat"                        # Minimal shadows (cel animation)
    CEL_SHADED = "cel_shaded"            # Sharp shadow edges (anime)
    AMBIENT_OCCLUSION = "ao"             # Soft contact shadows (Pixar)
    
    # Film industry advanced (14 new)
    BOOK_LIGHT = "book_light"            # Bounced through diffusion (soft cinema)
    NEGATIVE_FILL = "negative_fill"      # Black side to deepen shadows
    BOUNCE = "bounce"                    # Reflected/bounced fill light
    CLAMSHELL = "clamshell"              # Beauty: key above, fill below
    CROSS = "cross"                      # Two keys from opposite sides
    DAYLIGHT_TUNGSTEN = "mixed_temp"     # Mixing color temperatures
    HALO = "halo"                        # 360° rim around subject
    WINDOW = "window"                    # Simulated window light
    CAMPFIRE = "campfire"                # Warm flickering bottom light
    OVERHEAD = "overhead"                # Soft overhead panel (interview)
    EYE_LIGHT = "eye_light"              # Small light for eye sparkle
    VOLUMETRIC = "volumetric"            # Visible light beams/rays
    RING_LIGHT = "ring"                  # Even frontal (beauty/vlog)
    FRESNEL_SPOT = "spot"                # Hard focused spotlight



class TimeOfDay(Enum):
    """Time of day lighting conditions - 18 options."""
    # Dawn/Morning (4)
    DAWN = "dawn"                        # Blue hour, cool tones
    GOLDEN_HOUR_AM = "golden_am"         # Early morning warm light
    MORNING = "morning"                  # Soft morning light
    LATE_MORNING = "late_morning"        # Bright but not harsh
    
    # Day (4)
    MIDDAY = "midday"                    # Harsh overhead sun
    AFTERNOON = "afternoon"              # Warm afternoon
    LATE_AFTERNOON = "late_afternoon"    # Long shadows, warm
    GOLDEN_HOUR_PM = "golden_pm"         # Late afternoon magic hour
    
    # Evening/Night (6)
    DUSK = "dusk"                        # Twilight, cool/warm mix
    BLUE_HOUR = "blue_hour"              # Deep blue sky, city lights
    NIGHT = "night"                      # Dark, artificial light
    NEON_NIGHT = "neon_night"            # Urban, colorful lights
    MOONLIT = "moonlit"                  # Cool moonlight
    MIDNIGHT = "midnight"                # Deep night, minimal light
    
    # Weather/Interior (4)
    OVERCAST = "overcast"                # Soft, even, cloudy
    STORMY = "stormy"                    # Dark clouds, dramatic
    FOGGY = "foggy"                      # Diffused, low visibility
    INTERIOR = "interior"                # Controlled indoor lighting


class LightQuality(Enum):
    """Light quality/hardness."""
    HARD = "hard"                        # Sharp shadows
    MEDIUM = "medium"                    # Moderate shadows
    SOFT = "soft"                        # Gradient shadows
    DIFFUSED = "diffused"                # Very soft, even
    MIXED = "mixed"                      # Combination


class LightModifier(Enum):
    """
    Professional light modifiers used in film/photo production.
    These shape and control light quality.
    """
    # Diffusion (5)
    SOFTBOX = "softbox"                  # Large soft source
    SILK = "silk"                        # Frame with diffusion fabric
    SCRIM = "scrim"                      # Reduces intensity, softens
    CHINA_BALL = "china_ball"            # Spherical paper lantern
    BOOK = "book"                        # Bounce + diffusion combo
    
    # Reflection (3)
    BOUNCE_CARD = "bounce"               # White/silver reflector
    SILVER_REFLECTOR = "silver"          # Hard silver bounce
    GOLD_REFLECTOR = "gold"              # Warm gold bounce
    
    # Control/Shaping (5)
    FLAG = "flag"                        # Blocks light (solid black)
    CUTTER = "cutter"                    # Narrow flag, precise control
    BARNDOORS = "barndoors"              # 4-sided light control
    SNOOT = "snoot"                      # Narrow beam focus
    GRID = "grid"                        # Controls spill, directional
    
    # Special (2)
    GEL_CTB = "ctb"                      # Color temp blue (cool)
    GEL_CTO = "cto"                      # Color temp orange (warm)




# ============================================
# LIGHTING SPECIFICATION
# ============================================

@dataclass
class LightSource:
    """Individual light source configuration."""
    name: str                            # Key, fill, back, etc.
    intensity: float = 1.0               # Brightness (0-1)
    color_temperature: int = 5500        # Kelvin (2700-10000)
    color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB override
    quality: LightQuality = LightQuality.SOFT
    angle_h: float = 45.0                # Horizontal angle from subject
    angle_v: float = 30.0                # Vertical angle
    falloff: str = "natural"             # natural, harsh, none
    
    def get_color_description(self) -> str:
        """Get natural language description of light color."""
        if self.color_temperature <= 3000:
            return "warm tungsten"
        elif self.color_temperature <= 4000:
            return "warm"
        elif self.color_temperature <= 5000:
            return "neutral warm"
        elif self.color_temperature <= 6000:
            return "neutral"
        elif self.color_temperature <= 7000:
            return "cool"
        else:
            return "cool blue"


@dataclass
class LightingSpec:
    """
    Complete lighting specification for a scene.
    """
    # Setup
    setup: LightingSetup = LightingSetup.THREE_POINT
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    
    # Individual lights
    lights: List[LightSource] = field(default_factory=list)
    
    # Overall parameters
    ambient_intensity: float = 0.2       # Overall ambient level
    contrast_ratio: str = "normal"       # low, normal, high, extreme
    shadow_density: float = 0.5          # How dark shadows are
    shadow_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)  # Shadow tint
    
    # Atmosphere
    fog_density: float = 0.0             # Atmospheric haze
    bloom_intensity: float = 0.0         # Light bloom effect
    god_rays: bool = False               # Volumetric light rays
    
    # Special effects
    colored_gels: List[Tuple[str, Tuple[float, float, float]]] = field(default_factory=list)
    practical_sources: List[str] = field(default_factory=list)  # e.g., "lamp", "window"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Handle time_of_day being either enum or string
        time_value = self.time_of_day.value if hasattr(self.time_of_day, 'value') else str(self.time_of_day)
        
        return {
            "setup": self.setup.value,
            "time_of_day": time_value,
            "lights": [
                {
                    "name": l.name,
                    "intensity": l.intensity,
                    "color_temp": l.color_temperature,
                    "quality": l.quality.value,
                }
                for l in self.lights
            ],
            "ambient": self.ambient_intensity,
            "contrast": self.contrast_ratio,
            "shadow_density": self.shadow_density,
            "fog": self.fog_density,
            "bloom": self.bloom_intensity,
            "god_rays": self.god_rays,
        }
    
    def to_prompt_modifiers(self) -> List[str]:
        """Generate prompt modifiers for this lighting."""
        modifiers = []
        
        # Setup descriptions
        setup_prompts = {
            LightingSetup.THREE_POINT: ["three-point lighting", "professional lighting"],
            LightingSetup.REMBRANDT: ["Rembrandt lighting", "dramatic triangle shadow"],
            LightingSetup.SPLIT: ["split lighting", "half face in shadow"],
            LightingSetup.LOOP: ["loop lighting", "natural portrait lighting"],
            LightingSetup.BUTTERFLY: ["butterfly lighting", "beauty lighting"],
            LightingSetup.HIGH_KEY: ["high key lighting", "bright and airy"],
            LightingSetup.LOW_KEY: ["low key lighting", "dramatic shadows"],
            LightingSetup.SILHOUETTE: ["silhouette", "backlit figure"],
            LightingSetup.CHIAROSCURO: ["chiaroscuro lighting", "dramatic contrast"],
            LightingSetup.RIM: ["rim lighting", "edge light separation"],
            LightingSetup.PRACTICAL: ["practical lighting", "visible light sources"],
            LightingSetup.NATURAL: ["natural lighting", "realistic light"],
            LightingSetup.UNDER_LIGHTING: ["under lighting", "lit from below"],
            LightingSetup.FLAT: ["flat lighting", "minimal shadows"],
            LightingSetup.CEL_SHADED: ["cel-shaded lighting", "anime style shadows"],
            LightingSetup.AMBIENT_OCCLUSION: ["soft ambient occlusion", "Pixar style lighting"],
        }
        modifiers.extend(setup_prompts.get(self.setup, ["cinematic lighting"]))
        
        # Time of day
        time_prompts = {
            TimeOfDay.DAWN: ["dawn lighting", "blue hour", "early morning"],
            TimeOfDay.GOLDEN_HOUR_AM: ["golden hour", "warm morning light", "magic hour"],
            TimeOfDay.MORNING: ["soft morning light", "gentle daylight"],
            TimeOfDay.MIDDAY: ["midday sun", "harsh overhead light"],
            TimeOfDay.AFTERNOON: ["afternoon light", "warm daylight"],
            TimeOfDay.GOLDEN_HOUR_PM: ["golden hour", "sunset lighting", "magic hour"],
            TimeOfDay.DUSK: ["dusk", "twilight", "evening light"],
            TimeOfDay.BLUE_HOUR: ["blue hour", "deep blue sky"],
            TimeOfDay.NIGHT: ["night scene", "dark lighting", "nighttime"],
            TimeOfDay.OVERCAST: ["overcast", "soft diffused light", "cloudy day"],
            TimeOfDay.NEON_NIGHT: ["neon lights", "urban night", "colorful city lights"],
            TimeOfDay.MOONLIT: ["moonlight", "cool blue night", "lunar lighting"],
        }
        modifiers.extend(time_prompts.get(self.time_of_day, []))
        
        # Contrast
        if self.contrast_ratio == "high":
            modifiers.append("high contrast")
        elif self.contrast_ratio == "extreme":
            modifiers.extend(["extreme contrast", "dramatic shadows"])
        elif self.contrast_ratio == "low":
            modifiers.append("low contrast")
        
        # Atmosphere
        if self.fog_density > 0.3:
            modifiers.append("atmospheric haze")
        if self.god_rays:
            modifiers.append("god rays")
            modifiers.append("volumetric light")
        if self.bloom_intensity > 0.3:
            modifiers.append("soft bloom")
        
        return modifiers
    
    def get_mood_description(self) -> str:
        """Get natural language description of lighting mood."""
        mood_map = {
            LightingSetup.HIGH_KEY: "bright, cheerful, uplifting",
            LightingSetup.LOW_KEY: "moody, dramatic, mysterious",
            LightingSetup.SILHOUETTE: "mysterious, dramatic, anonymous",
            LightingSetup.CHIAROSCURO: "intense, dramatic, classical",
            LightingSetup.REMBRANDT: "sophisticated, dramatic, artistic",
            LightingSetup.FLAT: "neutral, clean, illustrative",
            LightingSetup.CEL_SHADED: "stylized, animated, bold",
            LightingSetup.THREE_POINT: "professional, balanced, natural",
        }
        return mood_map.get(self.setup, "cinematic")


# ============================================
# GENRE LIGHTING PRESETS
# ============================================

GENRE_LIGHTING = {
    "anime": {
        "setup": LightingSetup.CEL_SHADED,
        "contrast": "high",
        "shadow_density": 0.8,
        "ambient": 0.3,
        "bloom": 0.4,
        "specular_highlights": True,
        "shadow_color": (0.15, 0.12, 0.2),  # Purple tint
    },
    "pixar": {
        "setup": LightingSetup.AMBIENT_OCCLUSION,
        "contrast": "normal",
        "shadow_density": 0.4,
        "ambient": 0.4,
        "bloom": 0.2,
        "subsurface_scattering": True,
        "shadow_color": (0.2, 0.18, 0.22),
    },
    "noir": {
        "setup": LightingSetup.LOW_KEY,
        "contrast": "extreme",
        "shadow_density": 0.9,
        "ambient": 0.05,
        "bloom": 0.0,
        "venetian_blinds": True,
        "shadow_color": (0.02, 0.02, 0.02),
    },
    "cyberpunk": {
        "setup": LightingSetup.PRACTICAL,
        "contrast": "high",
        "shadow_density": 0.7,
        "ambient": 0.15,
        "bloom": 0.5,
        "neon_colors": True,
        "fog_density": 0.3,
        "shadow_color": (0.1, 0.05, 0.15),
    },
    "horror": {
        "setup": LightingSetup.UNDER_LIGHTING,
        "contrast": "high",
        "shadow_density": 0.85,
        "ambient": 0.1,
        "bloom": 0.1,
        "shadow_color": (0.05, 0.08, 0.05),  # Sickly green
    },
    "romance": {
        "setup": LightingSetup.THREE_POINT,
        "contrast": "low",
        "shadow_density": 0.3,
        "ambient": 0.4,
        "bloom": 0.3,
        "shadow_color": (0.2, 0.15, 0.18),  # Warm
    },
    "action": {
        "setup": LightingSetup.RIM,
        "contrast": "high",
        "shadow_density": 0.6,
        "ambient": 0.2,
        "bloom": 0.2,
        "shadow_color": (0.1, 0.1, 0.12),
    },
    "fantasy": {
        "setup": LightingSetup.MOTIVATED,
        "contrast": "normal",
        "shadow_density": 0.5,
        "ambient": 0.35,
        "bloom": 0.4,
        "god_rays": True,
        "shadow_color": (0.15, 0.12, 0.2),  # Magical purple
    },
    "documentary": {
        "setup": LightingSetup.AVAILABLE,
        "contrast": "normal",
        "shadow_density": 0.5,
        "ambient": 0.3,
        "bloom": 0.0,
        "shadow_color": (0.15, 0.15, 0.15),
    },
    "ghibli": {
        "setup": LightingSetup.FLAT,
        "contrast": "low",
        "shadow_density": 0.3,
        "ambient": 0.5,
        "bloom": 0.25,
        "shadow_color": (0.3, 0.25, 0.35),  # Soft purple
    },
    "realistic": {
        "setup": LightingSetup.NATURAL,
        "contrast": "normal",
        "shadow_density": 0.5,
        "ambient": 0.25,
        "bloom": 0.1,
        "shadow_color": (0.12, 0.12, 0.15),
    },
}


# ============================================
# LIGHTING ENGINE
# ============================================

class LightingEngine:
    """
    Professional lighting engine that selects appropriate lighting
    based on scene content, genre, and time of day.
    """
    
    # Keywords for lighting detection
    LIGHTING_KEYWORDS = {
        LightingSetup.LOW_KEY: ["dark", "shadow", "mysterious", "noir", "moody", "dramatic"],
        LightingSetup.HIGH_KEY: ["bright", "cheerful", "happy", "comedy", "light", "airy"],
        LightingSetup.SILHOUETTE: ["silhouette", "backlit", "anonymous", "mysterious figure"],
        LightingSetup.RIM: ["edge lit", "dramatic", "action", "hero", "powerful"],
        LightingSetup.NATURAL: ["natural", "outdoor", "daylight", "realistic"],
        LightingSetup.PRACTICAL: ["lamp", "candle", "fire", "screen", "neon"],
        LightingSetup.UNDER_LIGHTING: ["horror", "scary", "creepy", "monster"],
        LightingSetup.CHIAROSCURO: ["renaissance", "classical", "painting", "dramatic art"],
    }
    
    TIME_KEYWORDS = {
        TimeOfDay.DAWN: ["dawn", "sunrise", "early morning", "first light"],
        TimeOfDay.GOLDEN_HOUR_AM: ["golden hour", "morning glow", "warm morning"],
        TimeOfDay.MORNING: ["morning", "breakfast", "starting day"],
        TimeOfDay.MIDDAY: ["noon", "midday", "harsh sun", "bright day"],
        TimeOfDay.AFTERNOON: ["afternoon", "daytime", "sunny"],
        TimeOfDay.GOLDEN_HOUR_PM: ["sunset", "golden hour", "evening glow", "magic hour"],
        TimeOfDay.DUSK: ["dusk", "twilight", "evening"],
        TimeOfDay.BLUE_HOUR: ["blue hour", "after sunset", "before night"],
        TimeOfDay.NIGHT: ["night", "dark", "midnight", "late"],
        TimeOfDay.OVERCAST: ["overcast", "cloudy", "grey sky", "rain"],
        TimeOfDay.NEON_NIGHT: ["neon", "club", "city night", "urban night"],
        TimeOfDay.MOONLIT: ["moonlit", "moonlight", "lunar", "full moon"],
    }
    
    def __init__(self):
        self.default_setup = LightingSetup.THREE_POINT
        self.default_time = TimeOfDay.AFTERNOON
    
    def create_lighting(
        self,
        description: str,
        genre: str = "cinematic",
        override_time: Optional[TimeOfDay] = None,
    ) -> LightingSpec:
        """
        Create lighting specification for a scene.
        
        Args:
            description: Scene description
            genre: Visual style
            override_time: Force specific time of day
            
        Returns:
            LightingSpec with complete lighting configuration
        """
        desc_lower = description.lower()
        
        # Get genre preset as base
        genre_preset = GENRE_LIGHTING.get(genre, GENRE_LIGHTING["realistic"])
        
        # Determine time of day
        time = override_time or self._detect_time(desc_lower)
        
        # Determine setup
        setup = self._detect_setup(desc_lower, genre)
        
        # Build light sources
        lights = self._build_light_sources(setup, time, genre_preset)
        
        # Build spec
        spec = LightingSpec(
            setup=setup,
            time_of_day=time,
            lights=lights,
            ambient_intensity=genre_preset.get("ambient", 0.2),
            contrast_ratio=genre_preset.get("contrast", "normal"),
            shadow_density=genre_preset.get("shadow_density", 0.5),
            shadow_color=genre_preset.get("shadow_color", (0.1, 0.1, 0.15)),
            bloom_intensity=genre_preset.get("bloom", 0.0),
            fog_density=genre_preset.get("fog_density", 0.0),
            god_rays=genre_preset.get("god_rays", False),
        )
        
        # Detect practical sources
        spec.practical_sources = self._detect_practicals(desc_lower)
        
        return spec
    
    def _detect_time(self, desc: str) -> TimeOfDay:
        """Detect time of day from description."""
        for time, keywords in self.TIME_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                return time
        return self.default_time
    
    def _detect_setup(self, desc: str, genre: str) -> LightingSetup:
        """Detect appropriate lighting setup."""
        # Check genre preset first
        genre_preset = GENRE_LIGHTING.get(genre)
        if genre_preset:
            base_setup = genre_preset.get("setup", self.default_setup)
        else:
            base_setup = self.default_setup
        
        # Check for specific keywords that override genre
        for setup, keywords in self.LIGHTING_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                return setup
        
        return base_setup
    
    def _build_light_sources(
        self,
        setup: LightingSetup,
        time: TimeOfDay,
        genre_preset: Dict
    ) -> List[LightSource]:
        """Build individual light sources for setup."""
        lights = []
        
        # Get base color temperature for time
        temp = self._get_time_temperature(time)
        
        if setup == LightingSetup.THREE_POINT:
            lights = [
                LightSource("key", 1.0, temp, quality=LightQuality.MEDIUM, angle_h=45, angle_v=30),
                LightSource("fill", 0.5, temp + 500, quality=LightQuality.SOFT, angle_h=-30, angle_v=15),
                LightSource("back", 0.7, temp - 500, quality=LightQuality.HARD, angle_h=180, angle_v=40),
            ]
        elif setup == LightingSetup.REMBRANDT:
            lights = [
                LightSource("key", 1.0, temp, quality=LightQuality.MEDIUM, angle_h=45, angle_v=45),
                LightSource("fill", 0.2, temp + 500, quality=LightQuality.SOFT, angle_h=-60, angle_v=10),
            ]
        elif setup == LightingSetup.LOW_KEY:
            lights = [
                LightSource("key", 1.0, temp, quality=LightQuality.HARD, angle_h=60, angle_v=30),
                LightSource("fill", 0.1, temp, quality=LightQuality.SOFT, angle_h=-45, angle_v=15),
            ]
        elif setup == LightingSetup.HIGH_KEY:
            lights = [
                LightSource("key", 1.0, temp, quality=LightQuality.SOFT, angle_h=30, angle_v=45),
                LightSource("fill", 0.8, temp, quality=LightQuality.DIFFUSED, angle_h=-30, angle_v=30),
                LightSource("back", 0.5, temp, quality=LightQuality.SOFT, angle_h=180, angle_v=30),
            ]
        elif setup == LightingSetup.SILHOUETTE:
            lights = [
                LightSource("back", 1.0, temp, quality=LightQuality.HARD, angle_h=180, angle_v=20),
            ]
        elif setup == LightingSetup.RIM:
            lights = [
                LightSource("key", 0.6, temp, quality=LightQuality.MEDIUM, angle_h=45, angle_v=30),
                LightSource("rim_left", 0.8, temp - 1000, quality=LightQuality.HARD, angle_h=-120, angle_v=30),
                LightSource("rim_right", 0.8, temp - 1000, quality=LightQuality.HARD, angle_h=120, angle_v=30),
            ]
        elif setup == LightingSetup.CEL_SHADED:
            lights = [
                LightSource("key", 1.0, temp, quality=LightQuality.HARD, angle_h=45, angle_v=60),
                LightSource("rim", 0.6, temp + 2000, quality=LightQuality.HARD, angle_h=-150, angle_v=30),
            ]
        elif setup == LightingSetup.FLAT:
            lights = [
                LightSource("ambient", 1.0, temp, quality=LightQuality.DIFFUSED, angle_h=0, angle_v=90),
            ]
        elif setup == LightingSetup.UNDER_LIGHTING:
            lights = [
                LightSource("under", 1.0, 4500, quality=LightQuality.HARD, angle_h=0, angle_v=-45),
            ]
        else:
            # Default to simple key light
            lights = [
                LightSource("key", 1.0, temp, quality=LightQuality.SOFT, angle_h=45, angle_v=30),
            ]
        
        return lights
    
    def _get_time_temperature(self, time: TimeOfDay) -> int:
        """Get color temperature for time of day."""
        temps = {
            TimeOfDay.DAWN: 9000,
            TimeOfDay.GOLDEN_HOUR_AM: 3500,
            TimeOfDay.MORNING: 5000,
            TimeOfDay.MIDDAY: 6500,
            TimeOfDay.AFTERNOON: 5500,
            TimeOfDay.GOLDEN_HOUR_PM: 3000,
            TimeOfDay.DUSK: 4500,
            TimeOfDay.BLUE_HOUR: 9500,
            TimeOfDay.NIGHT: 4000,
            TimeOfDay.OVERCAST: 7000,
            TimeOfDay.NEON_NIGHT: 8000,  # Mix of colors
            TimeOfDay.MOONLIT: 9000,
        }
        return temps.get(time, 5500)
    
    def _detect_practicals(self, desc: str) -> List[str]:
        """Detect practical light sources mentioned in scene."""
        practicals = []
        practical_keywords = {
            "lamp": ["lamp", "lamplight"],
            "candle": ["candle", "candlelight", "candelabra"],
            "fire": ["fire", "firelight", "flames", "bonfire", "fireplace"],
            "window": ["window light", "sunlight through window"],
            "neon": ["neon sign", "neon lights"],
            "screen": ["computer screen", "tv", "monitor", "phone light"],
            "streetlight": ["streetlight", "street lamp"],
            "flashlight": ["flashlight", "torch"],
        }
        
        for practical, keywords in practical_keywords.items():
            if any(kw in desc for kw in keywords):
                practicals.append(practical)
        
        return practicals


# Singleton instance
_lighting_engine: Optional[LightingEngine] = None


def get_lighting_engine() -> LightingEngine:
    """Get or create the global lighting engine."""
    global _lighting_engine
    if _lighting_engine is None:
        _lighting_engine = LightingEngine()
    return _lighting_engine


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def create_lighting(description: str, genre: str = "cinematic") -> LightingSpec:
    """
    Convenience function to create lighting for a scene.
    
    Args:
        description: Scene description
        genre: Visual style
        
    Returns:
        LightingSpec with complete lighting configuration
    """
    engine = get_lighting_engine()
    return engine.create_lighting(description, genre)


def get_lighting_prompt(spec: LightingSpec) -> str:
    """
    Generate prompt modifiers from lighting spec.
    
    Args:
        spec: LightingSpec configuration
        
    Returns:
        String of prompt modifiers
    """
    modifiers = spec.to_prompt_modifiers()
    return ", ".join(modifiers)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROFESSIONAL LIGHTING ENGINE TEST")
    print("="*60)
    
    engine = LightingEngine()
    
    # Test various scenes
    test_scenes = [
        ("Dark mysterious figure in the shadows of the alley", "noir"),
        ("Bright sunny morning in the cheerful town square", "pixar"),
        ("Epic battle at sunset with golden hour lighting", "action"),
        ("Creepy ghost illuminated from below", "horror"),
        ("Romantic dinner scene lit by candlelight", "romance"),
        ("Futuristic city street with neon signs at night", "cyberpunk"),
        ("Anime character with dramatic rim lighting", "anime"),
        ("Wide landscape at blue hour with mountains", "fantasy"),
        ("Documentary interview with natural lighting", "documentary"),
        ("Soft Studio Ghibli meadow scene", "ghibli"),
    ]
    
    print("\n--- Lighting Selection Test ---\n")
    
    for desc, genre in test_scenes:
        spec = engine.create_lighting(desc, genre)
        print(f"Scene: \"{desc[:40]}...\"")
        print(f"  Genre: {genre}")
        print(f"  Setup: {spec.setup.value}")
        print(f"  Time: {spec.time_of_day.value}")
        print(f"  Lights: {len(spec.lights)}")
        print(f"  Contrast: {spec.contrast_ratio}")
        print(f"  Bloom: {spec.bloom_intensity:.2f}")
        if spec.practical_sources:
            print(f"  Practicals: {spec.practical_sources}")
        print()
    
    # Test all setups
    print(f"\n--- Available Lighting Setups: {len(LightingSetup)} ---")
    for setup in LightingSetup:
        print(f"  {setup.value}: {setup.name}")
    
    print(f"\n--- Time of Day Options: {len(TimeOfDay)} ---")
    for tod in TimeOfDay:
        temp = engine._get_time_temperature(tod)
        print(f"  {tod.value}: {temp}K")
    
    print(f"\n--- Genre Presets: {len(GENRE_LIGHTING)} ---")
    for genre in GENRE_LIGHTING.keys():
        print(f"  {genre}")
    
    print("\n✅ Lighting engine working!")
