"""
Professional Camera System

Industry-grade camera vocabulary and simulation for cinematic video generation.
Implements 25+ camera movements, lens characteristics, and professional techniques
used in film, anime, Pixar, and other production studios.

This module provides:
1. CameraMovement - All major camera movement types
2. CameraLens - Lens simulation with focal length characteristics
3. CameraRig - Rig types (steadicam, handheld, crane, dolly)
4. CameraSystem - Main orchestrator for camera behavior
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# ============================================
# CAMERA MOVEMENTS (25+ types)
# ============================================

class CameraMovement(Enum):
    """
    Professional camera movements used in film and animation.
    Each movement has specific emotional and narrative purposes.
    """
    # Static
    STATIC = "static"                   # No movement - contemplation, dialogue
    LOCKED_OFF = "locked_off"           # Tripod, completely still
    
    # Horizontal Movement
    PAN_LEFT = "pan_left"               # Rotate camera left
    PAN_RIGHT = "pan_right"             # Rotate camera right
    WHIP_PAN = "whip_pan"               # Fast pan for transition/disorientation
    SWISH_PAN = "swish_pan"             # Even faster, blurred pan
    TRUCK_LEFT = "truck_left"           # Camera moves left on tracks
    TRUCK_RIGHT = "truck_right"         # Camera moves right on tracks
    
    # Vertical Movement
    TILT_UP = "tilt_up"                 # Rotate camera up
    TILT_DOWN = "tilt_down"             # Rotate camera down
    PEDESTAL_UP = "pedestal_up"         # Camera rises vertically
    PEDESTAL_DOWN = "pedestal_down"     # Camera lowers vertically
    BOOM_UP = "boom_up"                 # Crane arm rises
    BOOM_DOWN = "boom_down"             # Crane arm lowers
    
    # Depth Movement
    DOLLY_IN = "dolly_in"               # Move toward subject on tracks
    DOLLY_OUT = "dolly_out"             # Move away on tracks
    PUSH_IN = "push_in"                 # Dramatic move toward
    PULL_OUT = "pull_out"               # Dramatic reveal back
    ZOOM_IN = "zoom_in"                 # Lens zoom (flattens perspective)
    ZOOM_OUT = "zoom_out"               # Lens zoom out
    DOLLY_ZOOM = "dolly_zoom"           # Vertigo effect (Hitchcock)
    CRASH_ZOOM = "crash_zoom"           # Rapid zoom for impact
    SNAP_ZOOM = "snap_zoom"             # Anime-style instant zoom
    
    # Orbital Movement
    ARC_LEFT = "arc_left"               # Camera orbits left around subject
    ARC_RIGHT = "arc_right"             # Camera orbits right
    ORBIT_360 = "orbit_360"             # Full orbit around subject
    
    # Complex Movement
    TRACKING_SHOT = "tracking_shot"     # Follow moving subject
    DOLLY_TRACK = "dolly_track"         # Combined dolly and track
    CRANE_SHOT = "crane_shot"           # Large vertical + horizontal
    JIB_SHOT = "jib_shot"               # Smaller crane movement
    STEADICAM = "steadicam"             # Smooth floating movement
    HANDHELD = "handheld"               # Organic shake
    FLOATING = "floating"               # Gentle drift (dream-like)
    DRIFT = "drift"                     # Subtle lateral movement
    
    # Special Movements
    PULL_FOCUS = "pull_focus"           # Rack focus between subjects
    FOLLOW = "follow"                   # Camera follows subject
    LEAD = "lead"                       # Camera leads subject
    REVEAL = "reveal"                   # Dramatic reveal movement
    
    # Anime-Specific
    SPEED_LINES = "speed_lines"         # Anime speed effect with camera
    IMPACT_FRAME = "impact_frame"       # Freeze + shake on impact


# ============================================
# CAMERA LENSES
# ============================================

class LensType(Enum):
    """Camera lens categories with focal length ranges."""
    ULTRA_WIDE = "ultra_wide"       # 14-24mm
    WIDE = "wide"                   # 24-35mm
    STANDARD = "standard"           # 35-50mm
    PORTRAIT = "portrait"           # 50-85mm
    TELEPHOTO = "telephoto"         # 85-200mm
    LONG_TELEPHOTO = "long_tele"    # 200mm+
    ANAMORPHIC = "anamorphic"       # Cinematic wide, lens flares
    MACRO = "macro"                 # Extreme close-up


@dataclass
class CameraLens:
    """
    Camera lens simulation with professional characteristics.
    
    Attributes:
        focal_length: Focal length in mm
        aperture: f-stop (lower = wider aperture, shallower DOF)
        lens_type: Category of lens
        distortion: Barrel/pincushion distortion (-1 to 1)
        vignette: Edge darkening amount (0-1)
        chromatic_aberration: Color fringing (0-1)
        bokeh_shape: Shape of out-of-focus highlights
    """
    focal_length: int = 50
    aperture: float = 2.8
    lens_type: LensType = LensType.STANDARD
    distortion: float = 0.0
    vignette: float = 0.1
    chromatic_aberration: float = 0.05
    bokeh_shape: str = "circular"
    anamorphic_squeeze: float = 1.0  # 2.0 for true anamorphic
    
    @classmethod
    def from_mm(cls, mm: int) -> "CameraLens":
        """Create lens from focal length."""
        if mm <= 24:
            lens_type = LensType.ULTRA_WIDE
            distortion = -0.15  # Barrel distortion
            vignette = 0.25
        elif mm <= 35:
            lens_type = LensType.WIDE
            distortion = -0.05
            vignette = 0.15
        elif mm <= 50:
            lens_type = LensType.STANDARD
            distortion = 0.0
            vignette = 0.1
        elif mm <= 85:
            lens_type = LensType.PORTRAIT
            distortion = 0.0
            vignette = 0.08
        elif mm <= 200:
            lens_type = LensType.TELEPHOTO
            distortion = 0.02  # Slight pincushion
            vignette = 0.05
        else:
            lens_type = LensType.LONG_TELEPHOTO
            distortion = 0.05
            vignette = 0.03
        
        return cls(
            focal_length=mm,
            lens_type=lens_type,
            distortion=distortion,
            vignette=vignette,
        )
    
    def get_depth_of_field(self) -> str:
        """Estimate depth of field description."""
        if self.aperture <= 1.8:
            return "extremely_shallow"
        elif self.aperture <= 2.8:
            return "shallow"
        elif self.aperture <= 5.6:
            return "moderate"
        elif self.aperture <= 11:
            return "deep"
        else:
            return "extremely_deep"
    
    def get_compression(self) -> str:
        """Get perspective compression description."""
        if self.focal_length <= 24:
            return "extreme_expansion"
        elif self.focal_length <= 35:
            return "natural_wide"
        elif self.focal_length <= 50:
            return "natural"
        elif self.focal_length <= 85:
            return "slight_compression"
        elif self.focal_length <= 135:
            return "moderate_compression"
        else:
            return "strong_compression"
    
    def to_prompt_modifiers(self) -> List[str]:
        """Generate prompt modifiers for this lens."""
        modifiers = []
        
        # Focal length effects
        if self.focal_length <= 24:
            modifiers.extend(["ultra wide angle", "expansive perspective", "distorted edges"])
        elif self.focal_length <= 35:
            modifiers.extend(["wide angle", "environmental context"])
        elif self.focal_length <= 50:
            modifiers.extend(["natural perspective", "standard lens"])
        elif self.focal_length <= 85:
            modifiers.extend(["portrait lens", "subject separation", "background blur"])
        else:
            modifiers.extend(["telephoto compression", "isolated subject", "compressed background"])
        
        # Aperture effects
        if self.aperture <= 2.8:
            modifiers.append("shallow depth of field")
            modifiers.append("bokeh background")
        
        # Anamorphic
        if self.anamorphic_squeeze > 1.0:
            modifiers.extend(["anamorphic lens flares", "oval bokeh", "cinematic aspect ratio"])
        
        return modifiers


# ============================================
# CAMERA RIGS
# ============================================

class CameraRig(Enum):
    """Professional camera mounting systems."""
    TRIPOD = "tripod"               # Static, locked shots
    DOLLY = "dolly"                 # Wheeled platform on tracks
    SLIDER = "slider"               # Small track for subtle movement
    STEADICAM = "steadicam"         # Vest-mounted stabilizer
    GIMBAL = "gimbal"               # Electronic stabilization
    CRANE = "crane"                 # Large arm for sweeping moves
    JIB = "jib"                     # Smaller crane arm
    DRONE = "drone"                 # Aerial shots
    HANDHELD = "handheld"           # Organic, human movement
    SHOULDER = "shoulder"           # Shoulder-mounted, documentary
    CABLE_CAM = "cable_cam"         # Suspended cable system
    TECHNOCRANE = "technocrane"     # Precise computer-controlled


@dataclass
class RigCharacteristics:
    """Characteristics of different camera rigs."""
    rig: CameraRig
    stability: float = 1.0          # 0-1, higher = more stable
    organic_shake: float = 0.0      # 0-1, handheld shake
    speed_range: Tuple[float, float] = (0.0, 1.0)  # Movement speed
    vertical_range: float = 1.0     # Vertical movement capability
    horizontal_range: float = 1.0   # Horizontal movement capability
    
    @classmethod
    def get_rig(cls, rig: CameraRig) -> "RigCharacteristics":
        """Get characteristics for a specific rig."""
        rigs = {
            CameraRig.TRIPOD: cls(CameraRig.TRIPOD, 1.0, 0.0, (0, 0), 0.1, 0.1),
            CameraRig.DOLLY: cls(CameraRig.DOLLY, 0.95, 0.02, (0.1, 0.5), 0.1, 1.0),
            CameraRig.SLIDER: cls(CameraRig.SLIDER, 0.98, 0.01, (0.1, 0.3), 0, 0.3),
            CameraRig.STEADICAM: cls(CameraRig.STEADICAM, 0.85, 0.08, (0.2, 0.7), 0.5, 1.0),
            CameraRig.GIMBAL: cls(CameraRig.GIMBAL, 0.92, 0.03, (0.1, 0.8), 0.8, 1.0),
            CameraRig.CRANE: cls(CameraRig.CRANE, 0.9, 0.02, (0.1, 0.6), 1.0, 1.0),
            CameraRig.JIB: cls(CameraRig.JIB, 0.88, 0.03, (0.1, 0.5), 0.7, 0.5),
            CameraRig.DRONE: cls(CameraRig.DRONE, 0.7, 0.1, (0.2, 1.0), 1.0, 1.0),
            CameraRig.HANDHELD: cls(CameraRig.HANDHELD, 0.5, 0.4, (0.1, 0.9), 0.6, 0.8),
            CameraRig.SHOULDER: cls(CameraRig.SHOULDER, 0.6, 0.25, (0.2, 0.8), 0.4, 0.8),
            CameraRig.CABLE_CAM: cls(CameraRig.CABLE_CAM, 0.85, 0.05, (0.3, 1.0), 0.8, 1.0),
            CameraRig.TECHNOCRANE: cls(CameraRig.TECHNOCRANE, 0.98, 0.01, (0.05, 0.4), 1.0, 1.0),
        }
        return rigs.get(rig, cls(CameraRig.TRIPOD))


# ============================================
# CAMERA SPEC (Complete Camera Configuration)
# ============================================

@dataclass
class CameraSpec:
    """
    Complete camera configuration for a shot.
    
    This is the output specification that tells the rendering
    system exactly how to configure the virtual camera.
    """
    # Movement
    movement: CameraMovement = CameraMovement.STATIC
    movement_speed: float = 0.5         # 0-1, relative speed
    movement_easing: str = "smooth"     # linear, ease_in, ease_out, ease_in_out, smooth
    
    # Lens
    lens: CameraLens = field(default_factory=lambda: CameraLens(50))
    
    # Rig
    rig: CameraRig = CameraRig.TRIPOD
    
    # Angles
    angle_horizontal: float = 0.0       # -180 to 180, pan angle
    angle_vertical: float = 0.0         # -90 to 90, tilt angle
    dutch_angle: float = 0.0            # -45 to 45, rotation
    
    # Height
    height: str = "eye_level"           # low, eye_level, high, birds_eye, worms_eye
    
    # Focus
    focus_distance: str = "subject"     # subject, foreground, background, infinity
    rack_focus: bool = False            # Whether to change focus during shot
    
    # Duration
    duration_sec: float = 4.0
    
    # Special effects
    shake_intensity: float = 0.0        # Handheld shake
    speed_ramp: bool = False            # Variable speed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "movement": self.movement.value,
            "movement_speed": self.movement_speed,
            "movement_easing": self.movement_easing,
            "lens": {
                "focal_length": self.lens.focal_length,
                "aperture": self.lens.aperture,
                "lens_type": self.lens.lens_type.value,
                "depth_of_field": self.lens.get_depth_of_field(),
            },
            "rig": self.rig.value,
            "angle_horizontal": self.angle_horizontal,
            "angle_vertical": self.angle_vertical,
            "dutch_angle": self.dutch_angle,
            "height": self.height,
            "focus_distance": self.focus_distance,
            "rack_focus": self.rack_focus,
            "duration_sec": self.duration_sec,
            "shake_intensity": self.shake_intensity,
            "speed_ramp": self.speed_ramp,
        }
    
    def to_prompt_modifiers(self) -> List[str]:
        """Generate prompt modifiers for this camera setup."""
        modifiers = []
        
        # Movement modifiers
        movement_prompts = {
            CameraMovement.STATIC: ["static shot", "locked camera"],
            CameraMovement.DOLLY_IN: ["dolly in", "camera moving forward"],
            CameraMovement.DOLLY_OUT: ["dolly out", "camera pulling back"],
            CameraMovement.PUSH_IN: ["dramatic push in", "intense forward movement"],
            CameraMovement.PULL_OUT: ["pull out reveal", "expanding view"],
            CameraMovement.TRACKING_SHOT: ["tracking shot", "camera following subject"],
            CameraMovement.CRANE_SHOT: ["crane shot", "sweeping camera movement"],
            CameraMovement.STEADICAM: ["steadicam shot", "smooth floating camera"],
            CameraMovement.HANDHELD: ["handheld camera", "organic movement"],
            CameraMovement.ARC_LEFT: ["camera orbiting left", "arc shot"],
            CameraMovement.ARC_RIGHT: ["camera orbiting right", "arc shot"],
            CameraMovement.WHIP_PAN: ["whip pan", "fast camera pan"],
            CameraMovement.DOLLY_ZOOM: ["dolly zoom", "vertigo effect"],
            CameraMovement.SNAP_ZOOM: ["snap zoom", "anime style zoom"],
        }
        modifiers.extend(movement_prompts.get(self.movement, ["cinematic camera"]))
        
        # Lens modifiers
        modifiers.extend(self.lens.to_prompt_modifiers())
        
        # Height modifiers
        height_prompts = {
            "low": ["low angle shot", "looking up"],
            "eye_level": ["eye level shot"],
            "high": ["high angle shot", "looking down"],
            "birds_eye": ["bird's eye view", "overhead shot"],
            "worms_eye": ["worm's eye view", "extreme low angle"],
        }
        modifiers.extend(height_prompts.get(self.height, ["eye level"]))
        
        # Dutch angle
        if abs(self.dutch_angle) > 5:
            modifiers.append("dutch angle")
            modifiers.append("tilted camera")
        
        # Rig characteristics
        if self.rig == CameraRig.HANDHELD:
            modifiers.append("handheld footage")
        elif self.rig == CameraRig.STEADICAM:
            modifiers.append("steadicam shot")
        elif self.rig == CameraRig.DRONE:
            modifiers.append("aerial shot")
        
        return modifiers


# ============================================
# CAMERA SYSTEM (Main Orchestrator)
# ============================================

class CameraSystem:
    """
    Professional camera system that selects appropriate camera
    configurations based on scene content and genre.
    
    This is the main interface for the cinematic system.
    """
    
    # Movement selection based on content keywords
    MOVEMENT_KEYWORDS = {
        CameraMovement.DOLLY_IN: ["approaching", "tension", "focus", "reveal character"],
        CameraMovement.PUSH_IN: ["dramatic", "intense", "realization", "shock", "confrontation"],
        CameraMovement.PULL_OUT: ["reveal", "context", "scale", "epic", "vastness"],
        CameraMovement.TRACKING_SHOT: ["walking", "running", "following", "chase", "movement"],
        CameraMovement.CRANE_SHOT: ["establishing", "epic", "overview", "sweeping", "grand"],
        CameraMovement.STEADICAM: ["following", "through", "corridor", "continuous"],
        CameraMovement.HANDHELD: ["documentary", "realistic", "raw", "intimate", "chaos"],
        CameraMovement.ARC_LEFT: ["circling", "surrounding", "examining", "orbit"],
        CameraMovement.ARC_RIGHT: ["circling", "surrounding", "examining", "orbit"],
        CameraMovement.WHIP_PAN: ["sudden", "surprise", "transition", "fast"],
        CameraMovement.DOLLY_ZOOM: ["vertigo", "realization", "horror", "dread"],
        CameraMovement.SNAP_ZOOM: ["impact", "hit", "punch", "anime", "action"],
        CameraMovement.STATIC: ["conversation", "dialogue", "contemplation", "peaceful"],
        CameraMovement.FLOATING: ["dream", "ethereal", "memory", "surreal"],
        CameraMovement.TILT_UP: ["looking up", "tower", "tall", "impressive", "reveal height"],
        CameraMovement.TILT_DOWN: ["looking down", "fallen", "ground", "despair"],
        CameraMovement.PAN_LEFT: ["scanning", "environment", "landscape", "left"],
        CameraMovement.PAN_RIGHT: ["scanning", "environment", "landscape", "right"],
    }
    
    # Shot height based on content
    HEIGHT_KEYWORDS = {
        "low": ["powerful", "imposing", "hero", "dominant", "threatening"],
        "high": ["vulnerable", "small", "weak", "observing", "overview"],
        "birds_eye": ["map", "layout", "tactical", "gods_view"],
        "worms_eye": ["extreme power", "monster", "intimidating"],
        "eye_level": ["neutral", "conversation", "dialogue", "normal"],
    }
    
    # Lens selection based on content
    LENS_KEYWORDS = {
        14: ["vast", "expansive", "distorted", "cramped", "claustrophobic"],
        24: ["wide", "environment", "establishing", "landscape"],
        35: ["group", "context", "environmental portrait"],
        50: ["natural", "standard", "dialogue", "neutral"],
        85: ["portrait", "close-up", "intimate", "face", "emotion"],
        135: ["telephoto", "compressed", "isolated", "far"],
        200: ["distant", "surveillance", "sports", "wildlife"],
    }
    
    def __init__(self):
        self.default_lens = CameraLens.from_mm(50)
        self.default_movement = CameraMovement.STATIC
        
    def select_camera_for_beat(
        self,
        description: str,
        genre: str = "cinematic",
        beat_type: str = "normal",
    ) -> CameraSpec:
        """
        Select appropriate camera configuration for a beat.
        
        Args:
            description: Text description of the beat
            genre: Visual style/genre
            beat_type: Type of beat (action, dialogue, establishing, etc.)
            
        Returns:
            Complete CameraSpec for the beat
        """
        desc_lower = description.lower()
        
        # Select movement
        movement = self._select_movement(desc_lower, genre)
        
        # Select lens
        lens = self._select_lens(desc_lower, genre)
        
        # Select rig based on movement
        rig = self._select_rig(movement)
        
        # Select height
        height = self._select_height(desc_lower)
        
        # Dutch angle for tension
        dutch = self._calculate_dutch_angle(desc_lower, genre)
        
        # Movement speed
        speed = self._calculate_speed(desc_lower, movement)
        
        # Shake intensity
        shake = self._calculate_shake(desc_lower, rig, genre)
        
        spec = CameraSpec(
            movement=movement,
            movement_speed=speed,
            lens=lens,
            rig=rig,
            height=height,
            dutch_angle=dutch,
            shake_intensity=shake,
        )
        
        return spec
    
    def _select_movement(self, desc: str, genre: str) -> CameraMovement:
        """Select camera movement based on description."""
        # Genre-specific defaults
        genre_defaults = {
            "anime": CameraMovement.STATIC,
            "action": CameraMovement.TRACKING_SHOT,
            "horror": CameraMovement.STEADICAM,
            "documentary": CameraMovement.HANDHELD,
            "epic": CameraMovement.CRANE_SHOT,
        }
        
        # Check keywords
        best_match = None
        best_score = 0
        
        for movement, keywords in self.MOVEMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc)
            if score > best_score:
                best_score = score
                best_match = movement
        
        if best_match and best_score > 0:
            return best_match
        
        # Fall back to genre default or static
        return genre_defaults.get(genre, CameraMovement.STATIC)
    
    def _select_lens(self, desc: str, genre: str) -> CameraLens:
        """Select appropriate lens."""
        # Genre defaults
        genre_lens = {
            "anime": 50,
            "pixar": 35,
            "noir": 35,
            "action": 24,
            "portrait": 85,
            "landscape": 24,
            "intimate": 85,
        }
        
        # Check keywords
        for mm, keywords in self.LENS_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                return CameraLens.from_mm(mm)
        
        # Genre default
        default_mm = genre_lens.get(genre, 50)
        return CameraLens.from_mm(default_mm)
    
    def _select_rig(self, movement: CameraMovement) -> CameraRig:
        """Select appropriate camera rig for movement."""
        movement_rigs = {
            CameraMovement.STATIC: CameraRig.TRIPOD,
            CameraMovement.DOLLY_IN: CameraRig.DOLLY,
            CameraMovement.DOLLY_OUT: CameraRig.DOLLY,
            CameraMovement.PUSH_IN: CameraRig.DOLLY,
            CameraMovement.PULL_OUT: CameraRig.DOLLY,
            CameraMovement.TRACKING_SHOT: CameraRig.STEADICAM,
            CameraMovement.STEADICAM: CameraRig.STEADICAM,
            CameraMovement.CRANE_SHOT: CameraRig.CRANE,
            CameraMovement.JIB_SHOT: CameraRig.JIB,
            CameraMovement.HANDHELD: CameraRig.HANDHELD,
            CameraMovement.ARC_LEFT: CameraRig.STEADICAM,
            CameraMovement.ARC_RIGHT: CameraRig.STEADICAM,
            CameraMovement.FLOATING: CameraRig.GIMBAL,
            CameraMovement.WHIP_PAN: CameraRig.HANDHELD,
        }
        return movement_rigs.get(movement, CameraRig.TRIPOD)
    
    def _select_height(self, desc: str) -> str:
        """Select camera height."""
        for height, keywords in self.HEIGHT_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                return height
        return "eye_level"
    
    def _calculate_dutch_angle(self, desc: str, genre: str) -> float:
        """Calculate dutch angle for unease/tension."""
        tension_keywords = ["unease", "tension", "horror", "creepy", "unstable", "chaos"]
        
        if genre == "noir":
            return 8.0  # Noir uses dutch angles frequently
        
        if any(kw in desc for kw in tension_keywords):
            return 12.0
        
        return 0.0
    
    def _calculate_speed(self, desc: str, movement: CameraMovement) -> float:
        """Calculate movement speed."""
        fast_movements = [
            CameraMovement.WHIP_PAN,
            CameraMovement.SNAP_ZOOM,
            CameraMovement.CRASH_ZOOM,
        ]
        slow_movements = [
            CameraMovement.DOLLY_IN,
            CameraMovement.PUSH_IN,
            CameraMovement.FLOATING,
        ]
        
        if movement in fast_movements:
            return 0.9
        elif movement in slow_movements:
            return 0.3
        
        # Check description
        if any(w in desc for w in ["fast", "quick", "rapid", "sudden"]):
            return 0.8
        elif any(w in desc for w in ["slow", "gentle", "subtle", "gradual"]):
            return 0.2
        
        return 0.5
    
    def _calculate_shake(self, desc: str, rig: CameraRig, genre: str) -> float:
        """Calculate camera shake intensity."""
        if rig == CameraRig.HANDHELD:
            if genre == "documentary":
                return 0.4
            elif genre == "action":
                return 0.6
            return 0.3
        elif rig == CameraRig.SHOULDER:
            return 0.25
        elif rig == CameraRig.STEADICAM:
            return 0.08
        elif rig == CameraRig.GIMBAL:
            return 0.03
        
        # Check for chaos/intensity
        if any(w in desc for w in ["earthquake", "explosion", "chaos", "impact"]):
            return 0.5
        
        return 0.0
    
    def get_movement_prompt(self, movement: CameraMovement) -> str:
        """Get prompt description for camera movement."""
        prompts = {
            CameraMovement.STATIC: "static locked camera shot",
            CameraMovement.DOLLY_IN: "dolly in camera movement, slowly pushing forward",
            CameraMovement.DOLLY_OUT: "dolly out camera movement, slowly pulling back",
            CameraMovement.PUSH_IN: "dramatic push in, intense forward camera movement",
            CameraMovement.PULL_OUT: "dramatic pull out reveal, camera moving backward",
            CameraMovement.TRACKING_SHOT: "tracking shot, camera following the subject",
            CameraMovement.CRANE_SHOT: "sweeping crane shot, elevated camera movement",
            CameraMovement.STEADICAM: "smooth steadicam shot, floating camera",
            CameraMovement.HANDHELD: "handheld camera, organic movement",
            CameraMovement.ARC_LEFT: "camera arc left, orbiting around subject",
            CameraMovement.ARC_RIGHT: "camera arc right, orbiting around subject",
            CameraMovement.WHIP_PAN: "whip pan, fast horizontal camera movement",
            CameraMovement.DOLLY_ZOOM: "dolly zoom vertigo effect",
            CameraMovement.SNAP_ZOOM: "snap zoom, anime style quick zoom",
            CameraMovement.TILT_UP: "camera tilting up, revealing height",
            CameraMovement.TILT_DOWN: "camera tilting down",
            CameraMovement.PAN_LEFT: "camera panning left",
            CameraMovement.PAN_RIGHT: "camera panning right",
            CameraMovement.FLOATING: "floating ethereal camera movement",
        }
        return prompts.get(movement, "cinematic camera movement")


# Singleton instance
_camera_system: Optional[CameraSystem] = None


def get_camera_system() -> CameraSystem:
    """Get or create the global camera system."""
    global _camera_system
    if _camera_system is None:
        _camera_system = CameraSystem()
    return _camera_system


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def select_camera(description: str, genre: str = "cinematic") -> CameraSpec:
    """
    Convenience function to select camera for a beat.
    
    Args:
        description: Beat description
        genre: Visual style
        
    Returns:
        CameraSpec with complete camera configuration
    """
    system = get_camera_system()
    return system.select_camera_for_beat(description, genre)


def get_camera_prompt(spec: CameraSpec) -> str:
    """
    Generate prompt modifiers from camera spec.
    
    Args:
        spec: CameraSpec configuration
        
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
    print("PROFESSIONAL CAMERA SYSTEM TEST")
    print("="*60)
    
    system = CameraSystem()
    
    # Test various beat descriptions
    test_beats = [
        ("Two warriors clash in an epic battle", "action"),
        ("The hero realizes the truth with growing horror", "thriller"),
        ("Wide establishing shot of the futuristic city", "cyberpunk"),
        ("Close-up on the character's tearful expression", "drama"),
        ("Characters walk through a dark corridor", "horror"),
        ("The villain reveals their master plan", "anime"),
        ("A romantic sunset scene on the beach", "romance"),
        ("Intense chase through narrow streets", "action"),
        ("The detective examines the crime scene", "noir"),
        ("Magical transformation sequence", "pixar"),
    ]
    
    print("\n--- Camera Selection Test ---\n")
    
    for desc, genre in test_beats:
        spec = system.select_camera_for_beat(desc, genre)
        print(f"Scene: \"{desc[:40]}...\"")
        print(f"  Genre: {genre}")
        print(f"  Movement: {spec.movement.value}")
        print(f"  Lens: {spec.lens.focal_length}mm ({spec.lens.lens_type.value})")
        print(f"  Rig: {spec.rig.value}")
        print(f"  Height: {spec.height}")
        if spec.dutch_angle > 0:
            print(f"  Dutch Angle: {spec.dutch_angle}°")
        print()
    
    # Test lens creation
    print("--- Lens Simulation ---\n")
    for mm in [14, 24, 35, 50, 85, 135, 200]:
        lens = CameraLens.from_mm(mm)
        print(f"{mm}mm: {lens.lens_type.value}, DOF={lens.get_depth_of_field()}, distortion={lens.distortion:.2f}")
    
    # Test all movements
    print(f"\n--- Available Movements: {len(CameraMovement)} ---")
    for i, mov in enumerate(CameraMovement):
        if i % 5 == 0:
            print()
        print(f"  {mov.value}", end="")
    print("\n")
    
    print("✅ Camera system working!")
