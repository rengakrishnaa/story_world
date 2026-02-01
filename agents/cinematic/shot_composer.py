"""
Shot Composer - Shot Type Intelligence

Professional shot type selection and composition rules for cinematic
video generation. Implements 20+ shot types and composition techniques
used in film, animation, and broadcast production.

This module provides:
1. ShotType - All major shot classifications
2. CompositionRule - Professional framing techniques
3. ShotComposer - Intelligent shot selection based on content
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# ============================================
# SHOT TYPES (20+ types)
# ============================================

class ShotType(Enum):
    """
    Professional shot types used in film and animation.
    Based on subject framing and narrative purpose.
    """
    # Distance-based shots
    EXTREME_LONG_SHOT = "els"       # Subject tiny in vast landscape
    LONG_SHOT = "ls"                 # Full body with environment
    FULL_SHOT = "fs"                 # Head to toe, minimal environment
    MEDIUM_LONG_SHOT = "mls"         # Knees up ("American shot")
    MEDIUM_SHOT = "ms"               # Waist up
    MEDIUM_CLOSE_UP = "mcu"          # Chest up
    CLOSE_UP = "cu"                  # Face fills frame
    EXTREME_CLOSE_UP = "ecu"         # Eyes or detail only
    
    # Character relationship shots
    TWO_SHOT = "two_shot"            # Two characters in frame
    THREE_SHOT = "three_shot"        # Three characters
    GROUP_SHOT = "group_shot"        # Many characters
    OVER_THE_SHOULDER = "ots"        # Behind one character to another
    CLEAN_SINGLE = "clean_single"    # One character, no others
    DIRTY_SINGLE = "dirty_single"    # One character with part of another
    
    # Point of view
    POV = "pov"                      # Character's viewpoint
    SUBJECTIVE = "subjective"        # Character's internal view
    OBJECTIVE = "objective"          # Neutral observer
    
    # Special purpose shots
    ESTABLISHING = "establishing"     # Sets location/time
    MASTER_SHOT = "master"           # Full scene coverage for editing
    INSERT_SHOT = "insert"           # Detail/object emphasis
    CUTAWAY = "cutaway"              # Away from main action
    REACTION_SHOT = "reaction"       # Character reacting
    
    # Transition shots
    BRIDGE_SHOT = "bridge"           # Connects scenes
    MONTAGE_SHOT = "montage"         # Part of rapid sequence
    
    # Anime/animation specific
    IMPACT_SHOT = "impact"           # Freeze frame for impact
    BEAUTY_SHOT = "beauty"           # Glamour/detail focus
    SPEED_LINE_SHOT = "speed_line"   # Motion lines
    TRANSFORMATION = "transformation" # Character changing


class CompositionType(Enum):
    """Composition and framing techniques."""
    RULE_OF_THIRDS = "thirds"
    CENTER_FRAME = "center"
    GOLDEN_RATIO = "golden"
    SYMMETRICAL = "symmetrical"
    ASYMMETRICAL = "asymmetrical"
    DYNAMIC_DIAGONAL = "diagonal"
    LEADING_LINES = "leading_lines"
    FRAME_WITHIN_FRAME = "frame_in_frame"
    DEPTH_LAYERING = "depth"
    NEGATIVE_SPACE = "negative_space"


# ============================================
# SHOT SPECIFICATION
# ============================================

@dataclass
class ShotSpec:
    """
    Complete specification for a shot's framing and composition.
    """
    # Shot type
    shot_type: ShotType = ShotType.MEDIUM_SHOT
    
    # Composition
    composition: CompositionType = CompositionType.RULE_OF_THIRDS
    
    # Subject placement (0-1, 0=left/top, 1=right/bottom)
    subject_x: float = 0.33          # Horizontal position
    subject_y: float = 0.33          # Vertical position
    
    # Headroom and look room (0-1)
    headroom: float = 0.1            # Space above subject
    look_room: float = 0.3           # Space in direction of gaze
    lead_room: float = 0.3           # Space in direction of movement
    
    # Depth
    foreground_element: bool = False  # Include foreground framing
    background_blur: float = 0.3      # Background blur amount
    depth_layers: int = 2             # Number of depth layers
    
    # Character count
    num_subjects: int = 1
    subject_relationship: str = "solo"  # solo, facing, same_direction, conflict
    
    # Special
    aspect_ratio: str = "16:9"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shot_type": self.shot_type.value,
            "composition": self.composition.value,
            "subject_position": {"x": self.subject_x, "y": self.subject_y},
            "headroom": self.headroom,
            "look_room": self.look_room,
            "lead_room": self.lead_room,
            "foreground_element": self.foreground_element,
            "background_blur": self.background_blur,
            "depth_layers": self.depth_layers,
            "num_subjects": self.num_subjects,
            "aspect_ratio": self.aspect_ratio,
        }
    
    def to_prompt_modifiers(self) -> List[str]:
        """Generate prompt modifiers for this shot."""
        modifiers = []
        
        # Shot type descriptions
        shot_prompts = {
            ShotType.EXTREME_LONG_SHOT: ["extreme wide shot", "vast landscape", "tiny figure in distance"],
            ShotType.LONG_SHOT: ["wide shot", "full body", "environmental context"],
            ShotType.FULL_SHOT: ["full shot", "head to toe", "full figure"],
            ShotType.MEDIUM_LONG_SHOT: ["medium long shot", "cowboy shot", "knees up"],
            ShotType.MEDIUM_SHOT: ["medium shot", "waist up", "half body"],
            ShotType.MEDIUM_CLOSE_UP: ["medium close up", "chest up", "upper body"],
            ShotType.CLOSE_UP: ["close up", "face shot", "head and shoulders"],
            ShotType.EXTREME_CLOSE_UP: ["extreme close up", "detail shot", "macro"],
            ShotType.TWO_SHOT: ["two shot", "two characters in frame"],
            ShotType.OVER_THE_SHOULDER: ["over the shoulder shot", "OTS"],
            ShotType.POV: ["point of view shot", "first person perspective"],
            ShotType.ESTABLISHING: ["establishing shot", "location shot"],
            ShotType.INSERT_SHOT: ["insert shot", "detail shot", "object focus"],
            ShotType.REACTION_SHOT: ["reaction shot", "character reaction"],
            ShotType.BEAUTY_SHOT: ["beauty shot", "glamour shot", "detailed"],
        }
        modifiers.extend(shot_prompts.get(self.shot_type, ["cinematic shot"]))
        
        # Composition
        comp_prompts = {
            CompositionType.RULE_OF_THIRDS: ["rule of thirds composition"],
            CompositionType.CENTER_FRAME: ["centered composition", "symmetrical framing"],
            CompositionType.GOLDEN_RATIO: ["golden ratio composition"],
            CompositionType.SYMMETRICAL: ["symmetrical composition"],
            CompositionType.DYNAMIC_DIAGONAL: ["dynamic diagonal composition"],
            CompositionType.LEADING_LINES: ["leading lines composition"],
            CompositionType.FRAME_WITHIN_FRAME: ["frame within frame"],
            CompositionType.DEPTH_LAYERING: ["layered depth", "foreground middleground background"],
        }
        modifiers.extend(comp_prompts.get(self.composition, []))
        
        # Depth
        if self.foreground_element:
            modifiers.append("foreground framing elements")
        if self.background_blur > 0.5:
            modifiers.append("shallow depth of field")
            modifiers.append("blurred background")
        if self.depth_layers >= 3:
            modifiers.append("layered composition")
        
        return modifiers

    def get_framing_description(self) -> str:
        """Get natural language description of framing."""
        shot_descriptions = {
            ShotType.EXTREME_LONG_SHOT: "an extreme wide shot showing the subject as a small figure in a vast environment",
            ShotType.LONG_SHOT: "a wide shot showing the subject's full body within the environment",
            ShotType.FULL_SHOT: "a full shot framing the subject from head to toe",
            ShotType.MEDIUM_LONG_SHOT: "a medium long shot framing the subject from the knees up",
            ShotType.MEDIUM_SHOT: "a medium shot framing the subject from the waist up",
            ShotType.MEDIUM_CLOSE_UP: "a medium close-up framing the subject from the chest up",
            ShotType.CLOSE_UP: "a close-up of the subject's face",
            ShotType.EXTREME_CLOSE_UP: "an extreme close-up focusing on a specific detail",
            ShotType.TWO_SHOT: "a two-shot with both characters in frame",
            ShotType.OVER_THE_SHOULDER: "an over-the-shoulder shot looking past one character to another",
            ShotType.POV: "a point-of-view shot from the character's perspective",
            ShotType.ESTABLISHING: "an establishing shot showing the location",
            ShotType.INSERT_SHOT: "an insert shot focusing on an important detail",
            ShotType.REACTION_SHOT: "a reaction shot capturing the character's response",
        }
        return shot_descriptions.get(self.shot_type, "a cinematic shot")


# ============================================
# SHOT COMPOSER
# ============================================

class ShotComposer:
    """
    Intelligent shot selection based on scene content and narrative.
    
    Analyzes beat descriptions to determine appropriate:
    - Shot type (close-up, wide, etc.)
    - Composition (rule of thirds, center, etc.)
    - Subject placement
    - Framing elements
    """
    
    # Keywords for shot type selection
    SHOT_KEYWORDS = {
        ShotType.EXTREME_LONG_SHOT: [
            "vast", "landscape", "horizon", "epic scale", "isolated", "tiny",
            "desert", "ocean", "mountain range", "cityscape from afar"
        ],
        ShotType.LONG_SHOT: [
            "walking", "approaching", "standing", "full body", "environment",
            "entering", "leaving", "establishing character"
        ],
        ShotType.FULL_SHOT: [
            "full figure", "head to toe", "stance", "posture", "outfit",
            "costume", "armor", "uniform"
        ],
        ShotType.MEDIUM_LONG_SHOT: [
            "cowboy shot", "action pose", "weapon draw", "gesture",
            "confrontation", "standing off"
        ],
        ShotType.MEDIUM_SHOT: [
            "conversation", "dialogue", "talking", "explaining",
            "gesturing", "interaction", "meeting"
        ],
        ShotType.MEDIUM_CLOSE_UP: [
            "speaking", "listening", "explaining", "upper body",
            "emphasis", "important dialogue"
        ],
        ShotType.CLOSE_UP: [
            "emotion", "expression", "face", "reaction", "fear",
            "joy", "sadness", "anger", "tear", "smile", "eyes"
        ],
        ShotType.EXTREME_CLOSE_UP: [
            "eye", "detail", "object", "hand", "fingers", "lips",
            "small item", "texture", "key", "ring", "button"
        ],
        ShotType.TWO_SHOT: [
            "two characters", "couple", "pair", "together", "side by side",
            "holding hands", "embrace", "confronting each other"
        ],
        ShotType.OVER_THE_SHOULDER: [
            "talking to", "facing", "listening to", "response",
            "conversation between", "dialogue scene"
        ],
        ShotType.POV: [
            "sees", "looking at", "perspective of", "through eyes",
            "from viewpoint", "first person"
        ],
        ShotType.ESTABLISHING: [
            "establishing", "location", "setting", "new scene",
            "meanwhile", "exterior", "interior of"
        ],
        ShotType.INSERT_SHOT: [
            "object", "item", "letter", "phone", "clock", "document",
            "important item", "detail of", "close on"
        ],
        ShotType.REACTION_SHOT: [
            "reacts", "responds", "shocked", "surprised", "realizes",
            "expression changes", "notices"
        ],
        ShotType.IMPACT_SHOT: [
            "impact", "hit", "explosion", "punch", "crash",
            "collision", "strike", "blow"
        ],
        ShotType.BEAUTY_SHOT: [
            "beautiful", "gorgeous", "stunning", "glamour",
            "hero shot", "dramatic pose", "iconic"
        ],
    }
    
    # Keywords for composition
    COMPOSITION_KEYWORDS = {
        CompositionType.SYMMETRICAL: [
            "symmetry", "balanced", "centered", "mirror",
            "formal", "throne", "altar", "doorway"
        ],
        CompositionType.DYNAMIC_DIAGONAL: [
            "action", "movement", "dynamic", "tension",
            "running", "falling", "flying"
        ],
        CompositionType.LEADING_LINES: [
            "corridor", "road", "path", "tunnel", "railway",
            "perspective", "vanishing point"
        ],
        CompositionType.FRAME_WITHIN_FRAME: [
            "doorway", "window", "arch", "frame", "through",
            "portal", "opening"
        ],
        CompositionType.DEPTH_LAYERING: [
            "foreground", "background", "layered", "depth",
            "silhouette in front", "objects between"
        ],
        CompositionType.NEGATIVE_SPACE: [
            "isolated", "alone", "empty", "vast", "minimal",
            "contemplation", "solitude"
        ],
    }
    
    # Character count detection
    CHARACTER_PATTERNS = {
        1: ["alone", "single", "solitary", "one character", "protagonist", "he/she stands"],
        2: ["two", "pair", "couple", "both", "together", "facing each other", "them"],
        3: ["three", "trio", "group of three"],
        4: ["four", "group of four", "quartet"],
    }
    
    def __init__(self):
        self.default_shot = ShotType.MEDIUM_SHOT
        self.default_composition = CompositionType.RULE_OF_THIRDS
    
    def compose_shot(
        self,
        description: str,
        genre: str = "cinematic",
        beat_type: str = "normal",
    ) -> ShotSpec:
        """
        Compose a shot specification for a beat.
        
        Args:
            description: Text description of the beat/scene
            genre: Visual style
            beat_type: Type of beat (action, dialogue, establishing)
            
        Returns:
            ShotSpec with complete framing configuration
        """
        desc_lower = description.lower()
        
        # Determine shot type
        shot_type = self._select_shot_type(desc_lower, genre)
        
        # Determine composition
        composition = self._select_composition(desc_lower, shot_type)
        
        # Determine subject placement
        subject_x, subject_y = self._calculate_subject_position(shot_type, composition)
        
        # Determine headroom
        headroom = self._calculate_headroom(shot_type)
        
        # Determine depth
        fg_element, bg_blur, depth_layers = self._calculate_depth(desc_lower, shot_type, genre)
        
        # Count subjects
        num_subjects = self._count_subjects(desc_lower)
        
        # Subject relationship
        relationship = self._detect_relationship(desc_lower, num_subjects)
        
        spec = ShotSpec(
            shot_type=shot_type,
            composition=composition,
            subject_x=subject_x,
            subject_y=subject_y,
            headroom=headroom,
            foreground_element=fg_element,
            background_blur=bg_blur,
            depth_layers=depth_layers,
            num_subjects=num_subjects,
            subject_relationship=relationship,
        )
        
        return spec
    
    def _select_shot_type(self, desc: str, genre: str) -> ShotType:
        """Select appropriate shot type."""
        # Genre-specific tendencies
        genre_defaults = {
            "anime": ShotType.MEDIUM_CLOSE_UP,
            "action": ShotType.MEDIUM_LONG_SHOT,
            "horror": ShotType.CLOSE_UP,
            "romance": ShotType.MEDIUM_SHOT,
            "epic": ShotType.LONG_SHOT,
            "pixar": ShotType.MEDIUM_SHOT,
        }
        
        # Find best keyword match
        best_match = None
        best_score = 0
        
        for shot_type, keywords in self.SHOT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc)
            if score > best_score:
                best_score = score
                best_match = shot_type
        
        if best_match and best_score > 0:
            return best_match
        
        return genre_defaults.get(genre, self.default_shot)
    
    def _select_composition(self, desc: str, shot_type: ShotType) -> CompositionType:
        """Select appropriate composition."""
        # Shot type tendencies
        shot_compositions = {
            ShotType.EXTREME_LONG_SHOT: CompositionType.RULE_OF_THIRDS,
            ShotType.ESTABLISHING: CompositionType.RULE_OF_THIRDS,
            ShotType.CLOSE_UP: CompositionType.CENTER_FRAME,
            ShotType.EXTREME_CLOSE_UP: CompositionType.CENTER_FRAME,
            ShotType.TWO_SHOT: CompositionType.SYMMETRICAL,
            ShotType.IMPACT_SHOT: CompositionType.CENTER_FRAME,
        }
        
        # Check keywords
        for comp, keywords in self.COMPOSITION_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                return comp
        
        return shot_compositions.get(shot_type, self.default_composition)
    
    def _calculate_subject_position(
        self,
        shot_type: ShotType,
        composition: CompositionType
    ) -> Tuple[float, float]:
        """Calculate subject position in frame (0-1)."""
        if composition == CompositionType.CENTER_FRAME:
            return (0.5, 0.5)
        elif composition == CompositionType.SYMMETRICAL:
            return (0.5, 0.4)  # Slightly above center
        elif composition == CompositionType.RULE_OF_THIRDS:
            # Prefer left or right third
            return (0.33, 0.33)
        elif composition == CompositionType.GOLDEN_RATIO:
            # Golden ratio position
            return (0.382, 0.382)
        elif composition == CompositionType.NEGATIVE_SPACE:
            # More on one side
            return (0.25, 0.4)
        
        return (0.33, 0.33)
    
    def _calculate_headroom(self, shot_type: ShotType) -> float:
        """Calculate appropriate headroom."""
        headroom_map = {
            ShotType.EXTREME_LONG_SHOT: 0.4,
            ShotType.LONG_SHOT: 0.25,
            ShotType.FULL_SHOT: 0.15,
            ShotType.MEDIUM_LONG_SHOT: 0.12,
            ShotType.MEDIUM_SHOT: 0.1,
            ShotType.MEDIUM_CLOSE_UP: 0.08,
            ShotType.CLOSE_UP: 0.05,
            ShotType.EXTREME_CLOSE_UP: 0.02,
        }
        return headroom_map.get(shot_type, 0.1)
    
    def _calculate_depth(
        self,
        desc: str,
        shot_type: ShotType,
        genre: str
    ) -> Tuple[bool, float, int]:
        """Calculate depth characteristics."""
        # Foreground elements
        fg_keywords = ["through", "peering", "foreground", "framed by", "silhouette"]
        foreground = any(kw in desc for kw in fg_keywords)
        
        # Background blur based on shot type
        blur_map = {
            ShotType.CLOSE_UP: 0.7,
            ShotType.EXTREME_CLOSE_UP: 0.9,
            ShotType.MEDIUM_CLOSE_UP: 0.5,
            ShotType.MEDIUM_SHOT: 0.3,
            ShotType.LONG_SHOT: 0.1,
            ShotType.EXTREME_LONG_SHOT: 0.0,
        }
        blur = blur_map.get(shot_type, 0.3)
        
        # Depth layers
        if "layered" in desc or genre == "pixar":
            layers = 3
        elif foreground:
            layers = 3
        elif shot_type in [ShotType.ESTABLISHING, ShotType.LONG_SHOT]:
            layers = 3
        else:
            layers = 2
        
        return foreground, blur, layers
    
    def _count_subjects(self, desc: str) -> int:
        """Count number of subjects in scene."""
        for count, keywords in sorted(self.CHARACTER_PATTERNS.items(), reverse=True):
            if any(kw in desc for kw in keywords):
                return count
        
        # Default based on common patterns
        if "group" in desc or "crowd" in desc:
            return 5
        
        return 1
    
    def _detect_relationship(self, desc: str, num_subjects: int) -> str:
        """Detect relationship between subjects."""
        if num_subjects == 1:
            return "solo"
        
        if any(w in desc for w in ["facing", "confronting", "arguing", "opposing"]):
            return "conflict"
        elif any(w in desc for w in ["together", "side by side", "walking", "looking same direction"]):
            return "same_direction"
        elif any(w in desc for w in ["talking", "conversation", "looking at each other"]):
            return "facing"
        
        return "facing"


# Singleton instance
_shot_composer: Optional[ShotComposer] = None


def get_shot_composer() -> ShotComposer:
    """Get or create the global shot composer."""
    global _shot_composer
    if _shot_composer is None:
        _shot_composer = ShotComposer()
    return _shot_composer


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def compose_shot(description: str, genre: str = "cinematic") -> ShotSpec:
    """
    Convenience function to compose a shot.
    
    Args:
        description: Scene/beat description
        genre: Visual style
        
    Returns:
        ShotSpec with complete framing configuration
    """
    composer = get_shot_composer()
    return composer.compose_shot(description, genre)


def get_shot_prompt(spec: ShotSpec) -> str:
    """
    Generate prompt modifiers from shot spec.
    
    Args:
        spec: ShotSpec configuration
        
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
    print("SHOT COMPOSER TEST")
    print("="*60)
    
    composer = ShotComposer()
    
    # Test various scenes
    test_scenes = [
        ("The hero stands alone in the vast desert landscape", "epic"),
        ("Close-up on Maya's tearful expression as she reads the letter", "drama"),
        ("Two warriors face each other in the arena", "action"),
        ("The detective examines the mysterious letter on the desk", "noir"),
        ("Wide establishing shot of the futuristic cityscape", "cyberpunk"),
        ("Over-the-shoulder shot as they have a heated conversation", "drama"),
        ("The villain reveals his shocked reaction to the news", "thriller"),
        ("Walking through the dark corridor towards the light", "horror"),
        ("Beautiful hero shot of the protagonist in their new armor", "anime"),
        ("Group of friends sitting together at the cafe", "romance"),
    ]
    
    print("\n--- Shot Composition Test ---\n")
    
    for desc, genre in test_scenes:
        spec = composer.compose_shot(desc, genre)
        print(f"Scene: \"{desc[:45]}...\"")
        print(f"  Genre: {genre}")
        print(f"  Shot Type: {spec.shot_type.value} ({spec.shot_type.name})")
        print(f"  Composition: {spec.composition.value}")
        print(f"  Subjects: {spec.num_subjects} ({spec.subject_relationship})")
        print(f"  Headroom: {spec.headroom:.2f}")
        print(f"  BG Blur: {spec.background_blur:.2f}")
        print()
    
    # Test all shot types
    print(f"\n--- Available Shot Types: {len(ShotType)} ---")
    for shot in ShotType:
        print(f"  {shot.value}: {shot.name}")
    
    print(f"\n--- Composition Types: {len(CompositionType)} ---")
    for comp in CompositionType:
        print(f"  {comp.value}: {comp.name}")
    
    print("\n✅ Shot composer working!")
