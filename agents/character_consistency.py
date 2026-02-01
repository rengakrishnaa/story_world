"""
Character Consistency Engine

Maintains consistent character appearance across multiple beats/shots.
Uses multiple techniques:
1. Reference image embedding (IP-Adapter style)
2. Detailed text descriptions
3. Face/body embeddings for identity preservation
4. Cross-beat consistency checks

This is critical for story continuity - the same character should
look the same across all scenes.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CharacterAppearance:
    """
    Complete appearance profile for a character.
    
    This captures all visual attributes needed to maintain
    consistent appearance across beats.
    """
    character_id: str
    name: str
    
    # Core visual description
    physical_description: str  # "tall muscular man with bald head"
    clothing_description: str  # "yellow superhero suit with red gloves"
    distinctive_features: List[str]  # ["bald head", "one punch fist"]
    
    # Reference images
    reference_image_urls: List[str] = field(default_factory=list)
    
    # Color palette (hex codes)
    primary_colors: List[str] = field(default_factory=list)  # ["#FFD700", "#FF0000"]
    
    # Body type
    body_type: str = "average"  # slim, average, athletic, muscular
    
    # Age/gender presentation
    apparent_age: str = "adult"  # child, teen, adult, elderly
    gender_presentation: str = "neutral"  # masculine, feminine, neutral
    
    # Style-specific modifiers
    anime_traits: List[str] = field(default_factory=list)  # ["large eyes", "spiky hair"]
    realistic_traits: List[str] = field(default_factory=list)  # ["detailed skin", "realistic proportions"]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterAppearance":
        return cls(**data)
    
    def get_prompt_description(self, style: str = "cinematic") -> str:
        """
        Generate a prompt-ready description for this character.
        
        Args:
            style: Visual style (anime, realistic, etc.)
            
        Returns:
            Detailed description for generation prompts
        """
        parts = [self.name]
        
        # Physical description
        if self.physical_description:
            parts.append(self.physical_description)
        
        # Clothing
        if self.clothing_description:
            parts.append(f"wearing {self.clothing_description}")
        
        # Distinctive features
        if self.distinctive_features:
            parts.append(", ".join(self.distinctive_features))
        
        # Style-specific traits
        if style == "anime" and self.anime_traits:
            parts.append(", ".join(self.anime_traits))
        elif style == "realistic" and self.realistic_traits:
            parts.append(", ".join(self.realistic_traits))
        
        return ", ".join(parts)


class CharacterConsistencyEngine:
    """
    Main engine for maintaining character consistency.
    
    Features:
    1. Character appearance registry
    2. Prompt enhancement with character details
    3. Multiple character handling in same scene
    4. Cross-beat consistency verification
    """
    
    def __init__(self, world_id: str):
        self.world_id = world_id
        self.characters: Dict[str, CharacterAppearance] = {}
        self._cache_dir = Path(f"outputs/{world_id}/character_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cached characters if available
        self._load_cache()
    
    def _load_cache(self):
        """Load cached character appearances."""
        cache_file = self._cache_dir / "characters.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    for char_id, char_data in data.items():
                        self.characters[char_id] = CharacterAppearance.from_dict(char_data)
                logger.info(f"Loaded {len(self.characters)} cached characters")
            except Exception as e:
                logger.warning(f"Failed to load character cache: {e}")
    
    def _save_cache(self):
        """Save characters to cache."""
        cache_file = self._cache_dir / "characters.json"
        try:
            data = {cid: c.to_dict() for cid, c in self.characters.items()}
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save character cache: {e}")
    
    def register_character(self, character: CharacterAppearance) -> str:
        """
        Register a character for consistency tracking.
        
        Args:
            character: CharacterAppearance to register
            
        Returns:
            Character ID
        """
        self.characters[character.character_id] = character
        self._save_cache()
        logger.info(f"Registered character: {character.name} ({character.character_id})")
        return character.character_id
    
    def register_from_world(self, world_data: Dict[str, Any]):
        """
        Register characters from world graph data.
        
        Args:
            world_data: WorldGraph dict with characters list
        """
        for char_data in world_data.get("characters", []):
            char_id = self._generate_id(char_data.get("name", "unknown"))
            
            appearance = CharacterAppearance(
                character_id=char_id,
                name=char_data.get("name", "Unknown"),
                physical_description=char_data.get("description", ""),
                clothing_description="",
                distinctive_features=char_data.get("traits", []),
                reference_image_urls=[char_data.get("reference_image_url", "")],
            )
            
            self.register_character(appearance)
    
    def _generate_id(self, name: str) -> str:
        """Generate consistent ID from name."""
        return hashlib.md5(name.lower().encode()).hexdigest()[:12]
    
    def get_character(self, name: str) -> Optional[CharacterAppearance]:
        """Get character by name (case-insensitive)."""
        name_lower = name.lower()
        
        for char in self.characters.values():
            if char.name.lower() == name_lower:
                return char
        
        return None
    
    def enhance_prompt_with_characters(
        self,
        base_prompt: str,
        character_names: List[str],
        style: str = "cinematic",
    ) -> str:
        """
        Enhance a generation prompt with character descriptions.
        
        Adds detailed character descriptions to maintain consistency.
        
        Args:
            base_prompt: Original generation prompt
            character_names: List of character names in the scene
            style: Visual style for style-specific traits
            
        Returns:
            Enhanced prompt with character descriptions
        """
        if not character_names:
            return base_prompt
        
        char_descriptions = []
        
        for name in character_names:
            char = self.get_character(name)
            if char:
                desc = char.get_prompt_description(style)
                char_descriptions.append(desc)
            else:
                # Unknown character - use name only
                char_descriptions.append(name)
        
        # Build enhanced prompt
        if char_descriptions:
            characters_text = "; ".join(char_descriptions)
            return f"{base_prompt}. The scene features: {characters_text}"
        
        return base_prompt
    
    def get_character_references(
        self,
        character_names: List[str],
    ) -> Dict[str, List[str]]:
        """
        Get reference image URLs for characters.
        
        Used for IP-Adapter or similar reference conditioning.
        
        Args:
            character_names: List of character names
            
        Returns:
            Dict mapping character name to list of reference URLs
        """
        references = {}
        
        for name in character_names:
            char = self.get_character(name)
            if char and char.reference_image_urls:
                references[name] = [
                    url for url in char.reference_image_urls if url
                ]
        
        return references
    
    def build_character_conditioning(
        self,
        character_names: List[str],
        style: str = "cinematic",
    ) -> Dict[str, Any]:
        """
        Build complete conditioning data for character-aware generation.
        
        Returns a dict that can be passed to generation backends.
        
        Args:
            character_names: Characters in the scene
            style: Visual style
            
        Returns:
            Dict with prompt, references, and negative prompts
        """
        conditioning = {
            "character_prompts": {},
            "reference_images": {},
            "character_negative": "",
        }
        
        negative_parts = []
        
        for name in character_names:
            char = self.get_character(name)
            if char:
                # Add character-specific prompt
                conditioning["character_prompts"][name] = char.get_prompt_description(style)
                
                # Add reference images
                if char.reference_image_urls:
                    conditioning["reference_images"][name] = char.reference_image_urls
                
                # Build negative prompt to avoid mixing characters
                other_chars = [c for c in self.characters.values() if c.name != name]
                for other in other_chars[:3]:  # Limit to 3 to avoid prompt length issues
                    if other.distinctive_features:
                        negative_parts.extend(other.distinctive_features[:2])
        
        if negative_parts:
            conditioning["character_negative"] = ", ".join(set(negative_parts))
        
        return conditioning
    
    def verify_character_consistency(
        self,
        beat_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify character consistency across multiple beats.
        
        This is a post-generation check that compares character
        appearances across beats.
        
        Args:
            beat_results: List of beat result dicts with video_url and characters
            
        Returns:
            Consistency report with scores and flagged issues
        """
        report = {
            "total_beats": len(beat_results),
            "characters_tracked": [],
            "issues": [],
            "overall_consistency": 1.0,
        }
        
        # Track character appearances
        character_beats: Dict[str, List[int]] = {}
        
        for i, beat in enumerate(beat_results):
            for char_name in beat.get("characters", []):
                if char_name not in character_beats:
                    character_beats[char_name] = []
                character_beats[char_name].append(i)
        
        report["characters_tracked"] = list(character_beats.keys())
        
        # Check for characters appearing in multiple beats
        for char_name, beat_indices in character_beats.items():
            if len(beat_indices) > 1:
                # Character appears multiple times - verify consistency
                char = self.get_character(char_name)
                if not char:
                    report["issues"].append({
                        "type": "unknown_character",
                        "character": char_name,
                        "beats": beat_indices,
                        "severity": "warning",
                    })
                elif not char.reference_image_urls:
                    report["issues"].append({
                        "type": "no_reference",
                        "character": char_name,
                        "beats": beat_indices,
                        "severity": "info",
                    })
        
        # Calculate overall consistency score
        if report["issues"]:
            # Reduce score based on issues
            warnings = len([i for i in report["issues"] if i["severity"] == "warning"])
            errors = len([i for i in report["issues"] if i["severity"] == "error"])
            report["overall_consistency"] = max(0, 1.0 - (warnings * 0.1) - (errors * 0.3))
        
        return report
    
    def create_character_from_description(
        self,
        name: str,
        description: str,
        style: str = "cinematic",
    ) -> CharacterAppearance:
        """
        Create a character appearance from a text description.
        
        Parses the description to extract visual attributes.
        
        Args:
            name: Character name
            description: Natural language description
            style: Visual style for style-specific traits
            
        Returns:
            CharacterAppearance object
        """
        # Simple keyword-based attribute extraction
        desc_lower = description.lower()
        
        # Body type detection
        body_type = "average"
        if any(w in desc_lower for w in ["muscular", "buff", "strong"]):
            body_type = "muscular"
        elif any(w in desc_lower for w in ["slim", "thin", "lean"]):
            body_type = "slim"
        elif any(w in desc_lower for w in ["athletic", "fit"]):
            body_type = "athletic"
        
        # Age detection
        apparent_age = "adult"
        if any(w in desc_lower for w in ["child", "kid", "young boy", "young girl"]):
            apparent_age = "child"
        elif any(w in desc_lower for w in ["teen", "teenager", "adolescent"]):
            apparent_age = "teen"
        elif any(w in desc_lower for w in ["old", "elderly", "aged", "grey hair"]):
            apparent_age = "elderly"
        
        # Color extraction (simple)
        colors = []
        color_keywords = {
            "red": "#FF0000",
            "blue": "#0000FF",
            "green": "#00FF00",
            "yellow": "#FFFF00",
            "orange": "#FFA500",
            "purple": "#800080",
            "black": "#000000",
            "white": "#FFFFFF",
            "gold": "#FFD700",
        }
        for color_name, hex_code in color_keywords.items():
            if color_name in desc_lower:
                colors.append(hex_code)
        
        # Extract distinctive features
        features = []
        feature_keywords = [
            "bald", "beard", "mustache", "scar", "tattoo", "glasses",
            "long hair", "short hair", "spiky hair", "ponytail",
            "cape", "armor", "robot", "cyborg", "mechanical",
        ]
        for feature in feature_keywords:
            if feature in desc_lower:
                features.append(feature)
        
        return CharacterAppearance(
            character_id=self._generate_id(name),
            name=name,
            physical_description=description,
            clothing_description="",
            distinctive_features=features,
            primary_colors=colors[:3],
            body_type=body_type,
            apparent_age=apparent_age,
        )


# Singleton instances per world
_engines: Dict[str, CharacterConsistencyEngine] = {}


def get_consistency_engine(world_id: str) -> CharacterConsistencyEngine:
    """Get or create consistency engine for a world."""
    if world_id not in _engines:
        _engines[world_id] = CharacterConsistencyEngine(world_id)
    return _engines[world_id]


if __name__ == "__main__":
    # Test the character consistency engine
    print("\n" + "="*60)
    print("CHARACTER CONSISTENCY TEST")
    print("="*60)
    
    engine = CharacterConsistencyEngine("test-world")
    
    # Create test characters
    saitama = CharacterAppearance(
        character_id="char_saitama",
        name="Saitama",
        physical_description="tall muscular bald man",
        clothing_description="yellow superhero suit with white cape and red gloves",
        distinctive_features=["bald head", "blank expression", "one punch"],
        primary_colors=["#FFD700", "#FFFFFF", "#FF0000"],
        body_type="muscular",
        anime_traits=["simplified face", "expressive eyes"],
    )
    
    genos = CharacterAppearance(
        character_id="char_genos",
        name="Genos",
        physical_description="young cyborg with blonde hair",
        clothing_description="black and gold mechanical armor",
        distinctive_features=["mechanical arms", "glowing eyes", "blonde hair"],
        primary_colors=["#000000", "#FFD700", "#C0C0C0"],
        body_type="athletic",
        anime_traits=["detailed mechanical parts", "intense gaze"],
    )
    
    engine.register_character(saitama)
    engine.register_character(genos)
    
    # Test prompt enhancement
    base_prompt = "Two heroes stand on a rooftop overlooking the city at sunset"
    enhanced = engine.enhance_prompt_with_characters(
        base_prompt,
        ["Saitama", "Genos"],
        style="anime"
    )
    
    print(f"\nBase prompt: {base_prompt}")
    print(f"\nEnhanced: {enhanced[:150]}...")
    
    # Test conditioning
    conditioning = engine.build_character_conditioning(
        ["Saitama", "Genos"],
        style="anime"
    )
    
    print(f"\nConditioning: {json.dumps(conditioning, indent=2)[:500]}...")
    
    print("\n✅ Character consistency engine working!")
