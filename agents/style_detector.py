"""
Style Detector

Automatically detects visual style from user intent/prompt.
Maps detected styles to rendering parameters for consistent generation.

Supported Styles:
- anime: Japanese animation style (vibrant colors, expressive, stylized)
- realistic: Photorealistic, cinematic (natural lighting, detailed textures)
- cartoon: Western animation style (bold colors, simple shapes)
- pixar: 3D animation style (round shapes, warm lighting)
- ghibli: Studio Ghibli style (soft colors, detailed backgrounds)
- noir: Film noir (high contrast, black & white, dramatic shadows)
- cyberpunk: Neon-lit, futuristic, high-tech
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class VisualStyle(Enum):
    """Supported visual styles for video generation."""
    ANIME = "anime"
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    PIXAR = "pixar"
    GHIBLI = "ghibli"
    NOIR = "noir"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    CINEMATIC = "cinematic"  # Default


@dataclass
class StyleProfile:
    """
    Complete style profile for rendering.
    
    Contains all parameters needed to maintain consistent style
    across an entire episode.
    """
    style: VisualStyle
    confidence: float
    
    # Prompt modifiers
    style_prefix: str
    style_suffix: str
    negative_prompt: str
    
    # Color grading
    color_temperature: str  # warm, cool, neutral
    saturation: str  # high, normal, low
    contrast: str  # high, normal, low
    
    # Lighting
    lighting_style: str  # dramatic, soft, natural, neon
    
    # Camera preferences
    preferred_shots: List[str]
    camera_movement: str  # smooth, dynamic, static
    
    # Model hints
    recommended_model: str  # veo, sdxl, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["style"] = self.style.value
        return result
    
    @classmethod
    def from_style(cls, style: VisualStyle, confidence: float = 0.8) -> "StyleProfile":
        """Create a full profile from just a style enum."""
        profiles = {
            VisualStyle.ANIME: cls(
                style=style,
                confidence=confidence,
                style_prefix="anime style, Japanese animation",
                style_suffix="vibrant colors, expressive characters, clean lines, dynamic action",
                negative_prompt="realistic, photographic, 3D render, western cartoon",
                color_temperature="neutral",
                saturation="high",
                contrast="normal",
                lighting_style="dramatic",
                preferred_shots=["close_up", "medium_shot", "action_shot"],
                camera_movement="dynamic",
                recommended_model="sdxl",
            ),
            VisualStyle.REALISTIC: cls(
                style=style,
                confidence=confidence,
                style_prefix="cinematic, photorealistic",
                style_suffix="detailed textures, natural lighting, 8K quality, film grain",
                negative_prompt="cartoon, anime, illustration, painting, 3D render",
                color_temperature="neutral",
                saturation="normal",
                contrast="normal",
                lighting_style="natural",
                preferred_shots=["wide_shot", "medium_shot", "close_up"],
                camera_movement="smooth",
                recommended_model="veo",
            ),
            VisualStyle.CARTOON: cls(
                style=style,
                confidence=confidence,
                style_prefix="cartoon style, western animation",
                style_suffix="bold colors, simple shapes, expressive faces, clean outlines",
                negative_prompt="realistic, photographic, anime, 3D render",
                color_temperature="warm",
                saturation="high",
                contrast="high",
                lighting_style="soft",
                preferred_shots=["medium_shot", "close_up", "establishing"],
                camera_movement="static",
                recommended_model="sdxl",
            ),
            VisualStyle.PIXAR: cls(
                style=style,
                confidence=confidence,
                style_prefix="Pixar style 3D animation",
                style_suffix="rounded shapes, subsurface scattering, warm lighting, expressive eyes",
                negative_prompt="2D, flat, anime, realistic, photographic",
                color_temperature="warm",
                saturation="normal",
                contrast="normal",
                lighting_style="soft",
                preferred_shots=["medium_shot", "close_up", "wide_shot"],
                camera_movement="smooth",
                recommended_model="veo",
            ),
            VisualStyle.GHIBLI: cls(
                style=style,
                confidence=confidence,
                style_prefix="Studio Ghibli style, Hayao Miyazaki",
                style_suffix="soft watercolor, detailed backgrounds, whimsical, peaceful, nature",
                negative_prompt="3D render, realistic, dark, violent, modern",
                color_temperature="warm",
                saturation="normal",
                contrast="low",
                lighting_style="soft",
                preferred_shots=["wide_shot", "establishing", "medium_shot"],
                camera_movement="smooth",
                recommended_model="sdxl",
            ),
            VisualStyle.NOIR: cls(
                style=style,
                confidence=confidence,
                style_prefix="film noir, black and white",
                style_suffix="high contrast, dramatic shadows, venetian blinds, rain, night",
                negative_prompt="colorful, bright, cartoon, anime, cheerful",
                color_temperature="cool",
                saturation="low",
                contrast="high",
                lighting_style="dramatic",
                preferred_shots=["close_up", "dutch_angle", "low_angle"],
                camera_movement="smooth",
                recommended_model="veo",
            ),
            VisualStyle.CYBERPUNK: cls(
                style=style,
                confidence=confidence,
                style_prefix="cyberpunk, neon-noir",
                style_suffix="neon lights, rain, holographics, futuristic city, high-tech",
                negative_prompt="natural, bright daylight, rural, historical, anime",
                color_temperature="cool",
                saturation="high",
                contrast="high",
                lighting_style="neon",
                preferred_shots=["wide_shot", "low_angle", "dutch_angle"],
                camera_movement="dynamic",
                recommended_model="veo",
            ),
            VisualStyle.FANTASY: cls(
                style=style,
                confidence=confidence,
                style_prefix="fantasy, magical",
                style_suffix="mystical, ethereal lighting, magical particles, epic landscapes",
                negative_prompt="modern, urban, realistic, mundane, technological",
                color_temperature="warm",
                saturation="high",
                contrast="normal",
                lighting_style="dramatic",
                preferred_shots=["wide_shot", "establishing", "aerial"],
                camera_movement="smooth",
                recommended_model="veo",
            ),
            VisualStyle.CINEMATIC: cls(
                style=style,
                confidence=confidence,
                style_prefix="cinematic, movie quality",
                style_suffix="professional cinematography, film look, detailed, high production value",
                negative_prompt="amateur, low quality, cartoon, illustration",
                color_temperature="neutral",
                saturation="normal",
                contrast="normal",
                lighting_style="natural",
                preferred_shots=["wide_shot", "medium_shot", "close_up"],
                camera_movement="smooth",
                recommended_model="veo",
            ),
        }
        
        return profiles.get(style, profiles[VisualStyle.CINEMATIC])


class StyleDetector:
    """
    Detects visual style from user intent using keyword matching
    and optional LLM analysis.
    """
    
    # Keyword patterns for each style
    STYLE_KEYWORDS = {
        VisualStyle.ANIME: [
            "anime", "manga", "japanese", "shounen", "shojo", "isekai",
            "one piece", "naruto", "dragon ball", "attack on titan",
            "demon slayer", "jujutsu", "my hero", "studio bones",
        ],
        VisualStyle.GHIBLI: [
            "ghibli", "miyazaki", "totoro", "spirited away", "howl",
            "mononoke", "ponyo", "kiki", "whimsical", "peaceful nature",
        ],
        VisualStyle.CARTOON: [
            "cartoon", "disney", "looney tunes", "animated", "toon",
            "rick and morty", "adventure time", "simpsons", "family guy",
            "spongebob", "2d animation",
        ],
        VisualStyle.PIXAR: [
            "pixar", "3d animation", "toy story", "finding nemo",
            "incredibles", "up", "coco", "wall-e", "dreamworks",
        ],
        VisualStyle.REALISTIC: [
            "realistic", "photorealistic", "live action", "documentary",
            "real life", "lifelike", "natural", "authentic",
        ],
        VisualStyle.NOIR: [
            "noir", "black and white", "detective", "mystery",
            "1940s", "femme fatale", "shadows", "moody",
        ],
        VisualStyle.CYBERPUNK: [
            "cyberpunk", "neon", "futuristic", "blade runner",
            "sci-fi city", "dystopian", "high-tech", "synth",
        ],
        VisualStyle.FANTASY: [
            "fantasy", "magical", "dragon", "wizard", "medieval",
            "lord of the rings", "game of thrones", "enchanted",
            "fairy tale", "mystical",
        ],
    }
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize style detector.
        
        Args:
            use_llm: Whether to use LLM for complex detection
        """
        self.use_llm = use_llm
        self._gemini_client = None
    
    @property
    def gemini(self):
        """Lazy Gemini client initialization."""
        if self._gemini_client is None:
            try:
                from google import genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    self._gemini_client = genai.Client(api_key=api_key)
            except ImportError:
                logger.warning("google-genai not installed, LLM detection disabled")
        return self._gemini_client
    
    def detect(self, intent: str) -> StyleProfile:
        """
        Detect style from user intent.
        
        Uses a multi-step approach:
        1. Keyword matching for explicit style mentions
        2. LLM analysis for implicit style detection
        3. Default to cinematic if uncertain
        
        Args:
            intent: User's story intent/description
            
        Returns:
            StyleProfile with complete rendering parameters
        """
        intent_lower = intent.lower()
        
        # Step 1: Keyword matching
        keyword_style, keyword_confidence = self._detect_by_keywords(intent_lower)
        
        if keyword_confidence >= 0.8:
            logger.info(f"Style detected by keywords: {keyword_style.value} ({keyword_confidence:.2f})")
            return StyleProfile.from_style(keyword_style, keyword_confidence)
        
        # Step 2: LLM analysis (if enabled and keywords unclear)
        if self.use_llm and keyword_confidence < 0.7:
            try:
                llm_style, llm_confidence = self._detect_by_llm(intent)
                
                # Prefer LLM if more confident
                if llm_confidence > keyword_confidence:
                    logger.info(f"Style detected by LLM: {llm_style.value} ({llm_confidence:.2f})")
                    return StyleProfile.from_style(llm_style, llm_confidence)
            except Exception as e:
                logger.warning(f"LLM style detection failed: {e}")
        
        # Step 3: Fall back to keyword result or default
        if keyword_style:
            return StyleProfile.from_style(keyword_style, keyword_confidence)
        
        logger.info("Using default cinematic style")
        return StyleProfile.from_style(VisualStyle.CINEMATIC, 0.5)
    
    def _detect_by_keywords(self, intent: str) -> tuple:
        """
        Detect style using keyword matching.
        
        Returns:
            Tuple of (VisualStyle, confidence)
        """
        scores = {}
        
        # Priority keywords that should boost confidence significantly
        PRIORITY_KEYWORDS = {
            "pixar": VisualStyle.PIXAR,
            "ghibli": VisualStyle.GHIBLI,
            "miyazaki": VisualStyle.GHIBLI,
            "disney": VisualStyle.CARTOON,
            "anime": VisualStyle.ANIME,
            "manga": VisualStyle.ANIME,
            "cyberpunk": VisualStyle.CYBERPUNK,
            "noir": VisualStyle.NOIR,
        }
        
        # Check priority keywords first
        for keyword, style in PRIORITY_KEYWORDS.items():
            if keyword in intent:
                # High confidence for explicit brand/style mention
                scores[style] = max(scores.get(style, 0), 0.85)
        
        # Then check all keywords
        for style, keywords in self.STYLE_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                if keyword in intent:
                    matches += 1
            
            if matches > 0:
                # Confidence based on number of matches
                base_confidence = min(0.5 + (matches * 0.15), 0.95)
                # Don't override priority keyword scores
                if style not in scores or base_confidence > scores[style]:
                    scores[style] = base_confidence
        
        if scores:
            best_style = max(scores, key=scores.get)
            return best_style, scores[best_style]
        
        return None, 0.0
    
    def _detect_by_llm(self, intent: str) -> tuple:
        """
        Detect style using Gemini LLM.
        
        Returns:
            Tuple of (VisualStyle, confidence)
        """
        if not self.gemini:
            return None, 0.0
        
        prompt = f"""Analyze this story intent and determine the most appropriate visual style for video generation.

Story Intent: {intent}

Return ONLY a JSON object with:
- "style": One of [anime, realistic, cartoon, pixar, ghibli, noir, cyberpunk, fantasy, cinematic]
- "confidence": A number 0-1 indicating confidence
- "reason": Brief explanation

Example response:
{{"style": "anime", "confidence": 0.85, "reason": "User mentions Japanese animation elements"}}

JSON response:"""

        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            
            # Parse response
            text = response.text.strip()
            
            # Clean up JSON if needed
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            
            style_str = data.get("style", "cinematic").lower()
            confidence = float(data.get("confidence", 0.7))
            
            # Map string to enum
            style_map = {s.value: s for s in VisualStyle}
            style = style_map.get(style_str, VisualStyle.CINEMATIC)
            
            return style, confidence
            
        except Exception as e:
            logger.error(f"LLM style detection error: {e}")
            return None, 0.0
    
    def enhance_prompt(self, prompt: str, style_profile: StyleProfile) -> str:
        """
        Enhance a generation prompt with style modifiers.
        
        Args:
            prompt: Original generation prompt
            style_profile: Detected style profile
            
        Returns:
            Enhanced prompt with style-specific modifiers
        """
        parts = []
        
        # Add style prefix
        if style_profile.style_prefix:
            parts.append(style_profile.style_prefix)
        
        # Add original prompt
        parts.append(prompt)
        
        # Add style suffix
        if style_profile.style_suffix:
            parts.append(style_profile.style_suffix)
        
        return ", ".join(parts)


# Singleton instance for easy access
_detector = None

def get_style_detector(use_llm: bool = True) -> StyleDetector:
    """Get or create the global StyleDetector instance."""
    global _detector
    if _detector is None:
        _detector = StyleDetector(use_llm=use_llm)
    return _detector


def detect_style(intent: str) -> StyleProfile:
    """
    Convenience function to detect style from intent.
    
    Usage:
        profile = detect_style("A cyberpunk story about hackers")
        print(profile.style)  # VisualStyle.CYBERPUNK
    """
    return get_style_detector().detect(intent)


if __name__ == "__main__":
    # Test the style detector
    test_intents = [
        "A dragon ball style fight scene between two super warriors",
        "A noir detective story set in 1940s Los Angeles",
        "A realistic documentary about climate change",
        "A Pixar style adventure about toys coming to life",
        "A cyberpunk heist in a neon-lit future city",
        "A Studio Ghibli inspired story about a girl and her forest friends",
        "An epic fantasy battle with dragons and wizards",
    ]
    
    print("\n" + "="*60)
    print("STYLE DETECTION TEST")
    print("="*60)
    
    detector = StyleDetector(use_llm=False)  # Keywords only for quick test
    
    for intent in test_intents:
        profile = detector.detect(intent)
        print(f"\nIntent: {intent[:50]}...")
        print(f"  Style: {profile.style.value}")
        print(f"  Confidence: {profile.confidence:.2f}")
        print(f"  Model: {profile.recommended_model}")
