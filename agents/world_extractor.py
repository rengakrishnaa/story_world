import google.genai as genai
from typing import List
import json
import base64
from pathlib import Path

# Import from models
try:
    from models.world import WorldGraph, Character, Location
except ImportError:
    from models.world import WorldGraph, Character, Location


class WorldExtractor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3-flash"

    def extract_from_images(
        self,
        character_images: List[str],
        location_images: List[str],
    ) -> WorldGraph:

        characters = []
        for i, img_path in enumerate(character_images[:3]):
            char_data = self._analyze_character_image(img_path)
            characters.append(
                Character(
                    name=char_data.get("name", f"Character_{i+1}"),
                    description=char_data["description"],
                    reference_image_url=img_path,
                    traits=char_data.get("traits", []),
                )
            )

        locations = []
        for i, img_path in enumerate(location_images[:3]):
            loc_data = self._analyze_location_image(img_path)
            locations.append(
                Location(
                    name=loc_data.get("name", f"Location_{i+1}"),
                    description=loc_data["description"],
                    reference_image_url=img_path,
                )
            )

        return WorldGraph(characters=characters, locations=locations)

    def _analyze_character_image(self, img_path: str) -> dict:
        """Analyze character image using Gemini multimodal API."""
        
        # Determine MIME type from file extension
        ext = Path(img_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Read and encode image
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Correct API format for Gemini
        prompt = """Analyze this character reference image for anime production.
Return ONLY valid JSON (no markdown, no backticks):
{"name": "Character Name", "description": "detailed physical description", "traits": ["trait1", "trait2", "trait3"]}
Focus on: appearance, clothing, age estimate, personality hints from visual cues."""
        
        contents = [
            {
                "text": prompt
            },
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
        ]
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )
            
            return self._safe_json(response.text)
        except Exception as e:
            print(f"Error analyzing character image: {e}")
            return {
                "name": "Unknown Character",
                "description": "Character analysis failed",
                "traits": ["unknown"]
            }

    def _analyze_location_image(self, img_path: str) -> dict:
        """Analyze location image using Gemini multimodal API."""
        
        # Determine MIME type
        ext = Path(img_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Read and encode image
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        prompt = """Analyze this location reference image for anime production.
Return ONLY valid JSON (no markdown, no backticks):
{"name": "Location Name", "description": "detailed description of setting, atmosphere, time of day, architectural style"}"""
        
        contents = [
            {
                "text": prompt
            },
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
        ]
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )
            
            return self._safe_json(response.text)
        except Exception as e:
            print(f"Error analyzing location image: {e}")
            return {
                "name": "Unknown Location",
                "description": "Location analysis failed"
            }

    def _safe_json(self, text: str) -> dict:
        """Bulletproof JSON extraction for Gemini responses."""

        if not text or not text.strip():
            raise ValueError("Empty response from model")

        cleaned = text.strip()

        # Remove Markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines).strip()

        # Remove language hints that might remain
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

        print(f"DEBUG: Raw response (first 120 chars): {repr(text[:120])}")
        print(f"DEBUG: Cleaned JSON (first 120 chars): {repr(cleaned[:120])}")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            import re
            
            print(f"JSON parse error: {e}")

            # LAST-RESORT extraction: find first complete JSON object
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Could not parse JSON from model output: {cleaned[:200]}")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment")
        exit(1)
    
    extractor = WorldExtractor(api_key)
    
    # Example paths
    char_images = [r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\uploads\img_1.webp"]
    loc_images = [r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\uploads\location.jpg"]
    
    try:
        world = extractor.extract_from_images(char_images, loc_images)
        print(world.to_json())
    except Exception as e:
        print(f"Error: {e}")

#-----------------------------
from models.world import WorldGraph, Character, Location

test_world_graph = WorldGraph(
    characters=[
        Character(
            name="Bald Hero",
            description=(
                "A lean, athletic adult male with a completely bald head and sharp, focused eyes. "
                "He wears a bright yellow bodysuit with a white cape fastened at the shoulders, "
                "red gloves, and red boots. His posture is relaxed but confident, conveying immense "
                "strength and emotional restraint."
            ),
            reference_image_url=r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\uploads\img_1.webp",
            traits=[
                "calm",
                "emotionally detached",
                "overpowered",
                "deadpan",
                "confident"
            ]
        ),
        Character(
            name="Eldritch Fire Entity",
            description=(
                "A colossal multi-limbed monster with a dark purple-black exoskeleton and glowing "
                "fiery core. Numerous tendrils and clawed limbs extend from its body, radiating heat "
                "and destructive energy. Its presence dominates the environment, suggesting ancient "
                "and near-godlike power."
            ),
            reference_image_url=r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\uploads\img_2.png",
            traits=[
                "aggressive",
                "ancient",
                "destructive",
                "intimidating",
                "inhuman"
            ]
        )
    ],
    locations=[
        Location(
            name="Volcanic Ruins",
            description=(
                "A vast volcanic landscape filled with molten lava rivers and crumbling stone ruins. "
                "The environment glows with intense orange and yellow light from magma below, while "
                "ash and heat distort the air. The setting feels ancient, hostile, and apocalyptic."
            ),
            reference_image_url=r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\uploads\location.jpg"
        )
    ]
)
