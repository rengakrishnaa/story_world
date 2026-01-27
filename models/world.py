"""
World model for StoryWorld video generation.
Represents the characters, locations, and relationships in a story world.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Character:
    """Represents a character in the story world."""
    name: str
    description: str
    reference_image_url: str
    traits: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Character":
        """Create Character from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            reference_image_url=data["reference_image_url"],
            traits=data.get("traits", [])
        )


@dataclass
class Location:
    """Represents a location in the story world."""
    name: str
    description: str
    reference_image_url: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Location":
        """Create Location from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            reference_image_url=data["reference_image_url"]
        )


@dataclass
class WorldGraph:
    """
    Represents the complete story world with all characters and locations.
    This is the primary data structure for world state.
    """
    characters: List[Character]
    locations: List[Location]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "characters": [c.to_dict() for c in self.characters],
            "locations": [l.to_dict() for l in self.locations]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldGraph":
        """Create WorldGraph from dictionary."""
        characters = [
            Character.from_dict(c) for c in data.get("characters", [])
        ]
        locations = [
            Location.from_dict(l) for l in data.get("locations", [])
        ]
        return cls(characters=characters, locations=locations)
    
    @classmethod
    def from_json(cls, json_str: str) -> "WorldGraph":
        """Create WorldGraph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_character(self, name: str) -> Optional[Character]:
        """Get character by name (case-insensitive)."""
        name_lower = name.lower()
        for char in self.characters:
            if char.name.lower() == name_lower:
                return char
        return None
    
    def get_location(self, name: str) -> Optional[Location]:
        """Get location by name (case-insensitive)."""
        name_lower = name.lower()
        for loc in self.locations:
            if loc.name.lower() == name_lower:
                return loc
        return None
    
    def character_names(self) -> List[str]:
        """Get list of all character names."""
        return [c.name for c in self.characters]
    
    def location_names(self) -> List[str]:
        """Get list of all location names."""
        return [l.name for l in self.locations]
    
    def __len__(self) -> int:
        """Total number of entities in the world."""
        return len(self.characters) + len(self.locations)
    
    def is_empty(self) -> bool:
        """Check if world has no characters or locations."""
        return len(self.characters) == 0 and len(self.locations) == 0


# Example usage and testing
if __name__ == "__main__":
    # Create test world
    world = WorldGraph(
        characters=[
            Character(
                name="Saitama",
                description="Bald hero in yellow suit with incredible strength",
                reference_image_url="/uploads/saitama.png",
                traits=["calm", "overpowered", "deadpan"]
            ),
            Character(
                name="Genos",
                description="Cyborg hero with yellow armor",
                reference_image_url="/uploads/genos.png",
                traits=["serious", "loyal", "analytical"]
            )
        ],
        locations=[
            Location(
                name="City Rooftop",
                description="Urban rooftop with city skyline view",
                reference_image_url="/uploads/rooftop.png"
            )
        ]
    )
    
    # Test serialization
    json_str = world.to_json()
    print("Serialized:")
    print(json_str)
    
    # Test deserialization
    world_restored = WorldGraph.from_json(json_str)
    print(f"\nRestored: {len(world_restored)} entities")
    print(f"Characters: {world_restored.character_names()}")
    print(f"Locations: {world_restored.location_names()}")
    
    # Test lookups
    char = world_restored.get_character("saitama")
    if char:
        print(f"\nFound character: {char.name} - {char.description}")
