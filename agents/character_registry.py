from pathlib import Path
from typing import Dict
from models.character_profile import CharacterProfile
from agents.embedding_provider import EmbeddingProvider
import numpy as np

class CharacterRegistry:
    def __init__(self, base_path="uploads/characters"):
        self.base_path = Path(base_path)
        self.embedder = EmbeddingProvider()
        self.characters: Dict[str, CharacterProfile] = {}

    def reload(self):
        self.characters.clear()
        self._load()


    def _load(self):
        for char_dir in self.base_path.iterdir():
            if not char_dir.is_dir():
                continue

            images = list(char_dir.glob("*.png"))
            embeddings = [
                self.embedder.embed_image(str(img))
                for img in images
            ]

            self.characters[char_dir.name] = CharacterProfile(
                name=char_dir.name,
                reference_images=[str(i) for i in images],
                reference_embeddings=embeddings
            )

    def get(self, name: str) -> CharacterProfile | None:
        return self.characters.get(name)
