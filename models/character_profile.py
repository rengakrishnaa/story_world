# models/character_profile.py
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class CharacterProfile:
    character_id: str
    reference_embeddings: List[np.ndarray] = field(default_factory=list)

    def add_reference(self, embedding: np.ndarray):
        self.reference_embeddings.append(embedding)
