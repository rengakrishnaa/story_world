import numpy as np
import importlib
redis = importlib.import_module("redis")

import json
from typing import Optional


class ContinuityMemory:
    """
    Stores last accepted embedding per character per world.
    Used ONLY for temporal continuity (not reference identity).
    """

    def __init__(self, world_id: str, redis_url="redis://localhost:6379"):
        self.world_id = world_id
        self.r = redis.from_url(redis_url)
        self._cache = {}

    def _key(self, character: str) -> str:
        return f"continuity:{self.world_id}:{character}"

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0 or np.isnan(norm):
            return v
        return v / norm

    def load(self, character: str) -> Optional[np.ndarray]:
        # Check in-memory first
        if character in self._cache:
            return self._cache[character]

        raw = self.r.get(self._key(character))
        if not raw:
            return None

        data = json.loads(raw)
        emb = np.array(data["embedding"], dtype=np.float32)
        emb = self._normalize(emb)

        self._cache[character] = emb
        return emb

    def similarity(self, character: str, embedding: np.ndarray) -> Optional[float]:
        prev = self.load(character)
        if prev is None:
            return None  # first appearance

        embedding = self._normalize(embedding)

        denom = np.linalg.norm(prev) * np.linalg.norm(embedding)
        if denom == 0:
            return 0.0

        return float(np.dot(prev, embedding) / denom)

    def update(self, character: str, embedding: np.ndarray, alpha=0.3):
        """
        EMA update — ONLY after beat is accepted.
        """
        embedding = self._normalize(embedding)

        prev = self.load(character)
        if prev is None:
            updated = embedding
        else:
            updated = (1 - alpha) * prev + alpha * embedding
            updated = self._normalize(updated)

        self._cache[character] = updated
        self.r.set(
            self._key(character),
            json.dumps({"embedding": updated.tolist()})
        )
