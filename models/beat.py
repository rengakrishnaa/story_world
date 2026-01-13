# models/beat.py

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Beat:
    beat_id: str
    episode_id: str
    description: str
    parameters: Dict[str, Any]
