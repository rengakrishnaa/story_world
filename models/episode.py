# models/episode.py

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


@dataclass
class Episode:
    episode_id: str
    world_id: str
    intent: str
    state: str
    policies: Dict[str, Any]
    created_at: datetime
