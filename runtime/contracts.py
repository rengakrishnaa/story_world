# runtime/contracts.py
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EpisodeRequest:
    intent: str
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]

@dataclass
class EpisodeStatus:
    episode_id: str
    state: str
    progress: float
    artifacts: List[Dict[str, Any]]
