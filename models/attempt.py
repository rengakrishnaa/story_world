# models/attempt.py

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


@dataclass
class Attempt:
    attempt_id: str
    episode_id: str
    beat_id: str
    model: str
    prompt: str
    success: bool
    metrics: Dict[str, Any]
    created_at: datetime
