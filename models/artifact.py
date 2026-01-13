# models/artifact.py

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Artifact:
    artifact_id: str
    beat_id: str
    attempt_id: str
    type: str              # "keyframe" | "video"
    uri: str
    metadata: Dict[str, Any]
