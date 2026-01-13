# models/location_profile.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LocationProfile:
    name: str
    description: str = ""
    metadata: Dict[str, Any] = None
