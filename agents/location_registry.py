from pathlib import Path
from typing import Dict
from models.location_profile import LocationProfile

class LocationRegistry:
    def __init__(self, base_path="uploads/locations"):
        self.base_path = Path(base_path)
        self.locations: Dict[str, LocationProfile] = {}
        self._load()

    def _load(self):
        for loc_dir in self.base_path.iterdir():
            if not loc_dir.is_dir():
                continue

            images = list(loc_dir.glob("*.png"))

            self.locations[loc_dir.name] = LocationProfile(
                name=loc_dir.name,
                reference_images=[str(i) for i in images]
            )

    def get(self, name: str) -> LocationProfile | None:
        return self.locations.get(name)
