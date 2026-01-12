from abc import ABC, abstractmethod
from typing import List, Dict

class PlannerInterface(ABC):
    @abstractmethod
    def generate_beats(self, intent: str) -> List[Dict]:
        """
        Returns a list of beat specs.
        Runtime does NOT care how they were produced.
        """
        pass
