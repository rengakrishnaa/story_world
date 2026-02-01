"""
Episode Plan Model

Defines the plan for composing an episode from individual beats.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EpisodePlan:
    """
    Plan for composing an episode from beats.
    
    Attributes:
        beats: Ordered list of beat IDs to include
        allow_gaps: Whether to skip missing beats or fail
        required_confidence: Minimum confidence score for inclusion
        transition: Transition type between beats (none, crossfade)
        transition_duration: Duration of transitions in seconds
    """
    beats: List[str] = field(default_factory=list)
    allow_gaps: bool = True
    required_confidence: float = 0.3
    transition: str = "none"
    transition_duration: float = 0.5
    
    @property
    def beat_count(self) -> int:
        return len(self.beats)
    
    def add_beat(self, beat_id: str) -> None:
        """Add a beat to the plan."""
        if beat_id not in self.beats:
            self.beats.append(beat_id)
    
    def remove_beat(self, beat_id: str) -> None:
        """Remove a beat from the plan."""
        if beat_id in self.beats:
            self.beats.remove(beat_id)
