"""
Composed Shot Model

Represents a shot that has been rendered and is ready for episode composition.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ComposedShot:
    """
    A completed shot ready for episode stitching.
    
    Attributes:
        beat_id: Unique identifier for this beat
        video_path: Path or URL to the rendered video file
        duration: Duration of the video in seconds
        confidence: Quality confidence score (0-1)
        order: Position in the episode sequence
    """
    beat_id: str
    video_path: Union[Path, str]  # Can be local Path or URL string
    duration: float = 5.0
    confidence: float = 0.5
    order: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if shot is valid for composition."""
        return self.confidence >= 0.3 and self.duration > 0
