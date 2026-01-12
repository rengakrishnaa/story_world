from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class HumanPose:
    body: np.ndarray        # (33, 3) body landmarks
    left_hand: Optional[np.ndarray]  # (21, 3)
    right_hand: Optional[np.ndarray] # (21, 3)
    face: Optional[np.ndarray]       # (468, 3)
    confidence: float
