# models/beat_observation.py

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class BeatObservation:
    beat_id: str
    success: bool
    confidence: float
    intent_satisfied: bool
    entity_presence: Dict[str, float]
    constraint_violations: list
    failure_type: Optional[str]
    explanation: Optional[str]
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "success": self.success,
            "confidence": self.confidence,
            "intent_satisfied": self.intent_satisfied,
            "entity_presence": self.entity_presence,
            "constraint_violations": self.constraint_violations,
            "failure_type": self.failure_type,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
        }
