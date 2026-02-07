"""
Observer Calibration & Reliability

Records verdicts for error-rate measurement.
Human labels can be added later for calibration loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class VerdictRecord:
    """Single verdict for calibration tracking."""
    beat_id: str
    episode_id: str
    verdict: str
    confidence: float
    intent: str = ""
    human_label: Optional[str] = None  # "correct" | "incorrect" | None
    observed_at: datetime = field(default_factory=datetime.utcnow)


class ObserverCalibration:
    """In-memory verdict history for error-rate measurement. Persistence optional."""

    def __init__(self, max_records: int = 1000):
        self._records: List[VerdictRecord] = []
        self.max_records = max_records

    def record(self, beat_id: str, episode_id: str, verdict: str, confidence: float, intent: str = "") -> None:
        r = VerdictRecord(beat_id=beat_id, episode_id=episode_id, verdict=verdict, confidence=confidence, intent=intent)
        self._records.append(r)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

    def add_human_label(self, beat_id: str, human_label: str) -> bool:
        for r in self._records:
            if r.beat_id == beat_id:
                r.human_label = human_label
                return True
        return False

    def get_error_rate(self) -> Optional[float]:
        """Error rate among human-labeled verdicts. Returns None if no labels."""
        labeled = [r for r in self._records if r.human_label]
        if not labeled:
            return None
        incorrect = sum(1 for r in labeled if r.human_label == "incorrect")
        return incorrect / len(labeled)

    def get_verdict_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for r in self._records:
            v = (r.verdict or "unknown").lower()
            dist[v] = dist.get(v, 0) + 1
        return dist

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        out = []
        for r in self._records[-n:]:
            out.append({
                "beat_id": r.beat_id,
                "episode_id": r.episode_id,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "human_label": r.human_label,
                "observed_at": r.observed_at.isoformat(),
            })
        return out


# Singleton for process-wide use
_calibration: Optional[ObserverCalibration] = None


def get_calibration() -> ObserverCalibration:
    global _calibration
    if _calibration is None:
        import os
        max_rec = int(os.getenv("OBSERVER_CALIBRATION_MAX_RECORDS", "1000"))
        _calibration = ObserverCalibration(max_records=max_rec)
    return _calibration
