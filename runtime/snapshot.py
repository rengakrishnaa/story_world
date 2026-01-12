from typing import Dict, List
from runtime.episode_state import EpisodeState
from runtime.beat_state import BeatState


class EpisodeSnapshot:
    def __init__(
        self,
        episode_id: str,
        state: EpisodeState,
        beats: List[Dict],
        artifacts: List[Dict],
        errors: List[Dict],
        total_cost: float,
    ):
        self.episode_id = episode_id
        self.state = state
        self.beats = beats
        self.artifacts = artifacts
        self.errors = errors
        self.total_cost = total_cost

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "state": self.state,
            "progress": self._progress(),
            "beats": self.beats,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "total_cost": self.total_cost,
        }

    def _progress(self) -> Dict:
        total = len(self.beats)
        completed = len(
            [b for b in self.beats if b["state"] == BeatState.ACCEPTED]
        )
        aborted = len(
            [b for b in self.beats if b["state"] == BeatState.ABORTED]
        )

        return {
            "total_beats": total,
            "completed": completed,
            "aborted": aborted,
            "percent": round((completed / total) * 100, 2) if total else 0.0,
        }

    # ---------- Factory ----------

    @classmethod
    def from_runtime(cls, runtime):
        sql = runtime.sql

        episode = sql.get_episode(runtime.episode_id)
        beats = sql.get_beats(runtime.episode_id)
        artifacts = sql.get_artifacts(runtime.episode_id)

        errors = [
            {
                "beat_id": b["beat_id"],
                "error": b["last_error"],
            }
            for b in beats
            if b["state"] == BeatState.ABORTED
        ]

        total_cost = sum(b.get("cost_spent", 0) for b in beats)

        return cls(
            episode_id=runtime.episode_id,
            state=episode["state"],
            beats=beats,
            artifacts=artifacts,
            errors=errors,
            total_cost=total_cost,
        )
