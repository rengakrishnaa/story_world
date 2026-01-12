# runtime/control/episode_lifecycle.py

from typing import List
from runtime.control.contracts import EpisodeState, BeatState


class EpisodeLifecycle:
    """
    Computes episode state from beat states.
    """

    @staticmethod
    def compute(beat_states: List[BeatState]) -> EpisodeState:
        if any(s == BeatState.ABORTED for s in beat_states):
            return EpisodeState.DEGRADED

        if all(s == BeatState.SUCCEEDED for s in beat_states):
            return EpisodeState.COMPLETED

        if any(s in {BeatState.RUNNING, BeatState.FAILED} for s in beat_states):
            return EpisodeState.EXECUTING

        return EpisodeState.PLANNED
