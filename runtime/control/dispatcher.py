# runtime/control/dispatcher.py

from typing import List
from runtime.control.contracts import BeatState, BeatSpec


class BeatDispatcher:
    """
    Decides which beats are eligible for execution.
    """

    @staticmethod
    def next_beats(
        beats: List[BeatSpec],
        beat_states: dict
    ) -> List[BeatSpec]:

        runnable = []

        for beat in beats:
            state = beat_states.get(beat.beat_id)
            if state == BeatState.PENDING:
                runnable.append(beat)

        return runnable
