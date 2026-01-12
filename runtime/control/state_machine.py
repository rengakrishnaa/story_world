# runtime/control/state_machine.py

from typing import Dict
from runtime.control.contracts import BeatState, ExecutionObservation
from runtime.control.beat_lifecycle import BeatLifecycle


class ControlStateMachine:
    """
    Central authority for beat state transitions.
    """

    def __init__(self, retry_limit: int):
        self.retry_limit = retry_limit
        self.retry_count: Dict[str, int] = {}

    def apply_observation(
        self,
        beat_id: str,
        current_state: BeatState,
        observation: ExecutionObservation
    ) -> BeatState:

        count = self.retry_count.get(beat_id, 0)
        retry_allowed = count < self.retry_limit

        next_state = BeatLifecycle.transition(
            current_state=current_state,
            observation=observation,
            retry_allowed=retry_allowed
        )

        if next_state == BeatState.FAILED:
            self.retry_count[beat_id] = count + 1

        return next_state
