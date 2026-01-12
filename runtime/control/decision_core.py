# runtime/control/decision_core.py

from typing import Dict

from runtime.control.contracts import (
    BeatState,
    EpisodeState,
    ExecutionObservation,
)
from runtime.control.state_machine import ControlStateMachine
from runtime.control.episode_lifecycle import EpisodeLifecycle


class DecisionCore:
    """
    Deterministic decision engine.
    No IO. No side effects.
    """

    def __init__(self, episode, beats, retry_limit: int):
        self.episode = episode
        self.beats = {b.beat_id: b for b in beats}

        self.beat_states: Dict[str, BeatState] = {
            b.beat_id: BeatState.PENDING for b in beats
        }

        self.state_machine = ControlStateMachine(retry_limit)

    def on_beat_started(self, beat_id: str):
        if beat_id not in self.beat_states:
            return
        self.beat_states[beat_id] = BeatState.RUNNING

    def on_observation(self, observation: ExecutionObservation):
        beat_id = observation.beat_id

        if beat_id not in self.beat_states:
            # late or foreign observation → ignore safely
            return

        current = self.beat_states[beat_id]

        next_state = self.state_machine.apply_observation(
            beat_id=beat_id,
            current_state=current,
            observation=observation,
        )

        self.beat_states[beat_id] = next_state

    def episode_state(self) -> EpisodeState:
        return EpisodeLifecycle.compute(list(self.beat_states.values()))
