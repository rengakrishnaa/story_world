# runtime/control/control_bridge.py

from story_world.decision_core import ControlRuntime
from runtime.control.dispatcher import BeatDispatcher
from runtime.control.observation_adapter import ObservationAdapter


class ControlBridge:
    """
    Orchestrates ControlRuntime <-> Worker execution.
    """

    def __init__(self, control_runtime: ControlRuntime):
        self.runtime = control_runtime

    def dispatch(self, enqueue_fn):
        """
        Enqueue eligible beats for execution.
        enqueue_fn: function(beat_spec) -> None
        """

        beats = BeatDispatcher.next_beats(
            beats=list(self.runtime.beats.values()),
            beat_states=self.runtime.beat_states
        )

        for beat in beats:
            self.runtime.on_beat_started(beat.beat_id)
            enqueue_fn(beat)

    def handle_worker_result(self, raw_payload: dict):
        observation = ObservationAdapter.from_worker_payload(raw_payload)
        self.runtime.on_observation(observation)
