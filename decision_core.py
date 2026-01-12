# runtime/control/decision_core.py

class DecisionCore:
    def __init__(self, episode, beats, retry_limit):
        self.episode = episode
        self.beats = {b.beat_id: b for b in beats}
        self.beat_states = {b.beat_id: BeatState.PENDING for b in beats}
        self.state_machine = ControlStateMachine(retry_limit)

    def on_beat_started(self, beat_id: str):
        self.beat_states[beat_id] = BeatState.RUNNING

    def on_observation(self, observation: ExecutionObservation):
        beat_id = observation.beat_id
        current = self.beat_states[beat_id]

        next_state = self.state_machine.apply_observation(
            beat_id=beat_id,
            current_state=current,
            observation=observation
        )

        self.beat_states[beat_id] = next_state

    def episode_state(self) -> EpisodeState:
        return EpisodeLifecycle.compute(list(self.beat_states.values()))
