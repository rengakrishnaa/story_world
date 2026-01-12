# runtime/control/beat_lifecycle.py

from runtime.control.contracts import BeatState, ExecutionObservation


class BeatLifecycle:
    """
    Pure state machine.
    No IO. No side effects.
    """

    @staticmethod
    def transition(
        current_state: BeatState,
        observation: ExecutionObservation,
        retry_allowed: bool
    ) -> BeatState:

        if current_state in {BeatState.SUCCEEDED, BeatState.ABORTED}:
            return current_state

        if observation.success:
            return BeatState.SUCCEEDED

        if not retry_allowed:
            return BeatState.ABORTED

        return BeatState.FAILED
