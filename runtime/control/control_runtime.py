# runtime/control/control_runtime.py

from datetime import datetime
from runtime.control.decision_core import DecisionCore
from runtime.control.event_log import RuntimeEvent
from runtime.control.branching import EpisodeBrancher

class ControlRuntime:
    """
    Authoritative control plane.
    Owns persistence, orchestration, branching.
    Delegates decisions to DecisionCore.
    """

    def __init__(
        self,
        episode,
        beats,
        retry_limit,
        event_log,
        state_store,
        queue,
    ):
        self.decision = DecisionCore(episode, beats, retry_limit)
        self.event_log = event_log
        self.state_store = state_store
        self.queue = queue
        self.brancher = EpisodeBrancher()

    # ---------- Scheduling ----------

    def schedule_beat(self, beat):
        self.queue.enqueue({
            "episode_id": self.decision.episode.episode_id,
            "beat_id": beat.beat_id,
            "execution_spec": beat.to_dict(),
        })

        self.decision.on_beat_started(beat.beat_id)

        self.event_log.append(RuntimeEvent(
            episode_id=self.decision.episode.episode_id,
            type="BEAT_SCHEDULED",
            payload={"beat_id": beat.beat_id},
            ts=datetime.utcnow(),
        ))

    # ---------- Worker Result Intake ----------

    def handle_worker_result(self, observation):
        """
        Single entry point from workers.
        """
        self.event_log.append(RuntimeEvent(
            episode_id=self.decision.episode.episode_id,
            type="BEAT_OBSERVED",
            payload=observation.to_dict(),
            ts=datetime.utcnow(),
        ))

        self.decision.on_observation(observation)

        state = self.decision.episode_state()

        if observation.success is False:
            return self._handle_failure(observation)

        return state

    # ---------- Failure & Branching ----------

    def _handle_failure(self, observation):
        fork = self.brancher.fork_episode(
            base_episode_id=self.decision.episode.episode_id,
            reason=observation.failure_type,
        )

        self.event_log.append(RuntimeEvent(
            episode_id=fork["new_episode_id"],
            type="EPISODE_FORKED",
            payload=fork,
            ts=datetime.utcnow(),
        ))

        return fork
