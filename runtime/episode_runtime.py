from datetime import datetime
from runtime.episode_state import EpisodeState
from runtime.persistence.sql_store import SQLStore
from runtime.persistence.redis_store import RedisStore
from runtime.snapshot import EpisodeSnapshot
from runtime.policies.retry_policy import RetryPolicy
from runtime.policies.quality_policy import QualityPolicy
from runtime.policies.cost_policy import CostPolicy
import uuid
from runtime.beat_state import BeatState
from runtime.episode_state import EpisodeState

class EpisodeRuntime:
    def __init__(self, episode_id, world_id, intent, policies, sql: SQLStore, redis: RedisStore):
        self.episode_id = episode_id
        self.world_id = world_id
        self.intent = intent
        self.policies = policies

        self.sql = sql
        self.redis = redis

        self.retry_policy = RetryPolicy(policies.get("retry", {}))
        self.quality_policy = QualityPolicy(policies.get("quality", {}))
        self.cost_policy = CostPolicy(policies.get("cost", {}))

        self.state = EpisodeState.CREATED

    @classmethod
    def create(cls, world_id, intent, policies, sql, redis):
        episode_id = str(uuid.uuid4())

        sql.create_episode(
            episode_id=episode_id,
            world_id=world_id,
            intent=intent,
            policies=policies,
            state=EpisodeState.CREATED
        )

        return cls(episode_id, world_id, intent, policies, sql, redis)

    

    def plan(self, planner):
        episode = self.sql.get_episode(self.episode_id)
        self.state = episode["state"]

        if self.state == EpisodeState.PLANNED:
            return

        if self.state != EpisodeState.CREATED:
            raise RuntimeError(f"Cannot plan in state {self.state}")

        beats = planner.generate_beats(self.intent)

        for beat in beats:
            beat["id"] = f"{self.episode_id}:{beat['id']}"
            self.sql.create_beat(self.episode_id, beat)

        self._advance(EpisodeState.PLANNED)



    def schedule(self):
        episode = self.sql.get_episode(self.episode_id)
        db_state = episode["state"]
        self.state = db_state

        if db_state == EpisodeState.EXECUTING:
            # Already executing → idempotent success
            return

        if db_state not in {
            EpisodeState.PLANNED,
            EpisodeState.PARTIALLY_COMPLETED,
        }:
            raise RuntimeError(f"Cannot execute episode in state {db_state}")

        beats = self.sql.get_pending_beats(self.episode_id)
        for beat in beats:
            self.redis.enqueue_beat(self.episode_id, beat)

        self._advance(EpisodeState.EXECUTING)

    def ingest_observation(self, beat_id, attempt_id, observation=None, error=None):
        # Mark beat observed
        self.sql.mark_beat_state(beat_id, BeatState.OBSERVED, error)

        attempts = self.sql.get_attempts(beat_id)

        decision = self.retry_policy.decide(
            beat={"id": beat_id},
            attempts=attempts,
            observation=observation,
            error=error,
        )

        if decision.action == "ACCEPT":
            self.sql.mark_beat_state(beat_id, BeatState.ACCEPTED)

        elif decision.action == "RETRY":
            self.sql.mark_beat_state(beat_id, BeatState.PENDING)
            self.redis.enqueue_retry(self.episode_id, decision.to_payload({"id": beat_id}))

        elif decision.action == "ABORT":
            self.sql.mark_beat_state(beat_id, BeatState.ABORTED, decision.reason)

        self._recompute_episode_state()


    def snapshot(self):
        return EpisodeSnapshot.from_runtime(self).to_dict()

    def _advance(self, new_state):
        self.state = new_state
        self.sql.update_episode_state(self.episode_id, new_state, datetime.utcnow())

    def _recompute_episode_state(self):
        if self.sql.all_beats_completed(self.episode_id):
            self._advance(EpisodeState.COMPLETED)
        elif self.sql.any_beats_failed(self.episode_id):
            self._advance(EpisodeState.DEGRADED)

    def is_terminal(self):
        return self.state in {
            EpisodeState.COMPLETED,
            EpisodeState.FAILED,
        }

    
    @classmethod
    def load(cls, episode_id, sql, redis, policies):
        """
        Rehydrate runtime from DB state.
        """
        episode = sql.get_episode(episode_id)
        if not episode:
            raise RuntimeError(f"Episode {episode_id} not found")

        policy_config = episode["policies"]
        runtime = cls(
            episode_id=episode_id,
            world_id=episode.get("world_id"),
            intent=episode.get("intent"),
            policies=policy_config,
            sql=sql,
            redis=redis,
        )

        runtime.state = episode["state"]
        return runtime

