# runtime/state_store.py
class EpisodeStateStore:
    def __init__(self, db):
        self.db = db

    def create_episode(self, episode_id, intent):
        self.db.insert("episodes", {
            "id": episode_id,
            "intent": intent,
            "state": "CREATED",
        })

    def update_state(self, episode_id, new_state):
        self.db.update(
            "episodes",
            where={"id": episode_id},
            values={"state": new_state},
        )

    def record_artifact(self, episode_id, beat_id, artifact):
        self.db.insert("artifacts", {
            "episode_id": episode_id,
            "beat_id": beat_id,
            **artifact,
        })
