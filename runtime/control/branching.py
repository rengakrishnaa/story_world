# runtime/branching.py
import uuid

class EpisodeBrancher:
    def fork_episode(self, base_episode_id, reason):
        new_episode_id = str(uuid.uuid4())

        return {
            "new_episode_id": new_episode_id,
            "parent_episode_id": base_episode_id,
            "reason": reason,
        }
