import time
from runtime.episode_runtime import EpisodeRuntime

class RuntimeDecisionLoop:
    def __init__(self, runtime: EpisodeRuntime):
        self.runtime = runtime

    def run(self):
        print(f"[runtime] Decision loop started for {self.runtime.episode_id}")

        while not self.runtime.is_terminal():
            try:
                observation = self.runtime.redis.pop_observation(
                    self.runtime.episode_id
                )

                if not observation:
                    continue

                self.runtime.ingest_observation(
                    beat_id=observation["beat_id"],
                    attempt_id=observation.get("attempt_id"),
                    observation=observation.get("observation"),
                    error=observation.get("error")
                )

            except Exception as e:
                print(f"[runtime] decision loop error: {e}")
                time.sleep(1)

