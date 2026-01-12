class RuntimeRegistry:
    def __init__(self):
        self._runtimes = {}

    def register(self, runtime):
        self._runtimes[runtime.episode_id] = runtime

    def get(self, episode_id):
        return self._runtimes.get(episode_id)
