class QualityDecision:
    def __init__(self, action):
        self.action = action


class QualityPolicy:
    def __init__(self, config=None):
        self.min_confidence = (config or {}).get("min_confidence", 0.0)

    def accept(self, observation):
        if observation is None:
            return False
        return observation.get("confidence", 0) >= self.min_confidence

    def accept_decision(self):
        return QualityDecision("ACCEPT")
