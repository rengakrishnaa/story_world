class CostPolicy:
    def __init__(self, config=None):
        self.max_cost = (config or {}).get("max_cost", float("inf"))

    def allow(self, attempt):
        return True  # default-allow for now

    def abort(self, reason):
        return {
            "action": "ABORT",
            "reason": reason,
        }
