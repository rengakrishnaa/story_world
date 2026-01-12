class TemporalConsistency:
    def __init__(self):
        self.last_beat = None

    def score(self, beat):
        if not self.last_beat:
            self.last_beat = beat
            return 1.0

        score = 1.0

        # Character continuity
        prev_chars = set(self.last_beat.get("characters", []))
        curr_chars = set(beat.get("characters", []))

        if not curr_chars.intersection(prev_chars):
            score -= 0.3  # hard cut

        # Action continuity
        if self.last_beat.get("action") and beat.get("action"):
            if self.last_beat["action"] != beat["action"]:
                score -= 0.2

        self.last_beat = beat
        return max(score, 0.0)
