class RetryDecision:
    def __init__(self, action, reason=None, updated_prompt=None, updated_model=None):
        self.action = action
        self.reason = reason
        self.updated_prompt = updated_prompt
        self.updated_model = updated_model

    def to_payload(self, beat, downgrade=False):
        # 1️⃣ Always construct spec safely
        spec = dict(beat.get("spec", beat))

        # 2️⃣ Ensure required fields exist
        spec.setdefault("description", "")
        spec.setdefault("estimated_duration_sec", 5)

        # 3️⃣ Apply retry modifications
        if self.updated_prompt:
            spec["description"] = self.updated_prompt

        if downgrade and self.updated_model:
            spec["force_model"] = self.updated_model

        # 4️⃣ Always emit a valid payload
        return {
            "beat_id": beat.get("beat_id", beat.get("id")),
            "execution_spec": spec,
        }


class RetryPolicy:
    def __init__(self, config=None):
        self.max_attempts = (config or {}).get("max_attempts", 2)

    def decide(self, beat, attempts, observation=None, error=None):
        num_attempts = len(attempts)

        # 🔴 If error keeps happening, ABORT after max_attempts
        if error is not None and num_attempts >= self.max_attempts:
            return RetryDecision(
                "ABORT",
                reason=f"Max attempts reached with error: {error}",
            )

        # If success observation exists, ACCEPT
        if observation is not None and getattr(observation, "success", False):
            return RetryDecision("ACCEPT")

        # Retry if attempts remain
        if num_attempts < self.max_attempts:
            return RetryDecision("RETRY", reason="Retry allowed")

        # Final safety abort
        return RetryDecision("ABORT", reason="Max attempts exceeded")
