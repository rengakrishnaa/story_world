"""
Adaptive Retry Policy - Failure-type-specific retry limits.

Uses failure classification to apply per-type max attempts.
"""

from runtime.failure_classifier import classify_failure, FailureClassification

RETRY_LIMITS = {
    "identity": 3,
    "intent": 3,
    "entity": 3,
    "framing": 2,
    "temporal": 2,
    "uncertain": 3,
    "infrastructure": 3,
    "unknown": 2,
}


class RetryDecision:
    def __init__(self, action, reason=None, updated_prompt=None, updated_model=None):
        self.action = action
        self.reason = reason
        self.updated_prompt = updated_prompt
        self.updated_model = updated_model

    def to_payload(self, beat, downgrade=False):
        spec = dict(beat.get("spec", beat))
        spec.setdefault("description", "")
        spec.setdefault("estimated_duration_sec", 5)
        if self.updated_prompt:
            spec["description"] = self.updated_prompt
        if downgrade and self.updated_model:
            spec["force_model"] = self.updated_model
        return {
            "beat_id": beat.get("beat_id", beat.get("id")),
            "execution_spec": spec,
        }


class RetryPolicy:
    def __init__(self, config=None):
        cfg = config or {}
        self.max_attempts = cfg.get("max_attempts", 3)
        self.limits = {**RETRY_LIMITS}
        for k, v in cfg.get("limits", {}).items():
            if k in self.limits:
                self.limits[k] = v

    def _get_limit(self, classification: FailureClassification) -> int:
        return self.limits.get(classification.failure_type, self.max_attempts)

    def decide(self, beat, attempts, observation=None, error=None, observer_verdict=None):
        num_attempts = len(attempts)
        classification = classify_failure(
            observer_verdict=observer_verdict,
            observation=observation,
            error=error,
        )
        limit = self._get_limit(classification)

        if classification.severity == "terminal":
            return RetryDecision(
                "ABORT",
                reason=f"{classification.reason}",
            )

        if observer_verdict in ("impossible", "contradicts", "blocks_intent"):
            return RetryDecision(
                "ABORT",
                reason=f"Observer verdict {observer_verdict}: do not retry",
            )

        if observer_verdict == "uncertain":
            if num_attempts < limit:
                return RetryDecision(
                    "RETRY",
                    reason=f"Observer uncertain; retry {num_attempts + 1}/{limit} ({classification.failure_type})",
                )
            return RetryDecision(
                "ABORT",
                reason=f"Observer uncertain after {limit} attempts",
            )

        if error is not None and num_attempts >= limit:
            return RetryDecision(
                "ABORT",
                reason=f"Max attempts ({limit}) reached: {error}",
            )

        if observation is not None and getattr(observation, "success", False):
            return RetryDecision("ACCEPT")

        if num_attempts < limit:
            return RetryDecision(
                "RETRY",
                reason=f"Retry {num_attempts + 1}/{limit} ({classification.failure_type})",
            )

        return RetryDecision("ABORT", reason=f"Max attempts ({limit}) exceeded")
