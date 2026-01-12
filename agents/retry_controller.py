from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RetryDecision:
    action: str                 # retry_prompt | retry_model | abort
    reason: str
    updated_prompt: Optional[str] = None
    updated_model: Optional[str] = None
    max_retries: int = 1


class RetryController:
    """
    Stateless decision engine.
    Reads BeatObservation → decides retry strategy.
    """

    def decide(self, beat: Dict, observation) -> RetryDecision:
        failure = observation.failure_type

        # ----------------------------
        # IDENTITY FAILURE
        # ----------------------------
        if failure == "identity":
            # Pick the weakest matching character
            character = min(
                observation.entity_presence,
                key=lambda k: observation.entity_presence[k]
            )

            return RetryDecision(
                action="retry_prompt",
                reason=f"Identity mismatch for {character}",
                updated_prompt=self._strengthen_identity_prompt(
                    beat, character
                ),
                max_retries=2
            )

        # ----------------------------
        # INTENT FAILURE
        # ----------------------------
        if failure == "intent":
            return RetryDecision(
                action="retry_prompt",
                reason="Intent not satisfied",
                updated_prompt=self._rewrite_intent_prompt(beat),
                max_retries=2
            )

        # ----------------------------
        # ENTITY PRESENCE FAILURE
        # ----------------------------
        if failure == "entity":
            return RetryDecision(
                action="retry_prompt",
                reason="Required entity missing",
                updated_prompt=self._inject_entities(beat),
                max_retries=2
            )

        # ----------------------------
        # FRAMING FAILURE
        # ----------------------------
        if failure == "framing":
            return RetryDecision(
                action="retry_prompt",
                reason="Bad framing / composition",
                updated_prompt=self._fix_framing(beat),
                max_retries=1
            )

        # ----------------------------
        # CONTINUITY FAILURE
        # ----------------------------
        if failure == "continuity":
            return RetryDecision(
                action="abort",
                reason="Continuity violation – unsafe to retry"
            )

        # ----------------------------
        # DEFAULT
        # ----------------------------
        return RetryDecision(
            action="abort",
            reason="Unknown failure"
        )

    # --------------------------------------------------
    # Prompt mutation helpers
    # --------------------------------------------------

    def _rewrite_intent_prompt(self, beat: Dict) -> str:
        return (
            f"Clearly depict the following action without ambiguity: "
            f"{beat['description']}. "
            f"Ensure the main action is visually dominant."
        )

    def _inject_entities(self, beat: Dict) -> str:
        entities = ", ".join(beat.get("characters", []))
        return (
            f"{beat['description']}. "
            f"The following characters MUST be visible: {entities}. "
            f"Do not omit or obscure them."
        )

    def _fix_framing(self, beat: Dict) -> str:
        return (
            f"{beat['description']}. "
            f"Use cinematic framing, rule-of-thirds, subject centered, "
            f"clear foreground/background separation."
        )

    def _strengthen_identity_prompt(self, beat: Dict, character: str) -> str:
        return (
            f"{beat['description']}. "
            f"The character {character} must closely resemble their reference appearance. "
            f"Maintain consistent facial features, outfit, hair, and proportions. "
            f"Do not alter identity."
        )
