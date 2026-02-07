"""
Failure Classifier - Categorizes beat failures for adaptive retry.

Classifies failure mode from observer verdict, observation, and error.
Used by retry policy to apply failure-type-specific retry limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FailureClassification:
    failure_type: str  # identity | intent | entity | framing | temporal | world | uncertain | impossible | infrastructure
    severity: str      # recoverable | marginal | terminal
    suggested_action: str  # retry_prompt | retry_model | augment_observability | abort
    reason: str


# Terminal failures - no retry
TERMINAL_TYPES = {"impossible", "contradicts", "blocks_intent"}

# Recoverable - prompt/model changes may help
RECOVERABLE_TYPES = {"identity", "intent", "entity", "framing"}

# Marginal - retry same spec may help
MARGINAL_TYPES = {"uncertain", "temporal"}


def classify_failure(
    observer_verdict: Optional[str] = None,
    observation: Optional[Any] = None,
    error: Optional[str] = None,
) -> FailureClassification:
    """
    Classify failure for adaptive retry strategy.
    """
    verdict = (observer_verdict or "").lower()

    if verdict in TERMINAL_TYPES:
        return FailureClassification(
            failure_type=verdict,
            severity="terminal",
            suggested_action="abort",
            reason=f"Observer verdict {verdict}: physics/logic violated",
        )

    if verdict == "uncertain":
        obs_type = getattr(observation, "failure_type", None) if observation else None
        if obs_type:
            return FailureClassification(
                failure_type=obs_type,
                severity="marginal",
                suggested_action="retry_prompt" if obs_type in RECOVERABLE_TYPES else "augment_observability",
                reason=f"Uncertain verdict with failure_type={obs_type}",
            )
        return FailureClassification(
            failure_type="uncertain",
            severity="marginal",
            suggested_action="augment_observability",
            reason="Observer uncertain; retry may obtain evidence",
        )

    if observation:
        obs_type = getattr(observation, "failure_type", None)
        if obs_type in RECOVERABLE_TYPES:
            return FailureClassification(
                failure_type=obs_type,
                severity="recoverable",
                suggested_action="retry_prompt",
                reason=f"Recoverable failure: {obs_type}",
            )
        if obs_type == "temporal":
            return FailureClassification(
                failure_type="temporal",
                severity="marginal",
                suggested_action="retry_prompt",
                reason="Temporal discontinuity",
            )
        if obs_type == "world":
            return FailureClassification(
                failure_type="world",
                severity="terminal",
                suggested_action="abort",
                reason="World invariant violated",
            )

    if error:
        err_lower = error.lower()
        if "timeout" in err_lower or "429" in err_lower:
            return FailureClassification(
                failure_type="infrastructure",
                severity="marginal",
                suggested_action="retry_model",
                reason=f"Infrastructure: {error[:100]}",
            )
        if "missing" in err_lower or "video" in err_lower:
            return FailureClassification(
                failure_type="infrastructure",
                severity="marginal",
                suggested_action="augment_observability",
                reason="Missing video or artifact",
            )

    return FailureClassification(
        failure_type="unknown",
        severity="marginal",
        suggested_action="retry_prompt",
        reason=error or "Unknown failure",
    )
