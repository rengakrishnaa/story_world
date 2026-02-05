"""
Intent Classification - Production Facade

Delegates to agents.intent_classifier (LLM-based, no keyword tricks).
Exposes stable API for epistemic evaluator and physics constraints.
"""

from typing import Optional

# Re-export for backward compatibility
from agents.intent_classifier import (
    IntentClassificationResult,
    classify_intent,
    PROBLEM_DOMAIN_VEHICLE,
    PROBLEM_DOMAIN_STATICS,
    PROBLEM_DOMAIN_STRUCTURAL,
    PROBLEM_DOMAIN_FLUID,
    PROBLEM_DOMAIN_GENERIC,
)

__all__ = [
    "IntentClassificationResult",
    "classify_intent",
    "requires_visual_verification",
    "get_observer_impact",
    "get_classification",
    "get_problem_domain",
    "PROBLEM_DOMAIN_VEHICLE",
    "PROBLEM_DOMAIN_STATICS",
    "PROBLEM_DOMAIN_STRUCTURAL",
    "PROBLEM_DOMAIN_FLUID",
    "PROBLEM_DOMAIN_GENERIC",
]


def requires_visual_verification(
    intent: str,
    *,
    override: Optional[bool] = None,
) -> bool:
    """
    True if the intent requires perceptual evidence from observer.
    False if it's a closed-form physics problem (solver-sufficient).

    Production: LLM-based classification. No keyword tricks.
    """
    result = classify_intent(intent, override_requires_visual=override)
    return result.requires_visual_verification


def get_observer_impact(
    intent: str,
    *,
    override_requires_visual: Optional[bool] = None,
) -> str:
    """
    "blocking" = observer required for epistemic validity
    "confidence_only" = observer failure only reduces confidence
    """
    result = classify_intent(intent, override_requires_visual=override_requires_visual)
    return result.observer_impact()


def get_classification(
    intent: str,
    *,
    override_requires_visual: Optional[bool] = None,
    override_problem_domain: Optional[str] = None,
    use_cache: bool = True,
) -> IntentClassificationResult:
    """
    Full classification result for physics constraints and audit.
    """
    return classify_intent(
        intent,
        override_requires_visual=override_requires_visual,
        override_problem_domain=override_problem_domain,
        use_cache=use_cache,
    )


def get_problem_domain(intent: str, *, override: Optional[str] = None) -> str:
    """
    Problem domain for constraint selection.
    vehicle_dynamics | statics | structural | fluid | generic
    """
    result = classify_intent(intent, override_problem_domain=override)
    return result.problem_domain


def get_intent_override_from_policies(policies: Optional[dict]) -> Optional[dict]:
    """
    Extract intent override from episode policies (API override).
    Returns dict with requires_visual_verification and/or problem_domain or None.
    """
    if not policies:
        return None
    override = policies.get("intent_override")
    if not override or not isinstance(override, dict):
        return None
    return override
