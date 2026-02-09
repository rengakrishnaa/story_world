"""
Epistemic Evaluator - Layer 2 & 3 Integration

Checks evidence availability before allowing state transitions.

INVARIANT: Observer is a witness, not a judge.
- Closed-form physics (mass, geometry, friction, gravity specified): observer optional
- Perceptual dynamics (vehicle turn, speed, etc.): observer required for epistemic validity
- Observer infrastructure failure must NOT block closed-form problems.
"""

from typing import Tuple, Optional
from models.epistemic import (
    EvidenceLedger,
    ConstraintEvaluator,
    EpistemicState,
    EpistemicSummary,
    ConfidenceLevel,
    compute_confidence_level,
)
from models.physics_constraints import get_constraints_for_intent
from models.intent_classification import requires_visual_verification, get_observer_impact


def evaluate_epistemic_state(
    evidence_ledger: Optional[EvidenceLedger],
    intent: str,
    verdict: str,
    confidence: float,
    *,
    observer_unavailable: bool = False,
    intent_override: Optional[dict] = None,
    risk_profile: Optional[str] = None,
) -> Tuple[EpistemicState, Optional[EpistemicSummary]]:
    """
    Evaluate epistemic state based on evidence availability and constraints.
    
    observer_unavailable: True when observer failed (429, timeout, mock fallback).
        For closed-form intents, this does NOT block - solver-only success is valid.
    intent_override: Optional API override {"requires_visual_verification": bool, "problem_domain": str}
    
    Returns:
        (epistemic_state, epistemic_summary)
    """
    over = intent_override or {}
    needs_visual = requires_visual_verification(intent, override=over.get("requires_visual_verification"))
    impact = get_observer_impact(intent, override_requires_visual=over.get("requires_visual_verification"))
    
    if not evidence_ledger:
        if not needs_visual:
            summary = EpistemicSummary(
                final_state=EpistemicState.ACCEPTED,
                confidence=compute_confidence_level(0.75),
                missing_evidence=[],
                justification=["Solver-sufficient: intent fully specified; observer unavailable for verification"],
                observer_status="unavailable",
                observer_impact="confidence_only",
                confidence_penalty_reason="observer_infrastructure_failure",
            )
            return EpistemicState.ACCEPTED, summary
        summary = EpistemicSummary(
            final_state=EpistemicState.EPISTEMICALLY_INCOMPLETE,
            confidence=compute_confidence_level(confidence),
            missing_evidence=[],
            justification=["No evidence ledger available from observer; intent requires visual verification"],
            observer_status="unavailable" if observer_unavailable else "failed",
            observer_impact="blocking",
        )
        return EpistemicState.EPISTEMICALLY_INCOMPLETE, summary
    
    constraints = get_constraints_for_intent(
        intent,
        override_problem_domain=over.get("problem_domain"),
        risk_profile=risk_profile,
    )
    evaluator = ConstraintEvaluator(constraints=constraints)
    
    can_proceed, missing_evidence = evaluator.can_proceed(evidence_ledger)
    
    if not can_proceed:
        if not needs_visual and impact != "blocking":
            summary = EpistemicSummary(
                final_state=EpistemicState.ACCEPTED,
                confidence=compute_confidence_level(0.75),
                missing_evidence=missing_evidence,
                constraints_triggered=["insufficient_observational_evidence"],
                justification=[
                    f"Closed-form intent: observer could not extract evidence ({', '.join(missing_evidence)}); solver-sufficient"
                ],
                observer_status="insufficient_observational_evidence",
                observer_impact="confidence_only",
                confidence_penalty_reason="observer_could_not_extract_evidence",
            )
            return EpistemicState.ACCEPTED, summary
        summary = EpistemicSummary(
            final_state=EpistemicState.EPISTEMICALLY_INCOMPLETE,
            confidence=compute_confidence_level(confidence),
            missing_evidence=missing_evidence,
            constraints_triggered=["insufficient_physical_evidence"],
            justification=[
                f"Cannot evaluate constraints without required evidence: {', '.join(missing_evidence)}"
            ],
        )
        return EpistemicState.EPISTEMICALLY_INCOMPLETE, summary
    
    if verdict in ("impossible", "contradicts", "blocks"):
        summary = EpistemicSummary(
            final_state=EpistemicState.REJECTED,
            confidence=compute_confidence_level(confidence),
            constraints_triggered=[verdict],
            justification=[f"Observer verdict: {verdict}"],
        )
        return EpistemicState.REJECTED, summary
    
    if verdict == "uncertain" and missing_evidence:
        summary = EpistemicSummary(
            final_state=EpistemicState.EPISTEMICALLY_INCOMPLETE,
            confidence=compute_confidence_level(confidence),
            missing_evidence=missing_evidence,
            constraints_triggered=["insufficient_physical_evidence"],
            justification=[
                f"Uncertain verdict with missing evidence: {', '.join(missing_evidence)}"
            ],
        )
        return EpistemicState.EPISTEMICALLY_INCOMPLETE, summary
    
    if verdict == "uncertain":
        summary = EpistemicSummary(
            final_state=EpistemicState.UNCERTAIN_TERMINATION,
            confidence=compute_confidence_level(confidence),
            justification=["Observer verdict uncertain despite available evidence"],
        )
        return EpistemicState.UNCERTAIN_TERMINATION, summary
    
    # Valid or degraded with all evidence - ACCEPTED
    summary = EpistemicSummary(
        final_state=EpistemicState.ACCEPTED,
        confidence=compute_confidence_level(confidence),
        justification=["All required evidence available, constraints satisfied"],
    )
    return EpistemicState.ACCEPTED, summary
