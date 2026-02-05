"""
Tests for observer-as-witness semantics.

Observer infrastructure failure must NOT block closed-form physics.
Observer is witness (reduces confidence), not judge (blocks completion).
"""

import pytest


def test_closed_form_box_stacking_no_observer_halt():
    """Box stacking with full spec: observer unavailable → ACCEPTED, not EPISTEMICALLY_INCOMPLETE."""
    from runtime.epistemic_evaluator import evaluate_epistemic_state
    from models.epistemic import EpistemicState

    intent = """A robotic arm stacks three identical boxes vertically on a flat surface without tipping.
Each box weighs 1.5 kg, measures 30 cm per side, and has a friction coefficient of 0.6.
Gravity is 9.8 m/s². The arm places each box gently with zero angular velocity."""
    
    state, summary = evaluate_epistemic_state(
        evidence_ledger=None,
        intent=intent,
        verdict="uncertain",
        confidence=0.3,
        observer_unavailable=True,
        intent_override={"requires_visual_verification": False},
    )
    assert state == EpistemicState.ACCEPTED, "Closed-form: observer failure must NOT block"
    assert summary.observer_status == "unavailable"
    assert summary.observer_impact == "confidence_only"
    assert "observer_infrastructure_failure" in (summary.confidence_penalty_reason or "")


def test_vehicle_turn_no_observer_halt():
    """Vehicle dynamics: observer unavailable → EPISTEMICALLY_INCOMPLETE (correct)."""
    from runtime.epistemic_evaluator import evaluate_epistemic_state
    from models.epistemic import EpistemicState

    intent = "A vehicle takes a sharp turn at increasing speed."
    
    state, summary = evaluate_epistemic_state(
        evidence_ledger=None,
        intent=intent,
        verdict="uncertain",
        confidence=0.3,
        observer_unavailable=True,
    )
    assert state == EpistemicState.EPISTEMICALLY_INCOMPLETE
    assert summary.observer_impact == "blocking"


def test_intent_classification_override():
    """Intent classification: API override bypasses LLM."""
    from models.intent_classification import requires_visual_verification, get_problem_domain

    intent = "Some arbitrary intent"
    assert requires_visual_verification(intent, override=False) is False
    assert requires_visual_verification(intent, override=True) is True
    assert get_problem_domain(intent, override="statics") == "statics"


def test_intent_classifier_result():
    """Intent classification: full result structure."""
    from agents.intent_classifier import classify_intent, IntentClassificationResult

    result = classify_intent("test intent", override_requires_visual=False, use_cache=False)
    assert isinstance(result, IntentClassificationResult)
    assert result.requires_visual_verification is False
    assert result.source == "override"


def test_llm_failure_fallback_closed_form():
    """
    CRITICAL: When LLM fails (429, etc.), fallback must use rule-based detection.
    Closed-form stacking + LLM unavailable + observer unavailable → COMPLETED.
    """
    from unittest.mock import patch
    from agents.intent_classifier import classify_intent, _fallback_classify

    intent = """A robotic arm stacks three identical boxes vertically on a flat surface without tipping.
Each box weighs 1.5 kg, measures 30 cm per side, and has a friction coefficient of 0.6.
Gravity is 9.8 m/s². The arm places each box gently with zero angular velocity."""

    result = _fallback_classify(intent)
    assert result.requires_visual_verification is False, (
        "Rule-based fallback must NOT default to observer required for closed-form"
    )
    assert result.source == "fallback"
    assert "solver sufficient" in result.reasoning.lower()

    with patch("agents.intent_classifier._call_llm_classify", return_value=None):
        result2 = classify_intent(intent, use_cache=False)
        assert result2.requires_visual_verification is False
        assert result2.source == "fallback"


def test_llm_failure_fallback_vehicle_still_blocking():
    """Vehicle dynamics: even with LLM failure, rule-based correctly returns blocking."""
    from agents.intent_classifier import _fallback_classify

    intent = "A vehicle takes a sharp turn at increasing speed."
    result = _fallback_classify(intent)
    assert result.requires_visual_verification is True
    assert result.source == "fallback"


def test_closed_form_no_override_llm_fails_completes():
    """
    Sanity test: closed-form stacking, LLM unavailable, observer unavailable.
    Expected: COMPLETED with confidence < 1.0 (solver-only success).
    """
    from unittest.mock import patch
    from runtime.epistemic_evaluator import evaluate_epistemic_state
    from models.epistemic import EpistemicState

    intent = """Stack three boxes. Each weighs 1.5 kg, 30 cm per side, friction 0.6. Gravity 9.8 m/s²."""

    with patch("agents.intent_classifier._call_llm_classify", return_value=None):
        state, summary = evaluate_epistemic_state(
            evidence_ledger=None,
            intent=intent,
            verdict="uncertain",
            confidence=0.3,
            observer_unavailable=True,
            intent_override=None,
        )
    assert state == EpistemicState.ACCEPTED, (
        "Closed-form + LLM fail + observer fail → must COMPLETE (solver sufficient), not block"
    )
    assert summary.observer_impact == "confidence_only"
