"""
Explainability of Verdicts

Builds human-readable evidence chain for observer verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class VerdictExplanation:
    """Human-readable explanation of why a verdict was issued."""
    verdict: str
    confidence: float
    causal_chain: List[str] = field(default_factory=list)
    evidence_cited: List[str] = field(default_factory=list)
    constraints_inferred: List[str] = field(default_factory=list)
    summary: str = ""


def build_verdict_explanation(
    verdict: str,
    confidence: float,
    observation: Any = None,
) -> VerdictExplanation:
    """
    Build explainability payload from observation.
    """
    chain: List[str] = []
    evidence: List[str] = []
    constraints: List[str] = []
    summary = ""

    if observation:
        v = (getattr(observation, "verdict", None) or verdict or "unknown").lower()
        c = getattr(observation, "confidence", confidence)
        causal = getattr(observation, "causal_explanation", None)
        if causal:
            chain.append(causal)
        inf = getattr(observation, "constraints_inferred", None) or []
        constraints = list(inf) if isinstance(inf, (list, tuple)) else []
        pv = getattr(observation, "physics_violation", None)
        sc = getattr(observation, "state_contradiction", None)
        ir = getattr(observation, "impossible_reason", None)
        if pv:
            chain.append(f"Physics violation: {pv}")
        if sc:
            chain.append(f"State contradiction: {sc}")
        if ir:
            chain.append(ir)
        if hasattr(observation, "evidence_ledger") and observation.evidence_ledger:
            try:
                avail = getattr(observation.evidence_ledger, "available", []) or []
                for e in avail[:5]:
                    n = getattr(e, "name", str(e))
                    evidence.append(n)
            except Exception:
                pass
    else:
        v = (verdict or "unknown").lower()
        c = confidence

    # Summary
    if v == "impossible":
        summary = "Observer concluded this action is physically impossible. " + (
            chain[0] if chain else "No causal detail available."
        )
    elif v == "contradicts":
        summary = "Observer detected a contradiction with prior state. " + (
            chain[0] if chain else ""
        )
    elif v == "uncertain":
        summary = "Observer could not determine outcome; evidence insufficient or conflicting."
    elif v in ("valid", "degraded"):
        summary = "Observer accepted the state transition."
    else:
        summary = f"Verdict: {v} (confidence {c:.2f})"

    return VerdictExplanation(
        verdict=v or verdict,
        confidence=c if observation else confidence,
        causal_chain=chain,
        evidence_cited=evidence,
        constraints_inferred=constraints,
        summary=summary.strip(),
    )
