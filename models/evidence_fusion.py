"""
Multi-Evidence Fusion

Combines video observer evidence with optional numeric simulation evidence
to produce a unified confidence and evidence ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from models.epistemic import (
    Evidence,
    EvidenceLedger,
    EvidenceSource,
    EvidenceResolution,
    calibrated_confidence,
)
from models.observation import ObservationResult


@dataclass
class NumericSimEvidence:
    """
    Optional evidence from numeric simulation (placeholder for future physics sim).
    """
    name: str
    value: Any
    units: Optional[str] = None
    confidence: float = 0.9
    frame_range: Optional[tuple] = None


def fuse_evidence(
    beat_id: str,
    video_observation: Optional[ObservationResult] = None,
    numeric_evidence: Optional[List[NumericSimEvidence]] = None,
) -> EvidenceLedger:
    """
    Fuse video observer + optional numeric sim into a single EvidenceLedger.

    Video evidence: verdict, constraints_inferred, confidence, quality.
    Numeric evidence: quantitative signals (velocity, position, etc.).
    """
    ledger = EvidenceLedger(beat_id=beat_id)

    # Add video observer evidence
    if video_observation:
        _add_video_evidence(ledger, video_observation)

    # Add numeric sim evidence (future: from physics engine)
    if numeric_evidence:
        for sim in numeric_evidence:
            ledger.add_evidence(Evidence(
                name=sim.name,
                source=EvidenceSource.INFERRED,  # Numeric sim is computed
                resolution=EvidenceResolution.FINE if sim.units else EvidenceResolution.COARSE,
                confidence=sim.confidence,
                value=sim.value,
                value_type=type(sim.value).__name__,
                units=sim.units,
                frame_range=sim.frame_range,
            ))

    return ledger


def _add_video_evidence(ledger: EvidenceLedger, obs: ObservationResult) -> None:
    """Extract Evidence from ObservationResult and add to ledger."""
    # Verdict as primary evidence
    ledger.add_evidence(Evidence(
        name="observer_verdict",
        source=EvidenceSource.OBSERVER,
        resolution=EvidenceResolution.COARSE,
        confidence=obs.confidence,
        value=obs.verdict,
        value_type="str",
    ))
    # Constraints inferred
    for i, c in enumerate(obs.constraints_inferred or []):
        ledger.add_evidence(Evidence(
            name=f"constraint_inferred_{i}",
            source=EvidenceSource.OBSERVER,
            resolution=EvidenceResolution.COARSE,
            confidence=obs.confidence,
            value=c,
            value_type="str",
        ))
    # Quality as evidence
    if obs.quality:
        ledger.add_evidence(Evidence(
            name="video_quality",
            source=EvidenceSource.OBSERVER,
            resolution=EvidenceResolution.COARSE,
            confidence=obs.quality.overall_quality,
            value=obs.quality.overall_quality,
            value_type="float",
        ))


def fused_confidence(
    video_observation: Optional[ObservationResult] = None,
    numeric_evidence: Optional[List[NumericSimEvidence]] = None,
) -> float:
    """
    Compute fused confidence from multiple evidence sources.
    Uses calibrated_confidence with evidence coverage.
    """
    evidence_count = 0
    missing_count = 0
    base = 0.5

    if video_observation:
        evidence_count += 1
        base = max(base, video_observation.confidence)
        if (video_observation.verdict or "").lower() in ("uncertain", "insufficient_evidence"):
            missing_count += 1

    if numeric_evidence:
        evidence_count += len(numeric_evidence)
        avg_sim = sum(s.confidence for s in numeric_evidence) / len(numeric_evidence)
        base = max(base, avg_sim)

    resolution_penalty = 0.0
    if video_observation and video_observation.quality:
        if video_observation.quality.artifacts_detected > 0:
            resolution_penalty += 0.1 * video_observation.quality.artifacts_detected

    return calibrated_confidence(
        base=base,
        evidence_count=evidence_count,
        missing_count=missing_count,
        resolution_penalty=resolution_penalty,
    )
