"""
Epistemic Architecture - Evidence Ledger & Constraint Evaluator

Layer 1: Evidence Ledger (Truth Source)
Layer 2: Constraint Evaluator (Gatekeeper)
Layer 3: World State Transition Engine (uses Layer 1 & 2)

CORE PRINCIPLE: No state transition without required evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime


class EvidenceSource(str, Enum):
    """Source of evidence."""
    USER = "user"  # From user input/goal
    OBSERVER = "observer"  # From video observation
    RENDERER = "renderer"  # From renderer metadata
    INFERRED = "inferred"  # Computed from other evidence (must declare uncertainty)
    UNKNOWN = "unknown"


class EvidenceResolution(str, Enum):
    """Resolution/quality of evidence."""
    COARSE = "coarse"  # Approximate, low precision
    FINE = "fine"  # High precision, quantitative
    UNKNOWN = "unknown"  # Resolution not determined


class ConfidenceLevel(str, Enum):
    """Bounded confidence levels."""
    BOUNDED_LOW = "bounded_low"  # 0.0-0.33
    BOUNDED_MEDIUM = "bounded_medium"  # 0.34-0.66
    BOUNDED_HIGH = "bounded_high"  # 0.67-1.0


def compute_confidence_level(confidence: float) -> ConfidenceLevel:
    """Convert numeric confidence to bounded level."""
    c = max(0.0, min(1.0, float(confidence)))
    if c <= 0.33:
        return ConfidenceLevel.BOUNDED_LOW
    elif c <= 0.66:
        return ConfidenceLevel.BOUNDED_MEDIUM
    else:
        return ConfidenceLevel.BOUNDED_HIGH


def calibrated_confidence(
    base: float,
    evidence_count: int = 0,
    missing_count: int = 0,
    resolution_penalty: float = 0.0,
) -> float:
    """
    Calibrate confidence with evidence quality and completeness.
    Returns 0.0-1.0.
    """
    c = max(0.0, min(1.0, float(base)))
    if evidence_count > 0 and missing_count > 0:
        coverage = evidence_count / (evidence_count + missing_count)
        c = c * (0.7 + 0.3 * coverage)
    if resolution_penalty > 0:
        c = c * (1.0 - min(0.3, resolution_penalty))
    return max(0.0, min(1.0, c))


@dataclass
class Evidence:
    """
    A structured, explicit signal required to justify a claim or transition.
    
    Evidence must be:
    - Named
    - Typed
    - Sourced
    - Bounded (confidence / resolution)
    """
    name: str
    source: EvidenceSource
    resolution: EvidenceResolution
    confidence: float  # 0.0-1.0
    
    # Value (if available)
    value: Optional[Any] = None
    value_type: Optional[str] = None  # e.g., "float", "vector3", "bool"
    units: Optional[str] = None  # e.g., "m/s", "rad/s", "N"
    
    # Metadata
    frame_range: Optional[tuple] = None  # (start_frame, end_frame) if video-derived
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # If inferred, declare uncertainty
    inferred_from: List[str] = field(default_factory=list)  # Names of evidence this was inferred from
    inference_uncertainty: Optional[str] = None  # Explanation of uncertainty in inference
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.value,
            "resolution": self.resolution.value,
            "confidence": self.confidence,
            "value": self.value,
            "value_type": self.value_type,
            "units": self.units,
            "frame_range": self.frame_range,
            "timestamp": self.timestamp.isoformat(),
            "inferred_from": self.inferred_from,
            "inference_uncertainty": self.inference_uncertainty,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(
            name=data["name"],
            source=EvidenceSource(data["source"]),
            resolution=EvidenceResolution(data["resolution"]),
            confidence=data["confidence"],
            value=data.get("value"),
            value_type=data.get("value_type"),
            units=data.get("units"),
            frame_range=tuple(data["frame_range"]) if data.get("frame_range") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.utcnow(),
            inferred_from=data.get("inferred_from", []),
            inference_uncertainty=data.get("inference_uncertainty"),
        )


@dataclass
class EvidenceLedger:
    """
    Layer 1: Evidence Ledger (Truth Source)
    
    Immutable ledger of available and missing evidence for a beat.
    """
    beat_id: str
    available: List[Evidence] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)  # Names of required but missing evidence
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence to the ledger."""
        # Remove from missing if it was there
        if evidence.name in self.missing:
            self.missing.remove(evidence.name)
        # Add to available (replace if exists)
        self.available = [e for e in self.available if e.name != evidence.name]
        self.available.append(evidence)
    
    def mark_missing(self, evidence_name: str):
        """Mark evidence as required but missing."""
        if evidence_name not in self.missing:
            self.missing.append(evidence_name)
    
    def has_evidence(self, name: str) -> bool:
        """Check if evidence exists."""
        return any(e.name == name for e in self.available)
    
    def get_evidence(self, name: str) -> Optional[Evidence]:
        """Get evidence by name."""
        for e in self.available:
            if e.name == name:
                return e
        return None
    
    def get_missing(self) -> List[str]:
        """Get list of missing evidence names."""
        return self.missing.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "available": [e.to_dict() for e in self.available],
            "missing": self.missing,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceLedger":
        return cls(
            beat_id=data["beat_id"],
            available=[Evidence.from_dict(e) for e in data.get("available", [])],
            missing=data.get("missing", []),
        )


@dataclass
class Constraint:
    """
    A rule that cannot be evaluated unless specific evidence exists.
    
    A constraint is not an error. It is an epistemic boundary.
    """
    name: str
    requires: List[str]  # Names of required evidence
    description: str
    
    def can_evaluate(self, ledger: EvidenceLedger) -> bool:
        """Check if all required evidence is available."""
        return all(ledger.has_evidence(name) for name in self.requires)
    
    def get_missing_evidence(self, ledger: EvidenceLedger) -> List[str]:
        """Get list of missing evidence needed to evaluate this constraint."""
        return [name for name in self.requires if not ledger.has_evidence(name)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "requires": self.requires,
            "description": self.description,
        }


class EpistemicState(str, Enum):
    """
    MANDATORY epistemic outcomes.
    
    No silent fallback. No probabilistic guessing.
    """
    ACCEPTED = "ACCEPTED"  # Evidence sufficient, constraints satisfied
    REJECTED = "REJECTED"  # Evidence sufficient, constraints violated
    EPISTEMICALLY_INCOMPLETE = "EPISTEMICALLY_INCOMPLETE"  # Evidence insufficient
    UNCERTAIN_TERMINATION = "UNCERTAIN_TERMINATION"  # Progression blocked by unresolved uncertainty


@dataclass
class ConstraintEvaluator:
    """
    Layer 2: Constraint Evaluator (Gatekeeper)
    
    Evaluates constraints only if all required evidence exists.
    """
    constraints: List[Constraint] = field(default_factory=list)
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to evaluate."""
        self.constraints.append(constraint)
    
    def evaluate(self, ledger: EvidenceLedger) -> Dict[str, Any]:
        """
        Evaluate all constraints that can be evaluated.
        
        Returns:
            {
                "evaluable": List[str],  # Constraint names that can be evaluated
                "blocked": List[Dict],   # Constraints blocked by missing evidence
                "violations": List[str], # Constraint names that were violated (if evaluable)
            }
        """
        evaluable = []
        blocked = []
        
        for constraint in self.constraints:
            if constraint.can_evaluate(ledger):
                evaluable.append(constraint.name)
            else:
                missing = constraint.get_missing_evidence(ledger)
                blocked.append({
                    "constraint": constraint.name,
                    "missing_evidence": missing,
                    "description": constraint.description,
                })
        
        return {
            "evaluable": evaluable,
            "blocked": blocked,
            "violations": [],  # Will be populated by actual constraint evaluation logic
        }
    
    def can_proceed(self, ledger: EvidenceLedger) -> tuple[bool, List[str]]:
        """
        Check if state transition can proceed.
        
        Returns:
            (can_proceed: bool, missing_evidence: List[str])
        """
        result = self.evaluate(ledger)
        
        # Cannot proceed if any required constraint is blocked
        if result["blocked"]:
            missing = set()
            for block in result["blocked"]:
                missing.update(block["missing_evidence"])
            return False, list(missing)
        
        # Can proceed if all constraints are evaluable (violations checked separately)
        return True, []


@dataclass
class EpistemicSummary:
    """
    Required output format for every episode.
    """
    final_state: EpistemicState
    confidence: ConfidenceLevel
    constraints_triggered: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    justification: List[str] = field(default_factory=list)
    # Observer as witness: infrastructure failure ≠ epistemic block for closed-form intents
    observer_status: Optional[str] = None  # "success" | "unavailable" | "failed"
    observer_impact: Optional[str] = None  # "blocking" | "confidence_only"
    confidence_penalty_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "final_state": self.final_state.value,
            "confidence": self.confidence.value,
            "constraints_triggered": self.constraints_triggered,
            "missing_evidence": self.missing_evidence,
            "justification": self.justification,
        }
        if self.observer_status is not None:
            d["observer_status"] = self.observer_status
        if self.observer_impact is not None:
            d["observer_impact"] = self.observer_impact
        if self.confidence_penalty_reason is not None:
            d["confidence_penalty_reason"] = self.confidence_penalty_reason
        return d
    
