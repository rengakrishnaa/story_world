"""
NLP Impossibility Gate (pre_simulation_veto)

This module is NOT a physics engine. It performs:
- Textual impossibility detection
- Explicit natural-language contradiction checks
- Hard-coded universal priors (conservation laws, causality)

CANONICAL RULE: This veto MAY NOT override observer verdicts and MAY NOT
finalize outcomes. It is a guardrail, not a judge.

Authority hierarchy:
  physics_veto.py  → NLP sanity filter (cheap pre-GPU gate)
  Planner         → Hypothesis generator
  Video           → Computation medium
  Observer        → Ground truth authority
  WorldStateGraph → Memory of truth

Usage: Only before GPU dispatch, to save compute. Block only universally
impossible cases. Never declare GOAL_ACHIEVED or final failure alone.
"""

import re
from enum import Enum
from typing import List, Tuple


class VetoLevel(str, Enum):
    """Severity of detected impossibility."""
    HARD = "hard"   # Universally impossible; block immediately
    SOFT = "soft"   # Observer should verify; do NOT block
    NONE = "none"


def evaluate_physics_veto(intent: str) -> Tuple[bool, List[str], str]:
    """
    NLP impossibility gate. Evaluates intent text for explicit contradictions.

    CRITICAL RULE: May only return should_block=True (GOAL_IMPOSSIBLE) when the goal
    is SELF-CONTRADICTORY IN LANGUAGE (e.g. "stone floats unsupported"), NOT when
    physically implausible. Physically implausible goals must reach the observer.
    This protects the "video as computation" principle.

    Returns (should_block, constraints, reason).
    should_block=True ONLY when HARD veto triggers (logical contradictions).
    SOFT veto adds constraints but does NOT block — observer must decide.

    This function NEVER sees video, motion, causality, or state transitions.
    """
    text = (intent or "").lower()
    hard_constraints: List[str] = []
    soft_constraints: List[str] = []
    reasons: List[str] = []

    # -------------------------------------------------------------------------
    # HARD: Universally impossible (conservation laws, causality)
    # -------------------------------------------------------------------------

    # Unsupported mass in mid-air — explicit textual contradiction
    if re.search(r"\bfloat|floating|suspended|levitate|hover\b", text) and re.search(
        r"\bno support|without support|unsupported|mid[- ]?air|in mid air\b", text
    ):
        hard_constraints.extend(["gravity_violation", "unsupported_mass"])
        reasons.append("Unsupported mass violates gravity")

    # Energy conservation: acceleration without force
    if re.search(r"\baccelerat|0 to 100|0-100|0–100\b", text) and re.search(
        r"\bwithout (fuel|force|energy|engine)|no (fuel|force|energy|engine)\b", text
    ):
        hard_constraints.append("energy_conservation")
        reasons.append("Acceleration without force violates energy conservation")

    # Teleportation / instant relocation
    if re.search(r"\bteleport|instantaneously|instantly\b", text) and re.search(
        r"\bmove|relocate|appear|disappear\b", text
    ):
        hard_constraints.append("causality_violation")
        reasons.append("Instant relocation violates causality")

    # Time reversal / reverse entropy
    if re.search(r"\btime\b", text) and re.search(r"\breverse|rewind\b", text):
        hard_constraints.append("time_reversal")
        reasons.append("Time reversal violates physical causality")

    # Perpetual motion / infinite energy
    if re.search(r"\bperpetual\b", text) or re.search(r"\binfinite energy\b", text):
        hard_constraints.append("energy_conservation")
        reasons.append("Perpetual motion violates energy conservation")

    # Instant stop from high speed without force
    if re.search(r"\b(stops?\s+instantly|instantly\s+stops?|instant\s+stop)\b", text) and re.search(
        r"\bwithout (force|brakes|friction)\b", text
    ):
        hard_constraints.append("inertia_violation")
        reasons.append("Instant stop without force violates inertia")

    # Lift without force/support — explicit contradiction
    if re.search(r"\blift|raise\b", text) and re.search(
        r"\bwithout (force|support|energy)\b", text
    ):
        hard_constraints.append("force_requirement")
        reasons.append("Lifting without force violates mechanics")

    # -------------------------------------------------------------------------
    # SOFT: Observer should discover — do NOT block
    # -------------------------------------------------------------------------

    # Human jump distance — probabilistic, context-dependent
    if re.search(r"\bhuman\b", text) and re.search(r"\bjump\b", text):
        m = re.search(r"(\d+(\.\d+)?)\s*(m|meter|meters)\b", text)
        if m:
            dist = float(m.group(1))
            if dist >= 4.0:
                soft_constraints.append("human_jump_limit")
                reasons.append(f"Human jump {dist}m exceeds typical bound (observer should verify)")

    # Balance on narrow ledge in wind
    if re.search(r"\bledge\b", text) and re.search(r"\bwind\b", text):
        m = re.search(r"(\d+(\.\d+)?)\s*(cm|centimeter|centimeters)\b", text)
        if m:
            width_cm = float(m.group(1))
            if width_cm <= 15:
                soft_constraints.append("balance_stability")
                reasons.append(f"Balance on {width_cm}cm ledge in wind (observer should verify)")

    # Thin glass / structural — OBSERVER should discover
    if re.search(r"\bthin\s+glass|glass\s+floor|thin glass\b", text) and re.search(
        r"\bsupported\s+only\s+at\s+the\s+edges|edges?\s+only|only\s+at\s+edges\b", text
    ):
        soft_constraints.extend(["structural_integrity", "load_distribution"])
        reasons.append("Thin glass supported only at edges (observer should verify)")

    if re.search(r"\bappears\s+solid\s+but\s+actually\s+thin\b", text):
        soft_constraints.append("structural_integrity")
        reasons.append("Floor lacks load-bearing support (observer should verify)")

    # Removing structural support / vase on table — observer should verify
    if re.search(r"\btable\b", text) and re.search(r"\bleg\b", text) and re.search(
        r"\bremove|removed|without\b", text
    ):
        soft_constraints.append("load_bearing_violation")
        reasons.append("Removing structural support (observer should verify)")
    if re.search(r"\bvase\b", text) and re.search(r"\btable\b", text) and re.search(r"\bleg\b", text):
        if "load_bearing_violation" not in soft_constraints:
            soft_constraints.append("load_bearing_violation")
            reasons.append("Support removal with vase present (observer should verify)")

    # Survive large fall without equipment — probabilistic
    if re.search(r"\bfall\b", text) and re.search(r"\bwithout (equipment|parachute|gear)\b", text):
        m = re.search(r"(\d+(\.\d+)?)\s*(m|meter|meters)\b", text)
        if m:
            dist = float(m.group(1))
            if dist >= 10:
                soft_constraints.append("human_survivability_limit")
                reasons.append(f"Surviving {dist}m fall (observer should verify)")

    # Drone heavy package + small battery — world-model hint
    if re.search(r"\bdrone\b", text) and re.search(r"\bheavy\s+package|heavy\s+load\b", text):
        if re.search(r"\b\d+\s+minutes?\b", text) and re.search(r"\bsmall\s+battery|limited\s+battery\b", text):
            soft_constraints.extend(["energy_capacity", "thrust_vs_mass"])
            reasons.append("Heavy payload + duration vs battery (observer should verify)")

    # Overloaded shelf — observer should detect
    if re.search(r"\bheavy\s+object|overloaded\s+shelf|shelf.*overload\b", text):
        if re.search(r"\bdoes\s+not\s+break\s+immediately|eventually\s+collapse\b", text):
            soft_constraints.extend(["material_fatigue", "load_over_time"])
            reasons.append("Overloaded shelf (observer should verify)")

    # -------------------------------------------------------------------------
    # Result: block ONLY on HARD veto
    # -------------------------------------------------------------------------
    all_constraints = list(dict.fromkeys(hard_constraints + soft_constraints))
    reason_str = "; ".join(reasons) if reasons else ""

    if hard_constraints:
        return True, all_constraints, reason_str
    return False, all_constraints, reason_str
