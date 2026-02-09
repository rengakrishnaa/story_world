"""
Physics Observability Contract.

If a beat is about dynamics, the render must expose motion.
- Rendering: camera angle, framing, reference frame
- Observer: explicit physics questions (slip, contact, trajectory)
- On insufficient_evidence: re-render with different camera, not abort
"""

from typing import List, Tuple, Optional

VEHICLE_TURN_TERMS = frozenset({"turn", "vehicle", "lateral", "slip", "tire", "corner", "curve"})
STRUCTURAL_TERMS = frozenset({"structural", "load", "shelf", "bending", "deformation", "collapse", "stress"})


def get_render_hints(description: str, intent: str, observability_attempt: int = 0) -> str:
    """
    Augment prompt with physics observability: camera/framing that exposes motion.
    Returns prefix to prepend to the beat description for rendering.
    """
    text = f"{description} {intent}".lower()

    if any(t in text for t in VEHICLE_TURN_TERMS):
        if observability_attempt == 0:
            return "Side view, tire-ground contact visible, road lines for trajectory reference. "
        return "Overhead or diagonal angle showing lateral slip and path deviation. "

    if any(t in text for t in STRUCTURAL_TERMS):
        if observability_attempt == 0:
            return "Clear view of structure under load, deformation visible. Fixed reference in frame. "
        return "Close-up showing stress points and bending. Background for scale. "

    return ""


def get_observer_physics_questions(description: str, intent: str) -> List[str]:
    """
    Physics-specific questions the observer MUST answer.
    If observer cannot answer, use observation_occluded and re-render.
    """
    text = f"{description} {intent}".lower()

    if any(t in text for t in VEHICLE_TURN_TERMS):
        return [
            "Is lateral slip visible?",
            "Is tire-ground contact maintained?",
            "Is trajectory deviating from intended path?",
        ]

    if any(t in text for t in STRUCTURAL_TERMS):
        return [
            "Is structural deformation or bending visible?",
            "Is contact/load application clear?",
            "Is failure or degradation observable?",
        ]

    return []


def should_augment_for_observability(constraints: List[str]) -> bool:
    """True if constraints indicate observer could not extract physics (re-render, do not abort)."""
    ep = frozenset({"insufficient_evidence", "video_unavailable", "missing_video", "observation_occluded"})
    norm = {str(c).strip().lower().replace("-", "_") for c in constraints if c}
    return bool(norm & ep)
