"""
Progressive constraint tightening: refine beat descriptions using prior observer output.

Beat N+1 must incorporate what was learned in Beat N.
Example: Beat 1 result "visible_bending" -> Beat 2 prompt becomes
"Increase acceleration slightly beyond previous level where bending was observed"
"""

from typing import Dict, List, Any

# Physics constraints that warrant beat refinement (not epistemic)
REFINEMENT_CONSTRAINTS = frozenset({
    "stress_limit_approached", "visible_bending", "tolerance_margin_low",
    "likely_failure", "structural_bending", "load_limit_approached",
})


def refine_beat_description(beat_spec: Dict[str, Any], prior_constraints: List[str]) -> Dict[str, Any]:
    """
    Augment beat description with prior physics constraints.
    Does not replace the description; adds context for progressive exploration.
    """
    physics = [c for c in prior_constraints if c and c.lower().replace("-", "_") in REFINEMENT_CONSTRAINTS]
    if not physics:
        return beat_spec

    desc = (beat_spec.get("description") or "").strip()
    prior_ctx = " [Prior: " + ", ".join(physics) + " observed; refine/advance beyond that level]"
    # Avoid appending multiple times
    if prior_ctx.rstrip("]") not in desc:
        beat_spec = dict(beat_spec)
        beat_spec["description"] = desc + prior_ctx
        beat_spec["prior_constraints"] = physics
    return beat_spec
