from typing import Dict, List, Optional
from runtime.episode_state import EpisodeState
from runtime.beat_state import BeatState
from models.epistemic import EpistemicState, EpistemicSummary, ConfidenceLevel, compute_confidence_level


class EpisodeSnapshot:
    def __init__(
        self,
        episode_id: str,
        state: EpisodeState,
        beats: List[Dict],
        artifacts: List[Dict],
        errors: List[Dict],
        total_cost: float,
        confidence: float = 1.0,
    ):
        self.episode_id = episode_id
        self.state = state
        self.beats = beats
        self.artifacts = artifacts
        self.errors = errors
        self.total_cost = total_cost
        self.confidence = confidence

    def to_dict(self) -> Dict:
        epistemic_summary = self._compute_epistemic_summary()
        # Confidence = conclusion certainty. No conclusion → 0.1 (bounded_low)
        confidence = self._epistemic_confidence(epistemic_summary)
        return {
            "episode_id": self.episode_id,
            "state": self.state,
            "status": self.state,  # UI expects status for graph terminal state
            "progress": self._progress(),
            "beats": self.beats,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "total_cost": self.total_cost,
            # UI Compat
            "budget_spent_usd": self.total_cost,
            "confidence": confidence,
            # Epistemic Architecture (Option C)
            "epistemic_summary": epistemic_summary.to_dict() if epistemic_summary else None,
            "outcome": self._epistemic_outcome(epistemic_summary),
            "constraints_discovered": epistemic_summary.constraints_triggered if epistemic_summary else [],
            "missing_evidence": epistemic_summary.missing_evidence if epistemic_summary else [],
        }
    
    def _epistemic_confidence(self, epistemic_summary: Optional[EpistemicSummary]) -> float:
        """Confidence = conclusion certainty. No conclusion → 0.1 (bounded_low)."""
        if not epistemic_summary:
            return 0.1
        if epistemic_summary.final_state in (
            EpistemicState.EPISTEMICALLY_INCOMPLETE,
            EpistemicState.UNCERTAIN_TERMINATION,
        ):
            return 0.1  # No conclusion made → low confidence
        if epistemic_summary.final_state == EpistemicState.REJECTED:
            return min(0.5, self.confidence)  # Conclusion made but negative
        if epistemic_summary.final_state == EpistemicState.ACCEPTED:
            return self.confidence  # Conclusion made with evidence
        return 0.1
    
    def _epistemic_outcome(self, epistemic_summary: Optional[EpistemicSummary]) -> str:
        """Human-readable outcome for Option C."""
        if not epistemic_summary:
            return "unknown"
        return epistemic_summary.final_state.value.lower()
    
    def _compute_epistemic_summary(self) -> Optional[EpistemicSummary]:
        """Compute epistemic summary from beat states and transitions."""
        # Check for epistemically incomplete beats
        incomplete_beats = [
            b for b in self.beats
            if b.get("state") == BeatState.EPISTEMICALLY_INCOMPLETE
        ]
        uncertain_beats = [
            b for b in self.beats
            if b.get("state") == BeatState.UNCERTAIN_TERMINATION
        ]
        rejected_beats = [
            b for b in self.beats
            if b.get("state") == BeatState.ABORTED
        ]
        accepted_beats = [
            b for b in self.beats
            if b.get("state") == BeatState.ACCEPTED
        ]
        
        # Determine final epistemic state
        if incomplete_beats:
            # Collect missing evidence from all incomplete beats
            missing_evidence = set()
            for beat in incomplete_beats:
                error = beat.get("last_error", "")
                if "Missing evidence:" in error:
                    # Extract evidence names from error message
                    evidence_str = error.split("Missing evidence:")[-1].strip()
                    missing_evidence.update([e.strip() for e in evidence_str.split(",")])
            
            return EpistemicSummary(
                final_state=EpistemicState.EPISTEMICALLY_INCOMPLETE,
                confidence=compute_confidence_level(self.confidence),
                missing_evidence=list(missing_evidence),
                constraints_triggered=["insufficient_physical_evidence"],
                justification=[
                    f"{len(incomplete_beats)} beat(s) blocked by missing evidence"
                ],
            )
        elif uncertain_beats:
            return EpistemicSummary(
                final_state=EpistemicState.UNCERTAIN_TERMINATION,
                confidence=compute_confidence_level(self.confidence),
                justification=[
                    f"{len(uncertain_beats)} beat(s) terminated with uncertain verdict"
                ],
            )
        elif rejected_beats and not accepted_beats:
            return EpistemicSummary(
                final_state=EpistemicState.REJECTED,
                confidence=compute_confidence_level(self.confidence),
                constraints_triggered=["observer_rejection"],
                justification=[
                    f"All {len(rejected_beats)} beat(s) rejected by observer"
                ],
            )
        elif accepted_beats:
            return EpistemicSummary(
                final_state=EpistemicState.ACCEPTED,
                confidence=compute_confidence_level(self.confidence),
                justification=[
                    f"{len(accepted_beats)} beat(s) accepted with sufficient evidence"
                ],
            )
        
        # Default: uncertain if no beats processed yet
        return EpistemicSummary(
            final_state=EpistemicState.UNCERTAIN_TERMINATION,
            confidence=ConfidenceLevel.BOUNDED_LOW,
            justification=["No beats processed yet"],
        )

    def _progress(self) -> Dict:
        total = len(self.beats)
        completed = len(
            [b for b in self.beats if b["state"] == BeatState.ACCEPTED]
        )
        aborted = len(
            [b for b in self.beats if b["state"] == BeatState.ABORTED]
        )

        return {
            "total_beats": total,
            "completed": completed,
            "aborted": aborted,
            "percent": round((completed / total) * 100, 2) if total else 0.0,
        }

    # ---------- Factory ----------

    @classmethod
    def from_runtime(cls, runtime):
        sql = runtime.sql

        episode = sql.get_episode(runtime.episode_id)
        beats = sql.get_beats(runtime.episode_id)
        artifacts = sql.get_artifacts(runtime.episode_id)

        errors = [
            {
                "beat_id": b["beat_id"],
                "error": b["last_error"],
            }
            for b in beats
            if b["state"] == BeatState.ABORTED
        ]

        total_cost = sum(b.get("cost_spent", 0) for b in beats)

        # Confidence: derive from observer metrics only. Do not default to high constants.
        # When epistemically blocked, confidence will be overridden to 0.1 in to_dict().
        conf_scores = [
            b.get("metrics", {}).get("confidence", 0.0)
            for b in beats
            if b.get("metrics")
        ]
        avg_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 0.5
        
        # State: if any beat is epistemically incomplete, episode should be EPISTEMICALLY_BLOCKED
        state = episode["state"]
        state_str = getattr(state, "value", state) if state else ""
        if state_str == EpisodeState.EXECUTING.value:
            if any(
                (b.get("state") or "") in (
                    BeatState.EPISTEMICALLY_INCOMPLETE.value,
                    BeatState.UNCERTAIN_TERMINATION.value,
                )
                for b in beats
            ):
                state = EpisodeState.EPISTEMICALLY_BLOCKED

        return cls(
            episode_id=runtime.episode_id,
            state=state,
            beats=beats,
            artifacts=artifacts,
            errors=errors,
            total_cost=total_cost,
            confidence=round(avg_confidence, 2),
        )
