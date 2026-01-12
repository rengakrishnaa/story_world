from typing import Dict, Any
import logging
import numpy as np
from models.beat_observation import BeatObservation
from agents.beat_observer_vision import detect_characters
from agents.continuity_memory import ContinuityMemory
from agents.embedding_provider import EmbeddingProvider
from agents.interaction_validator import InteractionValidator
from agents.temporal_consistency import TemporalConsistency
from agents.world_guard import WorldGuard
from agents.character_registry import CharacterRegistry
from agents.location_registry import LocationRegistry

logger = logging.getLogger(__name__)


class BeatObserver:
    def __init__(self, world_id: str):
        self.world_id = world_id

        # Core subsystems
        self.embedder = EmbeddingProvider()
        self.continuity = ContinuityMemory(world_id)
        self.interactions = InteractionValidator()
        self.temporal = TemporalConsistency()
        self.characters = CharacterRegistry()
        self.locations = LocationRegistry()

        # World invariants (plug-in system)
        self.world_guard = WorldGuard(invariants=[])

    def _cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


    def observe(
        self,
        beat: Dict[str, Any],
        generation_result: Dict[str, Any]
    ) -> BeatObservation:

        beat_id = beat["id"]
        image_path = generation_result.get("keyframe_path")

        # ------------------------------------------------
        # 1️⃣ Intent
        # ------------------------------------------------
        if not beat.get("description"):
            return BeatObservation(
                beat_id=beat_id,
                success=False,
                confidence=0.2,
                intent_satisfied=False,
                entity_presence={},
                constraint_violations=["intent_missing"],
                failure_type="intent",
                explanation="Beat description missing",
                recommended_action="retry_prompt"
            )

        # ------------------------------------------------
        # 2️⃣ Entity presence
        # ------------------------------------------------
        characters = beat.get("characters", [])
        entity_scores = {}

        if characters and image_path:
            entity_scores = detect_characters(image_path, characters)
            missing = [c for c, s in entity_scores.items() if s < 0.25]

            if missing:
                return BeatObservation(
                    beat_id=beat_id,
                    success=False,
                    confidence=0.3,
                    intent_satisfied=True,
                    entity_presence=entity_scores,
                    constraint_violations=missing,
                    failure_type="entity",
                    explanation=f"Missing characters: {missing}",
                    recommended_action="retry_prompt"
                )

        # ------------------------------------------------
        # 3️⃣ Identity continuity (ONCE)
        # ------------------------------------------------
        if characters and image_path:
            current_embedding = self.embedder.embed_image(image_path)


            for character in characters:
                profile = self.characters.get(character)

                if profile and profile.reference_embeddings:
                    ref_sims = [
                        self._cosine(current_embedding, ref)
                        for ref in profile.reference_embeddings
                    ]

                    ref_sim = max(ref_sims)


                    if ref_sim < 0.75:
                        return BeatObservation(
                            beat_id=beat_id,
                            success=False,
                            confidence=ref_sim,
                            intent_satisfied=True,
                            entity_presence={character: ref_sim},
                            constraint_violations=["identity_mismatch"],
                            failure_type="identity",
                            explanation=f"{character} does not match reference identity ({ref_sim:.2f})",
                            recommended_action="retry_model"
                
                        )

                # 3B — Temporal continuity check (RELATIVE)
                prev_sim = self.continuity.similarity(character, current_embedding)
                if prev_sim is not None and prev_sim < 0.8:
                    return BeatObservation(
                        beat_id=beat_id,
                        success=False,
                        confidence=prev_sim,
                        intent_satisfied=True,
                        entity_presence={character: prev_sim},
                        constraint_violations=["identity_drift"],
                        failure_type="continuity",
                        explanation=f"Temporal drift for {character} ({prev_sim:.2f})",
                        recommended_action="retry_model"
                    )
                
        # ------------------------------------------------
        # 4️⃣ Location validation
        # ------------------------------------------------
        location = beat.get("location")
        if location and not self.locations.get(location):
            return BeatObservation(
                beat_id=beat_id,
                success=False,
                confidence=0.3,
                intent_satisfied=True,
                entity_presence=entity_scores,
                constraint_violations=["unknown_location"],
                failure_type="location",
                explanation="Unknown location",
                recommended_action="retry_prompt"
            )

        # ------------------------------------------------
        # 5️⃣ Interactions
        # ------------------------------------------------
        interactions = beat.get("interactions", [])
        if interactions:
            violations = self.interactions.validate(interactions, entity_scores)
            if violations:
                return BeatObservation(
                    beat_id=beat_id,
                    success=False,
                    confidence=0.35,
                    intent_satisfied=True,
                    entity_presence=entity_scores,
                    constraint_violations=violations,
                    failure_type="interaction",
                    explanation=f"Interaction violations: {violations}",
                    recommended_action="retry_prompt"
                )

        # ------------------------------------------------
        # 6️⃣ Temporal consistency
        # ------------------------------------------------
        temporal_score = self.temporal.score(beat)
        if temporal_score < 0.6:
            return BeatObservation(
                beat_id=beat_id,
                success=False,
                confidence=temporal_score,
                intent_satisfied=True,
                entity_presence=entity_scores,
                constraint_violations=["temporal_jump"],
                failure_type="temporal",
                explanation="Temporal discontinuity",
                recommended_action="retry_prompt"
            )

        # ------------------------------------------------
        # 7️⃣ World invariants
        # ------------------------------------------------
        violations = self.world_guard.validate(beat)
        if violations:
            return BeatObservation(
                beat_id=beat_id,
                success=False,
                confidence=0.0,
                intent_satisfied=True,
                entity_presence=entity_scores,
                constraint_violations=violations,
                failure_type="world",
                explanation=f"World invariant violated: {violations}",
                recommended_action="abort"
            )

        # ------------------------------------------------
        # ✅ Accept
        # ------------------------------------------------
        confidence = self._compute_confidence(True, entity_scores)

        # After ALL checks passed
        for character in characters:
            self.continuity.update(character, current_embedding)


        return BeatObservation(
            beat_id=beat_id,
            success=True,
            confidence=confidence,
            intent_satisfied=True,
            entity_presence=entity_scores,
            constraint_violations=[],
            failure_type=None,
            explanation="Beat accepted",
            recommended_action="accept"
        )


    def _compute_confidence(self, intent_ok, entity_presence):
        score = 0.0
        if intent_ok:
            score += 0.5
        if entity_presence:
            score += 0.5 * (sum(entity_presence.values()) / len(entity_presence))
        return round(min(score, 1.0), 2)
