"""
Ollama Vision Observer

Fallback observer using Ollama with vision models (e.g. llava, llava:13b).
Used when Google Gemini returns 429 RESOURCE_EXHAUSTED.
"""

from __future__ import annotations

import os
import json
import uuid
import time
import base64
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from models.observation import (
    ObservationResult,
    CharacterObservation,
    EnvironmentObservation,
    ActionObservation,
    QualityMetrics,
    ContinuityError,
    ContinuityErrorType,
    ActionOutcome,
    EmotionState,
    TaskContext,
)

logger = logging.getLogger(__name__)


@dataclass
class OllamaObserverConfig:
    """Configuration for Ollama fallback observer."""
    base_url: str = "http://localhost:11434"
    model: str = "llava"
    timeout_sec: float = 30.0  # Reduced from 60s for faster failure
    connect_timeout_sec: float = 5.0  # Quick connection check


# Shared prompt - must match GeminiObserver format for consistent parsing
OBSERVATION_PROMPT = """You are a video analysis AI for a physics/causal simulation.
Analyze this video and extract structured observations. Focus on what is OBSERVABLE and physically verifiable.

CRITICAL PHYSICS RULES (MUST ENFORCE):
- If gravity is violated (unsupported mass floating), verdict = "impossible"
- If energy conservation is violated (acceleration with no force), verdict = "impossible"
- If action contradicts prior state, verdict = "contradicts"
- If evidence is insufficient or conflicting, verdict = "uncertain"

EPISTEMIC INTEGRITY (NON-NEGOTIABLE):
- Never assume values. If you cannot measure it, mark it as missing.
- Never interpolate physics without declaring uncertainty.
- If a human expert would say "I need more data", you must say the same.
- Explicitly state what evidence is missing or occluded.

QUANTITATIVE PHYSICS EXTRACTION (REQUIRED):
Extract quantitative measurements where observable. If not observable, mark as missing.
For vehicle dynamics: measure speed_profile (m/s over time), turn_radius (m), yaw_rate (rad/s), slip_angle (deg), roll_angle (deg).
For general motion: measure velocity_vector (m/s), acceleration_vector (m/s²), angular_velocity (rad/s).
If camera perspective or video quality prevents measurement, explicitly state "observation_occluded" for that evidence.

Never assume success because a video exists. The video is a hypothesis, not proof.

For inferred_constraints: when video shows degradation (bending, strain, deformation) use physics identifiers:
stress_limit_approached, visible_bending, tolerance_margin_low, likely_failure.
Use "insufficient_evidence" or "observation_occluded" ONLY when video is unclear, occluded, or camera does not expose the dynamics.

{physics_questions_section}

Return ONLY valid JSON in this exact format:
{{
    "characters": {{
        "<character_id>": {{
            "visible": true/false,
            "position": {{"x": 0.5, "y": 0.5}},
            "pose": "standing|sitting|running|fighting|etc",
            "emotion": "neutral|happy|angry|sad|fearful|surprised|determined|defeated|excited|confused",
            "motion_intensity": 0.0-1.0,
            "appearance_consistent": true/false
        }}
    }},
    "environment": {{
        "location_description": "brief description",
        "time_of_day": "dawn|morning|noon|afternoon|dusk|night",
        "lighting": "bright|dim|dramatic|natural",
        "mood": "tense|peaceful|chaotic|etc"
    }},
    "action": {{
        "action_description": "what happened in the video",
        "outcome": "success|partial|failed|interrupted|unknown",
        "action_type": "attack|dialogue|movement|etc",
        "participants": ["character_ids"],
        "narrative_beat_achieved": true/false,
        "narrative_implications": ["implication1", "implication2"]
    }},
    "quality": {{
        "visual_clarity": 0.0-1.0,
        "motion_smoothness": 0.0-1.0,
        "temporal_coherence": 0.0-1.0,
        "style_consistency": 0.0-1.0,
        "action_clarity": 0.0-1.0,
        "character_recognizability": 0.0-1.0,
        "artifacts_detected": 0
    }},
    "continuity_errors": [
        {{
            "error_type": "character_missing|location_mismatch|etc",
            "description": "what's wrong",
            "severity": 0.0-1.0,
            "affected_entities": ["entity_ids"]
        }}
    ],
    "evidence": {{
        "available": [
            {{
                "name": "speed_profile",
                "source": "observer",
                "resolution": "coarse|fine|unknown",
                "confidence": 0.0-1.0,
                "value": [10.5, 12.3, 15.1],
                "value_type": "array",
                "units": "m/s",
                "frame_range": [0, 30]
            }}
        ],
        "missing": ["turn_radius", "friction_estimate"],
        "occluded": ["slip_angle"]
    }},
    "causal_analysis": {{
        "explanation": "Why did the action succeed or fail? (Causal chain)",
        "inferred_constraints": ["constraint_1", "constraint_2"],
        "physical_validity_confidence": 0.0-1.0
    }},
    "verdict": "valid|degraded|failed|impossible|contradicts|blocks|uncertain",
    "confidence": 0.0-1.0
}}

CONTEXT:
- Expected characters: {expected_characters}
- Expected action: {expected_action}
- Expected location: {expected_location}
- Previous state: {previous_state}

Analyze the video frames provided and return ONLY the JSON, no explanation."""


def _normalize_physics_constraints(raw: List[str]) -> List[str]:
    """Map free-form observer output to canonical physics identifiers."""
    SOFT_PHYSICS_CONSTRAINTS = frozenset({
        "stress_limit_approached", "visible_bending", "tolerance_margin_low",
        "likely_failure", "structural_bending", "load_approached",
    })
    result = []
    raw_lower = " ".join(str(c).lower() for c in raw)
    if any(k in raw_lower for k in ("bending", "flex", "deformation", "strain")):
        result.append("visible_bending")
    if any(k in raw_lower for k in ("stress", "tolerance", "margin", "limit")):
        result.append("stress_limit_approached")
    if any(k in raw_lower for k in ("likely fail", "collapse", "progression")):
        result.append("likely_failure")
    for c in raw:
        cnorm = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if cnorm in SOFT_PHYSICS_CONSTRAINTS:
            result.append(cnorm)
    out = list(dict.fromkeys(result))
    return out if out else list(raw)


class OllamaObserver:
    """
    Observe video using Ollama with a vision model (e.g. llava).
    Fallback when Gemini returns 429 RESOURCE_EXHAUSTED.
    """

    def __init__(self, config: Optional[OllamaObserverConfig] = None):
        self.config = config or OllamaObserverConfig()
        self._session = None

    def _get_session(self):
        """Lazy import requests to avoid hard dependency."""
        import requests
        return requests

    def observe(
        self,
        frames: List[Tuple[int, bytes]],
        context: TaskContext,
    ) -> ObservationResult:
        """
        Analyze video frames and return structured observation.
        Same interface as GeminiObserver.observe().
        """
        start_time = time.time()
        observation_id = str(uuid.uuid4())

        if not frames:
            return self._mock_observation(observation_id, context)

        # Build prompt
        pq = getattr(context, "physics_questions", None) or []
        if pq:
            physics_questions_section = (
                "PHYSICS QUESTIONS (you MUST answer these; if camera does not expose them, use "
                'inferred_constraints: ["observation_occluded"] and verdict: "uncertain"):\n'
                + "\n".join(f"- {q}" for q in pq)
            )
        else:
            physics_questions_section = ""

        prompt = OBSERVATION_PROMPT.format(
            expected_characters=", ".join(context.expected_characters) or "any",
            expected_action=context.expected_action or "any action",
            expected_location=context.expected_location or "any location",
            previous_state=json.dumps(context.previous_world_state) if context.previous_world_state else "none",
            physics_questions_section=physics_questions_section,
        )

        # Ollama chat API: images as base64 in message
        images_b64 = [
            base64.standard_b64encode(fb).decode("utf-8")
            for _, fb in frames
        ]

        # Use model name with :latest suffix if not already present (Ollama prefers explicit tags)
        model_name = self.config.model
        if ":" not in model_name:
            model_name = f"{model_name}:latest"

        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
        }

        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        try:
            requests = self._get_session()
            
            # Quick connectivity check first (faster failure if Ollama isn't running)
            try:
                health_url = f"{self.config.base_url.rstrip('/')}/api/tags"
                requests.get(health_url, timeout=self.config.connect_timeout_sec)
            except Exception as health_err:
                logger.warning(
                    f"[ollama_observer] Ollama not reachable at {self.config.base_url}: {health_err}. "
                    f"Make sure Ollama is running: `ollama serve`"
                )
                return self._mock_observation(observation_id, context)
            
            # Full API call with longer timeout for model inference
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Check if it's a timeout error
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str:
                model_name = self.config.model if ":" in self.config.model else f"{self.config.model}:latest"
                logger.warning(
                    f"[ollama_observer] Request timed out after {self.config.timeout_sec}s. "
                    f"Model '{model_name}' may need to be loaded into memory. "
                    f"Try: `ollama run {model_name}` to preload it, or increase timeout."
                )
            else:
                logger.warning(f"[ollama_observer] API call failed: {e}")
            return self._mock_observation(observation_id, context)

        # Extract message content (Ollama returns {"message": {"content": "..."}})
        message = data.get("message", {}) if isinstance(data, dict) else {}
        raw_text = message.get("content", "") if isinstance(message, dict) else ""
        if not raw_text or not str(raw_text).strip():
            logger.warning("[ollama_observer] empty response from Ollama")
            return self._mock_observation(observation_id, context)

        latency_sec = time.time() - start_time
        observation = self._parse_response(raw_text, observation_id, context, latency_sec)
        observation.raw_response = raw_text

        logger.info(
            f"[ollama_observer] observation complete: "
            f"quality={observation.get_quality_score():.2f}, "
            f"confidence={observation.confidence:.2f}"
        )
        return observation

    def _parse_response(
        self,
        raw_text: str,
        observation_id: str,
        context: TaskContext,
        latency_sec: float,
    ) -> ObservationResult:
        """Parse Ollama response into ObservationResult (same schema as Gemini)."""
        from datetime import datetime

        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                text = text[start_idx : end_idx + 1]
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"[ollama_observer] JSON parse error: {e}")
            mock = self._mock_observation(observation_id, context)
            mock.verdict = "uncertain"
            mock.confidence = 0.3
            mock.constraints_inferred = ["insufficient_evidence", "json_parse_error"]
            return mock

        if not isinstance(data, dict):
            mock = self._mock_observation(observation_id, context)
            mock.verdict = "uncertain"
            mock.confidence = 0.3
            mock.constraints_inferred = ["insufficient_evidence", "invalid_response_structure"]
            return mock

        # Parse characters
        characters = {}
        chars_data = data.get("characters")
        if not isinstance(chars_data, dict):
            chars_data = {}
        for char_id, char_data in chars_data.items():
            if not isinstance(char_data, dict):
                continue
            emotion = None
            if char_data.get("emotion"):
                try:
                    emotion = EmotionState(char_data["emotion"])
                except ValueError:
                    emotion = EmotionState.NEUTRAL
            characters[char_id] = CharacterObservation(
                character_id=char_id,
                visible=char_data.get("visible", True),
                position=char_data.get("position"),
                pose=char_data.get("pose"),
                emotion=emotion,
                motion_intensity=char_data.get("motion_intensity", 0.0),
                appearance_consistent=char_data.get("appearance_consistent", True),
            )

        env_data = data.get("environment", {})
        environment = EnvironmentObservation(
            location_description=env_data.get("location_description"),
            time_of_day=env_data.get("time_of_day"),
            lighting=env_data.get("lighting"),
            mood=env_data.get("mood"),
        )

        action_data = data.get("action", {})
        action = None
        if action_data:
            outcome = ActionOutcome.UNKNOWN
            if action_data.get("outcome"):
                try:
                    outcome = ActionOutcome(action_data["outcome"])
                except ValueError:
                    pass
            action = ActionObservation(
                action_description=action_data.get("action_description", ""),
                outcome=outcome,
                action_type=action_data.get("action_type"),
                participants=action_data.get("participants", []),
                narrative_beat_achieved=action_data.get("narrative_beat_achieved", False),
                narrative_implications=action_data.get("narrative_implications", []),
            )

        quality_data = data.get("quality", {})
        quality = QualityMetrics(
            visual_clarity=quality_data.get("visual_clarity", 0.5),
            motion_smoothness=quality_data.get("motion_smoothness", 0.5),
            temporal_coherence=quality_data.get("temporal_coherence", 0.5),
            style_consistency=quality_data.get("style_consistency", 0.5),
            action_clarity=quality_data.get("action_clarity", 0.5),
            character_recognizability=quality_data.get("character_recognizability", 0.5),
            artifacts_detected=quality_data.get("artifacts_detected", 0),
        )
        quality.compute_overall()

        continuity_errors = []
        for err in data.get("continuity_errors", []):
            try:
                error_type = ContinuityErrorType(err.get("error_type", "temporal_artifact"))
            except ValueError:
                error_type = ContinuityErrorType.TEMPORAL_ARTIFACT
            continuity_errors.append(
                ContinuityError(
                    error_type=error_type,
                    description=err.get("description", ""),
                    severity=err.get("severity", 0.5),
                    affected_entities=err.get("affected_entities", []),
                )
            )

        causal_data = data.get("causal_analysis", {})
        constraints_inferred = causal_data.get("inferred_constraints", []) or []
        constraints_inferred = _normalize_physics_constraints(constraints_inferred)
        causal_explanation = causal_data.get("explanation")
        phys_conf = float(causal_data.get("physical_validity_confidence", 0.7) or 0.7)

        # Parse Phase 8: Evidence Ledger
        from models.epistemic import EvidenceLedger, Evidence, EvidenceSource, EvidenceResolution
        evidence_ledger = EvidenceLedger(beat_id=context.beat_id or observation_id)
        
        evidence_data = data.get("evidence", {})
        if isinstance(evidence_data, dict):
            # Parse available evidence
            available_evidence = evidence_data.get("available", [])
            if isinstance(available_evidence, list):
                for ev_data in available_evidence:
                    if isinstance(ev_data, dict):
                        try:
                            evidence = Evidence(
                                name=ev_data.get("name", ""),
                                source=EvidenceSource(ev_data.get("source", "observer")),
                                resolution=EvidenceResolution(ev_data.get("resolution", "unknown")),
                                confidence=float(ev_data.get("confidence", 0.5)),
                                value=ev_data.get("value"),
                                value_type=ev_data.get("value_type"),
                                units=ev_data.get("units"),
                                frame_range=tuple(ev_data["frame_range"]) if ev_data.get("frame_range") else None,
                            )
                            evidence_ledger.add_evidence(evidence)
                        except (ValueError, KeyError) as e:
                            logger.warning(f"[ollama_observer] Failed to parse evidence: {e}")
            
            # Mark missing evidence
            missing_evidence = evidence_data.get("missing", [])
            if isinstance(missing_evidence, list):
                for name in missing_evidence:
                    if isinstance(name, str):
                        evidence_ledger.mark_missing(name)
            
            # Mark occluded evidence as missing
            occluded_evidence = evidence_data.get("occluded", [])
            if isinstance(occluded_evidence, list):
                for name in occluded_evidence:
                    if isinstance(name, str):
                        evidence_ledger.mark_missing(name)
        
        verdict = (data.get("verdict") or "").lower()
        constraint_text = " ".join([str(c) for c in constraints_inferred]).lower()
        if any(k in constraint_text for k in ("gravity", "unsupported", "energy", "conservation")):
            verdict = "impossible"
        if not verdict:
            if phys_conf < 0.4:
                verdict = "impossible"
            elif phys_conf < 0.7:
                verdict = "uncertain"
            else:
                verdict = "valid"

        observation = ObservationResult(
            observation_id=observation_id,
            video_uri=context.beat_id or "unknown",
            beat_id=context.beat_id,
            created_at=datetime.utcnow(),
            observation_latency_ms=int(latency_sec * 1000),
            characters=characters,
            environment=environment,
            action=action,
            quality=quality,
            continuity_errors=continuity_errors,
            confidence=data.get("confidence", 0.7),
            observer_type="ollama",
            model_version=self.config.model,
            constraints_inferred=constraints_inferred,
            causal_explanation=causal_explanation,
            verdict=verdict,
            forces_termination=verdict in ("impossible", "contradicts", "blocks"),
            evidence_ledger=evidence_ledger,
        )
        
        return observation

    def _mock_observation(
        self,
        observation_id: str,
        context: TaskContext,
    ) -> ObservationResult:
        """Return mock observation when Ollama unavailable."""
        from datetime import datetime
        from models.epistemic import EvidenceLedger
        from models.physics_constraints import (
            EVIDENCE_SPEED_PROFILE,
            EVIDENCE_TURN_RADIUS,
            EVIDENCE_FRICTION_ESTIMATE,
            EVIDENCE_YAW_RATE,
            EVIDENCE_SLIP_ANGLE,
            EVIDENCE_ROLL_ANGLE,
            EVIDENCE_LATERAL_ACCELERATION,
            EVIDENCE_ANGULAR_VELOCITY,
        )

        characters = {}
        expected = (context.expected_characters or []) if context else []
        for char_id in expected:
            characters[char_id] = CharacterObservation(
                character_id=char_id,
                visible=True,
                emotion=EmotionState.NEUTRAL,
            )
        
        # Create evidence ledger with all required evidence marked as missing
        evidence_ledger = EvidenceLedger(beat_id=context.beat_id or observation_id)
        required_evidence = [
            EVIDENCE_SPEED_PROFILE,
            EVIDENCE_TURN_RADIUS,
            EVIDENCE_FRICTION_ESTIMATE,
            EVIDENCE_YAW_RATE,
            EVIDENCE_SLIP_ANGLE,
            EVIDENCE_ROLL_ANGLE,
            EVIDENCE_LATERAL_ACCELERATION,
            EVIDENCE_ANGULAR_VELOCITY,
        ]
        for ev_name in required_evidence:
            evidence_ledger.mark_missing(ev_name)

        return ObservationResult(
            observation_id=observation_id,
            video_uri=context.beat_id or "mock",
            beat_id=context.beat_id,
            characters=characters,
            environment=EnvironmentObservation(
                location_description=context.expected_location,
            ),
            action=ActionObservation(
                action_description=context.expected_action or "mock action",
                outcome=ActionOutcome.SUCCESS,
                narrative_beat_achieved=True,
            ),
            quality=QualityMetrics(
                overall_quality=0.75,
                visual_clarity=0.75,
                motion_smoothness=0.75,
                temporal_coherence=0.75,
                style_consistency=0.75,
                action_clarity=0.75,
                character_recognizability=0.75,
            ),
            confidence=0.3,
            observer_type="mock",
            model_version="mock",
            constraints_inferred=["insufficient_evidence"],
            verdict="uncertain",
            evidence_ledger=evidence_ledger,
        )
