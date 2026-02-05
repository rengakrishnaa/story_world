"""
Video Observer Agent

Watches rendered video and extracts structured observations.
Bootstrap with Gemini Vision, then internalize to local model.

Architecture:
    Video → Observer → ObservationResult → WorldStateGraph
"""

from __future__ import annotations

import os
import json
import uuid
import time
import base64
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

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

# Canonical soft physics constraints (not epistemic). Observer should emit these when video shows degradation.
SOFT_PHYSICS_CONSTRAINTS = frozenset({
    "stress_limit_approached", "visible_bending", "tolerance_margin_low",
    "likely_failure", "structural_bending", "load_limit_approached",
})


def _normalize_physics_constraints(raw: List[str]) -> List[str]:
    """Map free-form observer output to canonical physics identifiers."""
    result = []
    raw_lower = " ".join(str(c).lower() for c in raw)
    # Map physics keywords to canonical identifiers (so is_epistemic_only recognizes them)
    if any(k in raw_lower for k in ("bending", "flex", "deformation", "strain")):
        result.append("visible_bending")
    if any(k in raw_lower for k in ("stress", "tolerance", "margin", "limit")):
        result.append("stress_limit_approached")
    if any(k in raw_lower for k in ("likely fail", "collapse", "progression")):
        result.append("likely_failure")
    # Keep canonical constraints from raw
    for c in raw:
        cnorm = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if cnorm in SOFT_PHYSICS_CONSTRAINTS:
            result.append(cnorm)
        elif any(p in cnorm for p in ("structural", "load", "gravity", "stability", "energy")) and "insufficient" not in cnorm:
            result.append("stress_limit_approached" if "stress" in cnorm or "load" in cnorm else "visible_bending")
    out = list(dict.fromkeys(result))
    return out if out else list(raw)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ObserverConfig:
    """Configuration for VideoObserverAgent."""
    # Model selection
    use_gemini: bool = True
    gemini_model: str = "gemini-2.0-flash-lite"
    
    # Fallback open-source model (Ollama) when Gemini returns 429 RESOURCE_EXHAUSTED
    fallback_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava"
    
    # Local model (for internalization)
    use_local: bool = False
    local_model_path: Optional[str] = None
    local_confidence_threshold: float = 0.7
    
    # Frame extraction
    frames_to_analyze: int = 5  # Key frames per video
    frame_extraction_method: str = "uniform"  # uniform, keyframe, scene_change
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    
    # Training data
    record_for_training: bool = True
    training_data_dir: str = "training_data/observations"
    
    # Timeouts
    observation_timeout_sec: float = 30.0
    
    # Phase 7: Multi-Observer
    enable_multi_observer: bool = False
    multi_observer_count: int = 2
    disagreement_threshold: float = 0.3


# ============================================================================
# Frame Extraction
# ============================================================================

class FrameExtractor:
    """Extract key frames from video for analysis."""
    
    def __init__(self, method: str = "uniform"):
        self.method = method
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 5,
    ) -> List[Tuple[int, bytes]]:
        """
        Extract key frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of (frame_index, frame_bytes) tuples
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, using stub frames")
            return self._stub_frames(num_frames)
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return self._stub_frames(num_frames)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return self._stub_frames(num_frames)
        
        # Calculate frame indices based on method
        if self.method == "uniform":
            indices = self._uniform_indices(total_frames, num_frames)
        else:
            indices = self._uniform_indices(total_frames, num_frames)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append((idx, buffer.tobytes()))
        
        cap.release()
        return frames
    
    def _uniform_indices(self, total: int, count: int) -> List[int]:
        """Get uniformly distributed frame indices."""
        if count >= total:
            return list(range(total))
        step = total / count
        return [int(i * step) for i in range(count)]
    
    def _stub_frames(self, count: int) -> List[Tuple[int, bytes]]:
        """Return stub frames when video processing unavailable."""
        # Return empty list - observer will use text-based analysis
        return []


# ============================================================================
# Gemini Observer
# ============================================================================

class GeminiObserver:
    """
    Observe video using Google Gemini Vision API.
    This is the bootstrap observer before internalization.
    """
    
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

    def __init__(self, config: ObserverConfig, fallback_observer=None):
        self.config = config
        self.client = None
        self.fallback_observer = fallback_observer
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize Gemini client."""
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                logger.info(f"[gemini_observer] initialized with {self.config.gemini_model}")
            else:
                logger.warning("[gemini_observer] GEMINI_API_KEY not set")
        except ImportError:
            logger.warning("[gemini_observer] google-genai not installed")
    
    def observe(
        self,
        frames: List[Tuple[int, bytes]],
        context: TaskContext,
    ) -> ObservationResult:
        """
        Analyze video frames and return structured observation.
        
        Args:
            frames: List of (frame_index, frame_bytes) tuples
            context: Task context with expectations
            
        Returns:
            ObservationResult with structured observations
        """
        start_time = time.time()
        observation_id = str(uuid.uuid4())
        
        if not self.client:
            logger.warning("[gemini_observer] no client, returning mock observation")
            return self._mock_observation(observation_id, context)
        
        # Physics observability: explicit questions for dynamics beats
        pq = getattr(context, "physics_questions", None) or []
        if pq:
            physics_questions_section = (
                "PHYSICS QUESTIONS (you MUST answer these; if camera does not expose them, use "
                'inferred_constraints: ["observation_occluded"] and verdict: "uncertain"):\n'
                + "\n".join(f"- {q}" for q in pq)
            )
        else:
            physics_questions_section = ""

        # Build prompt with context
        prompt = self.OBSERVATION_PROMPT.format(
            expected_characters=", ".join(context.expected_characters) or "any",
            expected_action=context.expected_action or "any action",
            expected_location=context.expected_location or "any location",
            previous_state=json.dumps(context.previous_world_state) if context.previous_world_state else "none",
            physics_questions_section=physics_questions_section,
        )
        
        # Prepare content: google-genai expects parts with "text" or "inline_data"
        contents = [{"text": prompt}]
        for idx, frame_bytes in frames:
            contents.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.standard_b64encode(frame_bytes).decode("utf-8"),
                }
            })
        
        # Call Gemini
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
                )
                raw_text = self._extract_response_text(response)
                if not raw_text or not raw_text.strip():
                    raise ValueError("Gemini returned empty or blocked content")
                observation = self._parse_response(
                    raw_text,
                    observation_id,
                    context,
                    time.time() - start_time,
                )
                observation.raw_response = raw_text
                
                logger.info(
                    f"[gemini_observer] observation complete: "
                    f"quality={observation.get_quality_score():.2f}, "
                    f"confidence={observation.confidence:.2f}"
                )
                return observation
                
            except Exception as e:
                logger.warning(f"[gemini_observer] attempt {attempt+1} failed: {e}")
                # Check if this is a 429 error - if so, skip remaining retries and go straight to fallback
                err_str_check = (str(e) + " " + repr(e)).upper()
                if any(keyword in err_str_check for keyword in ["429", "RESOURCE_EXHAUSTED", "RATE", "QUOTA", "EXHAUSTED"]):
                    logger.warning(f"[gemini_observer] 429 detected on attempt {attempt+1}, skipping remaining retries")
                    last_error = e
                    break  # Exit retry loop immediately for 429 errors
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_sec)
                last_error = e

        # On 429 RESOURCE_EXHAUSTED, try open-source fallback before mock
        err_str = (str(last_error) + " " + repr(last_error)).upper() if last_error else ""
        err_code = getattr(last_error, "code", None) if last_error else None
        # google-genai ClientError has .code; httpx uses response.status_code
        if err_code is None and last_error:
            resp = getattr(last_error, "response", None)
            if resp is not None:
                err_code = getattr(resp, "status_code", None)
        if err_code is None and last_error:
            details = getattr(last_error, "details", None)
            if isinstance(details, dict):
                err_code = details.get("code") or (details.get("error") or {}).get("code")
        # Parse error dict from string representation (e.g., "{'error': {'code': 429}}")
        if err_code is None and last_error:
            import re
            err_repr = repr(last_error)
            # Look for 'code': 429 or "code": 429 patterns
            code_match = re.search(r"['\"]code['\"]:\s*429", err_repr)
            if code_match:
                err_code = 429
            # Also check for error dict structure
            if err_code is None:
                error_dict_match = re.search(r"\{['\"]error['\"]:\s*\{[^}]*['\"]code['\"]:\s*429", err_repr)
                if error_dict_match:
                    err_code = 429
        is_429 = (
            err_code == 429
            or "429" in err_str
            or "RESOURCE_EXHAUSTED" in err_str
            or "RATE" in err_str
            or "QUOTA" in err_str
            or "EXHAUSTED" in err_str
        )
        # Use print so this always appears (logging level may filter logger.warning)
        print(
            f"[gemini_observer] fallback check: fallback_observer={self.fallback_observer is not None} "
            f"is_429={is_429} err_code={err_code} err_sample={(err_str[:120] if err_str else '(none)')}"
        )
        logger.warning(
            "[gemini_observer] fallback check: fallback_observer=%s is_429=%s err_code=%s err_sample=%s",
            self.fallback_observer is not None,
            is_429,
            err_code,
            err_str[:120] if err_str else "(none)",
        )
        if is_429 and not self.fallback_observer:
            logger.warning(
                "[gemini_observer] Gemini 429 detected but fallback_observer is None "
                "(set OBSERVER_FALLBACK_ENABLED=true and ensure OllamaObserver initializes)"
            )
        if self.fallback_observer and is_429:
            logger.warning(
                "[gemini_observer] Gemini 429/RESOURCE_EXHAUSTED, trying open-source fallback"
            )
            try:
                fallback_result = self.fallback_observer.observe(frames, context)
                if fallback_result and fallback_result.observer_type != "mock":
                    logger.info("[gemini_observer] fallback observer succeeded")
                    return fallback_result
                logger.warning("[gemini_observer] fallback returned mock (Ollama unavailable?)")
            except Exception as fallback_e:
                logger.warning(f"[gemini_observer] fallback observer failed: {fallback_e}")

        # Fallback to mock
        logger.error("[gemini_observer] all attempts failed, returning mock")
        return self._mock_observation(observation_id, context)

    def _extract_response_text(self, response) -> str:
        """Safely extract text from Gemini response (handles blocked/empty candidates)."""
        try:
            return response.text or ""
        except (AttributeError, ValueError, KeyError, IndexError, TypeError) as e:
            logger.debug(f"[gemini_observer] response.text failed ({e}), trying candidates")
        try:
            candidates = getattr(response, "candidates", None) or []
            for c in candidates:
                content = getattr(c, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        return t
        except Exception as e:
            logger.debug(f"[gemini_observer] candidate extraction failed: {e}")
        return ""
    
    def _parse_response(
        self,
        raw_text: str,
        observation_id: str,
        context: TaskContext,
        latency_sec: float,
    ) -> ObservationResult:
        """Parse Gemini response into ObservationResult."""
        # Clean JSON from markdown
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            # Try to extract JSON from partial responses
            # Look for JSON object boundaries
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                text = text[start_idx:end_idx + 1]
            
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"[gemini_observer] JSON parse error: {e}")
            logger.error(f"[gemini_observer] Raw response (first 500 chars): {raw_text[:500]}")
            # Return mock observation with uncertain verdict
            mock = self._mock_observation(observation_id, context)
            mock.verdict = "uncertain"
            mock.confidence = 0.3
            mock.constraints_inferred = ["insufficient_evidence", "json_parse_error"]
            return mock

        if not isinstance(data, dict):
            logger.error(f"[gemini_observer] Expected dict, got {type(data).__name__}")
            mock = self._mock_observation(observation_id, context)
            mock.verdict = "uncertain"
            mock.confidence = 0.3
            mock.constraints_inferred = ["insufficient_evidence", "invalid_response_structure"]
            return mock
        
        # Parse characters (ensure we have a dict)
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
        
        # Parse environment
        env_data = data.get("environment", {})
        environment = EnvironmentObservation(
            location_description=env_data.get("location_description"),
            time_of_day=env_data.get("time_of_day"),
            lighting=env_data.get("lighting"),
            mood=env_data.get("mood"),
        )
        
        # Parse action
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
        
        # Parse quality
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
        
        # Parse continuity errors
        # Parse continuity errors
        continuity_errors = []
        for err in data.get("continuity_errors", []):
            try:
                error_type = ContinuityErrorType(err.get("error_type", "temporal_artifact"))
            except ValueError:
                error_type = ContinuityErrorType.TEMPORAL_ARTIFACT
            
            continuity_errors.append(ContinuityError(
                error_type=error_type,
                description=err.get("description", ""),
                severity=err.get("severity", 0.5),
                affected_entities=err.get("affected_entities", []),
            ))
        
        # Parse Phase 7 Causal Analysis
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
                            logger.warning(f"[gemini_observer] Failed to parse evidence: {e}")
            
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
            observer_type="gemini",
            model_version=self.config.gemini_model,
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
        """Return mock observation when Gemini unavailable."""
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
        # This ensures epistemic evaluator will correctly identify missing evidence
        evidence_ledger = EvidenceLedger(beat_id=context.beat_id or observation_id)
        # Mark all common physics evidence as missing for vehicle dynamics
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
        )


# ============================================================================
# Main Observer Agent
# ============================================================================

class VideoObserverAgent:
    """
    Main video observer agent.
    
    Strategy:
    1. Bootstrap with Gemini Vision
    2. Record observations for training
    3. After internalization, use local model with Gemini fallback
    
    Usage:
        observer = VideoObserverAgent()
        context = TaskContext(expected_characters=["saitama"])
        observation = observer.observe(video_url, context)
    """
    
    def __init__(self, config: Optional[ObserverConfig] = None):
        self.config = config or ObserverConfig()
        
        # Initialize components
        self.frame_extractor = FrameExtractor(self.config.frame_extraction_method)

        # Fallback open-source observer (Ollama) when Gemini returns 429
        fallback_observer = None
        _fallback_raw = os.getenv("OBSERVER_FALLBACK_ENABLED", "")
        _fallback_enabled = _fallback_raw.lower() in ("true", "1", "yes")
        if _fallback_enabled:
            try:
                from agents.backends.ollama_observer import (
                    OllamaObserver,
                    OllamaObserverConfig,
                )
                fallback_observer = OllamaObserver(
                    OllamaObserverConfig(
                        base_url=self.config.ollama_base_url,
                        model=self.config.ollama_model,
                    )
                )
                print(f"[video_observer] fallback enabled: ollama {self.config.ollama_model} at {self.config.ollama_base_url}")
                logger.info(
                    f"[video_observer] fallback enabled: ollama {self.config.ollama_model} "
                    f"at {self.config.ollama_base_url}"
                )
            except Exception as e:
                print(f"[video_observer] could not init fallback observer: {e}")
                logger.warning(f"[video_observer] could not init fallback observer: {e}")
        else:
            print(f"[video_observer] fallback disabled: OBSERVER_FALLBACK_ENABLED={repr(_fallback_raw)}")

        self.gemini_observer = GeminiObserver(self.config, fallback_observer=fallback_observer)
        self.local_model = None
        
        # Training data buffer
        self.training_buffer: List[Tuple[str, ObservationResult]] = []
        self._init_training_dir()
        
        # Load local model if available
        if self.config.use_local and self.config.local_model_path:
            self._load_local_model()
        
        logger.info(
            f"[video_observer] initialized: "
            f"gemini={self.config.use_gemini}, local={self.config.use_local}"
        )
    
    def _init_training_dir(self) -> None:
        """Initialize training data directory."""
        if self.config.record_for_training:
            Path(self.config.training_data_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_local_model(self) -> None:
        """Load internalized local model."""
        # Placeholder for local model loading
        # Will be implemented in observer_internalization.py
        pass
    
    def observe(
        self,
        video_path: str,
        context: TaskContext,
    ) -> ObservationResult:
        """
        Observe video and return structured observation.
        
        Args:
            video_path: Path or URL to video
            context: Task context with expectations
            
        Returns:
            ObservationResult with structured observations
        """
        logger.info(f"[video_observer] observing: {video_path[:50]}...")
        try:
            return self._observe_impl(video_path, context)
        except Exception as e:
            logger.error(f"[video_observer] observer failed: {type(e).__name__}: {e}", exc_info=True)
            return self.gemini_observer._mock_observation(str(uuid.uuid4()), context)

    def _observe_impl(
        self,
        video_path: str,
        context: TaskContext,
    ) -> Optional[ObservationResult]:
        """Internal observe implementation (wrapped by observe() for exception handling)."""
        # Download if URL; if download fails, return None (never parse JSON)
        local_path = self._ensure_local(video_path)
        if not local_path or not Path(local_path).exists():
            logger.warning("[video_observer] video unavailable, skipping observer (no parse)")
            return None

        # Extract frames
        frames = self.frame_extractor.extract_frames(
            local_path,
            self.config.frames_to_analyze,
        )
        if not frames:
            logger.warning("[video_observer] no frames extracted, skipping observer (no parse)")
            return None

        # Try local model first if available and confident enough
        if self._should_use_local(context):
            observation = self._observe_local(frames, context)
            if observation and observation.confidence >= self.config.local_confidence_threshold:
                logger.info(f"[video_observer] using local model (confidence: {observation.confidence:.2f})")
                return observation
        
        # Use Gemini
        if self.config.use_gemini:
            # Multi-Observer Logic (Phase 7)
            if self.config.enable_multi_observer and self.config.multi_observer_count > 1:
                observations = []
                # primary
                obs1 = self.gemini_observer.observe(frames, context)
                observations.append(obs1)
                
                # secondary (simulated by re-querying or using different model if avail)
                # For now, we query Gemini again (assuming non-deterministic temp)
                for _ in range(self.config.multi_observer_count - 1):
                    obs_n = self.gemini_observer.observe(frames, context)
                    observations.append(obs_n)
                
                # Consolidate
                return self._consolidate_observations(observations, context)
                
            observation = self.gemini_observer.observe(frames, context)
            
            # Record for training
            if self.config.record_for_training:
                self._record_for_training(video_path, observation)
            
            return observation
        
        # Fallback to mock
        return self.gemini_observer._mock_observation(str(uuid.uuid4()), context)
    
    def _should_use_local(self, context: TaskContext) -> bool:
        """Determine if local model should be tried."""
        if not self.config.use_local or not self.local_model:
            return False
        
        # Don't use local for high-stakes observations
        if context.is_branch_point:
            return False
        
        return True
    
    def _observe_local(
        self,
        frames: List[Tuple[int, bytes]],
        context: TaskContext,
    ) -> Optional[ObservationResult]:
        """Observe using local internalized model."""
        # Placeholder - will be implemented with actual local model
        return None
    
    def _ensure_local(self, video_path: str) -> str:
        """Ensure video is available locally."""
        if video_path.startswith(("http://", "https://")):
            # Download video
            return self._download_video(video_path)
        return video_path
    
    def _download_video(self, url: str) -> Optional[str]:
        """Download video from URL to temp file. Presigned URLs work for private R2."""
        try:
            # Presigned URLs (X-Amz- in query) work with plain GET; try first
            if "X-Amz-" in url or "x-amz-" in url.lower():
                import requests
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(response.content)
                    return f.name
            # Fallback: boto3 for raw S3/R2 URLs when credentials available
            try:
                from urllib.parse import urlparse
                import boto3
                endpoint = os.getenv("S3_ENDPOINT")
                bucket = os.getenv("S3_BUCKET")
                access_key = os.getenv("S3_ACCESS_KEY")
                secret_key = os.getenv("S3_SECRET_KEY")
                region = os.getenv("S3_REGION", "auto")
                if endpoint and bucket and access_key and secret_key:
                    parsed = urlparse(url)
                    path = (parsed.path or "").lstrip("/")
                    if path.startswith(bucket + "/"):
                        key = path[len(bucket) + 1 :]
                    else:
                        key = path.lstrip("/")
                    if "video" in key and not key.endswith(".mp4"):
                        key = f"{key}.mp4"
                    s3 = boto3.client(
                        "s3",
                        endpoint_url=endpoint,
                        region_name=region,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        s3.download_file(bucket, key, f.name)
                        return f.name
            except Exception as ex:
                logger.debug(f"[video_observer] boto3 download failed: {ex}")
                pass
            import requests
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                return f.name
        except Exception as e:
            logger.error(f"[video_observer] download failed: {e}")
            return None
    
    def _record_for_training(
        self,
        video_path: str,
        observation: ObservationResult,
    ) -> None:
        """Record observation for training internalized model."""
        if observation.observer_type != "gemini":
            return  # Only record Gemini observations
        
        self.training_buffer.append((video_path, observation))
        
        # Save to disk periodically
        if len(self.training_buffer) >= 10:
            self._flush_training_buffer()
    
    def _flush_training_buffer(self) -> None:
        """Save training buffer to disk."""
        if not self.training_buffer:
            return
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.training_data_dir) / f"batch_{timestamp}.jsonl"
        
        with open(output_file, "w") as f:
            for video_path, observation in self.training_buffer:
                record = {
                    "video_path": video_path,
                    "observation": observation.to_dict(),
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"[video_observer] saved {len(self.training_buffer)} training samples to {output_file}")
        self.training_buffer.clear()
    
    def _consolidate_observations(
        self,
        observations: List[ObservationResult],
        context: TaskContext,
    ) -> ObservationResult:
        """Consolidate multiple observations into one with disagreement score."""
        if not observations:
            return self.gemini_observer._mock_observation(str(uuid.uuid4()), context)
        
        # Select primary (highest confidence)
        primary = max(observations, key=lambda o: o.confidence)
        
        # Compute disagreement
        disagreement = self._compute_disagreement(observations)
        
        # Collect verdicts and constraints
        all_verdicts = [o.verdict for o in observations if o.verdict]
        all_constraints = set()
        for o in observations:
            all_constraints.update(o.constraints_inferred)
        
        # Create result based on primary, but enriched
        result = primary
        result.disagreement_score = disagreement
        result.verdicts = list(set(all_verdicts))
        result.constraints_inferred = list(all_constraints)
        
        # If high disagreement, force UNCERTAIN verdict or lower confidence
        if disagreement > self.config.disagreement_threshold:
            result.confidence *= (1.0 - disagreement)
            if "uncertain" not in result.verdicts:
                result.verdicts.append("uncertain")
        
        return result

    def _compute_disagreement(self, observations: List[ObservationResult]) -> float:
        """Compute disagreement score (0.0 - 1.0)."""
        if len(observations) < 2:
            return 0.0
            
        score = 0.0
        
        # 1. Verdict Mismatch (Weight: 0.5)
        verdicts = set(o.verdict for o in observations if o.verdict)
        if len(verdicts) > 1:
            score += 0.5
            
        # 2. Action Outcome Mismatch (Weight: 0.3)
        outcomes = set(o.action.outcome if o.action else None for o in observations)
        if len(outcomes) > 1:
            score += 0.3
            
        # 3. Constraint Mismatch (Weight: 0.2)
        # Compare Jaccard similarity of constraints
        constraints_sets = [set(o.constraints_inferred) for o in observations]
        # Pairwise Jaccard
        jaccard_sum = 0.0
        pairs = 0
        for i in range(len(constraints_sets)):
            for j in range(i + 1, len(constraints_sets)):
                s1, s2 = constraints_sets[i], constraints_sets[j]
                union = len(s1.union(s2))
                if union == 0:
                    jaccard_sum += 1.0 # Both empty = match
                else:
                    jaccard_sum += len(s1.intersection(s2)) / union
                pairs += 1
        
        avg_jaccard = jaccard_sum / max(1, pairs)
        score += 0.2 * (1.0 - avg_jaccard)
        
        return min(1.0, score)

    def get_training_sample_count(self) -> int:
        """Get total number of training samples collected."""
        count = len(self.training_buffer)
        
        training_dir = Path(self.config.training_data_dir)
        if training_dir.exists():
            for f in training_dir.glob("*.jsonl"):
                with open(f) as file:
                    count += sum(1 for _ in file)
        
        return count


# ============================================================================
# Convenience functions
# ============================================================================

def create_observer(
    use_gemini: bool = True,
    use_local: bool = False,
    local_model_path: Optional[str] = None,
) -> VideoObserverAgent:
    """
    Factory function to create observer with common configurations.
    
    Args:
        use_gemini: Whether to use Gemini API
        use_local: Whether to use local internalized model
        local_model_path: Path to local model weights
        
    Returns:
        Configured VideoObserverAgent
    """
    config = ObserverConfig(
        use_gemini=use_gemini,
        use_local=use_local,
        local_model_path=local_model_path,
    )
    return VideoObserverAgent(config)


def quick_observe(video_path: str, expected_characters: List[str] = None) -> ObservationResult:
    """
    Quick observation with minimal configuration.
    
    Args:
        video_path: Path or URL to video
        expected_characters: List of expected character IDs
        
    Returns:
        ObservationResult
    """
    observer = VideoObserverAgent()
    context = TaskContext(
        expected_characters=expected_characters or [],
    )
    return observer.observe(video_path, context)
