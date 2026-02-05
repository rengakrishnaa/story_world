"""
Production Intent Classifier - Gemini → Ollama → Rule-based

Fallback chain (same as observer):
1. Gemini (primary)
2. Ollama (when Gemini 429 / exhausted) - text model, same prompt
3. Rule-based (when both fail) - deterministic, never escalates epistemic requirements

CRITICAL: LLM failure must NEVER escalate epistemic requirements.
Rule-based is authority when all LLMs fail.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Cache size for production (per-process)
CLASSIFIER_CACHE_SIZE = int(os.getenv("INTENT_CLASSIFIER_CACHE_SIZE", "2000"))

# Problem domains for constraint selection (no keyword matching)
PROBLEM_DOMAIN_VEHICLE = "vehicle_dynamics"
PROBLEM_DOMAIN_STATICS = "statics"
PROBLEM_DOMAIN_STRUCTURAL = "structural"
PROBLEM_DOMAIN_FLUID = "fluid"
PROBLEM_DOMAIN_GENERIC = "generic"


@dataclass
class IntentClassificationResult:
    """Structured result from intent classification. Audit-ready."""
    requires_visual_verification: bool
    confidence: float
    problem_domain: str
    extracted_evidence: Dict[str, Any] = field(default_factory=dict)
    source: str = "llm"  # "llm" | "fallback" | "override"
    reasoning: str = ""

    def observer_impact(self) -> str:
        return "blocking" if self.requires_visual_verification else "confidence_only"


def _normalize_intent_for_cache(intent: str) -> str:
    """Normalize intent for cache key. Preserve meaning, collapse whitespace."""
    if not intent:
        return ""
    return " ".join(intent.strip().lower().split())[:2000]


def _cache_key(intent: str, override_hash: str) -> str:
    return hashlib.sha256((_normalize_intent_for_cache(intent) + "|" + override_hash).encode()).hexdigest()


def _call_llm_classify(intent: str) -> Optional[IntentClassificationResult]:
    """
    Call Gemini for intent classification. Returns None on failure.
    """
    try:
        from google import genai
    except ImportError:
        logger.warning("[intent_classifier] google-genai not installed")
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.debug("[intent_classifier] GEMINI_API_KEY not set")
        return None

    model = os.getenv("INTENT_CLASSIFIER_MODEL", "gemini-2.0-flash")
    prompt = _classification_prompt(intent)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = (getattr(response, "text", None) or str(response)).strip()
        return _parse_classification_response(text)
    except json.JSONDecodeError as e:
        logger.warning(f"[intent_classifier] JSON parse error: {e}")
        return None
    except Exception as e:
        err_str = (str(e) + " " + repr(e)).upper()
        is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "RATE" in err_str
        if is_429:
            logger.info(f"[intent_classifier] Gemini 429, trying Ollama fallback: {e}")
        else:
            logger.warning(f"[intent_classifier] LLM error: {type(e).__name__}: {e}")
        return None


def _classification_prompt(intent: str) -> str:
    """Shared prompt for Gemini and Ollama."""
    return f"""You are a physics simulation classifier. Analyze this user intent and determine:

1. Does this problem REQUIRE perceptual evidence from video observation to evaluate?
   - YES (requires_visual_verification=true): The problem involves dynamics that cannot be fully specified from text alone. Examples: vehicle turning, motion at unknown speed, fluid flow, collisions with unspecified masses.
   - NO (requires_visual_verification=false): The problem is fully specified with numeric physical parameters. Examples: stacking boxes with given mass/dimensions/friction/gravity, statics with known loads.

2. What is the problem domain? One of: vehicle_dynamics, statics, structural, fluid, generic

3. What physical quantities are explicitly specified in the intent? Extract as JSON: mass_kg, dimensions_m, friction_coef, gravity_ms2, velocity_ms, angular_velocity_rads, etc. Use null for unspecified.

Intent:
{intent[:3000]}

Return ONLY valid JSON, no markdown:
{{
  "requires_visual_verification": true or false,
  "confidence": 0.0 to 1.0,
  "problem_domain": "vehicle_dynamics" or "statics" or "structural" or "fluid" or "generic",
  "extracted_evidence": {{ "mass_kg": 1.5, "friction_coef": 0.6, ... }},
  "reasoning": "One sentence explaining the classification"
}}"""


def _parse_classification_response(text: str) -> Optional[IntentClassificationResult]:
    """Parse LLM response into IntentClassificationResult. Returns None on parse failure."""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        data = json.loads(text)
        requires = bool(data.get("requires_visual_verification", True))
        confidence = float(data.get("confidence", 0.7))
        domain = str(data.get("problem_domain", "generic")).lower()
        if domain not in (PROBLEM_DOMAIN_VEHICLE, PROBLEM_DOMAIN_STATICS, PROBLEM_DOMAIN_STRUCTURAL, PROBLEM_DOMAIN_FLUID, PROBLEM_DOMAIN_GENERIC):
            domain = PROBLEM_DOMAIN_GENERIC
        evidence = data.get("extracted_evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        reasoning = str(data.get("reasoning", ""))[:500]
        return IntentClassificationResult(
            requires_visual_verification=requires,
            confidence=confidence,
            problem_domain=domain,
            extracted_evidence=evidence,
            source="llm",
            reasoning=reasoning,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _call_ollama_classify(intent: str) -> Optional[IntentClassificationResult]:
    """
    Ollama fallback when Gemini returns 429. Uses text model (llama2, mistral, etc.).
    Same prompt, same JSON output. Returns None on failure.
    """
    fallback_enabled = os.getenv("INTENT_CLASSIFIER_FALLBACK_ENABLED", os.getenv("OBSERVER_FALLBACK_ENABLED", "")).lower() in ("true", "1", "yes")
    if not fallback_enabled:
        return None

    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    model = os.getenv("INTENT_CLASSIFIER_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL", "llama2")
    if model == "llava":
        model = "llama2"
    timeout = float(os.getenv("INTENT_CLASSIFIER_OLLAMA_TIMEOUT", "15"))

    prompt = _classification_prompt(intent)

    try:
        import urllib.request
        import urllib.error

        url = f"{base_url}/api/generate"
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        text = (data.get("response") or "").strip()
        if not text:
            return None
        result = _parse_classification_response(text)
        if result:
            result.source = "ollama"
            result.reasoning = f"Ollama fallback: {result.reasoning}"
            logger.info(f"[intent_classifier] Ollama fallback succeeded (model={model})")
        return result
    except Exception as e:
        logger.warning(f"[intent_classifier] Ollama fallback failed: {type(e).__name__}: {e}")
        return None


def _extract_physical_evidence(intent: str) -> Set[str]:
    """
    Deterministic extraction of physical quantities from intent text.
    Returns set of provided evidence names. LLM failure does NOT enter this.
    """
    text = (intent or "").lower()
    provided: Set[str] = set()

    if re.search(r"\d+\.?\d*\s*(kg|g)\b", text):
        provided.add("mass")
    if re.search(r"\d+\.?\d*\s*(cm|mm|m)\b", text):
        provided.add("dimensions")
    if "friction" in text and re.search(r"\d+\.?\d+", text):
        provided.add("friction")
    if "gravity" in text or "9.8" in text or "m/s" in text:
        provided.add("gravity")
    if "zero angular" in text or "angular velocity" in text or "gently" in text:
        provided.add("angular_spec")
    if "without tipping" in text or "no tipping" in text:
        provided.add("stability_constraint")
    if "flat surface" in text or "flat" in text:
        provided.add("surface")

    return provided


def _infer_required_for_domain(intent: str) -> Set[str]:
    """
    Infer required physical variables based on problem type (from intent content).
    Deterministic: no LLM.
    """
    text = (intent or "").lower()

    if any(k in text for k in ("vehicle", "car", "turn", "corner", "steer", "sharp turn", "speed", "velocity", "dynamics", "slip", "yaw", "tire", "lateral")):
        return {"speed_profile", "turn_radius", "friction"}
    if any(k in text for k in ("stack", "stacking", "boxes", "box", "without tipping")):
        return {"mass", "dimensions", "friction", "gravity"}
    if any(k in text for k in ("structural", "load", "bending", "collapse")):
        return {"mass", "dimensions", "load"}
    if any(k in text for k in ("fluid", "flow", "viscosity")):
        return {"velocity", "viscosity"}

    return {"mass", "dimensions", "gravity"}


def _rule_based_requires_visual(intent: str) -> tuple[bool, str, str]:
    """
    Deterministic rule-based classification. Used when LLM fails.
    Returns (requires_visual, problem_domain, reasoning).
    """
    provided = _extract_physical_evidence(intent)
    required = _infer_required_for_domain(intent)

    text = (intent or "").lower()
    is_statics = any(k in text for k in ("stack", "stacking", "boxes", "box", "without tipping"))
    has_sufficient_spec = (provided & {"mass", "dimensions"}) and (provided & {"friction", "gravity"})

    if is_statics and has_sufficient_spec:
        return False, PROBLEM_DOMAIN_STATICS, "Rule-based: statics with mass, dimensions, friction/gravity specified; solver sufficient"
    if required <= provided:
        return False, PROBLEM_DOMAIN_GENERIC, "Rule-based: required variables provided in intent; solver sufficient"
    if any(k in text for k in ("vehicle", "car", "turn", "corner", "speed", "dynamics")):
        return True, PROBLEM_DOMAIN_VEHICLE, "Rule-based: vehicle/dynamics domain; perceptual evidence required"
    if has_sufficient_spec and is_statics:
        return False, PROBLEM_DOMAIN_STATICS, "Rule-based: statics with numeric spec; solver sufficient"
    return True, PROBLEM_DOMAIN_GENERIC, "Rule-based: insufficient deterministic evidence; assume perceptual required"


def _fallback_classify(intent: str) -> IntentClassificationResult:
    """
    Rule-based fallback when LLM unavailable.
    LLM failure must NEVER escalate epistemic requirements.
    Uses deterministic evidence extraction - same authority as solver.
    """
    requires, domain, reasoning = _rule_based_requires_visual(intent)
    provided = _extract_physical_evidence(intent)
    evidence_dict = {k: True for k in provided}
    return IntentClassificationResult(
        requires_visual_verification=requires,
        confidence=0.5 if not requires else 0.3,
        problem_domain=domain,
        extracted_evidence=evidence_dict,
        source="fallback",
        reasoning=f"LLM unavailable; {reasoning}",
    )


def classify_intent(
    intent: str,
    *,
    override_requires_visual: Optional[bool] = None,
    override_problem_domain: Optional[str] = None,
    use_cache: bool = True,
) -> IntentClassificationResult:
    """
    Classify user intent. Production-grade: LLM-based, cached, auditable.

    Args:
        intent: User goal/description (e.g., "A robotic arm stacks three boxes...")
        override_requires_visual: If set, bypass LLM for this field (API override)
        override_problem_domain: If set, bypass LLM for constraint selection
        use_cache: Whether to use in-memory cache (disable for testing)

    Returns:
        IntentClassificationResult with requires_visual_verification, problem_domain, etc.
    """
    if not intent or not str(intent).strip():
        return IntentClassificationResult(
            requires_visual_verification=True,
            confidence=0.0,
            problem_domain=PROBLEM_DOMAIN_GENERIC,
            source="fallback",
            reasoning="Empty intent; no evidence to extract; assume perceptual required",
        )

    intent_str = str(intent).strip()

    # Explicit override from API
    if override_requires_visual is not None or override_problem_domain is not None:
        return IntentClassificationResult(
            requires_visual_verification=override_requires_visual if override_requires_visual is not None else True,
            confidence=1.0,
            problem_domain=override_problem_domain or PROBLEM_DOMAIN_GENERIC,
            source="override",
            reasoning="Explicit API override",
        )

    if use_cache:
        cache_key = _normalize_intent_for_cache(intent_str) or "__empty__"
        return _classify_cached(cache_key)
    return _classify_uncached(intent_str)


def _classify_uncached(intent: str) -> IntentClassificationResult:
    result = _call_llm_classify(intent)
    if result is None:
        result = _call_ollama_classify(intent)
    if result is None:
        return _fallback_classify(intent)
    return result


@lru_cache(maxsize=CLASSIFIER_CACHE_SIZE)
def _classify_cached(cache_key: str) -> IntentClassificationResult:
    """
    Cached classification. Key is normalized intent.
    LRU evicts oldest when full.
    """
    return _classify_uncached(cache_key if cache_key != "__empty__" else "")


def clear_classifier_cache() -> None:
    """Clear in-memory cache (e.g., for testing or config change)."""
    _classify_cached.cache_clear()
