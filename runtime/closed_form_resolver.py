"""
Cheap Path for Closed-Form Goals

When intent does not require visual verification, answer via LLM without GPU render.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ClosedFormResult:
    """Result from LLM-only closed-form resolution."""
    feasible: bool
    confidence: float
    explanation: str
    constraints_inferred: list
    source: str = "llm_closed_form"


def resolve_closed_form(intent: str) -> Optional[ClosedFormResult]:
    """
    Resolve a closed-form goal via LLM without video.
    Returns None if LLM unavailable.
    """
    try:
        from google import genai
    except ImportError:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    prompt = f"""You are a physics simulation advisor. This goal is fully specified (no video needed).
Goal: "{intent}"

Answer:
1. Is this physically feasible? (true/false)
2. Confidence 0.0-1.0
3. Brief explanation
4. Any physical constraints this scenario implies (list)

Return JSON only: {{"feasible": true/false, "confidence": 0.9, "explanation": "...", "constraints_inferred": ["c1", "c2"]}}"""

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = (getattr(resp, "text", None) or "").strip()
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            return ClosedFormResult(
                feasible=bool(data.get("feasible", True)),
                confidence=float(data.get("confidence", 0.8)),
                explanation=str(data.get("explanation", "")),
                constraints_inferred=data.get("constraints_inferred") or [],
            )
    except Exception as e:
        logger.warning(f"[closed_form] LLM failed: {e}")
    return None
