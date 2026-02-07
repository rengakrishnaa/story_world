"""
Risk Profile Adjuster

Applies risk_profile (low/medium/high) to simulation policies.
- Conservative (low): stricter quality, fewer retries, tighter budget
- Balanced (medium): defaults
- Exploratory (high): looser quality, more retries, higher budget tolerance
"""

from typing import Dict, Any


def apply_risk_profile(policies: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust retry, quality, and cost policies based on risk_profile.
    Mutates and returns the policies dict.
    """
    risk = (policies.get("risk_profile") or "medium").lower()
    if risk not in ("low", "medium", "high"):
        risk = "medium"

    retry = policies.get("retry") or {}
    quality = policies.get("quality") or {}
    cost = policies.get("cost") or {}

    if risk == "low":
        # Conservative: stricter quality, fewer retries
        base_ma = retry.get("max_attempts", 4)
        retry = dict(retry)
        retry["max_attempts"] = max(2, base_ma - 2)
        limits = dict(retry.get("limits", {}))
        for k in limits:
            limits[k] = max(1, limits[k] - 1)
        retry["limits"] = limits

        base_q = quality.get("min_confidence", 0.8)
        quality = dict(quality)
        quality["min_confidence"] = min(0.95, base_q + 0.1)

        base_c = cost.get("max_cost", 10.0)
        cost = dict(cost)
        cost["max_cost"] = base_c * 0.8

    elif risk == "high":
        # Exploratory: looser quality, more retries
        base_ma = retry.get("max_attempts", 4)
        retry = dict(retry)
        retry["max_attempts"] = base_ma + 2
        limits = dict(retry.get("limits", {}))
        for k in limits:
            limits[k] = limits[k] + 1
        retry["limits"] = limits

        base_q = quality.get("min_confidence", 0.8)
        quality = dict(quality)
        quality["min_confidence"] = max(0.5, base_q - 0.2)

        base_c = cost.get("max_cost", 10.0)
        cost = dict(cost)
        cost["max_cost"] = base_c * 1.2

    policies = dict(policies)
    policies["retry"] = retry
    policies["quality"] = quality
    policies["cost"] = cost
    return policies
