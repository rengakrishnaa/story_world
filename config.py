"""
Central configuration for StoryWorld.
All production values come from environment variables with sensible defaults.
"""

import os
from typing import Optional


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Redis & Queues
# ---------------------------------------------------------------------------
GPU_JOB_QUEUE = os.getenv("GPU_JOB_QUEUE", os.getenv("JOB_QUEUE", "storyworld:gpu:jobs"))
GPU_RESULT_QUEUE = os.getenv(
    "GPU_RESULT_QUEUE",
    os.getenv("RESULT_QUEUE", "storyworld:gpu:results"),
)
GPU_RESULT_QUEUE_PREFIX = os.getenv("GPU_RESULT_QUEUE_PREFIX")  # Optional per-episode suffix

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
USE_MOCK_PLANNER = _env_bool("USE_MOCK_PLANNER", False)
ENV = os.getenv("ENV", "local").lower()

# ---------------------------------------------------------------------------
# Reality Compiler Mode (product contract)
# ---------------------------------------------------------------------------
# When True: no cinematic/story specs, neutral style, observer-driven retry
REALITY_COMPILER_MODE = _env_bool("REALITY_COMPILER_MODE", True)

# Run observer on each successful render; populate constraints_discovered
USE_OBSERVER_IN_PRODUCTION = _env_bool("USE_OBSERVER_IN_PRODUCTION", True)


# ---------------------------------------------------------------------------
# Simulation Policies (used by /simulate and /episodes)
# ---------------------------------------------------------------------------
def get_simulation_policies() -> dict:
    """Policies for /simulate endpoint (stricter for production simulation)."""
    return {
        "retry": {
            "max_attempts": _env_int("SIM_RETRY_MAX_ATTEMPTS", 3),
        },
        "quality": {
            "min_confidence": _env_float("SIM_QUALITY_MIN_CONFIDENCE", 0.8),
        },
        "cost": {
            "max_cost": _env_float("SIM_COST_MAX_USD", 10.0),
        },
    }


def get_episode_policies() -> dict:
    """Policies for /episodes endpoint."""
    return {
        "retry": {
            "max_attempts": _env_int("EP_RETRY_MAX_ATTEMPTS", 2),
        },
        "quality": {
            "min_confidence": _env_float("EP_QUALITY_MIN_CONFIDENCE", 0.7),
        },
        "cost": {
            "max_cost": _env_float("EP_COST_MAX_USD", 5.0),
        },
    }


# ---------------------------------------------------------------------------
# Cost & Confidence
# ---------------------------------------------------------------------------
COST_PER_BEAT_USD = _env_float("COST_PER_BEAT_USD", 0.05)
DEFAULT_SUCCESS_CONFIDENCE = _env_float("DEFAULT_SUCCESS_CONFIDENCE", 0.85)
EPISODE_COMPOSE_REQUIRED_CONFIDENCE = _env_float(
    "EPISODE_COMPOSE_REQUIRED_CONFIDENCE", 0.3
)

# ---------------------------------------------------------------------------
# Paths & Logging
# ---------------------------------------------------------------------------
# On Vercel/Fly, /tmp is writable (ephemeral)
_use_tmp = os.getenv("VERCEL") or os.getenv("FLY_APP_NAME")
_default_log = "/tmp/decision_loop.log" if _use_tmp else "decision_loop.log"
DECISION_LOOP_LOG_PATH = os.getenv(
    "DECISION_LOOP_LOG_PATH",
    os.getenv("LOG_PATH", _default_log),
)
# Vercel/Fly: use /tmp for writable SQLite (ephemeral)
_default_db = "sqlite:////tmp/storyworld.db" if _use_tmp else "sqlite:///./local.db"
DATABASE_URL = os.getenv("DATABASE_URL", _default_db)
