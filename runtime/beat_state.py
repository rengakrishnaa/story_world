from enum import Enum


class BeatState(str, Enum):
    """
    Beat lifecycle in a GPU-job–based execution model.

    IMPORTANT:
    - GPU workers NEVER change beat state directly
    - Only control plane transitions states
    """

    # Beat exists but has not been submitted to GPU
    PENDING = "PENDING"

    # Beat has an active GPU job in-flight
    EXECUTING = "EXECUTING"

    # Beat completed successfully and accepted by policies
    ACCEPTED = "ACCEPTED"

    # Beat permanently failed after retries / policy abort
    ABORTED = "ABORTED"
    
    # ==========================================================================
    # Epistemic States (MANDATORY - No silent fallback, no probabilistic guessing)
    # ==========================================================================
    
    # Evidence insufficient - cannot evaluate constraints
    EPISTEMICALLY_INCOMPLETE = "EPISTEMICALLY_INCOMPLETE"
    
    # Progression blocked by unresolved uncertainty
    UNCERTAIN_TERMINATION = "UNCERTAIN_TERMINATION"