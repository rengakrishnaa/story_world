from enum import Enum


class EpisodeState(str, Enum):
    """
    Episode lifecycle driven by beat completion.
    """

    # Created but not planned
    CREATED = "CREATED"

    # Beats generated, nothing executed yet
    PLANNED = "PLANNED"

    # At least one beat is executing or pending execution
    EXECUTING = "EXECUTING"

    # Some beats succeeded, some aborted, episode still valid
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"

    # All beats completed successfully
    COMPLETED = "COMPLETED"

    # Episode execution failed irrecoverably
    FAILED = "FAILED"

    # Observer vetoed intent as impossible
    IMPOSSIBLE = "IMPOSSIBLE"

    # No valid actions remain
    DEAD_STATE = "DEAD_STATE"

    # Budget exhausted / abandoned
    ABANDONED = "ABANDONED"
    
    # ==========================================================================
    # Epistemic States (MANDATORY - No silent fallback, no probabilistic guessing)
    # ==========================================================================
    
    # Evidence sufficient, constraints satisfied
    ACCEPTED = "ACCEPTED"
    
    # Evidence sufficient, constraints violated
    REJECTED = "REJECTED"
    
    # Evidence insufficient - cannot evaluate constraints
    EPISTEMICALLY_INCOMPLETE = "EPISTEMICALLY_INCOMPLETE"
    
    # Progression blocked by unresolved uncertainty
    UNCERTAIN_TERMINATION = "UNCERTAIN_TERMINATION"
    
    # Episode-level: any beat epistemically incomplete → execution impossible
    # Cannot evaluate constraints, time progression not meaningful, transitions blocked
    EPISTEMICALLY_BLOCKED = "EPISTEMICALLY_BLOCKED"
