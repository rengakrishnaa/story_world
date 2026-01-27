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
