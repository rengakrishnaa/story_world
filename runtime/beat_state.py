from enum import Enum

class BeatState(str, Enum):
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    EXECUTING = "EXECUTING"
    OBSERVED = "OBSERVED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ABORTED = "ABORTED"
