# runtime/event_log.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
import json

@dataclass
class RuntimeEvent:
    episode_id: str
    type: str
    payload: Dict[str, Any]
    ts: datetime

class EventLog:
    def __init__(self, db):
        self.db = db

    def append(self, event: RuntimeEvent):
        self.db.insert(
            "event_log",
            {
                "episode_id": event.episode_id,
                "type": event.type,
                "payload": json.dumps(event.payload),
                "ts": event.ts.isoformat(),
            },
        )

    def replay(self, episode_id: str):
        return self.db.select(
            "event_log",
            where={"episode_id": episode_id},
            order_by="ts ASC",
        )
