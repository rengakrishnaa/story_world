class StubPlanner:
    """
    Minimal deterministic planner used ONLY to validate runtime.
    """

    def plan_episode(self, world_json, script):
        # Return a fake but valid structure
        return {
            "beats": [
                {
                    "id": "beat-1",
                    "description": script,
                    "characters": [],
                    "location": "nowhere",
                    "estimated_duration_sec": 5,
                }
            ]
        }
