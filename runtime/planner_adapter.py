from runtime.planner_interface import PlannerInterface

class PlannerAdapter(PlannerInterface):
    def __init__(self, narrative_planner):
        self.planner = narrative_planner

    def generate_beats(self, intent: str):
        plan = self.planner.plan_episode(
            world_json={},    # will expand later
            script=intent,
        )

        beats = []
        for act in plan.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    beats.append({
                        "id": beat.id,
                        "description": beat.description,
                        "characters": beat.characters,
                        "location": beat.location,
                        "estimated_duration_sec": beat.estimated_duration_sec,
                        "motion_type": "character" if beat.estimated_duration_sec > 0 else "static",
                    })


        return beats
