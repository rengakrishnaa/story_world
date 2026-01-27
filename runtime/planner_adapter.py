from runtime.planner_interface import PlannerInterface

class PlannerAdapter(PlannerInterface):
    def __init__(self, narrative_planner, redis_client=None):
        self.planner = narrative_planner
        self.redis = redis_client

    def generate_beats(self, intent: str):
        """
        Generate beats from user intent.
        
        The narrative planner's generate_beats() method handles:
        - Loading world data from world_id
        - Generating episode plan with Gemini/mock
        - Flattening to beat list
        - Adding backend selection
        
        Returns list of beat dictionaries ready for episode runtime.
        """
        # Set redis client for caching
        if self.redis:
            self.planner.redis = self.redis
        
        # Call the planner's generate_beats method
        # This handles world loading, planning, and beat generation
        beats = self.planner.generate_beats(intent)
        
        return beats

