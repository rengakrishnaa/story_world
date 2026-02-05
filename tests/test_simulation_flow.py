from fastapi.testclient import TestClient
from unittest import mock
from unittest.mock import MagicMock
import main
from main import app

# Mock the SQL store globally for the test
main.sql = MagicMock()
main.sql.create_episode.return_value = None 

# Define a side effect to handle different IDs
def get_episode_side_effect(episode_id):
    if episode_id == "sim-non-existent-999":
        return None
    return {
        "status": "created",
        "world_id": "default",
        "intent": "Test Intent",
        "policies": {},
        "state": "initialized"
    }

main.sql.get_episode.side_effect = get_episode_side_effect

client = TestClient(app)

class TestSimulationFlowFixes:
    
    def test_invalid_episode_returns_404(self):
        """Verify that requesting a non-existent episode returns 404, not 500."""
        response = client.get("/episodes/sim-non-existent-999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Simulation not found"

    def test_simulate_creation(self):
        """Verify that /simulate endpoint creates a valid session."""
        goal = "Test robot arm stacking"
        response = client.post(f"/simulate?world_id=default&goal={goal}")
        
        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert data["status"] == "executing"
        
        # Verify we can fetch it now
        sim_id = data["simulation_id"]
        status_res = client.get(f"/episodes/{sim_id}")
        assert status_res.status_code == 200

    @mock.patch("main.redis_store")
    @mock.patch("agents.narrative_planner.ProductionNarrativePlanner")
    def test_simulate_full_pipeline(self, mock_planner_cls, mock_redis):
        # Setup Planner Mock
        mock_planner_instance = mock_planner_cls.return_value
        mock_planner_instance.generate_beats.return_value = [{
            "id": "beat-1", 
            "description": "foo", 
            "duration_sec": 5,
        }]

        # Setup SQL Mock for BEATS
        # 1. get_beats_by_state must return the pending beat
        main.sql.get_beats_by_state.return_value = [{
            "id": "beat-1",
            "state": "PENDING"
        }]
        
        # 2. get_beat must return the full spec for build_gpu_job
        main.sql.get_beat.return_value = {
            "id": "beat-1",
            "state": "PENDING",
            "spec": {
                "backend": "stub",
                "description": "foo",
                "motion_strength": 0.5
            },
            "duration_sec": 5.0
        }

        # Setup Redis Mock
        mock_redis.push_gpu_job = mock.MagicMock()

        # Call Endpoint
        response = client.post("/simulate?world_id=default&goal=test")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "executing"
        
        # Verify Planner was called
        mock_planner_instance.generate_beats.assert_called_once()
        
        # Verify Redis push happened
        mock_redis.push_gpu_job.assert_called()
        call_args = mock_redis.push_gpu_job.call_args[0][0]
        assert call_args["job_id"]
        assert call_args["input"]["prompt"] == "foo"
            
        # Verify we can fetch it now
        sim_id = data["simulation_id"]
        status_res = client.get(f"/episodes/{sim_id}")
        assert status_res.status_code == 200
