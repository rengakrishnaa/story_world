
import pytest
from unittest.mock import MagicMock
from runtime.episode_runtime import EpisodeRuntime
from runtime.beat_state import BeatState

def test_gpu_job_payload_structure():
    # 1. Setup Mock SQL & Beat Data
    mock_sql = MagicMock()
    
    beat_id = "beat-1"
    beat_data = {
        "id": beat_id,
        "spec": {
            "backend": "veo",
            "description": "A robot arm moves to the left",
            "motion_strength": 0.5
        },
        "duration_sec": 4.0
    }
    
    mock_sql.get_beat.return_value = beat_data
    mock_sql.count_attempts.return_value = 0
    
    # 2. Initialize Runtime
    runtime = EpisodeRuntime("sim-test", "world-1", "intent", {}, mock_sql)
    
    # 3. Build Job
    job = runtime.build_gpu_job(beat_id, "job-123")
    
    # 4. Assert Structure (Critical for Worker Compatibility)
    assert job["job_id"] == "job-123"
    assert job["backend"] == "veo"
    
    # Check Input Spec
    inp = job["input"]
    assert inp["prompt"] == "A robot arm moves to the left"
    assert inp["duration_sec"] == 4.0
    assert "motion" in inp
    assert inp["motion"]["engine"] == "sparse"
    assert inp["motion"]["params"]["strength"] == 0.5
    
    # Check Meta (used for result routing)
    assert job["meta"]["episode_id"] == "sim-test"
    assert job["meta"]["beat_id"] == "beat-1"
    
def test_gpu_job_animatediff_fallback():
    # Test compatibility fix for Animatediff without frames
    mock_sql = MagicMock()
    beat_data = {
        "id": "beat-2",
        "spec": {
            "backend": "animatediff",
            "description": "foo"
            # Missing start_frame/end_frame
        }
    }
    mock_sql.get_beat.return_value = beat_data
    
    runtime = EpisodeRuntime("sim-test", "world-1", "intent", {}, mock_sql)
    job = runtime.build_gpu_job("beat-2", "job-456")
    
    # Should fallback to SVD
    assert job["backend"] == "svd"
