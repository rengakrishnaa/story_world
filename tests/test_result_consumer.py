
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import main
from main import app

client = TestClient(app)

# We need to test that the ResultConsumer (background task)
# correctly picks up a job from Redis and updates SQL.
# Since ResultConsumer runs in background loop, testing it with TestClient
# is tricky because TestClient doesn't run the full async loop the same way as uvicorn.
# However, we can unit test the consume ONE pass logic.

@pytest.mark.asyncio
async def test_result_consumer_logic():
    # 1. Setup Mocks
    mock_sql = MagicMock()
    mock_redis = MagicMock()
    
    # 2. Setup standard ResultConsumer
    from runtime.result_consumer import ResultConsumer
    consumer = ResultConsumer(mock_sql, mock_redis)
    
    # 3. Simulate a result in Redis
    fake_result = {
        "job_id": "test-job",
        "status": "success",
        "meta": {"episode_id": "sim-test", "beat_id": "beat-1"},
        "artifacts": {"video": "s3://foo"},
        "runtime": {"cost": 0.5}
    }
    
    # Mock redis pop to return this result once
    mock_redis.pop_gpu_result.return_value = fake_result
    
    # Mock SQL logic
    mock_runtime_instance = MagicMock()
    # When loading runtime, return our mock wrapper
    with pytest.helpers.mock_episode_runtime(mock_runtime_instance): 
        # CAUTION: We need to patch logic or reuse runtime logic. 
        # Simpler: just unit test _process_result directly
        
        # 4. Trigger processing manually
        consumer._process_result(fake_result)
        
        # 5. Assertions
        # It should try to load the runtime via EpisodeRuntime.load
        # And call mark_beat_success
        
        # Since we didn't patch EpisodeRuntime.load globally yet in this test function scope,
        # let's proceed to integration style test if possible, or refined unit test.
        pass

# Simplified Approach: Test the _process_result logic directly with patched Runtime
from unittest.mock import patch

def test_process_result_success():
    mock_sql = MagicMock()
    mock_redis = MagicMock()
    from runtime.result_consumer import ResultConsumer
    consumer = ResultConsumer(mock_sql, mock_redis)
    
    fake_result = {
        "status": "success",
        "meta": {"episode_id": "sim-123", "beat_id": "beat-45"},
        "artifacts": {"v": "path"},
        "runtime": {"c": 1}
    }

    with patch("runtime.result_consumer.EpisodeRuntime") as MockRuntime:
        mock_instance = MockRuntime.load.return_value
        
        consumer._process_result(fake_result)
        
        MockRuntime.load.assert_called_with("sim-123", mock_sql)
        mock_instance.mark_beat_success.assert_called_once()
        args = mock_instance.mark_beat_success.call_args
        assert args[1]["beat_id"] == "beat-45"

def test_process_result_failure():
    mock_sql = MagicMock()
    mock_redis = MagicMock()
    from runtime.result_consumer import ResultConsumer
    consumer = ResultConsumer(mock_sql, mock_redis)
    
    fake_result = {
        "status": "failure",
        "meta": {"episode_id": "sim-123", "beat_id": "beat-45"},
        "error": {"message": "GPU OOM"},
        "runtime": {}
    }

    with patch("runtime.result_consumer.EpisodeRuntime") as MockRuntime:
        mock_instance = MockRuntime.load.return_value
        
        consumer._process_result(fake_result)
        
        mock_instance.mark_beat_failure.assert_called_once()
        args = mock_instance.mark_beat_failure.call_args[1] # kwargs
        # method signature is (beat_id, error, metrics)
        # check args[0] or kwargs
        call_args = mock_instance.mark_beat_failure.call_args
        assert call_args.kwargs['beat_id'] == 'beat-45'
        assert call_args.kwargs['error'] == 'GPU OOM'
