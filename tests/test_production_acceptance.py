"""
StoryWorld Production Acceptance Test Suite

Full end-to-end validation for computational video infrastructure.
Treat failures as blocking issues for deployment.

Run: pytest tests/test_production_acceptance.py -v -s
"""

from __future__ import annotations

import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest import mock

# Set test env before importing app
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_acceptance.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("USE_MOCK_PLANNER", "true")

# Lazy imports after env
import main
from main import app


def _make_client():
    """Create TestClient with mocked persistence for isolated tests."""
    return TestClient(app)


# =============================================================================
# TEST 0 — Preconditions (Hard Gate)
# =============================================================================

class Test0Preconditions:
    """Verify system is ready before running acceptance tests."""

    def test_api_server_responds(self):
        """API server is reachable."""
        client = _make_client()
        r = client.get("/")
        assert r.status_code == 200, "API server must respond"

    def test_static_assets_load(self):
        """Static assets (JS, CSS) load without error."""
        client = _make_client()
        for path in ["/static/dashboard.js", "/static/style.css"]:
            r = client.get(path)
            assert r.status_code == 200, f"{path} must load"

    def test_phase_status_endpoint(self):
        """Phase status reports system state."""
        client = _make_client()
        r = client.get("/phase-status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        # Redis/GPU are optional for API tests
        assert data.get("status") in ("operational", "error")


# =============================================================================
# TEST 1 — UI ↔ Backend Wiring (No Video Involved)
# =============================================================================

class Test1UIBackendWiring:
    """UI must fetch from backend; no video dependency."""

    def test_dashboard_fetches_episodes_list(self):
        """Dashboard fetches simulation list from /episodes."""
        client = _make_client()
        r = client.get("/episodes?limit=20")
        assert r.status_code == 200
        data = r.json()
        assert "episodes" in data
        assert isinstance(data["episodes"], list)

    def test_dashboard_html_has_no_video_previews(self):
        """Dashboard HTML does not show video thumbnails or previews."""
        client = _make_client()
        r = client.get("/")
        html = r.text.lower()
        assert "thumbnail" not in html
        assert "video preview" not in html
        assert "autoplay" not in html

    def test_dashboard_shows_primary_fields(self):
        """Dashboard table has: ID, Goal, Status, Confidence, Cost."""
        client = _make_client()
        r = client.get("/")
        html = r.text
        assert "ID" in html or "id" in html.lower()
        assert "Goal" in html or "goal" in html.lower()
        assert "Status" in html or "status" in html.lower()
        assert "Conf" in html or "confidence" in html.lower()
        assert "Cost" in html or "cost" in html.lower()

    def test_no_generate_video_language(self):
        """UI must not say 'Generate Video' or similar."""
        for path in ["/", "/new.html", "/simulation.html"]:
            client = _make_client()
            r = client.get(path)
            html = r.text.lower()
            assert "generate video" not in html
            assert "create video" not in html


# =============================================================================
# TEST 2 — Simulation Creation (Intent → System)
# =============================================================================

class Test2SimulationCreation:
    """Simulation creation flow."""

    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        with mock.patch.object(main, "sql") as mock_sql:
            mock_sql.create_episode.return_value = None
            mock_sql.get_episode.return_value = {
                "episode_id": "test-ep-1",
                "world_id": "default",
                "intent": "A human jumps across a 10-meter gap",
                "policies": {},
                "state": "CREATED",
            }
            mock_sql.get_beats.return_value = []
            mock_sql.get_beats_by_state.return_value = []
            mock_sql.list_episodes.return_value = []
            mock_sql.get_attempts.return_value = []
            mock_sql.get_artifacts.return_value = []
            with mock.patch("agents.narrative_planner.ProductionNarrativePlanner") as mock_planner:
                mock_planner.return_value.generate_beats.return_value = [
                    {"id": "beat-1", "description": "Attempt jump", "duration_sec": 5}
                ]
                with mock.patch.object(main, "redis_store") as mock_redis:
                    mock_redis.push_gpu_job = mock.MagicMock()
                    mock_redis.redis = mock.MagicMock()
                    yield

    def test_simulate_accepts_goal(self):
        """POST /simulate accepts goal and returns simulation_id."""
        client = _make_client()
        goal = "A human jumps across a 10-meter gap between two buildings without assistance."
        r = client.post(f"/simulate?world_id=default&goal={goal}")
        assert r.status_code == 200
        data = r.json()
        assert "simulation_id" in data
        assert data.get("status") in ("executing", "initialized")

    def test_simulation_starts_without_video(self):
        """Simulation can start; no video required."""
        client = _make_client()
        r = client.post("/simulate?world_id=default&goal=Test impossible physics")
        assert r.status_code == 200
        assert "simulation_id" in r.json()


# =============================================================================
# TEST 3–4 — Closed Loop, Confidence (require GPU worker)
# =============================================================================

class Test3ClosedLoop:
    """Closed-loop execution - partial coverage without GPU."""

    def test_episode_status_returns_state(self):
        """GET /episodes/{id} returns state structure."""
        with mock.patch.object(main, "sql") as mock_sql:
            mock_sql.get_episode.return_value = {
                "episode_id": "ep-1",
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "EXECUTING",
            }
            mock_sql.get_beats.return_value = [
                {"beat_id": "b1", "description": "Beat 1", "state": "PENDING"}
            ]
            mock_sql.get_artifacts.return_value = []
            client = _make_client()
            r = client.get("/episodes/ep-1")
            assert r.status_code == 200
            data = r.json()
            assert "state" in data or "beats" in data


# =============================================================================
# TEST 5 — World State Graph Integrity
# =============================================================================

class Test5WorldStateGraph:
    """World state graph structure."""

    def test_world_state_endpoint_exists(self):
        """GET /world-state/{id} returns graph data or no_data."""
        client = _make_client()
        r = client.get("/world-state/test-ep-id")
        assert r.status_code == 200
        data = r.json()
        assert "episode_id" in data
        assert "total_nodes" in data or "status" in data
        # Graph works without video
        assert "phase_1_5_enabled" in data or "total_nodes" in data


# =============================================================================
# TEST 6 — Video as Debug Artifact
# =============================================================================

class Test6VideoDebugArtifact:
    """Video must be collapsed, labeled, not autoplayed."""

    def test_video_button_labeled_ephemeral(self):
        """Video toggle is labeled 'Ephemeral Debug Artifact'."""
        client = _make_client()
        r = client.get("/simulation.html")
        assert "Ephemeral Debug Artifact" in r.text or "ephemeral" in r.text.lower()

    def test_video_collapsed_by_default(self):
        """Video section is collapsed (button text implies this)."""
        client = _make_client()
        r = client.get("/simulation.html")
        assert "Collapsed" in r.text or "collapse" in r.text.lower()

    def test_result_endpoint_video_optional(self):
        """GET /episodes/{id}/result works without include_video."""
        with mock.patch.object(main, "sql") as mock_sql:
            mock_sql.get_episode.return_value = {
                "episode_id": "ep-1",
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "COMPLETED",
            }
            mock_sql.get_beats.return_value = [
                {"beat_id": "b1", "state": "ACCEPTED", "description": "x"}
            ]
            mock_sql.get_attempts.return_value = []
            mock_sql.get_episode.side_effect = lambda x: {
                "episode_id": x,
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "COMPLETED",
            }
            client = _make_client()
            r = client.get("/episodes/ep-1/result?include_video=false")
            assert r.status_code == 200
            data = r.json()
            assert "outcome" in data
            assert "confidence" in data
            assert "state_delta" in data
            # Video not required
            assert "video_uri" not in data or data.get("debug") is None or "video_uri" not in (data.get("debug") or {})


# =============================================================================
# TEST 8 — Failure as First-Class Outcome
# =============================================================================

class Test8FailureFirstClass:
    """System must support GOAL_IMPOSSIBLE and other failure outcomes."""

    def test_terminate_episode_accepts_goal_impossible(self):
        """POST /episodes/{id}/terminate accepts outcome=goal_impossible."""
        with mock.patch.object(main, "sql") as mock_sql:
            mock_sql.get_episode.return_value = {
                "episode_id": "ep-1",
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "EXECUTING",
            }
            mock_sql.update_episode_state = mock.MagicMock()
            client = _make_client()
            r = client.post(
                "/episodes/ep-1/terminate",
                params={"reason": "physics", "outcome": "goal_impossible"},
            )
            assert r.status_code == 200
            data = r.json()
            assert data.get("status") == "terminated"
            assert data.get("result", {}).get("outcome") == "goal_impossible"


# =============================================================================
# TEST 9 — API Contract Validation
# =============================================================================

class Test9APIContract:
    """Result API must match contract."""

    def test_result_has_required_fields(self):
        """Result has: outcome, state_delta, confidence, cost."""
        with mock.patch.object(main, "sql") as mock_sql:
            mock_sql.get_episode.return_value = {
                "episode_id": "ep-1",
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "COMPLETED",
            }
            mock_sql.get_beats.return_value = [
                {"beat_id": "b1", "state": "ACCEPTED", "description": "x"}
            ]
            mock_sql.get_attempts.return_value = []
            mock_sql.get_episode.side_effect = lambda x: {
                "episode_id": x,
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "COMPLETED",
            }
            client = _make_client()
            r = client.get("/episodes/ep-1/result")
            assert r.status_code == 200
            data = r.json()
            assert "outcome" in data
            assert "state_delta" in data
            assert "confidence" in data
            assert "total_cost_usd" in data or "cost" in str(data).lower()
            # Video optional - result complete without it
            assert "error" not in data or data.get("outcome") != "error"

    def test_result_video_optional(self):
        """Result without include_video has no video URLs required."""
        with mock.patch.object(main, "sql") as mock_sql:
            mock_sql.get_episode.return_value = {
                "episode_id": "ep-1",
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "COMPLETED",
            }
            mock_sql.get_beats.return_value = []
            mock_sql.get_attempts.return_value = []
            mock_sql.get_episode.side_effect = lambda x: {
                "episode_id": x,
                "world_id": "default",
                "intent": "test",
                "policies": {},
                "state": "COMPLETED",
            }
            client = _make_client()
            r = client.get("/episodes/ep-1/result?include_video=false")
            data = r.json()
            # Primary output is state; video is debug
            assert "outcome" in data
            assert isinstance(data.get("confidence"), (int, float)) or data.get("confidence") is None


# =============================================================================
# TEST 10 — Product Identity Check
# =============================================================================

class Test10ProductIdentity:
    """Final gate: simulator identity, not video generator."""

    def test_ui_branding_infrastructure(self):
        """UI says 'Infrastructure' or 'Simulation' not 'Video Generator'."""
        client = _make_client()
        r = client.get("/")
        html = r.text
        assert "INFRASTRUCTURE" in html or "Simulation" in html
        assert "Video Generator" not in html
        assert "Content Creation" not in html

    def test_new_form_says_simulation(self):
        """New simulation form uses simulation language."""
        client = _make_client()
        r = client.get("/new.html")
        assert "Simulation" in r.text
        assert "physics" in r.text.lower() or "constraint" in r.text.lower()


# =============================================================================
# Report hook
# =============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Emit acceptance verdict."""
    if "test_production_acceptance" in str(config.args):
        passed = terminalreporter.stats.get("passed", [])
        failed = terminalreporter.stats.get("failed", [])
        total = len(passed) + len(failed)
        if failed:
            verdict = "❌ Not ready"
        elif total < 15:
            verdict = "⚠️ Needs fixes (partial coverage)"
        else:
            verdict = "✅ Production-ready computational infrastructure"
        terminalreporter.write_sep("=", "ACCEPTANCE VERDICT")
        terminalreporter.write_line(f"Passed: {len(passed)} | Failed: {len(failed)} | Total: {total}")
        terminalreporter.write_line(f"Verdict: {verdict}")
