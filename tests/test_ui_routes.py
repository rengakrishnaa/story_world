"""
Tests for Infrastructure UI Routes (Phase 8).
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestUIRoutes:
    
    def test_dashboard_route(self):
        """Test that the dashboard (index.html) is served at root."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "STORYWORLD INFRASTRUCTURE" in response.text
        # Verify static asset linking
        assert '/static/style.css' in response.text
        assert '/static/dashboard.js' in response.text

    def test_dashboard_js_polling(self):
        """Verify that dashboard.js contains polling logic."""
        response = client.get("/static/dashboard.js")
        assert response.status_code == 200
        assert "setInterval" in response.text
        assert "fetchStatus" in response.text

    def test_new_simulation_route(self):
        """Test that the new simulation form is served."""
        response = client.get("/new.html")
        assert response.status_code == 200
        # HTML Title updated to "New Simulation"
        assert "New Simulation" in response.text
        assert "Initialize Simulation" in response.text

    def test_detail_view_route(self):
        """Test that the detail view is served."""
        response = client.get("/simulation.html")
        assert response.status_code == 200
        assert "World State Graph" in response.text
        assert "confidence" in response.text.lower()

    def test_static_css_served(self):
        """Test that CSS is served correctly."""
        response = client.get("/static/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]
        # Check for new pro variable
        assert "--bg-app" in response.text

    def test_simulate_api_availability(self):
        """Test that the /simulate API endpoint is reachable."""
        # Just check 422 for missing query params to verify route exists
        response = client.post("/simulate") 
        assert response.status_code == 422 
