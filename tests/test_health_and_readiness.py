"""Tests for application health and readiness probe endpoints."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import backend.main as main
from backend.main import check_ai_readiness, check_database_readiness


def test_legacy_health_endpoint(client):
    """Confirm backwards compatibility of the original /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Smart Internet Connectivity Analyser"


@pytest.mark.parametrize("path", ["/api/health/live", "/api/live", "/healthz"])
def test_liveness_endpoints(client, path):
    """Liveness probe confirms the API process responds without heavy I/O."""
    response = client.get(path)
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "alive"}


@pytest.mark.parametrize("path", ["/api/health/ready", "/api/ready", "/readyz"])
def test_readiness_healthy_state(client, path):
    """Readiness probe returns 200 and dependency status when healthy."""
    response = client.get(path)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "dependencies" in data
    assert data["dependencies"]["database"] == "connected"
    assert "ai_engine" in data["dependencies"]
    assert data["dependencies"]["ai_engine"] in ["ready", "unavailable", "degraded"]


def test_readiness_unavailable_when_database_fails(client):
    """Readiness probe returns 503 when required database dependency fails."""
    with patch("backend.main.get_connection", side_effect=sqlite3.OperationalError("Database disk I/O error")):
        response = client.get("/api/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["database"] == "unavailable"
        # Ensure no sensitive path or raw exception details are leaked
        assert "Database disk I/O error" not in response.text


def test_health_checks_do_not_trigger_network_diagnostics(client):
    """Confirm health and readiness checks never execute network probes or tests."""
    with patch("backend.main.subprocess.run") as mock_run:
        res_live = client.get("/api/health/live")
        res_ready = client.get("/api/health/ready")
        assert res_live.status_code == 200
        assert res_ready.status_code == 200
        mock_run.assert_not_called()


def test_check_ai_readiness_states():
    """Verify check_ai_readiness handles different model states."""
    with patch("backend.main.AI_MODELS_READY", True):
        assert check_ai_readiness() == "ready"

    with patch("backend.main.AI_MODELS_READY", False), patch("backend.main.torch", None):
        assert check_ai_readiness() == "unavailable"

    with patch("backend.main.AI_MODELS_READY", False), patch("backend.main.torch", MagicMock()):
        assert check_ai_readiness() == "degraded"
