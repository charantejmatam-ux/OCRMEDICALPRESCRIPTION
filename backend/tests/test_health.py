"""
backend/tests/test_health.py
Pytest tests for the GET /health endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Return a synchronous HTTPX-backed TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_status_code_is_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_is_json(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_status_field_is_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_service_field_present(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["service"] == "medical-prescription-ocr"

    def test_version_field_present(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "version" in data
        assert data["version"] != ""

    def test_timestamp_field_is_iso8601(self, client: TestClient) -> None:
        """Timestamp must be a valid ISO-8601 UTC string."""
        from datetime import datetime
        data = client.get("/health").json()
        assert "timestamp" in data
        # Will raise ValueError if not a valid ISO format
        dt = datetime.fromisoformat(data["timestamp"])
        assert dt.tzinfo is not None  # timezone-aware

    def test_schema_has_no_extra_unexpected_keys(self, client: TestClient) -> None:
        data = client.get("/health").json()
        expected_keys = {"status", "service", "version", "timestamp"}
        assert set(data.keys()) == expected_keys
