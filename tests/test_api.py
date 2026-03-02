"""Tests for the API layer."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestStoreStats:
    def test_get_stats(self):
        response = client.get("/store/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "collection_name" in data


class TestIngestEndpoint:
    def test_upload_unsupported_file(self):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_ingest_nonexistent_directory(self):
        response = client.post(
            "/ingest/directory",
            json={"directory": "/nonexistent/path"},
        )
        assert response.status_code == 404
