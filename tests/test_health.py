"""Smoke test for inference API health endpoint."""

from fastapi.testclient import TestClient

from src.inference.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
