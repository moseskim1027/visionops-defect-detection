"""Backward-compat smoke test — covered in full by test_inference_api.py."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_health():
    loader = MagicMock()
    loader.is_loaded = False
    with patch("src.inference.api._loader", loader):
        from src.inference.api import app

        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
