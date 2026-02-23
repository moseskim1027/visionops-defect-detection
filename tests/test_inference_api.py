"""Tests for src/inference/api.py"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from src.inference.model_loader import Detection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=(200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def _single_detection() -> list[Detection]:
    return [
        Detection(
            class_id=0,
            class_name="scratch",
            confidence=0.85,
            bbox=[10.0, 20.0, 110.0, 120.0],
        )
    ]


# ---------------------------------------------------------------------------
# Fixtures — split so tests can mutate the mock directly
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_loader():
    """A pre-loaded ModelLoader mock with one default detection."""
    loader = MagicMock()
    loader.is_loaded = True
    loader.predict.return_value = _single_detection()
    return loader


@pytest.fixture()
def client(mock_loader):
    """TestClient backed by mock_loader."""
    with patch("src.inference.api._loader", mock_loader):
        from src.inference.api import app

        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    loader = MagicMock()
    loader.is_loaded = False
    with patch("src.inference.api._loader", loader):
        from src.inference.api import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok_with_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "model_loaded": True}

    def test_ok_without_model(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_detections(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_detections"] == 1
        det = body["detections"][0]
        assert det["class_name"] == "scratch"
        assert det["confidence"] == pytest.approx(0.85)
        assert len(det["bbox"]) == 4

    def test_503_when_no_model(self, client_no_model):
        resp = client_no_model.post(
            "/predict",
            files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 503

    def test_500_on_model_error(self, client, mock_loader):
        mock_loader.predict.side_effect = RuntimeError("inference failed")
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 500

    def test_empty_detections(self, client, mock_loader):
        mock_loader.predict.return_value = []
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["num_detections"] == 0

    def test_inference_time_present(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert "inference_time_ms" in resp.json()


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "predictions_total" in resp.text
        assert "inference_latency_seconds" in resp.text
