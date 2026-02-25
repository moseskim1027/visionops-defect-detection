"""FastAPI inference service.

Endpoints
---------
POST /predict   Upload an image, receive bounding-box detections.
GET  /health    Liveness check — reports whether a model is loaded.
GET  /metrics   Prometheus metrics (latency, prediction count, errors).

Environment variables
---------------------
MLFLOW_TRACKING_URI   MLflow server URL (enables registry-based loading).
MLFLOW_MODEL_NAME     Registered model name.
MLFLOW_MODEL_ALIAS    Alias to load (default: "production").
YOLO_WEIGHTS_PATH     Local fallback weights path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel

from src.inference.model_loader import Detection as LoaderDetection
from src.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


def _get_or_create(metric_cls, name, doc, **kwargs):
    """Return an existing metric or create a new one.

    Prometheus raises if you register the same metric name twice (e.g. when a
    module is reloaded after a partial import failure during testing).
    """
    existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
    if existing is not None:
        return existing
    return metric_cls(name, doc, **kwargs)


INFERENCE_LATENCY = _get_or_create(
    Histogram,
    "inference_latency_seconds",
    "Per-request inference latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTIONS_TOTAL = _get_or_create(
    Counter, "predictions_total", "Total prediction requests served"
)
ERRORS_TOTAL = _get_or_create(
    Counter, "prediction_errors_total", "Total prediction errors"
)
MODEL_INFO = _get_or_create(Info, "active_model", "Currently loaded model metadata")

# ---------------------------------------------------------------------------
# Global model loader (replaced in tests via patching)
# ---------------------------------------------------------------------------

_loader = ModelLoader()


def _update_model_info() -> None:
    MODEL_INFO.info(
        {
            "run_id": _loader.run_id,
            "version": _loader.model_version,
            "alias": _loader.model_alias,
        }
    )


# ---------------------------------------------------------------------------
# Lifespan — model loading on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    model_name = os.environ.get("MLFLOW_MODEL_NAME")
    model_alias = os.environ.get("MLFLOW_MODEL_ALIAS", "production")
    weights_path = os.environ.get("YOLO_WEIGHTS_PATH")
    class_map = Path("data/processed/class_map.json")

    if tracking_uri and model_name:
        try:
            _loader.load_from_mlflow(
                model_name,
                model_alias,
                tracking_uri,
                class_map if class_map.exists() else None,
            )
            _update_model_info()
        except Exception as exc:
            logger.warning(
                "MLflow load failed (%s) — falling back to local weights", exc
            )
            _loader.load(
                Path(weights_path) if weights_path else None,
                class_map if class_map.exists() else None,
            )
    elif weights_path:
        _loader.load(
            Path(weights_path),
            class_map if class_map.exists() else None,
        )
    else:
        logger.warning("No model configured — /predict returns 503 until loaded")

    yield


# ---------------------------------------------------------------------------
# App + response schemas
# ---------------------------------------------------------------------------

app = FastAPI(title="VisionOps Inference", version="0.1.0", lifespan=lifespan)


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    detections: list[Detection]
    num_detections: int
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _loader.is_loaded}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not _loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    tmp_path: str | None = None

    try:
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        raw: list[LoaderDetection] = _loader.predict(tmp_path)
        inference_ms = (time.perf_counter() - start) * 1000

        PREDICTIONS_TOTAL.inc()
        INFERENCE_LATENCY.observe(inference_ms / 1000)

        return PredictResponse(
            detections=[
                Detection(
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                )
                for d in raw
            ],
            num_detections=len(raw),
            inference_time_ms=round(inference_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        ERRORS_TOTAL.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
async def reload_model() -> dict:
    """Reload the YOLO model from MLflow (or local weights fallback)."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    model_name = os.environ.get("MLFLOW_MODEL_NAME")
    model_alias = os.environ.get("MLFLOW_MODEL_ALIAS", "production")
    weights_path = os.environ.get("YOLO_WEIGHTS_PATH")
    class_map = Path("data/processed/class_map.json")

    def _do_reload() -> dict:
        if tracking_uri and model_name:
            try:
                _loader.load_from_mlflow(
                    model_name,
                    model_alias,
                    tracking_uri,
                    class_map if class_map.exists() else None,
                )
                _update_model_info()
                return {"status": "reloaded", "source": "mlflow", "alias": model_alias}
            except Exception as exc:
                logger.warning("MLflow reload failed (%s) — trying local weights", exc)
                if weights_path:
                    _loader.load(
                        Path(weights_path),
                        class_map if class_map.exists() else None,
                    )
                    return {
                        "status": "reloaded",
                        "source": "local",
                        "warning": str(exc),
                    }
                return {"status": "error", "detail": str(exc)}
        elif weights_path:
            _loader.load(
                Path(weights_path),
                class_map if class_map.exists() else None,
            )
            return {"status": "reloaded", "source": "local"}
        return {"status": "error", "detail": "No model source configured"}

    return await asyncio.to_thread(_do_reload)
