"""Deployment service — model registry management and inference service control."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import INFERENCE_URL, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI, RAW_DIR

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def list_model_versions() -> dict[str, Any]:
    """List all registered versions of the model with aliases and metrics."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        # Build version → [aliases] map from the registered model
        version_aliases: dict[str, list[str]] = {}
        try:
            reg_model = client.get_registered_model(MLFLOW_MODEL_NAME)
            for alias, ver in (reg_model.aliases or {}).items():
                version_aliases.setdefault(ver, []).append(alias)
        except Exception:
            pass

        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        result = []
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            metrics, experiment_id = _get_run_info(client, v.run_id)
            result.append(
                {
                    "version": v.version,
                    "status": v.status,
                    "aliases": version_aliases.get(v.version, []),
                    "run_id": v.run_id,
                    "experiment_id": experiment_id,
                    "creation_timestamp": v.creation_timestamp,
                    "metrics": metrics,
                }
            )
        return {"versions": result, "model_name": MLFLOW_MODEL_NAME}
    except Exception as exc:
        return {"versions": [], "model_name": MLFLOW_MODEL_NAME, "error": str(exc)}


def _get_run_info(client: Any, run_id: str) -> tuple[dict[str, float], str | None]:
    """Return (metrics, experiment_id) for a run."""
    try:
        run = client.get_run(run_id)
        m = run.data.metrics
        metrics = {
            "map50": round(m.get("map50") or m.get("metrics/mAP50(B)") or m.get("metrics/mAP50B") or 0, 4),
            "precision": round(m.get("precision") or m.get("metrics/precision(B)") or m.get("metrics/precisionB") or 0, 4),
            "recall": round(m.get("recall") or m.get("metrics/recall(B)") or m.get("metrics/recallB") or 0, 4),
        }
        return metrics, run.info.experiment_id
    except Exception:
        return {}, None


def promote_model(version: str) -> dict[str, Any]:
    """Set the 'production' alias on the given model version."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", version)
        return {"promoted": True, "version": version, "alias": "production"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------


def get_inference_status() -> dict[str, Any]:
    """Call the inference service health endpoint."""
    try:
        r = httpx.get(f"{INFERENCE_URL}/health", timeout=3.0)
        return r.json()
    except Exception as exc:
        return {"status": "unreachable", "model_loaded": False, "error": str(exc)}


def reload_inference() -> dict[str, Any]:
    """Trigger a model reload on the inference service."""
    try:
        r = httpx.post(f"{INFERENCE_URL}/reload", timeout=60.0)
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Inference testing (mirrors make predict / make predict-batch)
# ---------------------------------------------------------------------------


def _find_val_dir():
    """Return the first usable val image directory under RAW_DIR."""
    # Prefer Casting/val — same default as the Makefile
    preferred = RAW_DIR / "Casting" / "val"
    if preferred.exists() and any(preferred.glob("*.jpg")):
        return preferred
    for candidate in sorted(RAW_DIR.glob("*/val")):
        if any(candidate.glob("*.jpg")):
            return candidate
    return None


def test_single_predict() -> dict[str, Any]:
    """Send one image to /predict (mirrors `make predict`)."""
    val_dir = _find_val_dir()
    if val_dir is None:
        return {"error": "No val images found under data/raw/vision"}

    image_path = next(val_dir.glob("*.jpg"))
    try:
        with open(image_path, "rb") as f:
            r = httpx.post(
                f"{INFERENCE_URL}/predict",
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=30.0,
            )
        if r.status_code != 200:
            return {"error": f"Inference returned {r.status_code}: {r.text}"}
        data = r.json()
        return {
            "success": True,
            "image": image_path.name,
            "source_dir": val_dir.parent.name + "/val",
            "num_detections": data.get("num_detections", 0),
            "inference_time_ms": data.get("inference_time_ms"),
            "detections": [
                {"class_name": d["class_name"], "confidence": round(d["confidence"], 3)}
                for d in data.get("detections", [])
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


def test_batch_predict() -> dict[str, Any]:
    """Send all val images to /predict (mirrors `make predict-batch`)."""
    val_dir = _find_val_dir()
    if val_dir is None:
        return {"error": "No val images found under data/raw/vision"}

    images = sorted(val_dir.glob("*.jpg"))
    sent = 0
    errors = 0
    total_detections = 0

    for img in images:
        try:
            with open(img, "rb") as f:
                r = httpx.post(
                    f"{INFERENCE_URL}/predict",
                    files={"file": (img.name, f, "image/jpeg")},
                    timeout=30.0,
                )
            if r.status_code == 200:
                total_detections += r.json().get("num_detections", 0)
                sent += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    return {
        "success": errors == 0,
        "source_dir": val_dir.parent.name + "/val",
        "images_sent": sent,
        "errors": errors,
        "total_detections": total_detections,
        "avg_detections_per_image": round(total_detections / sent, 2) if sent else 0,
    }
