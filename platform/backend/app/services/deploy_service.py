"""Deployment service — model registry management and inference service control."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import INFERENCE_URL, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI

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
            result.append(
                {
                    "version": v.version,
                    "status": v.status,
                    "aliases": version_aliases.get(v.version, []),
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                    "metrics": _get_run_metrics(client, v.run_id),
                }
            )
        return {"versions": result, "model_name": MLFLOW_MODEL_NAME}
    except Exception as exc:
        return {"versions": [], "model_name": MLFLOW_MODEL_NAME, "error": str(exc)}


def _get_run_metrics(client: Any, run_id: str) -> dict[str, float]:
    try:
        run = client.get_run(run_id)
        return {
            "map50": round(run.data.metrics.get("map50", 0), 4),
            "precision": round(run.data.metrics.get("precision", 0), 4),
            "recall": round(run.data.metrics.get("recall", 0), 4),
        }
    except Exception:
        return {}


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
