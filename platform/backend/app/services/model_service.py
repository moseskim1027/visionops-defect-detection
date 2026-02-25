"""Model card service.

Provides per-class metrics and poor-performing sample images (with both
ground-truth and predicted bounding boxes) after training completes.
"""

from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.config import (
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
    PROCESSED_SUBSET_DIR,
    RUNS_DIR,
)
from app.services.training_service import _find_latest_run, _train_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_class_names(processed_dir: Path) -> list[str]:
    yaml_path = processed_dir / "dataset.yaml"
    if not yaml_path.exists():
        return []
    data = yaml.safe_load(yaml_path.read_text())
    names_raw = data.get("names", {})
    if isinstance(names_raw, dict):
        return [names_raw[k] for k in sorted(names_raw)]
    return list(names_raw)


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def _get_processed_dir() -> Path:
    """Return the dataset dir that matches the most recent training run.

    If training was run on a subset (products list set), use the subset dir;
    otherwise fall back to the full processed dataset.
    """
    if _train_state.get("products"):
        return PROCESSED_SUBSET_DIR
    return PROCESSED_DIR


# ---------------------------------------------------------------------------
# Best weights discovery
# ---------------------------------------------------------------------------


def _find_best_weights() -> Path | None:
    """Return path to the most recent best.pt in runs/detect/."""
    detect_dir = RUNS_DIR / "detect"
    if not detect_dir.exists():
        return None
    candidates = sorted(
        detect_dir.glob("train*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Per-class metrics via model.val()
# ---------------------------------------------------------------------------

_class_metrics_cache: dict[str, Any] | None = None
_class_metrics_started_at: float | None = None  # training start time at cache build


def get_class_metrics(
    run_id: str | None = None,
    processed_dir: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    global _class_metrics_cache, _class_metrics_started_at

    # Auto-invalidate cache when a new training run has started
    current_started_at = _train_state.get("started_at")
    if (
        _class_metrics_cache
        and _class_metrics_started_at is not None
        and current_started_at != _class_metrics_started_at
    ):
        _class_metrics_cache = None

    if _class_metrics_cache and not force_refresh:
        return _class_metrics_cache

    proc = Path(processed_dir) if processed_dir else _get_processed_dir()
    class_names = _load_class_names(proc)
    dataset_yaml = proc / "dataset.yaml"

    weights = _find_best_weights()
    if not weights or not dataset_yaml.exists():
        return {"error": "No trained model or dataset found.", "class_metrics": []}

    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        val_results = model.val(
            data=str(dataset_yaml),
            verbose=False,
            plots=False,
        )
        box = val_results.box
        ap50_per_class = box.ap50
        ap_per_class = box.ap
        seen_indices = (
            box.ap_class_index.tolist()
            if hasattr(box, "ap_class_index")
            else list(range(len(ap50_per_class)))
        )

        metrics_list = []
        for i, cls_id in enumerate(seen_indices):
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            metrics_list.append(
                {
                    "class_id": int(cls_id),
                    "class_name": name,
                    "ap50": round(float(ap50_per_class[i]), 4),
                    "ap50_95": round(float(ap_per_class[i]), 4),
                }
            )
        metrics_list.sort(key=lambda x: x["ap50"])

        overall = {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        }

        result = {
            "overall": overall,
            "class_metrics": metrics_list,
            "weights_path": str(weights),
        }
        _class_metrics_cache = result
        _class_metrics_started_at = current_started_at
        return result
    except Exception as exc:
        return {"error": str(exc), "class_metrics": []}


# ---------------------------------------------------------------------------
# Model card summary (from MLflow)
# ---------------------------------------------------------------------------


def get_model_card(run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or _find_latest_run(status="FINISHED")
    if not rid:
        return {"error": "No completed training run found."}

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run = client.get_run(rid)
        metrics = run.data.metrics
        params = run.data.params
        return {
            "run_id": rid,
            "status": run.info.status,
            "metrics": {
                "precision": round(metrics.get("precision", 0), 4),
                "recall": round(metrics.get("recall", 0), 4),
                "map50": round(metrics.get("map50", 0), 4),
                "map50_95": round(metrics.get("map50_95", 0), 4),
            },
            "params": params,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Poor-performing samples — GT + predicted bounding boxes
# ---------------------------------------------------------------------------

# BGR colours and drawing constants
_GT_COLOR = (50, 220, 50)  # green  — ground truth
_PRED_COLOR = (30, 120, 255)  # orange — model predictions
_BOX_THICKNESS = 3
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.4
_FONT_THICKNESS = 1


def get_poor_samples(
    processed_dir: str | None = None,
    n_classes: int = 5,
    n_per_class: int = 2,
) -> list[dict[str, Any]]:
    """Return sample val images for the worst-AP classes.

    Each image has ground-truth boxes (green) and model-predicted boxes
    (orange) overlaid so the user can visually compare them.
    """
    class_metrics = get_class_metrics(processed_dir=processed_dir)
    if "error" in class_metrics and not class_metrics.get("class_metrics"):
        return []

    worst = class_metrics["class_metrics"][:n_classes]
    proc = Path(processed_dir) if processed_dir else _get_processed_dir()
    class_names = _load_class_names(proc)
    images_dir = proc / "images" / "val"
    labels_dir = proc / "labels" / "val"

    # Load YOLO model once for inference
    yolo_model = None
    weights = _find_best_weights()
    if weights:
        try:
            from ultralytics import YOLO

            yolo_model = YOLO(str(weights))
        except Exception:
            pass

    results = []
    for cls_entry in worst:
        cls_id = cls_entry["class_id"]

        # Collect val images that contain this class
        candidates: list[Path] = []
        for lf in labels_dir.glob("*.txt") if labels_dir.exists() else []:
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if parts and int(parts[0]) == cls_id:
                    img_path = images_dir / lf.with_suffix(".jpg").name
                    if img_path.exists():
                        candidates.append(img_path)
                    break

        selected = random.sample(candidates, min(n_per_class, len(candidates)))
        for img_path in selected:
            label_path = labels_dir / img_path.with_suffix(".txt").name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # 1. Draw predicted boxes first (underneath GT)
            if yolo_model is not None:
                try:
                    preds = yolo_model.predict(img, verbose=False, conf=0.25)
                    for box in preds[0].boxes:
                        pred_cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1p, y1p, x2p, y2p = map(int, box.xyxy[0])
                        cv2.rectangle(
                            img, (x1p, y1p), (x2p, y2p), _PRED_COLOR, _BOX_THICKNESS
                        )
                        pred_name = (
                            class_names[pred_cls]
                            if pred_cls < len(class_names)
                            else str(pred_cls)
                        )
                        cv2.putText(
                            img,
                            f"{pred_name} {conf:.2f}",
                            (x1p, min(y2p + 12, h - 2)),
                            _FONT,
                            _FONT_SCALE,
                            _PRED_COLOR,
                            _FONT_THICKNESS,
                            cv2.LINE_AA,
                        )
                except Exception:
                    pass

            # 2. Draw ground-truth boxes on top
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), _GT_COLOR, _BOX_THICKNESS)
                    cname = class_names[cid] if cid < len(class_names) else str(cid)
                    cv2.putText(
                        img,
                        cname,
                        (x1, max(y1 - 4, 10)),
                        _FONT,
                        _FONT_SCALE,
                        _GT_COLOR,
                        _FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

            # Resize to max 512px on longest side
            max_side = 512
            scale = max_side / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            results.append(
                {
                    "filename": img_path.name,
                    "class_name": cls_entry["class_name"],
                    "ap50": cls_entry["ap50"],
                    "image_b64": _img_to_b64(img),
                }
            )

    return results
