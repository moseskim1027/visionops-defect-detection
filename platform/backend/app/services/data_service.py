"""Data preparation service.

Handles directory introspection, dataset preparation subprocess, and generating
annotated sample images from the processed YOLO dataset.
"""

from __future__ import annotations

import base64
import io
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from app.config import CONFIGS_DIR, PROCESSED_DIR, RAW_DIR, ROOT_DIR

# ---------------------------------------------------------------------------
# In-memory preparation state (single-server, no persistence needed)
# ---------------------------------------------------------------------------

_prep_state: dict[str, Any] = {
    "status": "idle",  # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "message": "",
    "error": None,
}

# Colours per-class index (HSV hue cycle)
_PALETTE = [
    (0, 200, 255),
    (30, 200, 255),
    (60, 200, 255),
    (90, 200, 255),
    (120, 200, 255),
    (150, 200, 255),
    (180, 200, 255),
    (210, 200, 255),
]


def _bgr_color(class_id: int) -> tuple[int, int, int]:
    hsv = np.uint8([[_PALETTE[class_id % len(_PALETTE)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Directory introspection
# ---------------------------------------------------------------------------


def get_directory_info(source_dir: str | None = None) -> dict[str, Any]:
    raw = Path(source_dir) if source_dir else RAW_DIR
    processed = PROCESSED_DIR

    categories: list[str] = []
    if raw.exists():
        categories = sorted(
            d.name for d in raw.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    is_prepared = (processed / "dataset.yaml").exists()
    num_train = num_val = 0
    if is_prepared:
        train_img = processed / "images" / "train"
        val_img = processed / "images" / "val"
        num_train = len(list(train_img.glob("*.jpg"))) if train_img.exists() else 0
        num_val = len(list(val_img.glob("*.jpg"))) if val_img.exists() else 0

    return {
        "raw_dir": str(raw),
        "processed_dir": str(processed),
        "categories": categories,
        "is_prepared": is_prepared,
        "num_train_images": num_train,
        "num_val_images": num_val,
        "prep_state": _prep_state.copy(),
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def start_preparation(source_dir: str | None = None, processed_dir: str | None = None) -> dict[str, Any]:
    if _prep_state["status"] == "running":
        return {"error": "Preparation already running", "state": _prep_state.copy()}

    raw = str(Path(source_dir) if source_dir else RAW_DIR)
    proc = str(Path(processed_dir) if processed_dir else PROCESSED_DIR)

    _prep_state.update(
        {
            "status": "running",
            "started_at": time.time(),
            "completed_at": None,
            "message": "Starting dataset preparation...",
            "error": None,
        }
    )

    def _run():
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT_DIR)
            cmd = [
                sys.executable,
                "-m",
                "src.data.prepare_dataset",
                "--src",
                raw,
                "--dst",
                proc,
            ]
            _prep_state["message"] = "Converting COCO annotations to YOLO format..."
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ROOT_DIR),
                env=env,
            )
            if result.returncode == 0:
                _prep_state.update(
                    {
                        "status": "completed",
                        "completed_at": time.time(),
                        "message": "Dataset preparation complete.",
                    }
                )
            else:
                _prep_state.update(
                    {
                        "status": "failed",
                        "error": result.stderr[-2000:] if result.stderr else "Unknown error",
                        "message": "Preparation failed.",
                    }
                )
        except Exception as exc:
            _prep_state.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "message": "Preparation failed with exception.",
                }
            )

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True, "state": _prep_state.copy()}


def get_preparation_status() -> dict[str, Any]:
    return _prep_state.copy()


# ---------------------------------------------------------------------------
# Annotated sample images
# ---------------------------------------------------------------------------


def _load_class_names(processed_dir: Path) -> list[str]:
    yaml_path = processed_dir / "dataset.yaml"
    if not yaml_path.exists():
        return []
    data = yaml.safe_load(yaml_path.read_text())
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def _image_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def get_annotated_samples(
    processed_dir: str | None = None,
    n_samples: int = 12,
    split: str = "val",
) -> list[dict[str, Any]]:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    images_dir = proc / "images" / split
    labels_dir = proc / "labels" / split

    if not images_dir.exists():
        return []

    class_names = _load_class_names(proc)
    all_images = list(images_dir.glob("*.jpg"))
    # prefer images that have at least one annotation
    annotated = [
        p for p in all_images if (labels_dir / p.with_suffix(".txt").name).exists()
    ]
    pool = annotated if annotated else all_images
    selected = random.sample(pool, min(n_samples, len(pool)))

    results = []
    for img_path in selected:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        annotations: list[dict] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = _bgr_color(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                label_text = cls_name
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    img, label_text, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
                annotations.append({"class_id": cls_id, "class_name": cls_name})

        # Resize to max 640 on long edge for bandwidth
        max_side = 640
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Extract category from filename prefix (e.g. "Cable_img001.jpg" → "Cable")
        parts_name = img_path.stem.split("_")
        category = parts_name[0] if parts_name else "unknown"

        results.append(
            {
                "filename": img_path.name,
                "category": category,
                "image_b64": _image_to_b64(img),
                "annotations": annotations,
                "num_annotations": len(annotations),
            }
        )
    return results
