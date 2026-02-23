"""Evidently-based image feature drift detection.

Extracts per-image statistics (brightness, contrast, sharpness) from two sets
of images and uses Evidently's DataDriftPreset to detect distribution shift.

Functions
---------
extract_image_features   Extract a 4-column DataFrame from a list of images.
run_drift_report         Compare reference vs. current image directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def extract_image_features(image_paths: list[Path]) -> pd.DataFrame:
    """Compute per-image statistics used as drift features.

    Features
    --------
    brightness_mean : float
        Mean normalised pixel intensity ([0, 1]).
    brightness_std : float
        Std of normalised pixel intensity.
    contrast : float
        Standard deviation of grayscale pixel values ([0, 1]).
    sharpness : float
        Mean gradient energy of the grayscale image.  Higher = sharper.

    Args:
        image_paths: Paths to JPEG/PNG images.

    Returns:
        DataFrame with one row per image and four feature columns.
    """
    rows = []
    for path in image_paths:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        gy, gx = np.gradient(gray)
        rows.append(
            {
                "brightness_mean": float(arr.mean()),
                "brightness_std": float(arr.std()),
                "contrast": float(gray.std()),
                "sharpness": float((gy**2 + gx**2).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_drift_report(
    reference_dir: Path,
    current_dir: Path,
    config_path: Path = Path("configs/drift.yaml"),
) -> dict:
    """Compare a current image batch against a reference set using Evidently.

    Args:
        reference_dir: Directory containing reference images (e.g.
                       ``data/processed/images/val``).
        current_dir:   Directory containing current-batch images (e.g. a
                       ``data/drift_batches/<batch>/images`` path).
        config_path:   Path to ``configs/drift.yaml``.

    Returns:
        Dictionary with keys:

        * ``drift_detected``   (bool)   — overall drift flag from Evidently.
        * ``drifted_features`` (list)   — names of drifted feature columns.
        * ``drift_share``      (float)  — fraction of features that drifted.
        * ``feature_stats``    (dict)   — per-feature drift score and flag.
        * ``skipped``          (bool)   — ``True`` if sample count too low.
    """
    from evidently.metric_preset import DataDriftPreset  # noqa: PLC0415
    from evidently.report import Report  # noqa: PLC0415

    cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    drift_cfg = cfg.get("drift", {})
    min_ref = drift_cfg.get("min_reference_samples", 50)
    min_cur = drift_cfg.get("min_current_samples", 20)

    ref_images = sorted(reference_dir.glob("*.jpg")) + sorted(
        reference_dir.glob("*.png")
    )
    cur_images = sorted(current_dir.glob("*.jpg")) + sorted(current_dir.glob("*.png"))

    logger.info(
        "Drift detection: %d reference, %d current images",
        len(ref_images),
        len(cur_images),
    )

    if len(ref_images) < min_ref or len(cur_images) < min_cur:
        logger.warning(
            "Insufficient samples (ref=%d min=%d, cur=%d min=%d) — skipping",
            len(ref_images),
            min_ref,
            len(cur_images),
            min_cur,
        )
        return {
            "drift_detected": False,
            "drifted_features": [],
            "drift_share": 0.0,
            "feature_stats": {},
            "skipped": True,
        }

    ref_df = extract_image_features(ref_images)
    cur_df = extract_image_features(cur_images)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()

    dataset_metric = next(
        (m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"),
        None,
    )
    if dataset_metric is None:
        return {
            "drift_detected": False,
            "drifted_features": [],
            "drift_share": 0.0,
            "feature_stats": {},
            "skipped": False,
        }

    res = dataset_metric["result"]
    drift_by_col: dict = res.get("drift_by_columns", {})
    drifted = [col for col, info in drift_by_col.items() if info.get("drift_detected")]
    drift_share = res.get(
        "share_of_drifted_columns",
        len(drifted) / max(1, len(drift_by_col)),
    )
    drift_detected = res.get("dataset_drift", len(drifted) > 0)

    logger.info(
        "Drift result: detected=%s, share=%.2f, features=%s",
        drift_detected,
        drift_share,
        drifted,
    )

    return {
        "drift_detected": bool(drift_detected),
        "drifted_features": drifted,
        "drift_share": float(drift_share),
        "feature_stats": {
            col: {
                "drift_score": info.get("drift_score"),
                "drift_detected": info.get("drift_detected"),
            }
            for col, info in drift_by_col.items()
        },
        "skipped": False,
    }
