"""Simulates distribution drift on the processed dataset.

Applies image-level transformations to a random sample and saves results to
data/drift_batches/.  The monitoring DAG uses these batches to trigger
retraining when drift is detected.

Drift types:
    brightness  — multiply pixel values by a constant factor
    noise       — add zero-mean Gaussian noise
    blur        — apply Gaussian blur
    mixed       — randomly combine two of the above
"""

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

DriftType = Literal["brightness", "noise", "blur", "mixed"]


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Multiply pixel values by factor  (< 1 = darker, > 1 = brighter)."""
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_noise(img: Image.Image, std: float) -> Image.Image:
    """Add zero-mean Gaussian noise with the given standard deviation."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_transform(
    img: Image.Image,
    drift_type: DriftType,
    severity: float,
) -> Image.Image:
    """Dispatch to the appropriate transform based on drift_type."""
    if drift_type == "brightness":
        return apply_brightness(img, factor=1.0 + severity)
    if drift_type == "noise":
        return apply_noise(img, std=severity * 50)
    if drift_type == "blur":
        return apply_blur(img, radius=severity * 5)

    # mixed: pick and chain two random transforms
    fns = [
        lambda i: apply_brightness(i, factor=1.0 + severity * 0.5),
        lambda i: apply_noise(i, std=severity * 25),
        lambda i: apply_blur(i, radius=severity * 2),
    ]
    first, second = random.sample(fns, 2)
    return second(first(img))


def simulate_drift(
    src_dir: Path,
    dst_dir: Path,
    drift_type: DriftType = "brightness",
    severity: float = 0.5,
    sample_fraction: float = 0.3,
    split: str = "val",
    seed: int = 42,
) -> Path:
    """Apply drift transforms to a random sample of processed images.

    Args:
        src_dir:         Processed dataset root   (e.g. data/processed).
        dst_dir:         Drift batches root        (e.g. data/drift_batches).
        drift_type:      One of brightness | noise | blur | mixed.
        severity:        Drift intensity in [0, 1].
        sample_fraction: Fraction of images to include in the batch.
        split:           Dataset split to sample from.
        seed:            Random seed for reproducibility.

    Returns:
        Path to the generated batch directory.
    """
    img_dir = src_dir / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"No images at {img_dir}. Run prepare_dataset first.")

    random.seed(seed)
    np.random.seed(seed)

    all_images = sorted(img_dir.glob("*.jpg"))
    sample_size = max(1, int(len(all_images) * sample_fraction))
    sampled = random.sample(all_images, sample_size)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = dst_dir / f"batch_{drift_type}_{timestamp}"
    batch_img_dir = batch_dir / "images"
    batch_img_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sampled:
        img = Image.open(img_path).convert("RGB")
        drifted = _apply_transform(img, drift_type, severity)
        drifted.save(batch_img_dir / img_path.name)

    metadata = {
        "timestamp": timestamp,
        "drift_type": drift_type,
        "severity": severity,
        "sample_fraction": sample_fraction,
        "split": split,
        "seed": seed,
        "num_images": len(sampled),
        "source_dir": str(src_dir),
    }
    (batch_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Drift batch created: %s  (%d images, type=%s, severity=%.2f)",
        batch_dir,
        len(sampled),
        drift_type,
        severity,
    )
    return batch_dir


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Simulate dataset drift")
    parser.add_argument("--src", default="data/processed")
    parser.add_argument("--dst", default="data/drift_batches")
    parser.add_argument(
        "--type",
        dest="drift_type",
        choices=["brightness", "noise", "blur", "mixed"],
        default="brightness",
    )
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    simulate_drift(
        Path(args.src),
        Path(args.dst),
        drift_type=args.drift_type,
        severity=args.severity,
        sample_fraction=args.fraction,
        split=args.split,
        seed=args.seed,
    )
