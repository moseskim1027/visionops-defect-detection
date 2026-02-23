"""Tests for src/data/drift_simulator.py"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.drift_simulator import (
    apply_blur,
    apply_brightness,
    apply_noise,
    simulate_drift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_image(size: tuple[int, int] = (64, 64), value: int = 128) -> Image.Image:
    arr = np.full((*size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def make_processed_images(base: Path, n: int = 5, split: str = "val") -> Path:
    img_dir = base / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make_image().save(img_dir / f"img_{i:03d}.jpg")
    return base


# ---------------------------------------------------------------------------
# apply_brightness
# ---------------------------------------------------------------------------


class TestApplyBrightness:
    def test_darkens(self):
        img = make_image(value=128)
        result = apply_brightness(img, factor=0.5)
        assert np.mean(np.array(result)) < np.mean(np.array(img))

    def test_brightens(self):
        img = make_image(value=64)
        result = apply_brightness(img, factor=2.0)
        assert np.mean(np.array(result)) > np.mean(np.array(img))

    def test_clamps_to_valid_range(self):
        img = make_image(value=200)
        result = apply_brightness(img, factor=100.0)
        arr = np.array(result)
        assert arr.max() <= 255
        assert arr.min() >= 0

    def test_identity_at_factor_one(self):
        img = make_image(value=100)
        result = apply_brightness(img, factor=1.0)
        np.testing.assert_array_equal(np.array(result), np.array(img))


# ---------------------------------------------------------------------------
# apply_noise
# ---------------------------------------------------------------------------


class TestApplyNoise:
    def test_changes_pixel_values(self):
        img = make_image(value=128)
        result = apply_noise(img, std=50.0)
        assert not np.array_equal(np.array(img), np.array(result))

    def test_output_in_valid_range(self):
        img = make_image(value=128)
        result = apply_noise(img, std=50.0)
        arr = np.array(result)
        assert arr.max() <= 255
        assert arr.min() >= 0

    def test_zero_std_is_identity(self):
        img = make_image(value=100)
        result = apply_noise(img, std=0.0)
        np.testing.assert_array_equal(np.array(result), np.array(img))


# ---------------------------------------------------------------------------
# apply_blur
# ---------------------------------------------------------------------------


class TestApplyBlur:
    def test_returns_image(self):
        img = make_image()
        result = apply_blur(img, radius=2.0)
        assert isinstance(result, Image.Image)
        assert result.size == img.size


# ---------------------------------------------------------------------------
# simulate_drift
# ---------------------------------------------------------------------------


class TestSimulateDrift:
    def test_creates_batch_directory(self, tmp_path):
        src = make_processed_images(tmp_path / "processed")
        batch_dir = simulate_drift(src, tmp_path / "batches")
        assert batch_dir.exists()

    def test_metadata_content(self, tmp_path):
        src = make_processed_images(tmp_path / "processed")
        batch_dir = simulate_drift(
            src, tmp_path / "batches", drift_type="noise", severity=0.3
        )
        meta = json.loads((batch_dir / "metadata.json").read_text())
        assert meta["drift_type"] == "noise"
        assert meta["severity"] == pytest.approx(0.3)
        assert "timestamp" in meta

    def test_sample_fraction(self, tmp_path):
        src = make_processed_images(tmp_path / "processed", n=10)
        batch_dir = simulate_drift(
            src, tmp_path / "batches", sample_fraction=0.5, seed=0
        )
        images = list((batch_dir / "images").glob("*.jpg"))
        assert len(images) == 5

    def test_minimum_one_image(self, tmp_path):
        src = make_processed_images(tmp_path / "processed", n=1)
        batch_dir = simulate_drift(src, tmp_path / "batches", sample_fraction=0.01)
        images = list((batch_dir / "images").glob("*.jpg"))
        assert len(images) == 1

    def test_all_drift_types(self, tmp_path):
        src = make_processed_images(tmp_path / "processed")
        for drift_type in ("brightness", "noise", "blur", "mixed"):
            batch_dir = simulate_drift(
                src,
                tmp_path / "batches",
                drift_type=drift_type,  # type: ignore[arg-type]
            )
            assert batch_dir.exists()

    def test_raises_if_src_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            simulate_drift(tmp_path / "nonexistent", tmp_path / "batches")
