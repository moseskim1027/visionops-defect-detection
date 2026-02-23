"""Tests for src/monitoring/drift_detection.py

Evidently path tests use real Casting val images (data/raw/vision/Casting/val/).
They are skipped automatically when the dataset is not present (e.g. CI without
the raw data volume), so the test suite always passes in both environments.

The only remaining mock is the edge-case test for a missing DatasetDriftMetric
in Evidently's output — that branch cannot be exercised with real data.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from src.data.drift_simulator import apply_brightness
from src.monitoring.drift_detection import extract_image_features, run_drift_report

# ---------------------------------------------------------------------------
# Real-data constants
# ---------------------------------------------------------------------------

CASTING_VAL = Path("data/raw/vision/Casting/val")
HAS_DATASET = CASTING_VAL.exists() and len(list(CASTING_VAL.glob("*.jpg"))) >= 50

# ---------------------------------------------------------------------------
# Helpers (used by skip-logic tests)
# ---------------------------------------------------------------------------


def _save_image(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (16, 16)
) -> None:
    PILImage.new("RGB", size, color=color).save(path)


def _make_image_dir(base: Path, name: str, n: int, color: tuple[int, int, int]) -> Path:
    d = base / name
    d.mkdir()
    for i in range(n):
        _save_image(d / f"img_{i:03d}.jpg", color)
    return d


# ---------------------------------------------------------------------------
# extract_image_features
# ---------------------------------------------------------------------------


class TestExtractImageFeatures:
    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        img_path = tmp_path / "a.jpg"
        _save_image(img_path, (128, 128, 128))
        df = extract_image_features([img_path])
        assert set(df.columns) == {
            "brightness_mean",
            "brightness_std",
            "contrast",
            "sharpness",
        }

    def test_one_row_per_image(self, tmp_path):
        for i in range(5):
            _save_image(tmp_path / f"img_{i}.jpg", (i * 40, i * 40, i * 40))
        paths = sorted(tmp_path.glob("*.jpg"))
        df = extract_image_features(paths)
        assert len(df) == 5

    def test_empty_list_returns_empty_dataframe(self):
        df = extract_image_features([])
        assert len(df) == 0

    def test_bright_image_higher_mean_than_dark(self, tmp_path):
        bright = tmp_path / "bright.jpg"
        dark = tmp_path / "dark.jpg"
        _save_image(bright, (240, 240, 240))
        _save_image(dark, (10, 10, 10))
        df = extract_image_features([bright, dark])
        assert df.iloc[0]["brightness_mean"] > df.iloc[1]["brightness_mean"]

    def test_uniform_image_has_zero_contrast(self, tmp_path):
        img_path = tmp_path / "uniform.jpg"
        _save_image(img_path, (200, 200, 200))
        df = extract_image_features([img_path])
        assert df.iloc[0]["contrast"] == pytest.approx(0.0, abs=1e-3)

    def test_sharpness_higher_for_edge_image(self, tmp_path):
        """An image with a sharp edge should have higher gradient energy."""
        flat = tmp_path / "flat.jpg"
        _save_image(flat, (128, 128, 128))

        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[:, 8:, :] = 255
        edge = tmp_path / "edge.png"
        PILImage.fromarray(arr).save(edge)

        df = extract_image_features([flat, edge])
        assert df.iloc[1]["sharpness"] > df.iloc[0]["sharpness"]

    def test_values_are_finite(self, tmp_path):
        img_path = tmp_path / "img.jpg"
        _save_image(img_path, (100, 150, 200))
        df = extract_image_features([img_path])
        assert df.notna().all().all()


# ---------------------------------------------------------------------------
# run_drift_report — skip behaviour (no Evidently needed)
# ---------------------------------------------------------------------------


class TestRunDriftReportSkip:
    def test_skips_when_too_few_reference_images(self, tmp_path):
        ref = _make_image_dir(tmp_path, "ref", n=3, color=(100, 100, 100))
        cur = _make_image_dir(tmp_path, "cur", n=25, color=(200, 200, 200))
        cfg = tmp_path / "drift.yaml"
        cfg.write_text(
            "drift:\n  min_reference_samples: 50\n  min_current_samples: 20\n"
        )
        result = run_drift_report(ref, cur, config_path=cfg)
        assert result["skipped"] is True
        assert result["drift_detected"] is False

    def test_skips_when_too_few_current_images(self, tmp_path):
        ref = _make_image_dir(tmp_path, "ref", n=55, color=(100, 100, 100))
        cur = _make_image_dir(tmp_path, "cur", n=5, color=(200, 200, 200))
        cfg = tmp_path / "drift.yaml"
        cfg.write_text(
            "drift:\n  min_reference_samples: 50\n  min_current_samples: 20\n"
        )
        result = run_drift_report(ref, cur, config_path=cfg)
        assert result["skipped"] is True

    def test_uses_defaults_when_config_missing(self, tmp_path):
        """With no config, defaults are 50 ref / 20 cur; 3 images → skip."""
        ref = _make_image_dir(tmp_path, "ref", n=3, color=(100, 100, 100))
        cur = _make_image_dir(tmp_path, "cur", n=3, color=(200, 200, 200))
        result = run_drift_report(ref, cur, config_path=tmp_path / "nonexistent.yaml")
        assert result["skipped"] is True


# ---------------------------------------------------------------------------
# run_drift_report — real Casting images + real Evidently
# ---------------------------------------------------------------------------

# Drift config that matches the actual Casting val set (51 images ≥ min 50)
_DRIFT_CFG_YAML = "drift:\n  min_reference_samples: 50\n  min_current_samples: 20\n"


@pytest.fixture(scope="module")
def drift_config(tmp_path_factory) -> Path:
    cfg = tmp_path_factory.mktemp("cfg") / "drift.yaml"
    cfg.write_text(_DRIFT_CFG_YAML)
    return cfg


@pytest.fixture(scope="module")
def bright_batch(tmp_path_factory) -> Path:
    """25 Casting val images with factor-2.5 brightness — extreme drift."""
    out = tmp_path_factory.mktemp("bright_batch")
    images = sorted(CASTING_VAL.glob("*.jpg"))[:25]
    for p in images:
        img = PILImage.open(p).convert("RGB")
        apply_brightness(img, factor=2.5).save(out / p.name)
    return out


@pytest.mark.skipif(not HAS_DATASET, reason="Casting val images not found")
class TestRunDriftReportRealData:
    def test_detects_brightness_drift(self, bright_batch, drift_config):
        """Extreme brightness shift on real images must be detected."""
        result = run_drift_report(CASTING_VAL, bright_batch, config_path=drift_config)
        assert result["skipped"] is False
        assert result["drift_detected"] is True

    def test_brightness_mean_flagged_as_drifted(self, bright_batch, drift_config):
        """brightness_mean should be among the drifted features."""
        result = run_drift_report(CASTING_VAL, bright_batch, config_path=drift_config)
        assert "brightness_mean" in result["drifted_features"]

    def test_no_drift_same_source(self, drift_config):
        """Reference and current from the same directory → no drift."""
        result = run_drift_report(CASTING_VAL, CASTING_VAL, config_path=drift_config)
        assert result["skipped"] is False
        assert result["drift_detected"] is False

    def test_return_structure(self, bright_batch, drift_config):
        result = run_drift_report(CASTING_VAL, bright_batch, config_path=drift_config)
        assert set(result.keys()) >= {
            "drift_detected",
            "drifted_features",
            "drift_share",
            "feature_stats",
            "skipped",
        }

    def test_drift_share_in_valid_range(self, bright_batch, drift_config):
        result = run_drift_report(CASTING_VAL, bright_batch, config_path=drift_config)
        assert 0.0 <= result["drift_share"] <= 1.0

    def test_feature_stats_keys(self, bright_batch, drift_config):
        result = run_drift_report(CASTING_VAL, bright_batch, config_path=drift_config)
        for key in ("brightness_mean", "brightness_std", "contrast", "sharpness"):
            assert key in result["feature_stats"]
            stats = result["feature_stats"][key]
            assert "drift_detected" in stats
            assert "drift_score" in stats


# ---------------------------------------------------------------------------
# run_drift_report — edge-case: Evidently returns no DatasetDriftMetric
# This branch cannot be exercised with real data; mock is intentional.
# ---------------------------------------------------------------------------


class TestRunDriftReportEdgeCases:
    def test_empty_metrics_returns_no_drift(self, tmp_path):
        """When Evidently returns no metrics at all, safe default is returned."""
        cfg = tmp_path / "drift.yaml"
        cfg.write_text("drift:\n  min_reference_samples: 5\n  min_current_samples: 3\n")
        ref = _make_image_dir(tmp_path, "ref", n=6, color=(100, 100, 100))
        cur = _make_image_dir(tmp_path, "cur", n=4, color=(200, 200, 200))

        mock_instance = MagicMock()
        mock_instance.as_dict.return_value = {"metrics": []}
        with patch("evidently.report.Report", MagicMock(return_value=mock_instance)):
            with patch("evidently.metric_preset.DataDriftPreset"):
                result = run_drift_report(ref, cur, config_path=cfg)

        assert result["drift_detected"] is False
        assert result["skipped"] is False
