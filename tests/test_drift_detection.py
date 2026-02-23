"""Tests for src/monitoring/drift_detection.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from src.monitoring.drift_detection import extract_image_features, run_drift_report

# ---------------------------------------------------------------------------
# Helpers
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
        # Pure red image — grayscale std should be ~0
        _save_image(img_path, (200, 200, 200))
        df = extract_image_features([img_path])
        assert df.iloc[0]["contrast"] == pytest.approx(0.0, abs=1e-3)

    def test_sharpness_higher_for_edge_image(self, tmp_path):
        """An image with a sharp edge should have higher gradient energy."""
        # Flat image
        flat = tmp_path / "flat.jpg"
        _save_image(flat, (128, 128, 128))

        # Edge image: left half black, right half white
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
# run_drift_report — Evidently path (mocked)
# ---------------------------------------------------------------------------

_EVIDENTLY_DRIFT_RESULT = {
    "metrics": [
        {
            "metric": "DatasetDriftMetric",
            "result": {
                "dataset_drift": True,
                "share_of_drifted_columns": 0.75,
                "drift_by_columns": {
                    "brightness_mean": {"drift_detected": True, "drift_score": 0.01},
                    "brightness_std": {"drift_detected": False, "drift_score": 0.15},
                    "contrast": {"drift_detected": True, "drift_score": 0.02},
                    "sharpness": {"drift_detected": True, "drift_score": 0.01},
                },
            },
        }
    ]
}

_EVIDENTLY_NO_DRIFT_RESULT = {
    "metrics": [
        {
            "metric": "DatasetDriftMetric",
            "result": {
                "dataset_drift": False,
                "share_of_drifted_columns": 0.0,
                "drift_by_columns": {
                    "brightness_mean": {"drift_detected": False, "drift_score": 0.5},
                    "brightness_std": {"drift_detected": False, "drift_score": 0.4},
                    "contrast": {"drift_detected": False, "drift_score": 0.3},
                    "sharpness": {"drift_detected": False, "drift_score": 0.4},
                },
            },
        }
    ]
}


@pytest.fixture()
def dirs_with_enough_images(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("drift:\n  min_reference_samples: 5\n  min_current_samples: 3\n")
    ref = _make_image_dir(tmp_path, "ref", n=6, color=(100, 100, 100))
    cur = _make_image_dir(tmp_path, "cur", n=4, color=(200, 200, 200))
    return ref, cur, cfg


class TestRunDriftReportEvidently:
    def _mock_report(self, as_dict_return: dict) -> MagicMock:
        mock_instance = MagicMock()
        mock_instance.as_dict.return_value = as_dict_return
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls

    def test_detects_drift(self, dirs_with_enough_images):
        ref, cur, cfg = dirs_with_enough_images
        mock_cls = self._mock_report(_EVIDENTLY_DRIFT_RESULT)
        with patch("evidently.report.Report", mock_cls):
            with patch("evidently.metric_preset.DataDriftPreset"):
                result = run_drift_report(ref, cur, config_path=cfg)
        assert result["drift_detected"] is True
        assert result["drift_share"] == pytest.approx(0.75)
        assert "brightness_mean" in result["drifted_features"]
        assert "brightness_std" not in result["drifted_features"]
        assert result["skipped"] is False

    def test_no_drift(self, dirs_with_enough_images):
        ref, cur, cfg = dirs_with_enough_images
        mock_cls = self._mock_report(_EVIDENTLY_NO_DRIFT_RESULT)
        with patch("evidently.report.Report", mock_cls):
            with patch("evidently.metric_preset.DataDriftPreset"):
                result = run_drift_report(ref, cur, config_path=cfg)
        assert result["drift_detected"] is False
        assert result["drifted_features"] == []
        assert result["drift_share"] == pytest.approx(0.0)

    def test_feature_stats_populated(self, dirs_with_enough_images):
        ref, cur, cfg = dirs_with_enough_images
        mock_cls = self._mock_report(_EVIDENTLY_DRIFT_RESULT)
        with patch("evidently.report.Report", mock_cls):
            with patch("evidently.metric_preset.DataDriftPreset"):
                result = run_drift_report(ref, cur, config_path=cfg)
        assert "brightness_mean" in result["feature_stats"]
        assert result["feature_stats"]["brightness_mean"][
            "drift_score"
        ] == pytest.approx(0.01)

    def test_missing_dataset_metric_returns_no_drift(self, dirs_with_enough_images):
        ref, cur, cfg = dirs_with_enough_images
        mock_cls = self._mock_report({"metrics": []})
        with patch("evidently.report.Report", mock_cls):
            with patch("evidently.metric_preset.DataDriftPreset"):
                result = run_drift_report(ref, cur, config_path=cfg)
        assert result["drift_detected"] is False
        assert result["skipped"] is False
