"""Tests for src/training/train.py and src/training/mlflow_utils.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.training.train import parse_yolo_metrics

# ---------------------------------------------------------------------------
# parse_yolo_metrics  (pure function — no mocking needed)
# ---------------------------------------------------------------------------


class TestParseYoloMetrics:
    def test_translates_known_keys(self):
        raw = {
            "metrics/mAP50(B)": 0.55,
            "metrics/mAP50-95(B)": 0.32,
            "metrics/precision(B)": 0.71,
            "metrics/recall(B)": 0.68,
        }
        metrics = parse_yolo_metrics(raw)
        assert metrics["map50"] == pytest.approx(0.55)
        assert metrics["map50_95"] == pytest.approx(0.32)
        assert metrics["precision"] == pytest.approx(0.71)
        assert metrics["recall"] == pytest.approx(0.68)

    def test_ignores_unknown_keys(self):
        raw = {"metrics/mAP50(B)": 0.4, "some/unknown_key": 99.0}
        metrics = parse_yolo_metrics(raw)
        assert "map50" in metrics
        assert "some/unknown_key" not in metrics
        assert "unknown_key" not in metrics

    def test_returns_empty_for_empty_input(self):
        assert parse_yolo_metrics({}) == {}

    def test_values_cast_to_float(self):
        raw = {"metrics/mAP50(B)": "0.45"}
        metrics = parse_yolo_metrics(raw)
        assert isinstance(metrics["map50"], float)


# ---------------------------------------------------------------------------
# promote_to_production  (threshold logic — MLflow client mocked)
# ---------------------------------------------------------------------------


class TestPromoteToProduction:
    _TRACKING_URI = "http://localhost:5000"
    _MODEL_NAME = "visionops-yolov8n"

    def _call(self, map50: float, threshold: float = 0.30) -> bool:
        from src.training.mlflow_utils import promote_to_production

        with (
            patch("src.training.mlflow_utils.mlflow.set_tracking_uri"),
            patch("src.training.mlflow_utils.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            result = promote_to_production(
                self._MODEL_NAME, 1, map50, threshold, self._TRACKING_URI
            )
            return result, mock_client

    def test_promotes_when_above_threshold(self):
        result, mock_client = self._call(map50=0.45)
        assert result is True
        mock_client.set_registered_model_alias.assert_called_once()

    def test_skips_when_below_threshold(self):
        result, mock_client = self._call(map50=0.10)
        assert result is False
        mock_client.set_registered_model_alias.assert_not_called()

    def test_promotes_when_equal_to_threshold(self):
        # Threshold is a minimum — exactly meeting it should promote
        result, mock_client = self._call(map50=0.30, threshold=0.30)
        assert result is True

    def test_sets_production_alias(self):
        _, mock_client = self._call(map50=0.50)
        call_args = mock_client.set_registered_model_alias.call_args
        assert call_args.args[1] == "production"


# ---------------------------------------------------------------------------
# register_to_staging  (MLflow register_model + alias mocked)
# ---------------------------------------------------------------------------


class TestRegisterToStaging:
    def test_returns_version_number(self):
        from src.training.mlflow_utils import register_to_staging

        mock_mv = MagicMock()
        mock_mv.version = "3"

        with (
            patch("src.training.mlflow_utils.mlflow.set_tracking_uri"),
            patch(
                "src.training.mlflow_utils.mlflow.register_model",
                return_value=mock_mv,
            ),
            patch("src.training.mlflow_utils.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            version = register_to_staging(
                "run-abc", "my-model", "http://localhost:5000"
            )

        assert version == 3

    def test_sets_staging_alias(self):
        from src.training.mlflow_utils import register_to_staging

        mock_mv = MagicMock()
        mock_mv.version = "1"

        with (
            patch("src.training.mlflow_utils.mlflow.set_tracking_uri"),
            patch(
                "src.training.mlflow_utils.mlflow.register_model",
                return_value=mock_mv,
            ),
            patch("src.training.mlflow_utils.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            register_to_staging("run-abc", "my-model", "http://localhost:5000")
            call_args = mock_client.set_registered_model_alias.call_args
            assert call_args.args[1] == "staging"
