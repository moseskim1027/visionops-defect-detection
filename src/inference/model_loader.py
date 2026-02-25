"""YOLO model loader — supports MLflow registry and local weights fallback.

Loading priority
----------------
1. MLflow registry  (``MLFLOW_TRACKING_URI`` + ``MLFLOW_MODEL_NAME`` set)
2. Local weights    (``YOLO_WEIGHTS_PATH`` env var)
3. Ultralytics default ``yolov8n.pt`` (downloaded on first use)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class ModelLoader:
    """Loads and wraps a YOLO model for structured inference."""

    def __init__(self) -> None:
        self._model = None
        self._class_names: dict[int, str] = {}
        self.run_id: str = ""
        self.model_version: str = ""
        self.model_alias: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        weights_path: Path | None = None,
        class_map_path: Path | None = None,
    ) -> None:
        """Load YOLO model from a local weights file.

        Args:
            weights_path:   Path to ``best.pt``.  Falls back to
                            ``YOLO_WEIGHTS_PATH`` env var, then ``yolov8n.pt``.
            class_map_path: Optional path to ``class_map.json`` produced by
                            the data pipeline.
        """
        from ultralytics import YOLO  # noqa: PLC0415

        path = weights_path or Path(os.environ.get("YOLO_WEIGHTS_PATH", "yolov8n.pt"))
        self._model = YOLO(str(path))
        self._load_class_map(class_map_path)
        logger.info("Model loaded from %s", path)

    def load_from_mlflow(
        self,
        model_name: str,
        alias: str,
        tracking_uri: str,
        class_map_path: Path | None = None,
    ) -> None:
        """Download YOLO weights from MLflow registry and load.

        The model must have been logged with :class:`YOLOPyfuncWrapper` which
        stores weights under the ``weights`` artifact key.
        """
        import mlflow  # noqa: PLC0415
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        mv = client.get_model_version_by_alias(model_name, alias)
        run_id = mv.run_id

        local_path = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/model/artifacts/weights"
        )
        candidates = list(Path(local_path).glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt weights found in {local_path}")

        self.load(candidates[0], class_map_path)
        self.run_id = run_id
        self.model_version = mv.version
        self.model_alias = alias

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> list[Detection]:
        """Run inference and return structured detections.

        Args:
            image_path: Path to the image file.

        Returns:
            List of :class:`Detection` objects.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self._model.predict(image_path, verbose=False)
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=round(conf, 4),
                        bbox=[round(v, 2) for v in [x1, y1, x2, y2]],
                    )
                )

        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_class_map(self, path: Path | None) -> None:
        if path and path.exists():
            class_map: dict[str, int] = json.loads(path.read_text())
            self._class_names = {v: k for k, v in class_map.items()}
            logger.info("Loaded %d class names from %s", len(self._class_names), path)
