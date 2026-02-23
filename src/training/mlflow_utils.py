"""MLflow helpers: experiment management, model registration, and promotion.

Design notes
------------
- Model versions use *aliases* (MLflow 2.x recommended) rather than the
  deprecated stage transitions.  Aliases used: ``staging``, ``production``.
- The ``YOLOPyfuncWrapper`` lets the inference API load the model via the
  standard ``mlflow.pyfunc.load_model("models:/<name>@staging")`` call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"


# ---------------------------------------------------------------------------
# MLflow pyfunc wrapper
# ---------------------------------------------------------------------------


class YOLOPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Thin MLflow wrapper around an Ultralytics YOLO model.

    Artifacts expected in the logged model:
        ``weights``: path to ``best.pt``
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        self.model = YOLO(context.artifacts["weights"])

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input
    ):  # noqa: ANN001
        """Run inference.

        Args:
            model_input: pandas DataFrame with an ``image_path`` (str) column.

        Returns:
            List of raw detection results (one per row).
        """
        results = []
        for _, row in model_input.iterrows():
            preds = self.model.predict(row["image_path"], verbose=False)
            results.append(preds[0].boxes.data.tolist())
        return results


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------


def get_or_create_experiment(name: str, tracking_uri: str) -> str:
    """Return the experiment ID, creating the experiment if it does not exist."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    experiment_id = mlflow.create_experiment(name)
    logger.info("Created MLflow experiment '%s' (id=%s)", name, experiment_id)
    return experiment_id


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------


def register_to_staging(run_id: str, model_name: str, tracking_uri: str) -> int:
    """Register the model logged in ``run_id`` and set the *staging* alias.

    Args:
        run_id:       MLflow run that contains the logged ``model`` artifact.
        model_name:   Registered model name in the MLflow registry.
        tracking_uri: MLflow tracking server URI.

    Returns:
        The new model version number (int).
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    mv: ModelVersion = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.set_registered_model_alias(model_name, STAGING_ALIAS, mv.version)
    logger.info("Registered '%s' v%s → @%s", model_name, mv.version, STAGING_ALIAS)
    return int(mv.version)


def promote_to_production(
    model_name: str,
    version: int,
    map50: float,
    threshold: float,
    tracking_uri: str,
) -> bool:
    """Promote model version to *production* if ``map50`` exceeds threshold.

    Args:
        model_name:   Registered model name.
        version:      Model version to promote.
        map50:        mAP@0.5 achieved during evaluation.
        threshold:    Minimum mAP@0.5 required for promotion.
        tracking_uri: MLflow tracking server URI.

    Returns:
        ``True`` if the model was promoted, ``False`` otherwise.
    """
    if map50 < threshold:
        logger.info(
            "map50=%.4f below threshold=%.4f — skipping promotion",
            map50,
            threshold,
        )
        return False

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
    logger.info(
        "Promoted '%s' v%d → @%s  (map50=%.4f)",
        model_name,
        version,
        PRODUCTION_ALIAS,
        map50,
    )
    return True
