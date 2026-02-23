"""YOLOv8n training script with MLflow experiment tracking.

Responsibilities
----------------
1. Load training config from ``configs/model.yaml``.
2. Run YOLOv8n training via the Ultralytics API.
3. Log hyperparameters, metrics, and the model artefact to MLflow.
4. Return the MLflow run ID and final metrics dict so the Airflow DAG can
   pass them downstream (register_model / conditional_promote tasks).

Registration (staging → production) is intentionally *not* performed here —
that is delegated to the Airflow DAG to maintain clean task separation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import yaml

from src.training.mlflow_utils import YOLOPyfuncWrapper, get_or_create_experiment

logger = logging.getLogger(__name__)

# Ultralytics metric keys → clean names logged to MLflow
_METRIC_ALIASES: dict[str, str] = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
}


def parse_yolo_metrics(results_dict: dict) -> dict[str, float]:
    """Translate Ultralytics ``results_dict`` keys to clean MLflow metric names.

    Unknown keys are ignored so the function remains forward-compatible with
    new Ultralytics releases.
    """
    return {
        clean: float(results_dict[raw])
        for raw, clean in _METRIC_ALIASES.items()
        if raw in results_dict
    }


def run_training(
    config_path: Path = Path("configs/model.yaml"),
    dataset_yaml: Path = Path("data/processed/dataset.yaml"),
) -> tuple[str, dict[str, float]]:
    """Train YOLOv8n and log everything to MLflow.

    Args:
        config_path:  Path to ``configs/model.yaml``.
        dataset_yaml: Path to the processed ``dataset.yaml`` for Ultralytics.

    Returns:
        ``(run_id, metrics)`` where ``metrics`` contains at minimum ``map50``.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    config = yaml.safe_load(config_path.read_text())
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}. "
            "Run src/data/prepare_dataset.py first."
        )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg["tracking_uri"]
    mlflow.autolog(disable=True)
    # ray is installed as a transitive dep but its version is incompatible with the
    # Ultralytics raytune callback. Patch the missing symbol so the callback is a no-op.
    try:
        import ray.train._internal.session as _ray_session
        if not hasattr(_ray_session, "_get_session"):
            _ray_session._get_session = lambda: None
    except Exception:
        pass
    experiment_id = get_or_create_experiment(
        mlflow_cfg["experiment_name"],
        tracking_uri,
    )

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_params(
            {
                "model_variant": model_cfg["variant"],
                "imgsz": model_cfg["imgsz"],
                "epochs": train_cfg["epochs"],
                "batch": train_cfg["batch"],
                "lr0": train_cfg["lr0"],
                "patience": train_cfg["patience"],
                "device": train_cfg["device"],
                "dataset_yaml": str(dataset_yaml),
            }
        )

        logger.info("Starting YOLOv8n training (run_id=%s)", run.info.run_id)
        model = YOLO(model_cfg["variant"])
        results = model.train(
            data=str(dataset_yaml),
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch"],
            imgsz=model_cfg["imgsz"],
            lr0=train_cfg["lr0"],
            patience=train_cfg["patience"],
            workers=train_cfg["workers"],
            device=train_cfg["device"],
            verbose=False,
        )

        metrics = parse_yolo_metrics(results.results_dict)
        mlflow.log_metrics(metrics)
        logger.info("Training metrics: %s", metrics)

        # Log supplementary artefacts
        save_dir = Path(results.save_dir)
        for artefact, subfolder in [
            ("results.csv", "training_logs"),
            ("confusion_matrix.png", "plots"),
            ("PR_curve.png", "plots"),
        ]:
            path = save_dir / artefact
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=subfolder)

        # Log the YOLO model as a pyfunc so it can be registered and loaded
        # later via  mlflow.pyfunc.load_model("models:/<name>@staging")
        best_weights = save_dir / "weights" / "best.pt"
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YOLOPyfuncWrapper(),
            artifacts={"weights": str(best_weights)},
            pip_requirements=["ultralytics"],
        )

    return run.info.run_id, metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train YOLOv8n and log to MLflow")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--dataset", default="data/processed/dataset.yaml")
    args = parser.parse_args()

    run_id, metrics = run_training(Path(args.config), Path(args.dataset))
    print(f"run_id: {run_id}")
    print(f"map50:  {metrics.get('map50', 'N/A'):.4f}")
