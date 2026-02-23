"""Training pipeline DAG.

Topology
--------
validate_data → train_model → register_model → conditional_promote
                                                        │
                                          ┌─────────────┴─────────────┐
                                 promote_to_production          skip_promotion

Tasks
-----
validate_data         Verify the dataset.yaml exists.
train_model           Run YOLOv8n training and log to MLflow.
register_model        Push the run to the MLflow registry at @staging.
conditional_promote   Branch: map50 >= threshold → promote; else skip.
promote_to_production Set the @production alias on the registered model.
skip_promotion        No-op terminal task.

Scheduling
----------
``schedule=None`` — triggered manually or by ``monitoring_pipeline``.

Trigger params
--------------
dataset_yaml   Path to the YOLO dataset.yaml (default: data/processed_subset/dataset.yaml).
resume_from    Optional path to a .pt checkpoint to resume training from.
               Example: runs/detect/train3/weights/last.pt
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.standard.operators.empty import EmptyOperator


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "mlflow"],
    params={
        "dataset_yaml": Param(
            "data/processed_subset/dataset.yaml",
            type="string",
            description="Path to dataset.yaml (relative to /opt/airflow in Docker)",
        ),
        "resume_from": Param(
            None,
            type=["null", "string"],
            description="Optional checkpoint path to resume training from, "
                        "e.g. runs/detect/train3/weights/last.pt",
        ),
    },
)
def training_pipeline() -> None:
    @task()
    def validate_data(**context) -> str:
        dataset_yaml = context["params"]["dataset_yaml"]
        path = Path(dataset_yaml)
        if not path.exists():
            raise FileNotFoundError(
                f"{dataset_yaml} not found — run prepare_dataset first"
            )
        return dataset_yaml

    @task()
    def train_model(dataset_yaml: str, **context) -> dict:
        import yaml as _yaml  # noqa: PLC0415

        from src.training.train import run_training  # noqa: PLC0415

        resume_from = context["params"].get("resume_from")
        config_path = Path("configs/model.yaml")
        cfg = _yaml.safe_load(config_path.read_text())

        orig_variant = cfg["model"]["variant"]
        if resume_from:
            cfg["model"]["variant"] = resume_from
            config_path.write_text(_yaml.dump(cfg))

        try:
            run_id, metrics = run_training(
                config_path=config_path,
                dataset_yaml=Path(dataset_yaml),
            )
        finally:
            if resume_from:
                cfg["model"]["variant"] = orig_variant
                config_path.write_text(_yaml.dump(cfg))

        return {"run_id": run_id, "metrics": metrics}

    @task()
    def register_model(result: dict) -> dict:
        from src.training.mlflow_utils import register_to_staging  # noqa: PLC0415

        config = yaml.safe_load(Path("configs/model.yaml").read_text())
        mlflow_cfg = config["mlflow"]
        version = register_to_staging(
            run_id=result["run_id"],
            model_name=mlflow_cfg["model_name"],
            tracking_uri=mlflow_cfg["tracking_uri"],
        )
        return {"version": version, "metrics": result["metrics"]}

    @task.branch()
    def conditional_promote(registered: dict) -> str:
        config = yaml.safe_load(Path("configs/model.yaml").read_text())
        threshold = config["promotion"]["map50_threshold"]
        map50 = registered["metrics"].get("map50", 0.0)
        if map50 >= threshold:
            return "promote_to_production"
        return "skip_promotion"

    @task()
    def promote_to_production(registered: dict) -> None:
        from src.training.mlflow_utils import (  # noqa: PLC0415
            promote_to_production as _promote,
        )

        config = yaml.safe_load(Path("configs/model.yaml").read_text())
        mlflow_cfg = config["mlflow"]
        threshold = config["promotion"]["map50_threshold"]
        _promote(
            model_name=mlflow_cfg["model_name"],
            version=registered["version"],
            map50=registered["metrics"].get("map50", 0.0),
            threshold=threshold,
            tracking_uri=mlflow_cfg["tracking_uri"],
        )

    skip = EmptyOperator(task_id="skip_promotion")

    # --- wire up ---
    dataset_yaml = validate_data()
    result = train_model(dataset_yaml)
    registered = register_model(result)
    branch = conditional_promote(registered)
    promote_task = promote_to_production(registered)
    branch >> [promote_task, skip]


training_pipeline()
