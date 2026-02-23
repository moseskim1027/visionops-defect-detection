"""Monitoring pipeline DAG — runs daily to detect image distribution drift.

Topology
--------
simulate_drift → run_drift_detection → branch_on_drift
                                               │
                                ┌──────────────┴──────────────┐
                        trigger_retraining              end_no_drift

Tasks
-----
simulate_drift_task      Apply a drift transform to a sample of val images,
                         save batch to ``data/drift_batches/``.
run_drift_detection_task Extract image features and run Evidently drift report
                         comparing val reference vs. the new batch.
branch_on_drift          Return ``trigger_retraining`` when drift is detected,
                         ``end_no_drift`` otherwise.
trigger_retraining       Fire the ``training_pipeline`` DAG (non-blocking).
end_no_drift             No-op terminal task.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from airflow.decorators import dag, task
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="monitoring_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
)
def monitoring_pipeline() -> None:
    @task()
    def simulate_drift_task() -> str:
        from src.data.drift_simulator import simulate_drift  # noqa: PLC0415

        config = yaml.safe_load(Path("configs/data.yaml").read_text())
        drift_cfg = config.get("drift_simulation", {})
        batch_dir = simulate_drift(
            src_dir=Path(config["dataset"]["processed_dir"]),
            dst_dir=Path(config["dataset"]["drift_batches_dir"]),
            drift_type=drift_cfg.get("default_type", "brightness"),
            severity=drift_cfg.get("severity", 0.5),
            sample_fraction=drift_cfg.get("sample_fraction", 0.3),
            split=drift_cfg.get("split", "val"),
            seed=drift_cfg.get("seed", 42),
        )
        return str(batch_dir)

    @task()
    def run_drift_detection_task(batch_dir: str) -> dict:
        from src.monitoring.drift_detection import run_drift_report  # noqa: PLC0415

        config = yaml.safe_load(Path("configs/data.yaml").read_text())
        processed_dir = Path(config["dataset"]["processed_dir"])
        return run_drift_report(
            reference_dir=processed_dir / "images" / "val",
            current_dir=Path(batch_dir) / "images",
            config_path=Path("configs/drift.yaml"),
        )

    @task.branch()
    def branch_on_drift(report: dict) -> str:
        if report.get("drift_detected", False):
            return "trigger_retraining"
        return "end_no_drift"

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )
    end_no_drift = EmptyOperator(task_id="end_no_drift")

    batch_dir = simulate_drift_task()
    report = run_drift_detection_task(batch_dir)
    branch = branch_on_drift(report)
    branch >> [trigger_retraining, end_no_drift]


monitoring_pipeline()
