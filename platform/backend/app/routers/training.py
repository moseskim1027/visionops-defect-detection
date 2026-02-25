from typing import Any

from fastapi import APIRouter

from app.services import training_service

router = APIRouter()


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    return training_service.get_model_info()


@router.get("/data-card")
def data_card(processed_dir: str | None = None) -> dict[str, Any]:
    distribution = training_service.get_class_distribution(processed_dir)
    return {"distribution": distribution, "total_classes": len(distribution)}


@router.get("/config")
def training_config() -> dict[str, Any]:
    return training_service.get_training_config()


@router.put("/config")
def update_config(body: dict[str, Any]) -> dict[str, Any]:
    return training_service.update_training_config(body)


@router.post("/start")
def start_training(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    return training_service.start_training(
        dataset_yaml=body.get("dataset_yaml"),
        config_overrides=body.get("config"),
    )


@router.get("/status")
def training_status() -> dict[str, Any]:
    return training_service.get_training_status()


@router.get("/metrics/{run_id}")
def mlflow_metrics(run_id: str) -> dict[str, Any]:
    return training_service.get_mlflow_metrics(run_id)


@router.get("/runs")
def list_runs(max_results: int = 10) -> dict[str, Any]:
    runs = training_service.list_mlflow_runs(max_results)
    return {"runs": runs}
