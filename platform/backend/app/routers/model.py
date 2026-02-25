from typing import Any

from fastapi import APIRouter

from app.services import model_service

router = APIRouter()


@router.get("/card")
def model_card(run_id: str | None = None) -> dict[str, Any]:
    return model_service.get_model_card(run_id)


@router.get("/class-metrics")
def class_metrics(
    run_id: str | None = None,
    processed_dir: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    return model_service.get_class_metrics(
        run_id=run_id,
        processed_dir=processed_dir,
        force_refresh=force_refresh,
    )


@router.get("/prediction-distribution")
def prediction_distribution(processed_dir: str | None = None) -> dict[str, Any]:
    dist = model_service.get_prediction_distribution(processed_dir)
    return {"distribution": dist}


@router.get("/poor-samples")
def poor_samples(
    processed_dir: str | None = None,
    n_classes: int = 5,
    n_per_class: int = 2,
    class_name: str | None = None,
) -> dict[str, Any]:
    samples = model_service.get_poor_samples(
        processed_dir=processed_dir,
        n_classes=n_classes,
        n_per_class=n_per_class,
        class_name=class_name,
    )
    return {"samples": samples, "count": len(samples)}
