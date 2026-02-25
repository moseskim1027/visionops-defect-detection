from typing import Any

from fastapi import APIRouter

from app.services import data_service

router = APIRouter()


@router.get("/directory-info")
def directory_info(source_dir: str | None = None) -> dict[str, Any]:
    return data_service.get_directory_info(source_dir)


@router.post("/prepare")
def prepare_dataset(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    return data_service.start_preparation(
        source_dir=body.get("source_dir"),
        processed_dir=body.get("processed_dir"),
    )


@router.get("/preparation-status")
def preparation_status() -> dict[str, Any]:
    return data_service.get_preparation_status()


@router.get("/samples")
def annotated_samples(
    n_samples: int = 12,
    split: str = "val",
    processed_dir: str | None = None,
    class_name: str | None = None,
) -> dict[str, Any]:
    samples = data_service.get_annotated_samples(
        processed_dir=processed_dir,
        n_samples=n_samples,
        split=split,
        class_name=class_name,
    )
    return {"samples": samples, "count": len(samples)}
