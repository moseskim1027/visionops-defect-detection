"""Deploy router — model registry and inference service control."""

from typing import Any

from fastapi import APIRouter

from app.services import deploy_service

router = APIRouter()


@router.get("/model-versions")
def model_versions() -> dict[str, Any]:
    return deploy_service.list_model_versions()


@router.post("/promote/{version}")
def promote(version: str) -> dict[str, Any]:
    return deploy_service.promote_model(version)


@router.get("/inference-status")
def inference_status() -> dict[str, Any]:
    return deploy_service.get_inference_status()


@router.post("/reload")
def reload() -> dict[str, Any]:
    return deploy_service.reload_inference()
