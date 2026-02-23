"""FastAPI inference service — stub (implemented in feat/inference-api)."""

from fastapi import FastAPI

app = FastAPI(title="VisionOps Inference", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
