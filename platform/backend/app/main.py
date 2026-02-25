from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import data, model, training

app = FastAPI(
    title="VisionOps Platform API",
    version="1.0.0",
    description="MLOps platform for defect detection pipeline management",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(model.router, prefix="/api/model", tags=["model"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
