import os
from pathlib import Path

ROOT_DIR = Path(os.getenv("ROOT_DIR", "/workspace"))
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "vision"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = ROOT_DIR / "runs"
CONFIGS_DIR = ROOT_DIR / "configs"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "visionops-defect-detection"
)
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "visionops-yolov8n")

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
