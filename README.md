# visionops-defect-detection

An end-to-end MLOps platform for industrial defect detection using the [VISION dataset](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets). Designed to demonstrate production engineering patterns — not just model accuracy.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Apache Airflow                          │
│   ┌──────────────────────────┐  ┌───────────────────────────┐   │
│   │     Training Pipeline    │  │   Monitoring Pipeline     │   │
│   │  validate → prepare →    │  │  collect → drift_check →  │   │
│   │  train → eval → register │  │  branch → [retrain|end]   │   │
│   └──────────────────────────┘  └───────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────┘
                                 │ triggers / registers
                   ┌─────────────▼──────────────┐
                   │         MLflow             │
                   │  Experiment Tracking       │
                   │  Model Registry            │
                   │  Artifact Storage          │
                   └─────────────┬──────────────┘
                                 │ loads model
                   ┌─────────────▼──────────────┐
                   │    FastAPI Inference       │
                   │  POST /predict             │
                   │  GET  /health              │
                   │  GET  /metrics             │
                   └────────────┬───────────────┘
                                │ deployed to
                   ┌────────────▼───────────────┐
                   │     Kubernetes (kind)      │
                   │  Deployment + Service      │
                   └────────────┬───────────────┘
                                │ metrics scraped by
                   ┌────────────▼───────────────┐
                   │  Prometheus + Grafana      │
                   │  Evidently Drift Reports   │
                   └────────────────────────────┘
```

---

## Tech Stack

| Component        | Technology              |
|------------------|-------------------------|
| Dataset          | VISION (HuggingFace)    |
| Model            | YOLOv8n (Ultralytics)   |
| Orchestration    | Apache Airflow 2.10     |
| Model Registry   | MLflow 2.19             |
| Serving          | FastAPI + Uvicorn       |
| Containerisation | Docker + Kubernetes     |
| Drift Monitoring | Evidently               |
| Infra Monitoring | Prometheus + Grafana    |
| Linting          | Ruff + Black            |

---

## MLOps Capabilities

- **Dataset validation** — checks integrity before every pipeline run
- **COCO → YOLO conversion** — automated annotation transformation
- **Experiment tracking** — hyperparams, metrics, and artifacts in MLflow
- **Model versioning** — staging → production promotion with metric gate
- **Containerised serving** — Docker image with CPU-only PyTorch
- **Kubernetes deployment** — local `kind` cluster
- **Drift detection** — Evidently monitors bbox, confidence, and pixel distributions
- **Automated retraining** — monitoring DAG triggers training DAG on drift
- **CI checks** — lint (ruff + black) and Docker build on every push

---

## Repository Structure

```
visionops-defect-detection/
├── .github/workflows/      # CI: lint + docker build
├── configs/                # YAML configs (model, training, drift thresholds)
├── dags/                   # Airflow DAGs
├── data/
│   ├── raw/vision/         # Dataset (not committed — .gitkeep)
│   ├── processed/          # YOLO-format annotations
│   └── drift_batches/      # Simulated drift data
├── docker/                 # Dockerfiles + inference requirements
├── k8s/                    # Kubernetes manifests
├── src/
│   ├── data/               # prepare_dataset, drift_simulator
│   ├── features/           # Feature extraction utilities
│   ├── training/           # YOLOv8n training + MLflow logging
│   ├── inference/          # FastAPI service
│   └── monitoring/         # Evidently drift detection
└── tests/                  # pytest test suite
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- `kind` (Kubernetes in Docker)
- `kubectl`

### 1. Clone and install dependencies

```bash
git clone https://github.com/moseskim1027/visionops-defect-detection.git
cd visionops-defect-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Download VISION dataset

Download from [HuggingFace](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets) and extract to `data/raw/vision/`.

### 3. Start MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --port 5000
```

### 4. Start Airflow

```bash
# See docs/airflow-setup.md for full instructions
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow webserver --port 8080 &
airflow scheduler &
```

### 5. Run the full stack with Docker Compose

```bash
docker compose up -d
```

---

## End-to-End Docker Walkthrough

### Prerequisites

Make sure Docker Desktop is running and the processed dataset exists:

```bash
python -m src.data.prepare_dataset \
  --src data/raw/vision \
  --dst data/processed

ls data/processed/images/train | wc -l   # should be 54
```

### Step 1 — Start the stack

```bash
docker compose up --build -d
```

This starts four services:

| Service | URL | Purpose |
|---|---|---|
| mlflow | http://localhost:5000 | Experiment tracking + model registry |
| inference | http://localhost:8000 | FastAPI + Prometheus metrics |
| prometheus | http://localhost:9090 | Scrapes inference every 15 s |
| grafana | http://localhost:3000 | Dashboards (anonymous access) |

Wait for the services to be healthy:

```bash
docker compose ps
# inference and mlflow should show "(healthy)"
```

### Step 2 — Run a training job

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

python -m src.training.train \
  --data data/processed/dataset.yaml \
  --config configs/model.yaml
```

The run appears at http://localhost:5000. The script logs `map50`, `precision`, and `recall` and registers the model as **visionops-yolov8n**. The promotion threshold is **mAP@0.5 ≥ 0.30** (configured in `configs/model.yaml`).

If the run meets the threshold, promote it to production:

```bash
python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()
# replace <VERSION> with the version number shown in the MLflow UI
client.set_registered_model_alias("visionops-yolov8n", "production", "<VERSION>")
EOF
```

> When running via Airflow, the `training_pipeline` DAG handles promotion automatically.

### Step 3 — Reload the inference service

The inference container loads the `production` alias on startup. Restart it after promoting:

```bash
docker compose restart inference

curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,...}
```

### Step 4 — Send predictions

```bash
# Single image
curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/vision/Casting/val/casting_def_0_1.jpg" | python -m json.tool

# Batch — loop over all val images
for img in data/raw/vision/Casting/val/*.jpg; do
  curl -s -X POST http://localhost:8000/predict \
    -F "file=@$img" > /dev/null
done
```

Each request increments `predictions_total` and records latency in `inference_latency_seconds`.

### Step 5 — Simulate drift

```bash
# Generate a drifted batch (extreme brightness shift)
python - <<'EOF'
from pathlib import Path
from PIL import Image
from src.data.drift_simulator import apply_brightness

src = Path("data/raw/vision/Casting/val")
dst = Path("data/drift_batches/bright_batch")
dst.mkdir(parents=True, exist_ok=True)

for p in sorted(src.glob("*.jpg"))[:25]:
    img = Image.open(p).convert("RGB")
    apply_brightness(img, factor=2.5).save(dst / p.name)

print(f"Created {len(list(dst.glob('*.jpg')))} drifted images")
EOF

# Run drift detection
python - <<'EOF'
from pathlib import Path
from src.monitoring.drift_detection import run_drift_report

result = run_drift_report(
    reference_dir=Path("data/raw/vision/Casting/val"),
    current_dir=Path("data/drift_batches/bright_batch"),
    config_path=Path("configs/drift.yaml"),
)

print(f"Drift detected  : {result['drift_detected']}")
print(f"Drift share     : {result['drift_share']:.0%}")
print(f"Drifted features: {result['drifted_features']}")
EOF
```

Expected output:

```
Drift detected  : True
Drift share     : 75%
Drifted features: ['brightness_mean', 'brightness_std', 'contrast']
```

### Step 6 — View everything in Grafana

Open http://localhost:3000 (no login required). Navigate to **Dashboards → VisionOps Inference**:

| Panel | What to look for |
|---|---|
| Request Rate | Spikes from the batch sent in Step 4 |
| Error Rate | Should be 0 unless a request failed |
| Inference Latency | p50 / p95 / p99 lines per request |
| Service Up | Green `1` = inference is alive |

Prometheus scrapes every 15 s. To verify scraping: http://localhost:9090/targets — the `inference` job should show **State: UP**.

### Step 7 — Tear down

```bash
docker compose down        # stop and remove containers
docker compose down -v     # also remove the mlflow-data volume
```

---

## Model Selection Rationale

YOLOv8n (~3.2M parameters, ~3MB weights) was chosen over heavier alternatives (Faster R-CNN, YOLOv8m) because:

1. This is an **MLOps** project — the system design is the artefact, not SOTA accuracy
2. Trains in minutes on CPU, making the full pipeline runnable locally
3. Built-in export to ONNX for edge deployment
4. Ultralytics API integrates cleanly with MLflow custom logging

---

## Future Improvements

- Active learning loop (label uncertain predictions)
- Shadow deployment for A/B model comparison
- Canary releases via Kubernetes rolling update
- GPU autoscaling with KEDA
- CI/CD model promotion via GitHub Actions
