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

## Planned PRs

| Branch                       | Description                          |
|------------------------------|--------------------------------------|
| `feat/repo-scaffold-ci-cd`   | Repo structure + CI workflows        |
| `feat/data-pipeline`         | COCO→YOLO conversion + drift sim     |
| `feat/training-mlflow`       | YOLOv8n training + MLflow logging    |
| `feat/inference-api`         | FastAPI service + Prometheus metrics |
| `feat/airflow-dags`          | Training + monitoring DAGs           |
| `feat/k8s-deployment`        | Kubernetes manifests + Helm values   |
| `feat/monitoring-stack`      | Evidently + Prometheus + Grafana     |
| `docs/readme`                | Architecture diagram + full docs     |

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
