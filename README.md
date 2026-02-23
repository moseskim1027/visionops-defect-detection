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
| Orchestration    | Apache Airflow 3.1.7    |
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

### 1. Clone and create an environment

```bash
git clone https://github.com/moseskim1027/visionops-defect-detection.git
cd visionops-defect-detection
```

**Option A — venv:**
```bash
make venv
source .venv/bin/activate
```

**Option B — conda:**
```bash
make conda-env
conda activate visionops
```

### 2. Install dependencies

```bash
make install
```

### 3. Download VISION dataset

Download from [HuggingFace](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets) and extract to `data/raw/vision/`.

### 4. Prepare the dataset

```bash
make data
```

### 5. Start the full stack

```bash
make up
```

---

## Makefile Reference

Run `make help` to list all available targets.

| Target | Description |
|---|---|
| `make venv` | Create `.venv` virtual environment |
| `make conda-env` | Create conda env named `visionops` |
| `make install` | Install dev dependencies (activate env first) |
| `make data` | Prepare COCO→YOLO dataset |
| `make up` | Build and start the Docker stack |
| `make down` | Stop and remove containers |
| `make down-v` | Stop containers and delete volumes |
| `make ps` | Show service status |
| `make logs` | Tail all service logs |
| `make train` | Run a YOLOv8n training job on the full dataset (44 classes) |
| `make train-subset` | Train on 11-class subset: Casting, Console, Groove, Ring |
| `make promote MODEL_VERSION=<n>` | Promote a model version to production |
| `make reload` | Restart the inference container |
| `make predict` | Send a single image to `/predict` |
| `make predict-batch` | Send all Casting val images to `/predict` |
| `make drift-sim` | Generate a brightness-shifted batch |
| `make drift-check` | Run Evidently drift check (terminal output) |
| `make drift-report` | Generate full Evidently HTML report |
| `make test` | Run the full pytest suite |
| `make lint` | Run ruff + black checks |

---

## End-to-End Walkthrough

### Prerequisites

Make sure Docker Desktop is running, then:

```bash
make install   # install Python deps
make data      # convert COCO → YOLO (1894 images across 14 products)
```

### Step 1 — Start the stack

```bash
make up
make ps   # wait until mlflow and inference show "(healthy)"
```

This starts four services:

| Service | URL | Purpose |
|---|---|---|
| mlflow | http://localhost:5001 | Experiment tracking + model registry |
| inference | http://localhost:8000 | FastAPI + Prometheus metrics |
| prometheus | http://localhost:9090 | Scrapes inference every 15 s |
| grafana | http://localhost:3000 | Dashboards (anonymous access) |

### Step 2 — Run a training job

**Option A — full dataset (44 classes, ~30 min on CPU):**

```bash
make train
```

**Option B — 11-class subset (Casting, Console, Groove, Ring, ~5 min on CPU):**

```bash
make train-subset                              # 5 epochs, threshold 0.10
make train-subset EPOCHS=3 THRESHOLD=0.05     # custom overrides
```

The run appears at http://localhost:5001. The script logs `map50`, `precision`, and `recall`, registers the model as **visionops-yolov8n**, and prints a promotion recommendation. Promotion is always a separate step.

> **If you used `train-subset`**, the downstream `predict`, `predict-batch`, `drift-sim`, and `drift-check` targets default to the full dataset paths. Override them at runtime to match the subset products:
> ```bash
> make predict       PREDICT_IMAGE=data/raw/vision/Casting/val/<img>.jpg
> make predict-batch DRIFT_SRC=data/raw/vision/Console/val
> make drift-sim     DRIFT_SRC=data/raw/vision/Groove/val  DRIFT_DST=data/drift_batches/groove_batch
> make drift-check   DRIFT_SRC=data/raw/vision/Groove/val  DRIFT_DST=data/drift_batches/groove_batch
> ```

### Step 2a — Promote to production

Check the MLflow UI at http://localhost:5001 and promote when ready:

```bash
make promote MODEL_VERSION=1
```

> When running via Airflow, the `training_pipeline` DAG handles promotion automatically.

### Step 3 — Reload the inference service

```bash
make reload
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### Step 4 — Send predictions

```bash
make predict                          # single image
make predict-batch                    # all 51 Casting val images
```

Each request increments `predictions_total` and records latency in `inference_latency_seconds`.

### Step 5 — Simulate drift

```bash
make drift-sim     # generate 25 brightness-shifted images
make drift-check   # run Evidently drift report
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
make down     # stop and remove containers
make down-v   # also remove the mlflow-data volume
```

---

## Airflow Automation

The Airflow stack runs on top of the base stack and automates training and drift monitoring. It requires no additional tools beyond Docker.

### Start the Airflow stack

```bash
make airflow-up     # starts base stack + Airflow services
```

Wait for all services to become healthy (~60 s), then open the Airflow UI:

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |

To stop:

```bash
make airflow-down    # stop containers, keep volumes
make airflow-down-v  # stop containers and delete volumes
```

### DAGs

#### `training_pipeline`

Validates the dataset, trains YOLOv8n, registers the model in MLflow, and conditionally promotes it to the `production` alias if it meets the `map50_threshold` defined in `configs/model.yaml`.

```
validate_data → train_model → register_model → conditional_promote
                                                      │
                                        ┌─────────────┴──────────────┐
                                  promote_model               skip_promotion
```

Trigger manually from the Airflow UI, or override epochs at runtime:

```bash
# trigger via UI → training_pipeline → Trigger DAG ▶
```

After a successful run, check the model version at http://localhost:5001.

#### `monitoring_pipeline`

Runs on a daily schedule. Simulates a drift batch from the validation set, runs an Evidently feature-drift check, and triggers `training_pipeline` if drift is detected.

```
simulate_drift → run_drift_detection → branch_on_drift
                                              │
                               ┌──────────────┴──────────────┐
                       trigger_retraining              end_no_drift
```

To run a drift check locally without Airflow:

```bash
make drift-sim     # generate 25 brightness-shifted images in data/drift_batches/bright_batch/
make drift-check   # print drift_detected, drift_share, drifted_features
make drift-report  # save drift_report.html (open in browser)
```

### Architecture notes

- **LocalExecutor** — tasks run in the same container as the scheduler; no Celery or Redis needed
- **PostgreSQL** — Airflow metadata DB (`airflow-db` service); data persists in the `airflow-db-data` volume
- **Shared volumes** — `./src`, `./configs`, `./data`, `./runs`, and `./dags` are bind-mounted so DAG and source changes are live without a rebuild
- **MLflow integration** — containers use `MLFLOW_TRACKING_URI=http://mlflow:5000` (internal Docker DNS); the host uses `http://localhost:5001`

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
