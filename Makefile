.PHONY: help venv conda-env install data up down down-v ps logs airflow-up airflow-down airflow-down-v train promote reload predict predict-batch drift-sim drift-check test lint

MLFLOW_URI     ?= http://localhost:5001
DATA_SRC       ?= data/raw/vision
DATA_DST       ?= data/processed
PREDICT_IMAGE  ?= data/raw/vision/Casting/val/000054.jpg
DRIFT_SRC      ?= data/raw/vision/Casting/val
DRIFT_DST      ?= data/drift_batches/bright_batch
DRIFT_CONFIG   ?= configs/drift.yaml
MODEL_VERSION  ?= 1
EPOCHS         ?= 5
THRESHOLD      ?= 0.10
RESUME_FROM    ?=
CONDA_ENV      ?= visionops
PYTHON_VERSION ?= 3.11

help:
	@echo ""
	@echo "  Environment (choose one):"
	@echo "  venv           Create .venv with Python venv"
	@echo "  conda-env      Create conda env named '$(CONDA_ENV)'"
	@echo "  install        Install dev dependencies (activate env first)"
	@echo ""
	@echo "  data           Prepare COCO→YOLO dataset"
	@echo ""
	@echo "  up             Build and start the base stack (mlflow/inference/prometheus/grafana)"
	@echo "  down           Stop and remove base stack containers"
	@echo "  down-v         Stop base stack containers and delete volumes"
	@echo "  ps             Show service status"
	@echo "  logs           Tail all service logs"
	@echo ""
	@echo "  airflow-up     Build and start the Airflow stack (+ base stack)"
	@echo "  airflow-down   Stop Airflow stack containers"
	@echo "  airflow-down-v Stop Airflow stack containers and delete volumes"
	@echo ""
	@echo "  train          Run a YOLOv8n training job on the full dataset"
	@echo "  train-subset   Train on 11-class subset (Casting/Console/Groove/Ring)"
	@echo "               Usage: make train-subset EPOCHS=5 THRESHOLD=0.10"
	@echo "               Resume: make train-subset RESUME_FROM=runs/detect/train2/weights/last.pt"
	@echo "  promote        Promote model version to production alias"
	@echo "               Usage: make promote MODEL_VERSION=<n>"
	@echo "  reload         Restart the inference container"
	@echo ""
	@echo "  predict        Send a single image to /predict"
	@echo "               Usage: make predict PREDICT_IMAGE=path/to/img.jpg"
	@echo "  predict-batch  Send all Casting val images to /predict"
	@echo ""
	@echo "  drift-sim      Generate a brightness-shifted batch"
	@echo "  drift-check    Run Evidently drift report"
	@echo ""
	@echo "  test           Run the full pytest suite"
	@echo "  lint           Run ruff + black checks"
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

venv:
	python$(PYTHON_VERSION) -m venv .venv
	@echo "Run: source .venv/bin/activate"

conda-env:
	conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	@echo "Run: conda activate $(CONDA_ENV)"

install:
	pip install -r requirements-dev.txt

data:
	python -m src.data.prepare_dataset --src $(DATA_SRC) --dst $(DATA_DST)

# ---------------------------------------------------------------------------
# Docker stack
# ---------------------------------------------------------------------------

up:
	docker compose up --build -d

down:
	docker compose down

down-v:
	docker compose down -v

ps:
	docker compose ps

logs:
	docker compose logs -f

airflow-up:
	docker compose --profile airflow up --build -d

airflow-down:
	docker compose --profile airflow down

airflow-down-v:
	docker compose --profile airflow down -v

# ---------------------------------------------------------------------------
# Training & promotion
# ---------------------------------------------------------------------------

train:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) python -m src.training.train \
		--data $(DATA_DST)/dataset.yaml \
		--config configs/model.yaml

train-subset:
	# NOTE: if using train-subset, downstream defaults must match the subset products.
	# Override at runtime, e.g.:
	#   make predict        PREDICT_IMAGE=data/raw/vision/Casting/val/<img>.jpg
	#   make predict-batch  DRIFT_SRC=data/raw/vision/Console/val
	#   make drift-sim      DRIFT_SRC=data/raw/vision/Groove/val  DRIFT_DST=data/drift_batches/groove_batch
	#   make drift-check    DRIFT_SRC=data/raw/vision/Groove/val  DRIFT_DST=data/drift_batches/groove_batch
	conda run -n visionops env PYTHONPATH=. python scripts/train_subset.py \
		--epochs $(EPOCHS) \
		--threshold $(THRESHOLD) \
		--mlflow-uri $(MLFLOW_URI) \
		$(if $(RESUME_FROM),--resume-from $(RESUME_FROM),)

promote:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) python -c "\
import mlflow; \
mlflow.set_tracking_uri('$(MLFLOW_URI)'); \
c = mlflow.MlflowClient(); \
c.set_registered_model_alias('visionops-yolov8n', 'production', '$(MODEL_VERSION)'); \
print('Promoted version $(MODEL_VERSION) to production')"

reload:
	docker compose restart inference

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

predict:
	curl -s -X POST http://localhost:8000/predict \
		-F "file=@$(PREDICT_IMAGE)" | python3 -m json.tool

predict-batch:
	@echo "Sending all Casting val images..."
	@for img in $(DRIFT_SRC)/*.jpg; do \
		curl -s -X POST http://localhost:8000/predict -F "file=@$$img" > /dev/null; \
	done
	@echo "Done."

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

drift-sim:
	conda run -n $(CONDA_ENV) env PYTHONPATH=. python -c "\
from pathlib import Path; \
from PIL import Image; \
from src.data.drift_simulator import apply_brightness; \
src = Path('$(DRIFT_SRC)'); dst = Path('$(DRIFT_DST)'); \
dst.mkdir(parents=True, exist_ok=True); \
imgs = sorted(src.glob('*.jpg'))[:25]; \
[apply_brightness(Image.open(p).convert('RGB'), factor=2.5).save(dst / p.name) for p in imgs]; \
print(f'Created {len(imgs)} drifted images in $(DRIFT_DST)')"

drift-report:
	conda run -n $(CONDA_ENV) env PYTHONPATH=. python scripts/generate_drift_report.py \
		--reference $(DRIFT_SRC) \
		--current $(DRIFT_DST) \
		--output drift_report.html
	@echo "Open drift_report.html in your browser."

drift-check:
	conda run -n $(CONDA_ENV) env PYTHONPATH=. python -c "\
from pathlib import Path; \
from src.monitoring.drift_detection import run_drift_report; \
r = run_drift_report(Path('$(DRIFT_SRC)'), Path('$(DRIFT_DST)'), Path('$(DRIFT_CONFIG)')); \
print(f'Drift detected  : {r[\"drift_detected\"]}'); \
print(f'Drift share     : {r[\"drift_share\"]:.0%}'); \
print(f'Drifted features: {r[\"drifted_features\"]}')"

# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	black --check src/ tests/
