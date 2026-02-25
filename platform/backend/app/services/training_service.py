"""Training service.

Manages training lifecycle, config updates, MLflow metric polling,
and model information retrieval.
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from app.config import (
    CONFIGS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
    PROCESSED_SUBSET_DIR,
    RAW_DIR,
    ROOT_DIR,
    RUNS_DIR,
)

# ---------------------------------------------------------------------------
# In-memory training state
# ---------------------------------------------------------------------------

_train_state: dict[str, Any] = {
    "status": "idle",  # idle | running | completed | failed | stopped
    "run_id": None,
    "started_at": None,
    "completed_at": None,
    "error": None,
    "pid": None,
    "configured_epochs": None,
    "products": None,  # list[str] | None — None means full dataset
}

# ---------------------------------------------------------------------------
# Model information (static facts about YOLOv8n)
# ---------------------------------------------------------------------------

YOLOV8N_INFO = {
    "name": "YOLOv8n",
    "full_name": "You Only Look Once v8 Nano",
    "parameters": "3.2M",
    "gflops": "8.7",
    "input_size": "640×640",
    "task": "Object Detection",
    "framework": "Ultralytics / PyTorch",
    "architecture": "CSPDarknet backbone + PANet neck + Decoupled head",
    "description": (
        "YOLOv8n is the nano variant of the YOLOv8 family — designed for "
        "edge inference and fast CPU training while retaining strong accuracy "
        "on dense detection tasks. Anchor-free detection with a decoupled head "
        "enables precise localisation of small industrial defects."
    ),
    "strengths": [
        "Real-time CPU inference",
        "Small memory footprint",
        "Strong small-object detection",
        "Easy fine-tuning on custom datasets",
    ],
}


def get_model_info() -> dict[str, Any]:
    config = _load_training_config()
    return {
        **YOLOV8N_INFO,
        "training_config": config,
    }


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return CONFIGS_DIR / "model.yaml"


def _load_training_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def get_training_config() -> dict[str, Any]:
    cfg = _load_training_config()
    train = cfg.get("training", {})
    mlflow = cfg.get("mlflow", {})
    model = cfg.get("model", {})
    return {
        "epochs": train.get("epochs", 10),
        "batch": train.get("batch", 8),
        "lr0": train.get("lr0", 0.01),
        "patience": train.get("patience", 5),
        "workers": train.get("workers", 2),
        "device": train.get("device", "cpu"),
        "imgsz": model.get("imgsz", 640),
        "experiment_name": mlflow.get("experiment_name", MLFLOW_EXPERIMENT_NAME),
    }


def update_training_config(updates: dict[str, Any]) -> dict[str, Any]:
    cfg = _load_training_config()
    train = cfg.setdefault("training", {})
    model = cfg.setdefault("model", {})
    mlflow_cfg = cfg.setdefault("mlflow", {})

    field_map = {
        "epochs": (train, "epochs"),
        "batch": (train, "batch"),
        "lr0": (train, "lr0"),
        "patience": (train, "patience"),
        "workers": (train, "workers"),
        "device": (train, "device"),
        "imgsz": (model, "imgsz"),
        "experiment_name": (mlflow_cfg, "experiment_name"),
    }
    for key, value in updates.items():
        if key in field_map:
            section, field = field_map[key]
            section[field] = value

    _config_path().write_text(yaml.dump(cfg, default_flow_style=False))
    return get_training_config()


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def get_class_distribution(processed_dir: str | None = None) -> list[dict[str, Any]]:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    yaml_path = proc / "dataset.yaml"
    if not yaml_path.exists():
        return []

    data = yaml.safe_load(yaml_path.read_text())
    names_raw = data.get("names", {})
    if isinstance(names_raw, dict):
        class_names = [names_raw[k] for k in sorted(names_raw)]
    else:
        class_names = list(names_raw)

    counts: Counter[int] = Counter()
    for split in ("train", "val"):
        labels_dir = proc / "labels" / split
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        counts[int(parts[0])] += 1
                    except ValueError:
                        pass

    distribution = []
    for cls_id, count in sorted(counts.items()):
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        distribution.append({"class_id": cls_id, "class_name": name, "count": count})

    # Sort descending by count
    distribution.sort(key=lambda x: x["count"], reverse=True)
    return distribution


# ---------------------------------------------------------------------------
# Available products
# ---------------------------------------------------------------------------


def get_available_products() -> list[str]:
    """Return sorted product category names from the raw dataset directory."""
    if not RAW_DIR.exists():
        return []
    return sorted(
        p.name for p in RAW_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# Training lifecycle
# ---------------------------------------------------------------------------


def start_training(
    dataset_yaml: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    products: list[str] | None = None,
) -> dict[str, Any]:
    if _train_state["status"] == "running":
        return {"error": "Training already running", "state": _train_state.copy()}

    if config_overrides:
        update_training_config(config_overrides)

    # Snapshot configured epochs AFTER applying any overrides
    configured_epochs = get_training_config().get("epochs", 10)

    # Resolve dataset path: subset when products specified, full otherwise
    subset_products = products if products else None
    if subset_products:
        dataset = dataset_yaml or str(PROCESSED_SUBSET_DIR / "dataset.yaml")
    else:
        dataset = dataset_yaml or str(PROCESSED_DIR / "dataset.yaml")

    _train_state.update(
        {
            "status": "running",
            "run_id": None,
            "started_at": time.time(),
            "completed_at": None,
            "error": None,
            "pid": None,
            "configured_epochs": configured_epochs,
            "products": subset_products,
        }
    )

    def _run():
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT_DIR)
            env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

            # Prepare subset dataset first if products were specified
            if subset_products:
                prep_cmd = [
                    sys.executable,
                    "-m",
                    "src.data.prepare_dataset",
                    "--src",
                    str(RAW_DIR),
                    "--dst",
                    str(PROCESSED_SUBSET_DIR),
                    "--products",
                    *subset_products,
                ]
                prep = subprocess.run(
                    prep_cmd,
                    cwd=str(ROOT_DIR),
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if prep.returncode != 0:
                    _train_state.update(
                        {
                            "status": "failed",
                            "error": prep.stderr[-2000:] or prep.stdout[-2000:],
                            "completed_at": time.time(),
                        }
                    )
                    return

            cmd = [
                sys.executable,
                "-m",
                "src.training.train",
                "--config",
                str(_config_path()),
                "--dataset",
                dataset,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT_DIR),
                env=env,
            )
            _train_state["pid"] = proc.pid

            # Try to grab the run_id from MLflow as soon as training begins
            _poll_for_run_id()

            stdout, _ = proc.communicate()
            if proc.returncode == 0:
                # Parse run_id from stdout as fallback
                for line in (stdout or "").splitlines():
                    if line.startswith("run_id:"):
                        _train_state["run_id"] = line.split(":", 1)[1].strip()
                _train_state.update(
                    {
                        "status": "completed",
                        "completed_at": time.time(),
                    }
                )
                # Re-discover run_id from MLflow if we missed it
                if not _train_state["run_id"]:
                    _train_state["run_id"] = _find_latest_run(status="FINISHED")
            else:
                _train_state.update(
                    {
                        "status": "failed",
                        "error": (stdout or "")[-2000:],
                        "completed_at": time.time(),
                    }
                )
        except Exception as exc:
            _train_state.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": time.time(),
                }
            )

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True, "state": _train_state.copy()}


def _poll_for_run_id(max_wait: int = 60, interval: int = 3) -> None:
    """Background thread: find the active MLflow run and store its ID."""

    def _worker():
        deadline = time.time() + max_wait
        while time.time() < deadline:
            run_id = _find_latest_run(status="RUNNING")
            if run_id:
                _train_state["run_id"] = run_id
                return
            time.sleep(interval)

    threading.Thread(target=_worker, daemon=True).start()


def _find_latest_run(status: str = "RUNNING") -> str | None:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.status = '{status}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0].info.run_id if runs else None
    except Exception:
        return None


def stop_training() -> dict[str, Any]:
    """Send SIGTERM to the training subprocess."""
    if _train_state["status"] != "running":
        return {"error": "No training is currently running."}
    pid = _train_state.get("pid")
    if not pid:
        return {"error": "PID not available yet; retry in a moment."}
    try:
        os.kill(pid, signal.SIGTERM)
        _train_state.update(
            {
                "status": "failed",
                "error": "Training stopped by user.",
                "completed_at": time.time(),
            }
        )
        return {"stopped": True}
    except ProcessLookupError:
        _train_state["status"] = "failed"
        return {"error": "Process already exited."}
    except Exception as exc:
        return {"error": str(exc)}


def get_training_status() -> dict[str, Any]:
    state = _train_state.copy()
    if state["started_at"]:
        end = state["completed_at"] or time.time()
        state["elapsed_seconds"] = int(end - state["started_at"])
    return state


# ---------------------------------------------------------------------------
# Per-epoch results (from Ultralytics results.csv)
# ---------------------------------------------------------------------------


def get_epoch_results() -> list[dict[str, Any]]:
    """Read per-epoch metrics from the most recently modified results.csv.

    Ultralytics writes one row per completed epoch progressively, so this
    reflects live progress during training without needing MLflow.
    """
    detect_dir = RUNS_DIR / "detect"
    if not detect_dir.exists():
        return []

    csvs = sorted(
        detect_dir.glob("train*/results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csvs:
        return []

    # Guard against serving stale data from a previous training run.
    # If the most-recent CSV was last modified before this run started,
    # the current run hasn't written its first epoch yet — return empty.
    started_at = _train_state.get("started_at")
    if started_at and csvs[0].stat().st_mtime < started_at:
        return []

    results = []
    try:
        with open(csvs[0], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Ultralytics CSV keys have leading/trailing whitespace
                cleaned = {k.strip(): v.strip() for k, v in row.items() if k}
                try:
                    results.append(
                        {
                            "epoch": int(cleaned.get("epoch", 0)),
                            "train_box_loss": float(cleaned.get("train/box_loss") or 0),
                            "train_cls_loss": float(cleaned.get("train/cls_loss") or 0),
                            "val_box_loss": float(cleaned.get("val/box_loss") or 0),
                            "val_cls_loss": float(cleaned.get("val/cls_loss") or 0),
                            "precision": float(
                                cleaned.get("metrics/precision(B)") or 0
                            ),
                            "recall": float(cleaned.get("metrics/recall(B)") or 0),
                            "map50": float(cleaned.get("metrics/mAP50(B)") or 0),
                            "map50_95": float(cleaned.get("metrics/mAP50-95(B)") or 0),
                        }
                    )
                except (ValueError, KeyError):
                    pass
    except (OSError, csv.Error):
        pass

    return results


# ---------------------------------------------------------------------------
# MLflow metrics
# ---------------------------------------------------------------------------


def get_mlflow_metrics(run_id: str) -> dict[str, Any]:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run = client.get_run(run_id)
        metrics: dict[str, list[dict]] = {}
        for key in run.data.metrics:
            history = client.get_metric_history(run_id, key)
            metrics[key] = [
                {"step": m.step, "value": round(m.value, 6), "timestamp": m.timestamp}
                for m in history
            ]
        return {
            "run_id": run_id,
            "status": run.info.status,
            "metrics": metrics,
            "params": run.data.params,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
    except Exception as exc:
        return {"error": str(exc), "run_id": run_id}


def list_mlflow_runs(max_results: int = 10) -> list[dict[str, Any]]:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        return [
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "metrics": r.data.metrics,
                "params": r.data.params,
            }
            for r in runs
        ]
    except Exception as exc:
        return [{"error": str(exc)}]
