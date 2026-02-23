"""Train YOLOv8n on a subset of classes.

Builds a filtered dataset from a chosen set of products, trains the model,
and reports whether the result meets the promotion threshold.
Promotion is a separate step — run `make promote MODEL_VERSION=<n>` if the
validation check passes.

Default subset: Casting + Console + Groove + Ring  →  11 classes

Usage
-----
python scripts/train_subset.py                                      # 11-class default, 5 epochs
python scripts/train_subset.py --products Casting Groove
python scripts/train_subset.py --epochs 3 --threshold 0.10
python scripts/train_subset.py --dry-run                            # prepare data only
python scripts/train_subset.py --resume-from runs/detect/train2/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PRODUCTS  = ["Casting", "Console", "Groove", "Ring"]  # 2+4+2+3 = 11 classes
DEFAULT_EPOCHS    = 5
DEFAULT_THRESHOLD = 0.10
DEFAULT_URI       = "http://localhost:5001"
DEFAULT_SRC       = Path("data/raw/vision")
DEFAULT_DST       = Path("data/processed_subset")
DEFAULT_MODEL     = "visionops-yolov8n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"\033[1m[train_subset]\033[0m {msg}", flush=True)


def step_prepare(src: Path, dst: Path, products: list[str]) -> Path:
    """Build a YOLO dataset containing only the chosen products."""
    from src.data.prepare_dataset import prepare_dataset  # noqa: PLC0415

    _log(f"Products : {products}")
    yaml_path = prepare_dataset(src_dir=src, dst_dir=dst, products=products)

    class_map = json.loads((dst / "class_map.json").read_text())
    _log(f"Classes  : {len(class_map)}  →  {sorted(class_map)}")
    _log(f"Dataset  : {yaml_path}")
    return yaml_path


def step_train(
    yaml_path: Path,
    epochs: int,
    mlflow_uri: str,
    resume_from: Path | None = None,
) -> tuple[str, dict]:
    """Patch configs/model.yaml, run training, then restore the original values."""
    import yaml  # noqa: PLC0415
    from src.training.train import run_training  # noqa: PLC0415

    if resume_from:
        _log(f"Resuming from  {resume_from}")
    _log(f"Training  epochs={epochs}  classes=subset  mlflow={mlflow_uri}")

    config_path = Path("configs/model.yaml")
    cfg = yaml.safe_load(config_path.read_text())

    orig_epochs  = cfg["training"]["epochs"]
    orig_uri     = cfg["mlflow"]["tracking_uri"]
    orig_variant = cfg["model"]["variant"]

    cfg["training"]["epochs"]     = epochs
    cfg["mlflow"]["tracking_uri"] = mlflow_uri
    if resume_from:
        cfg["model"]["variant"] = str(resume_from)
    config_path.write_text(yaml.dump(cfg))

    try:
        run_id, metrics = run_training(config_path, yaml_path)
    finally:
        cfg["training"]["epochs"]     = orig_epochs
        cfg["mlflow"]["tracking_uri"] = orig_uri
        cfg["model"]["variant"]       = orig_variant
        config_path.write_text(yaml.dump(cfg))

    _log(f"run_id     = {run_id}")
    _log(f"mAP@50     = {metrics.get('map50', 0):.4f}")
    _log(f"mAP@50-95  = {metrics.get('map50_95', 0):.4f}")
    _log(f"precision  = {metrics.get('precision', 0):.4f}")
    _log(f"recall     = {metrics.get('recall', 0):.4f}")

    # Print the last checkpoint path so it can be passed to --resume-from next time
    last_ckpt = sorted(Path("runs/detect").glob("*/weights/last.pt"))
    if last_ckpt:
        _log(f"Last checkpoint: {last_ckpt[-1]}")
        _log(f"To continue training: add --resume-from {last_ckpt[-1]}")

    return run_id, metrics


def step_validate(metrics: dict, threshold: float, mlflow_uri: str, run_id: str) -> None:
    """Print a promotion recommendation — does NOT promote."""
    import mlflow  # noqa: PLC0415

    map50  = metrics.get("map50", 0.0)
    passed = map50 >= threshold
    label  = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
    _log(f"Validation [{label}]  mAP@50={map50:.4f}  threshold={threshold}")

    if passed:
        mlflow.set_tracking_uri(mlflow_uri)
        client   = mlflow.MlflowClient()
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            version_num = versions[0].version
            _log(f"Model registered as version {version_num}.")
            _log(f"To promote:  make promote MODEL_VERSION={version_num}")
        else:
            _log("Model not yet registered in MLflow — check the UI.")
    else:
        _log("Does not meet the promotion threshold.")
        _log(f"Review the run at {mlflow_uri} before promoting manually.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--products", nargs="+", default=DEFAULT_PRODUCTS,
        help="Products to include (default: Casting Console Groove Ring = 11 classes)",
    )
    p.add_argument("--src",       type=Path,  default=DEFAULT_SRC,
                   help="Raw VISION dataset root")
    p.add_argument("--dst",       type=Path,  default=DEFAULT_DST,
                   help="Output directory for the prepared subset dataset")
    p.add_argument("--epochs",    type=int,   default=DEFAULT_EPOCHS,
                   help="Number of training epochs (default: 5)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="mAP@0.50 threshold for promotion recommendation (default: 0.10)")
    p.add_argument("--mlflow-uri", default=DEFAULT_URI,
                   help="MLflow tracking URI (default: http://localhost:5001)")
    p.add_argument("--model-name", default=DEFAULT_MODEL,
                   help="Registered model name in MLflow")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Path to a checkpoint (.pt) to continue training from, "
                        "e.g. runs/detect/train2/weights/last.pt")
    p.add_argument("--dry-run",   action="store_true",
                   help="Prepare the dataset only — skip training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    yaml_path = step_prepare(args.src, args.dst, args.products)

    if args.dry_run:
        _log("Dry run complete — skipping training.")
        sys.exit(0)

    run_id, metrics = step_train(yaml_path, args.epochs, args.mlflow_uri, args.resume_from)
    step_validate(metrics, args.threshold, args.mlflow_uri, run_id)

    _log("Done. Check MLflow at http://localhost:5001 and promote when ready.")


if __name__ == "__main__":
    main()
