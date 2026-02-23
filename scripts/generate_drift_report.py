"""Generate an Evidently HTML drift report.

Usage
-----
python scripts/generate_drift_report.py \
    --reference data/processed_subset/images/val \
    --current   data/drift_batches/<batch_dir>/images \
    --output    drift_report.html

Then open drift_report.html in your browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument(
        "--reference",
        default="data/processed_subset/images/val",
        help="Reference image directory (val split)",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="Current (drifted) image directory. Defaults to most recent drift batch.",
    )
    parser.add_argument(
        "--output",
        default="drift_report.html",
        help="Output HTML file path",
    )
    args = parser.parse_args()

    from evidently.metric_preset import DataDriftPreset  # noqa: PLC0415
    from evidently.report import Report  # noqa: PLC0415

    from src.monitoring.drift_detection import extract_image_features  # noqa: PLC0415

    ref_dir = Path(args.reference)

    if args.current:
        cur_dir = Path(args.current)
    else:
        batches = sorted(Path("data/drift_batches").glob("batch_*"))
        if not batches:
            raise FileNotFoundError(
                "No drift batches found in data/drift_batches/. "
                "Run the monitoring_pipeline DAG first, or pass --current."
            )
        cur_dir = batches[-1] / "images"
        print(f"Using most recent batch: {cur_dir.parent.name}")

    print(f"Reference : {ref_dir}  ({len(list(ref_dir.glob('*.jpg')))} images)")
    print(f"Current   : {cur_dir}  ({len(list(cur_dir.glob('*.jpg')))} images)")
    print("Extracting features...")

    ref_df = extract_image_features(sorted(ref_dir.glob("*.jpg")))
    cur_df = extract_image_features(sorted(cur_dir.glob("*.jpg")))

    print("Running Evidently drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    out = Path(args.output)
    report.save_html(str(out))
    print(f"\nReport saved → {out.resolve()}")
    print("Open it in your browser to view the drift analysis.")


if __name__ == "__main__":
    main()
