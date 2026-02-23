"""Converts VISION dataset (COCO format) to YOLO format for training.

Input structure:
    data/raw/vision/{Product}/train/_annotations.coco.json
    data/raw/vision/{Product}/train/*.jpg
    data/raw/vision/{Product}/val/_annotations.coco.json
    data/raw/vision/{Product}/val/*.jpg

Output structure:
    data/processed/images/train/{Product}_{filename}.jpg
    data/processed/images/val/{Product}_{filename}.jpg
    data/processed/labels/train/{Product}_{stem}.txt
    data/processed/labels/val/{Product}_{stem}.txt
    data/processed/dataset.yaml
    data/processed/class_map.json
"""

import json
import logging
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SPLITS = ("train", "val")


def build_class_map(
    base_dir: Path,
    products: list[str] | None = None,
) -> dict[str, int]:
    """Return a sorted, unified {class_name: index} map across all products.

    Class names are normalised to lowercase to avoid duplicates caused by
    inconsistent capitalisation in the VISION annotations (e.g. 'Scratch' vs
    'scratch').
    """
    names: set[str] = set()
    selected = products or sorted(p.name for p in base_dir.iterdir() if p.is_dir())

    for product in selected:
        for split in SPLITS:
            ann_file = base_dir / product / split / "_annotations.coco.json"
            if not ann_file.exists():
                continue
            data = json.loads(ann_file.read_text())
            for cat in data["categories"]:
                names.add(cat["name"].lower())

    return {name: idx for idx, name in enumerate(sorted(names))}


def coco_to_yolo(
    bbox: list[float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert COCO bbox to YOLO format.

    COCO:  [x_min, y_min, width, height]  (absolute pixels)
    YOLO:  [cx, cy, w, h]                 (normalised 0–1)

    Values are clamped to [0, 1] to handle annotations that slightly exceed
    image boundaries.
    """
    x_min, y_min, w, h = bbox
    cx = min(1.0, max(0.0, (x_min + w / 2) / img_w))
    cy = min(1.0, max(0.0, (y_min + h / 2) / img_h))
    nw = min(1.0, max(0.0, w / img_w))
    nh = min(1.0, max(0.0, h / img_h))
    return cx, cy, nw, nh


def process_split(
    product: str,
    split: str,
    class_map: dict[str, int],
    src_dir: Path,
    dst_dir: Path,
) -> int:
    """Convert one (product, split) pair.  Returns the number of images written."""
    ann_file = src_dir / product / split / "_annotations.coco.json"
    if not ann_file.exists():
        logger.warning("Missing annotation: %s — skipping", ann_file)
        return 0

    data = json.loads(ann_file.read_text())
    img_src_dir = src_dir / product / split

    # Map local category_id → global class index
    local_map: dict[int, int] = {}
    for cat in data["categories"]:
        name = cat["name"].lower()
        if name in class_map:
            local_map[cat["id"]] = class_map[name]

    images = {img["id"]: img for img in data["images"]}

    ann_by_image: dict[int, list] = {img_id: [] for img_id in images}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id in ann_by_image and ann["category_id"] in local_map:
            ann_by_image[img_id].append(ann)

    out_img_dir = dst_dir / "images" / split
    out_lbl_dir = dst_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_id, img_info in images.items():
        src_img = img_src_dir / img_info["file_name"]
        if not src_img.exists():
            logger.warning("Image not found: %s — skipping", src_img)
            continue

        # Prefix with product name to avoid filename collisions across products
        stem = f"{product}_{Path(img_info['file_name']).stem}"
        shutil.copy2(src_img, out_img_dir / f"{stem}.jpg")

        lines = []
        for ann in ann_by_image[img_id]:
            cls_idx = local_map[ann["category_id"]]
            cx, cy, nw, nh = coco_to_yolo(
                ann["bbox"], img_info["width"], img_info["height"]
            )
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        written += 1

    return written


def prepare_dataset(
    src_dir: Path,
    dst_dir: Path,
    products: list[str] | None = None,
) -> Path:
    """Convert VISION COCO annotations to YOLO format.

    Args:
        src_dir:  Raw VISION dataset root   (e.g. data/raw/vision).
        dst_dir:  Output directory          (e.g. data/processed).
        products: Product categories to include.  None means all 14.

    Returns:
        Path to the generated dataset.yaml.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    selected = sorted(products or [p.name for p in src_dir.iterdir() if p.is_dir()])
    logger.info("Processing %d products: %s", len(selected), selected)

    class_map = build_class_map(src_dir, selected)
    logger.info("Unified class map: %d classes", len(class_map))

    total = 0
    for product in selected:
        for split in SPLITS:
            n = process_split(product, split, class_map, src_dir, dst_dir)
            logger.info("  %s / %-5s  %d images", product, split, n)
            total += n

    # Persist class map for downstream reference (training, monitoring)
    (dst_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))

    # dataset.yaml consumed by ultralytics YOLO trainer
    names_list = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
    config = {
        "path": str(dst_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_map),
        "names": names_list,
    }
    dataset_yaml = dst_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )

    logger.info("Dataset prepared: %d total images → %s", total, dst_dir)
    return dataset_yaml


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare VISION dataset for YOLO")
    parser.add_argument("--src", default="data/raw/vision")
    parser.add_argument("--dst", default="data/processed")
    parser.add_argument(
        "--products", nargs="*", help="Products to include (default: all)"
    )
    args = parser.parse_args()

    prepare_dataset(Path(args.src), Path(args.dst), args.products)
