"""Tests for src/data/prepare_dataset.py"""

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.data.prepare_dataset import (
    build_class_map,
    coco_to_yolo,
    prepare_dataset,
    process_split,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CATEGORIES = [
    {"id": 1, "name": "defect", "supercategory": "defect"},
    {"id": 2, "name": "scratch", "supercategory": "scratch"},
]

SAMPLE_IMAGE = {"id": 1, "file_name": "000000.jpg", "width": 200, "height": 100}

SAMPLE_ANNOTATION = {
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [20, 10, 60, 40],
    "area": 2400,
    "iscrowd": 0,
    "segmentation": [],
}


def _write_coco_split(
    base: Path,
    product: str,
    split: str,
    images: list,
    annotations: list,
    categories: list,
) -> None:
    split_dir = base / product / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for img_info in images:
        Image.new("RGB", (img_info["width"], img_info["height"])).save(
            split_dir / img_info["file_name"]
        )

    ann = {"images": images, "annotations": annotations, "categories": categories}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(ann))


def make_minimal_dataset(tmp_path: Path, product: str = "PartA") -> Path:
    """Create a minimal two-split COCO dataset under tmp_path."""
    for split in ("train", "val"):
        _write_coco_split(
            tmp_path,
            product,
            split,
            images=[SAMPLE_IMAGE],
            annotations=[SAMPLE_ANNOTATION],
            categories=SAMPLE_CATEGORIES,
        )
    return tmp_path


# ---------------------------------------------------------------------------
# coco_to_yolo
# ---------------------------------------------------------------------------


class TestCocoToYolo:
    def test_centre_conversion(self):
        # 100×100 box at origin in a 200×200 image
        cx, cy, nw, nh = coco_to_yolo([0, 0, 100, 100], 200, 200)
        assert cx == pytest.approx(0.25)
        assert cy == pytest.approx(0.25)
        assert nw == pytest.approx(0.5)
        assert nh == pytest.approx(0.5)

    def test_full_image_box(self):
        cx, cy, nw, nh = coco_to_yolo([0, 0, 640, 480], 640, 480)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert nw == pytest.approx(1.0)
        assert nh == pytest.approx(1.0)

    def test_clamps_out_of_bounds(self):
        cx, cy, nw, nh = coco_to_yolo([-10, -10, 700, 500], 640, 480)
        for v in (cx, cy, nw, nh):
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# build_class_map
# ---------------------------------------------------------------------------


class TestBuildClassMap:
    def test_single_product(self, tmp_path):
        _write_coco_split(tmp_path, "PartA", "train", [], [], SAMPLE_CATEGORIES)
        class_map = build_class_map(tmp_path)
        assert "defect" in class_map
        assert "scratch" in class_map
        assert len(class_map) == 2

    def test_alphabetically_sorted_indices(self, tmp_path):
        _write_coco_split(tmp_path, "PartA", "train", [], [], SAMPLE_CATEGORIES)
        class_map = build_class_map(tmp_path)
        # 'defect' < 'scratch' alphabetically → defect=0, scratch=1
        assert class_map["defect"] < class_map["scratch"]

    def test_deduplicates_across_products(self, tmp_path):
        cats = [{"id": 1, "name": "scratch", "supercategory": "scratch"}]
        for product in ("PartA", "PartB"):
            _write_coco_split(tmp_path, product, "train", [], [], cats)
        class_map = build_class_map(tmp_path)
        assert len(class_map) == 1

    def test_normalises_to_lowercase(self, tmp_path):
        cats = [
            {"id": 0, "name": "Scratch", "supercategory": "Scratch"},
            {"id": 1, "name": "scratch", "supercategory": "scratch"},
        ]
        _write_coco_split(tmp_path, "PartA", "train", [], [], cats)
        class_map = build_class_map(tmp_path)
        assert len(class_map) == 1
        assert "scratch" in class_map

    def test_products_filter(self, tmp_path):
        cats_a = [{"id": 1, "name": "defect", "supercategory": "defect"}]
        cats_b = [{"id": 1, "name": "scratch", "supercategory": "scratch"}]
        _write_coco_split(tmp_path, "PartA", "train", [], [], cats_a)
        _write_coco_split(tmp_path, "PartB", "train", [], [], cats_b)
        class_map = build_class_map(tmp_path, products=["PartA"])
        assert "defect" in class_map
        assert "scratch" not in class_map


# ---------------------------------------------------------------------------
# process_split
# ---------------------------------------------------------------------------


class TestProcessSplit:
    def test_writes_image_and_label(self, tmp_path):
        src = make_minimal_dataset(tmp_path / "src")
        dst = tmp_path / "dst"
        class_map = {"defect": 0, "scratch": 1}

        n = process_split("PartA", "train", class_map, src, dst)

        assert n == 1
        assert (dst / "images" / "train" / "PartA_000000.jpg").exists()
        assert (dst / "labels" / "train" / "PartA_000000.txt").exists()

    def test_label_format(self, tmp_path):
        src = make_minimal_dataset(tmp_path / "src")
        dst = tmp_path / "dst"
        class_map = {"defect": 0, "scratch": 1}

        process_split("PartA", "train", class_map, src, dst)
        label = (dst / "labels" / "train" / "PartA_000000.txt").read_text().strip()
        parts = label.split()

        assert len(parts) == 5
        assert parts[0] == "0"
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0

    def test_missing_annotation_returns_zero(self, tmp_path):
        n = process_split("NoProduct", "train", {}, tmp_path, tmp_path / "dst")
        assert n == 0


# ---------------------------------------------------------------------------
# prepare_dataset (integration)
# ---------------------------------------------------------------------------


class TestPrepareDataset:
    def test_generates_dataset_yaml(self, tmp_path):
        src = make_minimal_dataset(tmp_path / "src")
        dst = tmp_path / "dst"

        yaml_path = prepare_dataset(src, dst)

        assert yaml_path.exists()
        config = yaml.safe_load(yaml_path.read_text())
        assert config["nc"] == 2
        assert "train" in config
        assert "val" in config

    def test_names_list_matches_class_map(self, tmp_path):
        src = make_minimal_dataset(tmp_path / "src")
        dst = tmp_path / "dst"

        prepare_dataset(src, dst)

        class_map = json.loads((dst / "class_map.json").read_text())
        config = yaml.safe_load((dst / "dataset.yaml").read_text())
        assert len(config["names"]) == len(class_map)

    def test_raises_on_missing_source(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare_dataset(tmp_path / "nonexistent", tmp_path / "dst")
