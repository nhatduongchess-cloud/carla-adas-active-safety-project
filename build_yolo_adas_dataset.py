"""Build the unified eight-class YOLO dataset from validated local sources."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil

from audit_week2_dataset import TAXONOMY
from modules.dataset_labels import require_training_review


CLASS_TO_ID = {name: index for index, name in enumerate(TAXONOMY)}
SPLIT_DIR = {"train": "train", "validation": "val", "test": "test"}
SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def safe_token(value):
    return SAFE_TOKEN.sub("_", str(value)).strip("._") or "unknown"


def xyxy_to_yolo(box, image_size):
    if len(box) != 4 or len(image_size) != 2:
        raise ValueError("bbox_xyxy and image_size must have lengths 4 and 2")
    x1, y1, x2, y2 = (float(value) for value in box)
    width, height = (float(value) for value in image_size)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2, width, height)):
        raise ValueError("non-finite box or image size")
    if width <= 0 or height <= 0 or not (0 <= x1 < x2 <= width - 1
                                         and 0 <= y1 < y2 <= height - 1):
        raise ValueError("invalid bbox_xyxy for image size")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def parse_adas_yolo(line, source):
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"malformed YOLO row in {source}: {line!r}")
    class_id = int(fields[0])
    values = tuple(float(value) for value in fields[1:])
    if class_id not in range(len(TAXONOMY)):
        raise ValueError(f"class id outside ADAS taxonomy in {source}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite YOLO coordinate in {source}")
    x, y, width, height = values
    tolerance = 1e-4
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1
            and x - width / 2 >= -tolerance and x + width / 2 <= 1 + tolerance
            and y - height / 2 >= -tolerance and y + height / 2 <= 1 + tolerance):
        raise ValueError(f"YOLO box crosses image boundary in {source}")
    return class_id, values


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def ensure_output_dirs(root):
    for split in SPLIT_DIR.values():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def add_carla_sources(roots, output, counts, frames, materialization):
    seen_names = set()
    source_reports = []
    for source_index, root_value in enumerate(roots):
        root = Path(root_value).resolve()
        integrity_path = root / "dataset_integrity.json"
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if not integrity.get("valid"):
            raise ValueError(f"CARLA source integrity failed: {root}")
        source_frames = 0
        for annotations in sorted(root.rglob("annotations.jsonl")):
            manifest_path = annotations.with_name('manifest.json')
            capture_manifest = (json.loads(manifest_path.read_text(encoding='utf-8'))
                                if manifest_path.is_file() else {})
            with open(annotations, "r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    require_training_review(record, capture_manifest)
                    split = str(record.get("split"))
                    if split not in SPLIT_DIR:
                        raise ValueError(
                            f"invalid split at {annotations}:{line_number}: {split}")
                    image_size = record.get("image_size")
                    sample_id = str(record.get("sample_id") or
                                    f"{int(record['frame_id']):010d}")
                    source_image = annotations.parent / "rgb" / f"{sample_id}.jpg"
                    if not source_image.is_file():
                        raise FileNotFoundError(source_image)
                    name = "_".join((
                        f"carla{source_index}", safe_token(split),
                        safe_token(record.get("town")),
                        safe_token(record.get("weather")), safe_token(sample_id),
                    ))
                    if name in seen_names:
                        raise ValueError(f"duplicate output sample name: {name}")
                    seen_names.add(name)
                    target_split = SPLIT_DIR[split]
                    mode = link_or_copy(
                        source_image, output / target_split / "images" / f"{name}.jpg")
                    materialization[mode] += 1
                    label_rows = []
                    for obj in record.get("objects", []):
                        class_name = str(obj.get("class"))
                        if class_name not in CLASS_TO_ID:
                            raise ValueError(
                                f"unknown class {class_name} at {annotations}:{line_number}")
                        values = xyxy_to_yolo(obj.get("bbox_xyxy", []), image_size)
                        class_id = CLASS_TO_ID[class_name]
                        label_rows.append(
                            f"{class_id} "
                            + " ".join(f"{value:.8f}" for value in values))
                        counts[target_split][class_name] += 1
                    (output / target_split / "labels" / f"{name}.txt").write_text(
                        "\n".join(label_rows) + ("\n" if label_rows else ""),
                        encoding="utf-8")
                    frames[target_split] += 1
                    source_frames += 1
        source_reports.append({"kind": "carla", "source": str(root),
                               "frames": source_frames,
                               "integrity": str(integrity_path)})
    return source_reports, seen_names


def add_yolo_train_sources(roots, output, counts, frames, materialization,
                           seen_names):
    source_reports = []
    for source_index, root_value in enumerate(roots):
        root = Path(root_value).resolve()
        images_dir = root / "train" / "images"
        labels_dir = root / "train" / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"YOLO train source missing images/labels: {root}")
        source_frames = 0
        for label_path in sorted(labels_dir.glob("*.txt")):
            images = list(images_dir.glob(label_path.stem + ".*"))
            if len(images) != 1:
                raise ValueError(
                    f"expected one image for {label_path}, got {len(images)}")
            name = f"yolo{source_index}_{safe_token(label_path.stem)}"
            if name in seen_names:
                raise ValueError(f"duplicate output sample name: {name}")
            seen_names.add(name)
            rows = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, values = parse_adas_yolo(line, label_path)
                rows.append(f"{class_id} "
                            + " ".join(f"{value:.8f}" for value in values))
                counts["train"][TAXONOMY[class_id]] += 1
            image_suffix = images[0].suffix.lower()
            mode = link_or_copy(
                images[0], output / "train" / "images" / f"{name}{image_suffix}")
            materialization[mode] += 1
            (output / "train" / "labels" / f"{name}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            frames["train"] += 1
            source_frames += 1
        source_reports.append({"kind": "yolo_train", "source": str(root),
                               "frames": source_frames})
    return source_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-source", action="append", default=[])
    parser.add_argument("--yolo-train-source", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.carla_source:
        raise SystemExit("at least one --carla-source is required")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    ensure_output_dirs(output)
    counts = {split: Counter() for split in SPLIT_DIR.values()}
    frames = Counter()
    materialization = Counter()
    try:
        carla_reports, seen_names = add_carla_sources(
            args.carla_source, output, counts, frames, materialization)
        yolo_reports = add_yolo_train_sources(
            args.yolo_train_source, output, counts, frames,
            materialization, seen_names)
        missing = {
            split: [name for name in TAXONOMY if counts[split][name] <= 0]
            for split in SPLIT_DIR.values()
        }
        if any(missing.values()):
            raise ValueError(f"unified dataset has missing classes: {missing}")
        yaml_lines = [
            f"path: {output.as_posix()}",
            "train: train/images", "val: val/images", "test: test/images",
            f"nc: {len(TAXONOMY)}", "names:",
        ]
        yaml_lines.extend(f"  {index}: {name}"
                          for index, name in enumerate(TAXONOMY))
        (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n",
                                          encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "output": str(output),
            "taxonomy": list(TAXONOMY),
            "frames_by_split": dict(sorted(frames.items())),
            "class_counts_by_split": {
                split: dict(sorted(split_counts.items()))
                for split, split_counts in counts.items()
            },
            "image_materialization": dict(sorted(materialization.items())),
            "sources": carla_reports + yolo_reports,
            "training_ready_8_class": True,
        }
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    except Exception:
        (output / "BUILD_FAILED.txt").write_text(
            "Dataset build failed. This directory is incomplete and must not be used.\n",
            encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
