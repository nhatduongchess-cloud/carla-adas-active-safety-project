"""Check CARLA dataset integrity, class balance, duplicates, and split leakage."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

TAXONOMY = {"Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck",
            "TrafficLight", "StopSign"}

MODALITIES = {
    "rgb": ("jpg", "image"),
    "road_line": ("png", "image"),
    "semantic": ("png", "image"),
    "depth": ("png", "image"),
    "lidar": ("npz", "points"),
    "radar": ("npz", "points"),
}


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _read_declared_file(path, kind):
    if kind == "image":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return
    with np.load(path, allow_pickle=False) as archive:
        points = archive["points"]
        if points.ndim != 2 or points.shape[1:] != (4,):
            raise ValueError(f"points must have shape (N, 4), got {points.shape}")
        if points.dtype != np.float32:
            raise ValueError(f"points must be float32, got {points.dtype}")
        if not np.isfinite(points).all():
            raise ValueError('points contain nonfinite coordinates')


def inspect_dataset(dataset):
    """Return the dataset integrity report without writing files or exiting."""
    root = Path(dataset).resolve()
    issues, class_counts = [], Counter()
    hashes, split_towns = defaultdict(list), defaultdict(set)
    frames = 0
    sample_ids = set()
    for annotations in root.rglob("annotations.jsonl"):
        relative = annotations.relative_to(root).parts
        split = relative[0] if relative else "unknown"
        with open(annotations, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                frames += 1
                split_towns[split].add(record.get("town"))
                if record.get("split") != split:
                    issues.append({"type": "split_mismatch", "record": record.get("split"),
                                   "directory": split})
                frame_id = int(record["frame_id"])
                stem = str(record.get("sample_id") or f"{frame_id:010d}")
                if Path(stem).name != stem:
                    issues.append({"type": "unsafe_sample_id", "sample_id": stem})
                    continue
                scoped_id = (split, record.get("town"), record.get("weather"), stem)
                if scoped_id in sample_ids:
                    issues.append({"type": "duplicate_sample_id", "sample_id": stem})
                sample_ids.add(scoped_id)
                base = annotations.parent
                expected = {
                    name: base / name / f"{stem}.{extension}"
                    for name, (extension, _) in MODALITIES.items()
                }
                missing = [str(path) for path in expected.values()
                           if not path.is_file()]
                if missing:
                    issues.append({"type": "missing_files", "frame_id": frame_id,
                                   "files": missing})
                    continue
                unreadable = set()
                for name, path in expected.items():
                    try:
                        _read_declared_file(path, MODALITIES[name][1])
                    except Exception as exc:
                        unreadable.add(name)
                        issues.append({"type": "unreadable_file",
                                       "frame_id": frame_id,
                                       "modality": name,
                                       "file": str(path),
                                       "error": f"{type(exc).__name__}: {exc}"})
                if "rgb" not in unreadable:
                    image_hash = digest(expected["rgb"])
                    hashes[image_hash].append(
                        (split, frame_id, str(expected["rgb"])))
                for obj in record.get("objects", []):
                    class_name = obj.get("class", "Unknown")
                    class_counts[class_name] += 1
                    if class_name not in TAXONOMY:
                        issues.append({"type": "invalid_class", "frame_id": frame_id,
                                       "class": class_name})
                    box = obj.get("bbox_xyxy", [])
                    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                        issues.append({"type": "invalid_bbox", "frame_id": frame_id,
                                       "bbox": box})
                    distance = float(obj.get("distance_m", float("inf")))
                    if not math.isfinite(distance) or not 0.0 < distance <= 80.0:
                        issues.append({"type": "invalid_distance", "frame_id": frame_id,
                                       "distance_m": distance})
                    if (class_name == "StopSign"
                            and str(obj.get("type_id", "")).startswith("traffic.stop")):
                        issues.append({"type": "invisible_stop_trigger",
                                       "frame_id": frame_id})
    if not frames:
        issues.append({'type': 'empty_dataset'})
    duplicates = [records for records in hashes.values() if len(records) > 1]
    leakage = [records for records in duplicates
               if len({record[0] for record in records}) > 1]
    town_leakage = {town: sorted(split for split, towns in split_towns.items() if town in towns)
                    for towns in split_towns.values() for town in towns
                    if sum(town in values for values in split_towns.values()) > 1}
    report = {"dataset": str(root), "frames": frames,
              "class_counts": dict(sorted(class_counts.items())),
              "duplicate_groups": len(duplicates), "split_leakage_groups": len(leakage),
              "town_split_leakage": town_leakage, "issues": issues,
              "valid": not issues and not leakage and not town_leakage}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="logs/dataset_integrity.json")
    args = parser.parse_args()
    report = inspect_dataset(args.dataset)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
