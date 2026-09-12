"""Depth-filter existing CARLA annotations while preserving raw backups."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import cv2

from collect_carla_dataset import TAXONOMY, annotation_rejection_reason


def sanitize_file(path: Path):
    backup = path.with_name("annotations.raw.jsonl")
    if not backup.exists():
        shutil.copy2(path, backup)
    kept_classes, removed_reasons = Counter(), Counter()
    frames, kept_objects, removed_objects = 0, 0, 0
    temporary = path.with_suffix(".jsonl.tmp")
    with open(path, "r", encoding="utf-8") as source, open(
            temporary, "w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            frames += 1
            stem = str(record.get("sample_id") or f"{int(record['frame_id']):010d}")
            depth_path = path.parent / "depth" / f"{stem}.png"
            depth_cm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_cm is None:
                raise FileNotFoundError(f"missing depth map: {depth_path}")
            depth_values_m = depth_cm.astype("float32") / 100.0
            cleaned = []
            for annotation in record.get("objects", []):
                class_name = annotation.get("class")
                if class_name not in TAXONOMY:
                    reason = "invalid_class"
                else:
                    reason = annotation_rejection_reason(
                        annotation, depth_values_m)
                if reason is not None:
                    removed_reasons[reason] += 1
                    removed_objects += 1
                    continue
                cleaned.append(annotation)
                kept_classes[class_name] += 1
                kept_objects += 1
            record["objects"] = cleaned
            record.setdefault("image_size", [int(depth_cm.shape[1]), int(depth_cm.shape[0])])
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return {
        "annotation": str(path), "backup": str(backup), "frames": frames,
        "kept_objects": kept_objects, "removed_objects": removed_objects,
        "kept_classes": dict(sorted(kept_classes.items())),
        "removed_reasons": dict(sorted(removed_reasons.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="logs/dataset_sanitization.json")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    reports = [sanitize_file(path) for path in root.rglob("annotations.jsonl")]
    summary = {
        "dataset": str(root), "files": len(reports),
        "frames": sum(report["frames"] for report in reports),
        "kept_objects": sum(report["kept_objects"] for report in reports),
        "removed_objects": sum(report["removed_objects"] for report in reports),
        "reports": reports,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
