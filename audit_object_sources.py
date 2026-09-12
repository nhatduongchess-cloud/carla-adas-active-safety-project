"""Audit combined CARLA object-detection sources before YOLO conversion."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from audit_week2_dataset import TAXONOMY
from validate_dataset import digest


SPLITS = ("train", "validation", "test")


def evaluate_source_gates(class_counts_by_split, sources_valid,
                          cross_source_duplicates=0, split_leakage=0):
    missing = {
        split: [name for name in TAXONOMY
                if int(class_counts_by_split.get(split, {}).get(name, 0)) <= 0]
        for split in SPLITS
    }
    checks = {
        "all_source_integrity_reports_valid": bool(sources_valid),
        "no_cross_source_duplicates": int(cross_source_duplicates) == 0,
        "no_cross_split_leakage": int(split_leakage) == 0,
        "all_splits_cover_8_class_taxonomy": not any(missing.values()),
    }
    return {
        "checks": checks,
        "training_ready_8_class": all(checks.values()),
        "missing_classes_by_split": missing,
    }


def build_report(source_paths, external_manifest_paths=()):
    roots = [Path(path).resolve() for path in source_paths]
    if not roots:
        raise ValueError("at least one dataset source is required")
    if len(set(roots)) != len(roots):
        raise ValueError("dataset sources must be unique")

    class_counts_by_split = defaultdict(Counter)
    frames_by_split = Counter()
    weather_counts = Counter()
    hashes = defaultdict(list)
    source_reports = []
    issues = []

    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        integrity_path = root / "dataset_integrity.json"
        integrity = (json.loads(integrity_path.read_text(encoding="utf-8"))
                     if integrity_path.is_file() else None)
        source_valid = bool(integrity and integrity.get("valid"))
        source_frames = 0
        source_classes = Counter()
        for annotations in sorted(root.rglob("annotations.jsonl")):
            with open(annotations, "r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    split = str(record.get("split", "unknown"))
                    if split not in SPLITS:
                        issues.append({"type": "invalid_split", "source": str(root),
                                       "file": str(annotations), "line": line_number,
                                       "split": split})
                        continue
                    source_frames += 1
                    frames_by_split[split] += 1
                    weather_counts[(split, str(record.get("weather", "unknown")))] += 1
                    stem = str(record.get("sample_id") or
                               f"{int(record['frame_id']):010d}")
                    image_path = annotations.parent / "rgb" / f"{stem}.jpg"
                    if not image_path.is_file():
                        issues.append({"type": "missing_rgb", "path": str(image_path)})
                    else:
                        hashes[digest(image_path)].append(
                            (str(root), split, str(image_path)))
                    for obj in record.get("objects", []):
                        name = str(obj.get("class", "Unknown"))
                        source_classes[name] += 1
                        class_counts_by_split[split][name] += 1
        source_reports.append({
            "kind": "carla",
            "source": str(root),
            "frames": source_frames,
            "class_counts": dict(sorted(source_classes.items())),
            "integrity_report": str(integrity_path) if integrity_path.is_file() else None,
            "integrity_valid": source_valid,
        })

    for manifest_value in external_manifest_paths:
        manifest_path = Path(manifest_value).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = str(manifest.get("split", "unknown"))
        selected_images = int(manifest.get("selected_images", 0))
        image_count = len(list((manifest_path.parent / split / "images").glob("*.jpg")))
        label_count = len(list((manifest_path.parent / split / "labels").glob("*.txt")))
        external_valid = bool(
            manifest.get("integrity_valid")
            and split == "train"
            and selected_images > 0
            and image_count == selected_images
            and label_count == selected_images
            and not manifest.get("heldout_collisions")
            and manifest.get("archive_sha256")
            == manifest.get("expected_archive_sha256")
        )
        if not external_valid:
            issues.append({"type": "invalid_external_manifest",
                           "path": str(manifest_path)})
        counts = Counter({
            str(name): int(count)
            for name, count in manifest.get("mapped_class_counts", {}).items()
        })
        class_counts_by_split["train"].update(counts)
        frames_by_split["train"] += selected_images
        source_reports.append({
            "kind": "external_yolo",
            "source": str(manifest_path.parent),
            "frames": selected_images,
            "class_counts": dict(sorted(counts.items())),
            "integrity_report": str(manifest_path),
            "integrity_valid": external_valid,
            "declared_license": manifest.get("declared_license"),
            "license_provenance_risk": manifest.get("license_provenance_risk"),
        })

    cross_source_groups = [
        records for records in hashes.values()
        if len({record[0] for record in records}) > 1
    ]
    leakage_groups = [
        records for records in cross_source_groups
        if len({record[1] for record in records}) > 1
    ]
    gates = evaluate_source_gates(
        class_counts_by_split,
        all(source["integrity_valid"] for source in source_reports) and not issues,
        len(cross_source_groups), len(leakage_groups))
    return {
        "schema_version": 1,
        "sources": source_reports,
        "frames": sum(frames_by_split.values()),
        "frames_by_split": dict(sorted(frames_by_split.items())),
        "frames_by_split_weather": {
            f"{split}/{weather}": count
            for (split, weather), count in sorted(weather_counts.items())
        },
        "class_counts_by_split": {
            split: dict(sorted(class_counts_by_split[split].items()))
            for split in SPLITS
        },
        "cross_source_duplicate_groups": len(cross_source_groups),
        "cross_split_leakage_groups": len(leakage_groups),
        "issues": issues,
        "gates": gates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--external-manifest", action="append", default=[])
    parser.add_argument("--output", default="logs/object_source_audit.json")
    args = parser.parse_args()
    report = build_report(args.sources, args.external_manifest)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "frames": report["frames"],
        "frames_by_split": report["frames_by_split"],
        "training_ready_8_class": report["gates"]["training_ready_8_class"],
        "missing_classes_by_split": report["gates"]["missing_classes_by_split"],
        "cross_source_duplicate_groups": report["cross_source_duplicate_groups"],
        "cross_split_leakage_groups": report["cross_split_leakage_groups"],
        "output": str(output),
    }, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["gates"]["training_ready_8_class"] else 2)


if __name__ == "__main__":
    main()
