"""Validate and curate the approved StopSign YOLO training supplement."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from audit_dataset_archive import sha256_file


SOURCE_TO_ADAS = {0: 6, 1: 6, 14: 7}
ADAS_NAMES = [
    "Person", "Bicycle", "Car", "Motorcycle",
    "Bus", "Truck", "TrafficLight", "StopSign",
]
SAFE_STEM = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_yolo_row(line, path, line_number):
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"{path}:{line_number}: expected 5 fields")
    try:
        class_id = int(fields[0])
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: non-numeric YOLO row") from exc
    values = (x, y, width, height)
    if class_id < 0 or class_id > 14:
        raise ValueError(f"{path}:{line_number}: class id outside [0, 14]")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: non-finite coordinate")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
    tolerance = 1e-4
    if (x - width / 2 < -tolerance or x + width / 2 > 1.0 + tolerance
            or y - height / 2 < -tolerance
            or y + height / 2 > 1.0 + tolerance):
        raise ValueError(f"{path}:{line_number}: box crosses image boundary")
    return class_id, values


def image_digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def heldout_hashes(roots):
    hashes = defaultdict(list)
    for root_value in roots:
        root = Path(root_value).resolve()
        for annotations in sorted(root.rglob("annotations.jsonl")):
            with open(annotations, "r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("split") == "train":
                        continue
                    stem = str(record.get("sample_id") or
                               f"{int(record['frame_id']):010d}")
                    image = annotations.parent / "rgb" / f"{stem}.jpg"
                    if image.is_file():
                        hashes[image_digest(image)].append(str(image))
    return hashes


def deduplicate_records(records):
    """Keep the first deterministic record for each exact image hash."""
    seen = set()
    kept, dropped = [], []
    for record in records:
        digest_value = record["sha256"]
        if digest_value in seen:
            dropped.append(record["stem"])
            continue
        seen.add(digest_value)
        kept.append(record)
    return kept, dropped


def validate_source(source, archive, expected_sha256, heldout_roots):
    import cv2

    source = Path(source).resolve()
    images_dir = source / "images"
    labels_dir = source / "labels"
    issues = []
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("source must contain images/ and labels/")
    actual_archive_hash = sha256_file(archive)
    if actual_archive_hash.lower() != expected_sha256.lower():
        issues.append({"type": "archive_sha256_mismatch",
                       "expected": expected_sha256.lower(),
                       "actual": actual_archive_hash.lower()})

    image_paths = {path.stem: path for path in images_dir.glob("*.jpg")}
    label_paths = {path.stem: path for path in labels_dir.glob("*.txt")}
    missing_images = sorted(set(label_paths) - set(image_paths))
    missing_labels = sorted(set(image_paths) - set(label_paths))
    if missing_images:
        issues.append({"type": "missing_images", "count": len(missing_images)})
    if missing_labels:
        issues.append({"type": "missing_labels", "count": len(missing_labels)})

    selected = []
    source_class_counts = Counter()
    for stem in sorted(set(image_paths) & set(label_paths)):
        if not SAFE_STEM.fullmatch(stem):
            issues.append({"type": "unsafe_stem", "stem": stem})
            continue
        rows = []
        try:
            for line_number, line in enumerate(
                    label_paths[stem].read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    rows.append(parse_yolo_row(line, label_paths[stem], line_number))
        except (OSError, ValueError) as exc:
            issues.append({"type": "invalid_label", "stem": stem,
                           "error": str(exc)})
            continue
        source_class_counts.update(class_id for class_id, _ in rows)
        if not any(class_id == 14 for class_id, _ in rows):
            continue
        image = cv2.imread(str(image_paths[stem]), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or min(image.shape[:2]) < 32:
            issues.append({"type": "corrupt_or_tiny_image", "stem": stem})
            continue
        mapped_rows = []
        for class_id, values in rows:
            target_id = SOURCE_TO_ADAS.get(class_id)
            if target_id is None:
                continue
            mapped_rows.append((target_id, values))
        selected.append({"stem": stem, "image": image_paths[stem],
                         "rows": mapped_rows,
                         "sha256": image_digest(image_paths[stem])})

    selected_hashes = defaultdict(list)
    for record in selected:
        selected_hashes[record["sha256"]].append(record["stem"])
    internal_duplicates = [values for values in selected_hashes.values()
                           if len(values) > 1]
    selected, duplicate_images_removed = deduplicate_records(selected)
    mapped_class_counts = Counter()
    stop_areas = []
    for record in selected:
        for target_id, values in record["rows"]:
            mapped_class_counts[ADAS_NAMES[target_id]] += 1
            if target_id == 7:
                stop_areas.append(values[2] * values[3])
    heldout = heldout_hashes(heldout_roots)
    heldout_collisions = [
        {"stem": record["stem"], "heldout": heldout[record["sha256"]]}
        for record in selected if record["sha256"] in heldout
    ]
    if heldout_collisions:
        issues.append({"type": "heldout_image_leakage",
                       "count": len(heldout_collisions)})
    if mapped_class_counts["StopSign"] < 100:
        issues.append({"type": "insufficient_stopsign_boxes",
                       "count": mapped_class_counts["StopSign"], "minimum": 100})
    return {
        "source": str(source),
        "archive": str(Path(archive).resolve()),
        "archive_sha256": actual_archive_hash,
        "expected_archive_sha256": expected_sha256.lower(),
        "source_pairs": len(set(image_paths) & set(label_paths)),
        "selected_images": len(selected),
        "duplicate_selected_groups": len(internal_duplicates),
        "duplicate_selected_images_removed": len(duplicate_images_removed),
        "source_class_counts": dict(sorted(source_class_counts.items())),
        "mapped_class_counts": dict(sorted(mapped_class_counts.items())),
        "stop_box_area": {
            "min": min(stop_areas) if stop_areas else None,
            "max": max(stop_areas) if stop_areas else None,
        },
        "heldout_roots": [str(Path(root).resolve()) for root in heldout_roots],
        "heldout_hashes_checked": len(heldout),
        "heldout_collisions": heldout_collisions,
        "issues": issues,
        "integrity_valid": not issues,
        "records": selected,
    }


def materialize(report, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    images_dir = output / "train" / "images"
    labels_dir = output / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    link_count = copy_count = 0
    for record in report["records"]:
        filename = f"hf7073297_{record['stem']}"
        image_output = images_dir / f"{filename}.jpg"
        try:
            os.link(record["image"], image_output)
            link_count += 1
        except OSError:
            shutil.copy2(record["image"], image_output)
            copy_count += 1
        lines = [
            f"{class_id} " + " ".join(f"{value:.8f}" for value in values)
            for class_id, values in record["rows"]
        ]
        (labels_dir / f"{filename}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    manifest = {key: value for key, value in report.items() if key != "records"}
    manifest.update({
        "schema_version": 1,
        "output": str(output),
        "split": "train",
        "taxonomy": ADAS_NAMES,
        "mapping": {str(key): value for key, value in SOURCE_TO_ADAS.items()},
        "source_revision": "7073297e955abcc414ee24dde0f31648ce665ec7",
        "declared_license": "CC BY 4.0",
        "license_provenance_risk": "medium",
        "hardlinked_images": link_count,
        "copied_images": copy_count,
    })
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "SOURCE_ATTRIBUTION.md").write_text(
        "# StopSign training supplement attribution\n\n"
        "Source: ayoubsa/Sign_Road_Detection_Dataset on Hugging Face\n\n"
        "Pinned revision: `7073297e955abcc414ee24dde0f31648ce665ec7`\n\n"
        "Declared license: CC BY 4.0. The dataset card does not provide "
        "per-image provenance; see `docs/stopsign_dataset_security_audit.md`.\n",
        encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--heldout-root", action="append", default=[])
    parser.add_argument("--report")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = validate_source(args.source, args.archive, args.expected_sha256,
                             args.heldout_root)
    printable = {key: value for key, value in report.items() if key != "records"}
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(printable, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    if not report["integrity_valid"]:
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    if args.dry_run:
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return
    manifest = materialize(report, args.output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
