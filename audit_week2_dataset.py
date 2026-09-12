"""Create a reproducible Week-2 CARLA dataset quality report.

The audit is read-only with respect to captured samples. It summarizes the
split, weather, class, modality and sensor-synchronization evidence needed
before model training starts.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path


TAXONOMY = (
    "Person", "Bicycle", "Car", "Motorcycle",
    "Bus", "Truck", "TrafficLight", "StopSign",
)
EXPECTED_SPLITS = {"train": 14000, "validation": 3000, "test": 3000}
EXPECTED_WEATHERS = {"clear", "light_rain", "heavy_rain", "fog", "storm"}
MODALITIES = {
    "rgb": ".jpg",
    "road_line": ".png",
    "depth": ".png",
    "lidar": ".npz",
    "radar": ".npz",
}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _distance_bucket(distance_m):
    if distance_m <= 20.0:
        return "0-20m"
    if distance_m <= 40.0:
        return "20-40m"
    return "40-80m"


def evaluate_gates(frames, split_frames, split_weathers, modality_counts,
                   radar_available_frames, sync_errors, radar_sync_errors,
                   completed_jobs, integrity, class_counts,
                   split_class_counts=None):
    """Return distinct collection and training-readiness gates."""
    present_classes = {
        name for name, count in class_counts.items() if int(count) > 0}
    missing_classes = sorted(set(TAXONOMY) - present_classes)
    split_class_counts = split_class_counts or {}
    missing_classes_by_split = {
        split: sorted(set(TAXONOMY) - {
            name for name, count in split_class_counts.get(split, {}).items()
            if int(count) > 0})
        for split in EXPECTED_SPLITS
    }
    gates = {
        "target_frames": frames == sum(EXPECTED_SPLITS.values()),
        "split_targets": dict(split_frames) == EXPECTED_SPLITS,
        "weather_coverage": all(
            set(split_weathers.get(split, ())) == EXPECTED_WEATHERS
            for split in EXPECTED_SPLITS),
        "all_modalities_present": all(
            modality_counts.get(name, 0) == frames for name in MODALITIES),
        "radar_availability_99_5pct": (
            frames > 0 and radar_available_frames / frames >= 0.995),
        "sensor_frame_errors_zero": sync_errors == 0 and radar_sync_errors == 0,
        "all_jobs_completed": completed_jobs == 30,
        "no_duplicates_or_split_leakage": bool(integrity.get("valid")),
        "all_taxonomy_classes_present": not missing_classes,
        "all_splits_cover_taxonomy": not any(missing_classes_by_split.values()),
    }
    training_only = {"all_taxonomy_classes_present", "all_splits_cover_taxonomy"}
    collection_keys = tuple(key for key in gates if key not in training_only)
    return {
        "checks": gates,
        "collection_pass": all(gates[key] for key in collection_keys),
        "training_ready_8_class": all(gates.values()),
        "missing_classes": missing_classes,
        "missing_classes_by_split": missing_classes_by_split,
    }


def build_report(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")

    split_frames = Counter()
    town_frames = Counter()
    weather_frames = Counter()
    split_weather_frames = Counter()
    split_weathers = defaultdict(set)
    class_counts = Counter()
    split_class_counts = defaultdict(Counter)
    distance_counts = Counter()
    traffic_light_states = Counter()
    frames = radar_available_frames = empty_object_frames = 0

    annotation_files = sorted(root.rglob("annotations.jsonl"))
    for annotations in annotation_files:
        with open(annotations, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {annotations}:{line_number}") from exc
                split = str(record.get("split", "unknown"))
                town = str(record.get("town", "unknown"))
                weather = str(record.get("weather", "unknown"))
                frames += 1
                split_frames[split] += 1
                town_frames[town] += 1
                weather_frames[weather] += 1
                split_weather_frames[(split, weather)] += 1
                split_weathers[split].add(weather)
                radar_available_frames += bool(record.get("radar_available"))
                objects = record.get("objects", [])
                empty_object_frames += not objects
                for obj in objects:
                    class_name = str(obj.get("class", "Unknown"))
                    class_counts[class_name] += 1
                    split_class_counts[split][class_name] += 1
                    distance_counts[_distance_bucket(float(obj["distance_m"]))] += 1
                    if class_name == "TrafficLight":
                        traffic_light_states[str(
                            obj.get("traffic_light_state", "unknown"))] += 1

    modality_counts = Counter()
    modality_bytes = Counter()
    for annotations in annotation_files:
        job_dir = annotations.parent
        for modality, extension in MODALITIES.items():
            for path in (job_dir / modality).glob(f"*{extension}"):
                modality_counts[modality] += 1
                modality_bytes[modality] += path.stat().st_size

    sync_errors = radar_sync_errors = stale_frames = radar_stale_frames = 0
    max_sync_p99_ms = max_radar_sync_p99_ms = 0.0
    manifests = sorted(root.rglob("manifest.json"))
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        sync = manifest.get("sync", {})
        radar_sync = manifest.get("radar_sync", {})
        sync_errors += int(sync.get("frame_errors", 0))
        radar_sync_errors += int(radar_sync.get("frame_errors", 0))
        stale_frames += int(sync.get("stale_discarded", 0))
        radar_stale_frames += int(radar_sync.get("stale_discarded", 0))
        max_sync_p99_ms = max(max_sync_p99_ms, float(sync.get("wait_p99_ms", 0.0)))
        max_radar_sync_p99_ms = max(
            max_radar_sync_p99_ms, float(radar_sync.get("wait_p99_ms", 0.0)))

    state_path = root / "collection_state.json"
    integrity_path = root / "dataset_integrity.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    completed_jobs = len(set(state.get("completed_jobs", [])))
    gates = evaluate_gates(
        frames, split_frames, split_weathers, modality_counts,
        radar_available_frames, sync_errors, radar_sync_errors,
        completed_jobs, integrity, class_counts, split_class_counts)

    warnings = []
    if gates["missing_classes"]:
        warnings.append(
            "Missing visual labels for: " + ", ".join(gates["missing_classes"])
            + ". Do not substitute CARLA traffic.stop trigger volumes; they are not "
              "visible sign assets. Add an audited visual source before 8-class training.")
    for split, missing in gates["missing_classes_by_split"].items():
        if missing:
            warnings.append(
                f"{split} split has no instances for: {', '.join(missing)}.")
    for split, counts in sorted(split_class_counts.items()):
        sparse = sorted(
            f"{name}={count}" for name, count in counts.items() if count < 100)
        if sparse:
            warnings.append(
                f"{split} split has classes below the 100-instance diagnostic "
                f"threshold: {', '.join(sparse)}.")

    return {
        "schema_version": 1,
        "dataset": str(root),
        "frames": frames,
        "jobs": {"manifests": len(manifests), "completed": completed_jobs},
        "frames_by_split": dict(sorted(split_frames.items())),
        "frames_by_town": dict(sorted(town_frames.items())),
        "frames_by_weather": dict(sorted(weather_frames.items())),
        "frames_by_split_weather": {
            f"{split}/{weather}": value
            for (split, weather), value in sorted(split_weather_frames.items())
        },
        "objects": {
            "total": sum(class_counts.values()),
            "empty_frame_count": empty_object_frames,
            "class_counts": dict(sorted(class_counts.items())),
            "class_counts_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(split_class_counts.items())
            },
            "distance_counts": dict(sorted(distance_counts.items())),
            "traffic_light_states": dict(sorted(traffic_light_states.items())),
        },
        "modalities": {
            name: {"files": modality_counts[name],
                   "size_gib": round(modality_bytes[name] / 1024 ** 3, 3)}
            for name in MODALITIES
        },
        "dataset_size_gib": round(sum(modality_bytes.values()) / 1024 ** 3, 3),
        "sensor_integrity": {
            "radar_available_frames": radar_available_frames,
            "radar_availability_pct": round(
                100.0 * radar_available_frames / max(1, frames), 4),
            "sync_frame_errors": sync_errors,
            "radar_frame_errors": radar_sync_errors,
            "sync_stale_discarded": stale_frames,
            "radar_stale_discarded": radar_stale_frames,
            "max_sync_wait_p99_ms": max_sync_p99_ms,
            "max_radar_wait_p99_ms": max_radar_sync_p99_ms,
        },
        "integrity": {
            "valid": bool(integrity.get("valid")),
            "duplicate_groups": int(integrity.get("duplicate_groups", 0)),
            "split_leakage_groups": int(integrity.get("split_leakage_groups", 0)),
            "town_split_leakage": integrity.get("town_split_leakage", {}),
            "issue_count": len(integrity.get("issues", [])),
        },
        "gates": gates,
        "warnings": warnings,
    }


def render_markdown(report):
    checks = report["gates"]["checks"]
    lines = [
        "# CARLA Week-2 Dataset Audit", "",
        f"Dataset: `{report['dataset']}`", "",
        f"- Frames: **{report['frames']:,}**",
        f"- Captured modalities: **{report['dataset_size_gib']:.3f} GiB**",
        f"- Collection gate: **{'PASS' if report['gates']['collection_pass'] else 'FAIL'}**",
        f"- 8-class training ready: **{'YES' if report['gates']['training_ready_8_class'] else 'NO'}**",
        "", "## Gates", "", "| Check | Result |", "|---|:---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items())
    lines.extend(["", "## Split", "", "| Split | Frames |", "|---|---:|"])
    lines.extend(
        f"| {split} | {count:,} |"
        for split, count in report["frames_by_split"].items())
    lines.extend(["", "## Classes", "", "| Class | Instances |", "|---|---:|"])
    lines.extend(
        f"| {class_name} | {report['objects']['class_counts'].get(class_name, 0):,} |"
        for class_name in TAXONOMY)
    sensor = report["sensor_integrity"]
    lines.extend([
        "", "## Sensor integrity", "",
        f"- Radar availability: **{sensor['radar_availability_pct']:.4f}%**",
        f"- Camera/depth/road-line/LiDAR frame errors: **{sensor['sync_frame_errors']}**",
        f"- Radar frame errors: **{sensor['radar_frame_errors']}**",
        f"- Worst required-sensor wait p99: **{sensor['max_sync_wait_p99_ms']:.2f} ms**",
        f"- Worst radar wait p99: **{sensor['max_radar_wait_p99_ms']:.2f} ms**",
    ])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    json_output = Path(args.json_output).resolve() if args.json_output else root / "week2_audit.json"
    markdown_output = (Path(args.markdown_output).resolve()
                       if args.markdown_output else root / "week2_audit.md")
    report = build_report(root)
    _atomic_json(json_output, report)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_output.with_suffix(markdown_output.suffix + ".tmp")
    temporary.write_text(render_markdown(report), encoding="utf-8")
    os.replace(temporary, markdown_output)
    print(json.dumps({
        "frames": report["frames"],
        "collection_pass": report["gates"]["collection_pass"],
        "training_ready_8_class": report["gates"]["training_ready_8_class"],
        "missing_classes": report["gates"]["missing_classes"],
        "json_output": str(json_output),
        "markdown_output": str(markdown_output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
