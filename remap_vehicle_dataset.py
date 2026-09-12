"""Map the existing 12 Roboflow vehicle classes into the 8-class ADAS taxonomy."""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil


TAXONOMY = ["Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck",
            "TrafficLight", "StopSign"]
SOURCE_TO_TARGET = {
    0: 4, 1: 5, 2: 4, 3: 4, 4: 2, 5: 5,
    6: 4, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5,
}
EXPECTED_SOURCE_NAMES = [
    "big bus", "big truck", "bus-l-", "bus-s-", "car", "mid truck",
    "small bus", "small truck", "truck-l-", "truck-m-", "truck-s-",
    "truck-xl-",
]


def source_names(data_yaml):
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            value = ast.literal_eval(line.split(":", 1)[1].strip())
            return list(value)
    raise ValueError(f"names field missing from {data_yaml}")


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents:
        raise SystemExit("output must be outside the source dataset")
    data_yaml = source / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"source data.yaml missing: {data_yaml}")
    actual_names = source_names(data_yaml)
    if actual_names != EXPECTED_SOURCE_NAMES:
        raise SystemExit(
            "source taxonomy differs from the audited Roboflow-100 mapping: "
            f"{actual_names}")
    split_names = {"train": "train", "valid": "val", "test": "test"}
    plan = {"source": str(source), "output": str(output),
            "mapping": SOURCE_TO_TARGET, "taxonomy": TAXONOMY}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    counts = {name: 0 for name in TAXONOMY}
    image_modes = {"hardlink": 0, "copy": 0}
    samples = 0
    for source_split, output_split in split_names.items():
        images = source / source_split / "images"
        labels = source / source_split / "labels"
        if not images.is_dir() or not labels.is_dir():
            continue
        out_images, out_labels = output / output_split / "images", output / output_split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        for label_path in labels.glob("*.txt"):
            mapped = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"malformed YOLO row in {label_path}: {line!r}")
                source_id = int(fields[0])
                if source_id not in SOURCE_TO_TARGET:
                    raise ValueError(
                        f"unexpected class id {source_id} in {label_path}")
                coordinates = [float(value) for value in fields[1:]]
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise ValueError(f"YOLO coordinate outside [0, 1] in {label_path}")
                target_id = SOURCE_TO_TARGET[source_id]
                mapped.append(" ".join([str(target_id), *fields[1:]]))
                counts[TAXONOMY[target_id]] += 1
            (out_labels / label_path.name).write_text(
                "\n".join(mapped) + ("\n" if mapped else ""), encoding="utf-8")
            candidates = list(images.glob(label_path.stem + ".*"))
            if len(candidates) != 1:
                raise ValueError(
                    f"expected exactly one image for {label_path}, got {len(candidates)}")
            mode = link_or_copy(candidates[0], out_images / candidates[0].name)
            image_modes[mode] += 1
            samples += 1
    yaml_text = (
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(TAXONOMY)}\nnames: {TAXONOMY!r}\n")
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (output / "remap_manifest.json").write_text(json.dumps(
        {**plan, "source_names": actual_names, "class_counts": counts,
         "samples": samples, "image_materialization": image_modes,
         "declared_license": "CC BY 4.0"}, indent=2), encoding="utf-8")
    print(json.dumps({"class_counts": counts, "samples": samples,
                      "image_materialization": image_modes}, indent=2))


if __name__ == "__main__":
    main()
