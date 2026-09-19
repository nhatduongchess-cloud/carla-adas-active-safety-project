"""Reproducible YOLOv8n training/promotion for the unified 8-class taxonomy."""

import argparse
import json
import os
from pathlib import Path
import shutil

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import YAML
import torch

from modules.provenance import file_record
from modules.training_utils import collect_image_paths, deterministic_subset


def _posttrain_data_yaml(data_yaml, run_dir, fraction, seed):
    """Create a deterministic test subset YAML because val() ignores fraction."""
    if fraction == 1.0:
        return str(Path(data_yaml).resolve()), None
    resolved = check_det_dataset(str(data_yaml), autodownload=False, split="test")
    all_test_images = collect_image_paths(resolved["test"])
    selected = deterministic_subset(all_test_images, fraction, seed)
    subset_list = run_dir / "posttrain_test_subset.txt"
    subset_list.write_text(
        "".join(f"{path.as_posix()}\n" for path in selected), encoding="utf-8")
    subset_yaml = run_dir / "posttrain_subset.yaml"
    subset_config = {
        "path": str(resolved["path"]),
        "train": _yaml_value(resolved["train"]),
        "val": _yaml_value(resolved["val"]),
        "test": subset_list.resolve().as_posix(),
        "nc": int(resolved["nc"]),
        "names": resolved["names"],
    }
    YAML.save(subset_yaml, subset_config)
    return str(subset_yaml), len(selected)


def _yaml_value(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="merged 8-class data.yaml")
    parser.add_argument("--base", default="yolov8n.pt")
    parser.add_argument("--resume", action="store_true",
                        help="tiếp tục từ checkpoint --base (khuyến nghị weights/last.pt)")
    parser.add_argument("--project", default="runs/object_perception")
    parser.add_argument("--name", default="yolov8n_adas_seed42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of training images; use 0.02 for a smoke run")
    parser.add_argument("--val-fraction", type=float, default=1.0,
                        help="fraction of held-out images used by post-train validation")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, or a CUDA device such as 0")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--promote-to", default=None)
    parser.add_argument("--min-map50", type=float, default=0.75)
    parser.add_argument("--max-latency-ms", type=float, default=50.0)
    args = parser.parse_args()

    if not os.path.isfile(args.base):
        raise SystemExit(f"base weight missing; auto-download forbidden: {os.path.abspath(args.base)}")
    if not os.path.isfile(args.data):
        raise SystemExit(f"dataset yaml missing: {os.path.abspath(args.data)}")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")
    if not 0.0 < args.val_fraction <= 1.0:
        raise SystemExit("--val-fraction must be in (0, 1]")
    if args.workers < 0:
        raise SystemExit("--workers must be non-negative")
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_amp = bool(torch.cuda.is_available() and str(device).lower() != "cpu")
    project_dir = Path(args.project).resolve()
    print(f"[train] device={device}, seed={args.seed}, AMP={use_amp}, "
          f"fraction={args.fraction}")
    model = YOLO(args.base)
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, workers=args.workers, amp=use_amp, patience=args.patience,
        fraction=args.fraction,
        seed=args.seed, deterministic=True, project=str(project_dir), name=args.name,
        # A resumed run intentionally reuses its existing save directory; a
        # fresh run remains protected from accidental overwrite.
        exist_ok=bool(args.resume), plots=True)
    if args.resume:
        print(f"[train] resuming checkpoint={os.path.abspath(args.base)}")
        train_kwargs["resume"] = True
    model.train(**train_kwargs)

    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if not trainer_save_dir:
        raise SystemExit("training completed without an Ultralytics save directory")
    run_dir = Path(trainer_save_dir).resolve()
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training completed without checkpoint: {best}")
    validation_data, validation_image_count = _posttrain_data_yaml(
        args.data, run_dir, args.val_fraction, args.seed)
    if validation_image_count is not None:
        print(f"[train] post-train test subset={validation_image_count} images")
    candidate = YOLO(str(best))
    validation = candidate.val(
        data=validation_data, split="test", imgsz=args.imgsz,
        batch=args.batch, device=device, workers=args.workers,
        plots=True)
    map50 = float(validation.box.map50)
    latency_ms = float(validation.speed.get("inference", float("inf")))
    baseline_map50 = -1.0
    if args.baseline_metrics:
        with open(args.baseline_metrics, "r", encoding="utf-8") as stream:
            baseline_map50 = float(json.load(stream).get("map50", -1.0))
    promoted = bool(map50 >= args.min_map50 and map50 > baseline_map50
                    and latency_ms <= args.max_latency_ms)
    if args.promote_to and promoted:
        destination = Path(args.promote_to).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
    report = {"map50": map50, "latency_ms": latency_ms,
              "baseline_map50": baseline_map50, "promoted": promoted,
              "candidate": file_record(best), "base": file_record(args.base),
              "run_dir": str(run_dir),
              "posttrain_validation_images": validation_image_count,
              "settings": vars(args)}
    with open(run_dir / "promotion_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    # Fix lỗi multiprocessing trên Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()
