"""Create a pinned environment/model manifest; performs no downloads."""

import argparse
import json
import os

import config as cfg
from modules.provenance import runtime_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/runtime_manifest.json")
    parser.add_argument("--carla-version", default=None)
    args = parser.parse_args()
    models = [cfg.YOLO_MODEL_A]
    if cfg.USE_ENSEMBLE:
        models.append(cfg.YOLO_MODEL_B)
    if cfg.LEARNED_LANE_MODEL_PATH:
        models.append(cfg.LEARNED_LANE_MODEL_PATH)
    manifest = runtime_manifest(models, args.carla_version, config={
        "fps": cfg.FPS, "camera": [cfg.CAM_WIDTH, cfg.CAM_HEIGHT, cfg.CAM_FOV],
        "yolo_imgsz": cfg.YOLO_IMGSZ, "detect_every_n": cfg.DETECT_EVERY_N,
        "lane_every_n": cfg.LANE_EVERY_N, "radar_enabled": cfg.ENABLE_RADAR,
    })
    write_manifest(args.output, manifest)
    print(json.dumps({"written": os.path.abspath(args.output),
                      "models": manifest["models"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

