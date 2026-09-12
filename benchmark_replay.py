"""Run object/LiDAR/radar perception against a recorded replay."""

import argparse
import json
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (CURRENT_DIR, os.path.join(CURRENT_DIR, "modules")):
    if path not in sys.path:
        sys.path.insert(0, path)

import config as cfg
from lidar_processor import LidarProcessor
from object_tracking import EnsembleVehicleTracker
from pipeline_metrics import PipelineMetrics
from radar_processor import RadarProcessor
from replay_io import ReplayReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", default="logs/replay_benchmark.json")
    args = parser.parse_args()

    detector = EnsembleVehicleTracker(
        device=cfg.YOLO_DEVICE, use_ensemble=cfg.USE_ENSEMBLE,
        model_a=cfg.YOLO_MODEL_A, model_b=cfg.YOLO_MODEL_B,
        imgsz=cfg.YOLO_IMGSZ, half=cfg.YOLO_HALF,
        use_optimized=cfg.YOLO_USE_OPTIMIZED,
        target_classes=cfg.DETECTED_CLASSES, class_names=cfg.CLASS_NAMES,
        conf_threshold=cfg.CONF_THRES, nms_threshold=cfg.IOU_THRES,
        traffic_light_model_path=cfg.TRAFFIC_LIGHT_MODEL_PATH)
    lidar = LidarProcessor(eps=0.6, min_samples=5, max_points=cfg.LIDAR_MAX_POINTS)
    radar = RadarProcessor(cfg.RADAR_X, cfg.RADAR_Y, cfg.RADAR_Z,
                           reference_x_m=cfg.LIDAR_X,
                           velocity_sign=cfg.RADAR_VELOCITY_SIGN,
                           min_target_height_m=cfg.RADAR_MIN_TARGET_HEIGHT_M)
    detector.warmup(cfg.CAM_WIDTH, cfg.CAM_HEIGHT, runs=2)
    metrics = PipelineMetrics({"object": cfg.PERCEPTION_DEADLINE_MS,
                               "safety_geometry": cfg.SAFETY_DEADLINE_MS})
    count = 0
    started = time.perf_counter()
    for record, arrays in ReplayReader(args.replay):
        with metrics.stage("object"):
            detector.process(arrays["rgb"], None, record["frame_id"])
        with metrics.stage("safety_geometry"):
            lidar.extract_perception_obstacles(arrays["lidar"])
            radar.process(arrays.get("radar"))
        metrics.sample_gpu_memory()
        count += 1
        if args.max_frames and count >= args.max_frames:
            break
    duration = time.perf_counter() - started
    result = {"frames": count, "duration_s": round(duration, 3),
              "throughput_fps": round(count / max(duration, 1e-9), 3),
              **metrics.summary(),
              "model_provenance": detector.model_provenance,
              "provenance_errors": detector.provenance_errors}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
