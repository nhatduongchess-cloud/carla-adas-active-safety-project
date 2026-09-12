"""Neural perception composition and bounded asynchronous scheduling."""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

from inference_scheduler import (LatestFrameScheduler,
                                 usable_inference_payload)
from lane_detection import LaneDetector
from lane_source import LaneSourceArbiter
from perception_contracts import LaneEstimate
from learned_lane import GroundProjector, LearnedLaneAdapter
from lidar_processor import LidarProcessor
from object_tracking import EnsembleVehicleTracker
from sensor_fusion import SensorFusion
from ufldv2_backend import UFLDv2Backend


class NeuralPerceptionRuntime:
    """Own YOLO/UFLDv2 initialization, fusion workers and GPU queue."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.object_tracker = EnsembleVehicleTracker(
            device=cfg.YOLO_DEVICE,
            use_ensemble=cfg.USE_ENSEMBLE,
            model_a=cfg.YOLO_MODEL_A,
            model_b=cfg.YOLO_MODEL_B,
            imgsz=cfg.YOLO_IMGSZ,
            half=cfg.YOLO_HALF,
            use_optimized=cfg.YOLO_USE_OPTIMIZED,
            target_classes=cfg.DETECTED_CLASSES,
            class_names=cfg.CLASS_NAMES,
            conf_threshold=cfg.CONF_THRES,
            nms_threshold=cfg.IOU_THRES,
            traffic_light_model_path=cfg.TRAFFIC_LIGHT_MODEL_PATH,
        )
        self.object_tracker.warmup(cfg.CAM_WIDTH, cfg.CAM_HEIGHT, runs=2)
        self.lidar_processor = LidarProcessor(
            eps=0.6, min_samples=5, max_points=cfg.LIDAR_MAX_POINTS)
        # The classical detector remains a shadow/debug channel.
        self.lane_detector = LaneDetector()
        self.sensor_fusion = SensorFusion(
            width=cfg.CAM_WIDTH,
            height=cfg.CAM_HEIGHT,
            fov=cfg.CAM_FOV,
            lidar_y_sign=cfg.LIDAR_Y_SIGN,
            real_heights_m=cfg.REAL_HEIGHTS_M,
            default_height_m=cfg.DEFAULT_HEIGHT_M,
            depth_min_m=cfg.DEPTH_MIN_M,
            depth_max_m=cfg.DEPTH_MAX_M,
            camera_pose=(cfg.CAM_X, cfg.CAM_Y, cfg.CAM_Z,
                         cfg.CAM_ROLL, cfg.CAM_PITCH, cfg.CAM_YAW),
            lidar_pose=(cfg.LIDAR_X, cfg.LIDAR_Y, cfg.LIDAR_Z,
                        cfg.LIDAR_ROLL, cfg.LIDAR_PITCH, cfg.LIDAR_YAW),
        )

        fx, fy, cx, cy = cfg.camera_intrinsics(
            cfg.CAM_WIDTH, cfg.CAM_HEIGHT, cfg.CAM_FOV)
        lane_backend = None
        self.lane_backend_error: Optional[str] = None
        if cfg.LEARNED_LANE_ENABLED:
            try:
                lane_backend = UFLDv2Backend(
                    cfg.LEARNED_LANE_MODEL_PATH,
                    cfg.LEARNED_LANE_MODEL_SHA256,
                    device=cfg.LEARNED_LANE_DEVICE,
                    use_fp16=cfg.LEARNED_LANE_FP16,
                )
                print("[Lane] UFLDv2 Tusimple ResNet18 ready "
                      f"({cfg.LEARNED_LANE_DEVICE}, "
                      f"{'FP16' if lane_backend.use_fp16 else 'FP32'}).")
            except Exception as exc:
                self.lane_backend_error = f"{type(exc).__name__}: {exc}"
                print("[Lane] UFLDv2 unavailable; map fallback active: "
                      f"{self.lane_backend_error}")

        self.lane_adapter = LearnedLaneAdapter(
            GroundProjector(fx, fy, cx, cy, cfg.CAM_Z, cfg.CAM_X, cfg.LIDAR_X),
            backend=lane_backend,
            model_path=cfg.LEARNED_LANE_MODEL_PATH,
            input_size=(cfg.LEARNED_LANE_INPUT_WIDTH,
                        cfg.LEARNED_LANE_INPUT_HEIGHT),
        )
        if self.lane_backend_error:
            self.lane_adapter.provenance_error = self.lane_backend_error
        self.lane_arbiter = LaneSourceArbiter(
            cfg.LEARNED_LANE_CONFIDENCE,
            cfg.LEARNED_LANE_STABLE_FRAMES,
            cfg.LEARNED_LANE_INVALID_FRAMES,
            cfg.LANE_WIDTH_MIN_M,
            cfg.LANE_WIDTH_MAX_M,
            cfg.LANE_GEOMETRY_JUMP_M,
        )
        # Same two-worker concurrency as before, but lane has no queued backlog.
        # Object keeps the only replaceable pending slot; a busy lane skips work.
        self._lane_scheduler = (
            LatestFrameScheduler(self._learned_lane_worker, "learned-lane",
                                 deadline_ms=cfg.PERCEPTION_DEADLINE_MS)
            if self.lane_adapter.ready else None)
        self._scheduler = LatestFrameScheduler(
            self._neural_worker, deadline_ms=cfg.PERCEPTION_DEADLINE_MS)
        self._closed = False

    def _learned_lane_worker(self, payload, frame_id: int):
        image, capture_timestamp = payload
        started = time.perf_counter()
        estimate = self.lane_adapter.infer(
            image, frame_id, capture_timestamp)
        return {"lane": estimate, "stage_latency_ms": {
            "learned_lane": (time.perf_counter() - started) * 1000.0}}

    def _neural_worker(self, payload, frame_id: int):
        image, same_frame_point_cloud, run_lane, capture_timestamp = payload
        if run_lane and self._lane_scheduler is not None and not self._closed:
            self._lane_scheduler.submit(
                frame_id, capture_timestamp, (image, capture_timestamp), only_if_idle=True)

        stage_started = time.perf_counter()
        annotated, detections, worker_metrics = self.object_tracker.process(
            image, None, frame_id)
        object_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        obstacles = self.lidar_processor.extract_perception_obstacles(
            same_frame_point_cloud)
        fused = self.sensor_fusion.fuse(detections, obstacles)
        fusion_ms = (time.perf_counter() - stage_started) * 1000.0
        return {
            "annotated": annotated,
            "detections": detections,
            "fused": fused,
            "metrics": worker_metrics,
            "stage_latency_ms": {
                "object_detection": object_ms,
                "lidar_fusion": fusion_ms,
            },
        }

    def submit(self, frame_id: int, timestamp: float, image, point_cloud,
               run_lane: bool) -> None:
        self._scheduler.submit(
            frame_id,
            timestamp,
            (image.copy(), point_cloud.copy(), bool(run_lane), timestamp),
        )

    def poll(self, current_frame_id: int, now_s: float,
             max_age_ms: float) -> Tuple[Any, Any]:
        result = self._scheduler.latest(
            current_frame_id, now_s, max_age_ms)
        return result, usable_inference_payload(result)

    def scheduler_stats(self) -> dict:
        return {**self._scheduler.stats(),
                "lane": self._lane_scheduler.stats() if self._lane_scheduler is not None else None}

    def poll_lane(self, current_frame_id: int, now_s: float,
                  max_age_ms: float) -> Tuple[Any, Any]:
        if self._lane_scheduler is None:
            return None, None
        result = self._lane_scheduler.latest(current_frame_id, now_s, max_age_ms)
        if result is None:
            return None, None
        if result.error is not None:
            # A lane fault must not erase successful objects or retain an active lane.
            return result, LaneEstimate(result.frame_id, result.timestamp, "learned",
                                        valid=False, reason=result.error)
        return result, result.payload["lane"]

    def start_measurement(self) -> None:
        self._scheduler.start_measurement()
        if self._lane_scheduler is not None:
            self._lane_scheduler.start_measurement()

    def stop_measurement(self) -> None:
        self._scheduler.stop_measurement()
        if self._lane_scheduler is not None:
            self._lane_scheduler.stop_measurement()

    def record_consumed(self, result, now_s: float) -> None:
        self._scheduler.record_consumed(result, now_s)

    def record_lane_consumed(self, result, now_s: float) -> None:
        if self._lane_scheduler is not None:
            self._lane_scheduler.record_consumed(result, now_s)

    def close(self, timeout=2.0) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_measurement()
        deadline = time.monotonic() + max(0., float(timeout))
        self._scheduler.close(timeout=max(0., deadline-time.monotonic()))
        if self._lane_scheduler is not None:
            self._lane_scheduler.close(timeout=max(0., deadline-time.monotonic()))
