"""Runtime profile and command-line configuration for the ADAS application.

This module owns policy-free startup configuration.  The central orchestrator
selects a profile and receives the effective values; perception and sensor
modules never parse command-line arguments themselves.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple


PERFORMANCE_PROFILES = {
    "quality": {},
    "scale-85": {
        "CAM_WIDTH": 896,
        "CAM_HEIGHT": 504,
        "YOLO_IMGSZ": 448,
        "LIDAR_POINTS_PER_SECOND": 51000,
        "LIDAR_MAX_POINTS": 1700,
    },
    "balanced": {
        "CAM_WIDTH": 800,
        "CAM_HEIGHT": 450,
        "YOLO_IMGSZ": 448,
        "LIDAR_POINTS_PER_SECOND": 50000,
        "LIDAR_MAX_POINTS": 1600,
        "RENDER_EVERY_N": 4,
    },
    "low-memory": {
        "CAM_WIDTH": 640,
        "CAM_HEIGHT": 360,
        "YOLO_IMGSZ": 320,
        "LIDAR_POINTS_PER_SECOND": 40000,
        "LIDAR_MAX_POINTS": 1200,
        "RENDER_EVERY_N": 4,
    },
}


def configure_inference_device(cfg: Any, requested: str,
                               runtime_mode: str) -> Dict[str, str]:
    """Select the neural device without allowing an unavailable CUDA device.

    ``auto`` is deliberately conservative on Windows ``async-stable`` runs:
    CARLA's UE4/D3D11 renderer and PyTorch CUDA share the same physical GPU,
    and the project has a reproducible device-lost failure in that combination.
    The user can opt back into CUDA explicitly after validating their driver.
    """
    requested = str(requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("inference_device must be 'auto', 'cpu' or 'cuda'")

    device = requested
    reason = "explicit selection"
    if requested == "auto":
        if os.name == "nt" and runtime_mode == "async-stable":
            device = "cpu"
            reason = "safe Windows CARLA mode (avoid CUDA/D3D11 contention)"
        else:
            device = str(getattr(cfg, "YOLO_DEVICE", "cpu") or "cpu").lower()
            reason = "existing project configuration"

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[Inference] CUDA unavailable; falling back to CPU.")
                device = "cpu"
                reason = "CUDA unavailable"
        except Exception as exc:
            print(f"[Inference] CUDA probe failed ({exc}); falling back to CPU.")
            device = "cpu"
            reason = "CUDA probe failed"

    cfg.YOLO_DEVICE = device
    cfg.YOLO_HALF = bool(getattr(cfg, "YOLO_HALF", False) and device == "cuda")
    cfg.LEARNED_LANE_DEVICE = device
    cfg.LEARNED_LANE_FP16 = bool(
        getattr(cfg, "LEARNED_LANE_FP16", False) and device == "cuda")
    print(f"[Inference] device={device} requested={requested} ({reason}).")
    return {
        "requested": requested,
        "effective": device,
        "reason": reason,
    }


def configure_runtime(cfg: Any, profile_name: str,
                      runtime_mode: str) -> Tuple[bool, Dict[str, Any]]:
    """Apply one runtime profile and return ``(stable_async, effective)``."""
    profile_name = str(profile_name or "low-memory").lower()
    if profile_name not in PERFORMANCE_PROFILES:
        raise ValueError(
            f"performance profile không hợp lệ: {profile_name}; "
            f"chọn {sorted(PERFORMANCE_PROFILES)}")
    if runtime_mode not in {"async-stable", "synchronous"}:
        raise ValueError("runtime_mode must be 'async-stable' or 'synchronous'")

    cfg.apply_performance_overrides(PERFORMANCE_PROFILES[profile_name])
    stable_async = runtime_mode == "async-stable"
    if stable_async:
        # Conservative renderer profile; these are configured, not measured rates.
        # LiDAR/control target 40 Hz and RGB 10 Hz; async world time is variable.
        cfg.CAMERA_SENSOR_TICK_S = 0.1
        cfg.CAMERA_POSTPROCESS = False
        cfg.LIDAR_SENSOR_TICK_S = cfg.FIXED_DELTA
        # B18: scheduled radar skips world ticks, but this build divides its
        # sensor displacement by the current world delta, inflating ego velocity.
        cfg.RADAR_SENSOR_TICK_S = 0.0

    effective = {
        "name": profile_name,
        "camera": [cfg.CAM_WIDTH, cfg.CAM_HEIGHT, cfg.CAM_FOV],
        "yolo_imgsz": cfg.YOLO_IMGSZ,
        "detect_hz": (1.0 / cfg.CAMERA_SENSOR_TICK_S
                      if stable_async else cfg.FPS / cfg.DETECT_EVERY_N),
        "safety_control_hz": cfg.FPS,
        "lidar_points_per_second": cfg.LIDAR_POINTS_PER_SECOND,
        "lidar_max_points": cfg.LIDAR_MAX_POINTS,
        "dashboard_hz": cfg.FPS / cfg.RENDER_EVERY_N,
        "runtime_mode": runtime_mode,
        "camera_sensor_hz": _sensor_hz(cfg.CAMERA_SENSOR_TICK_S, cfg.FPS),
        "camera_postprocess": bool(cfg.CAMERA_POSTPROCESS),
        "lidar_sensor_hz": _sensor_hz(cfg.LIDAR_SENSOR_TICK_S, cfg.FPS),
        "radar_sensor_tick_s": cfg.RADAR_SENSOR_TICK_S,
        "radar_cadence": ("fixed_interval" if cfg.RADAR_SENSOR_TICK_S > 0.0
                          else "every_world_tick"),
        "radar_sensor_hz": _sensor_hz(
            cfg.RADAR_SENSOR_TICK_S, None if stable_async else cfg.FPS),
        "learned_lane_hz": (1.0 / cfg.CAMERA_SENSOR_TICK_S
                            if stable_async else cfg.FPS / cfg.LANE_EVERY_N),
    }
    return stable_async, effective


def _sensor_hz(sensor_tick_s: float, world_fps: float | None) -> float | None:
    return 1.0 / sensor_tick_s if sensor_tick_s > 0.0 else world_fps


def build_argument_parser(cfg: Any) -> argparse.ArgumentParser:
    """Build the public CLI without coupling it to the CARLA runtime loop."""
    parser = argparse.ArgumentParser(
        description="CARLA ADAS pipeline (custom ego control + AEB/avoidance; TM is NPC/explicit opt-in)")
    parser.add_argument('--vehicles', type=int, default=cfg.NUM_NPC_VEHICLES,
                        help="số xe NPC ngẫu nhiên (mặc định %(default)s; 0 = không sinh)")
    parser.add_argument('--spawn-index', type=int, default=0,
                        help="chỉ số spawn point cho ego (mặc định 0; đổi nếu điểm 0 có vật cản/xe đỗ phía trước)")
    parser.add_argument('--hazard', action='store_true',
                        help="đặt một xe ĐỨNG YÊN ngay trước ego để test phanh gấp/tránh né")
    parser.add_argument('--seed', type=int, default=None,
                        help="seed ngẫu nhiên để tái lập kịch bản giao thông")
    parser.add_argument('--weather', type=str, default='clear',
                        help="hồ sơ thời tiết: clear | light_rain | heavy_rain | fog | storm")
    parser.add_argument('--driver-takeover', action='store_true',
                        help="mô phỏng tài xế tiếp quản khi có Takeover Request")
    parser.add_argument('--turn', type=str, default=None, choices=['left', 'right'],
                        help="ý định rẽ -> chủ động chuyển làn phía đó trước giao lộ")
    parser.add_argument('--town', type=str, default='Town02',
                        help="map chạy (mặc định Town02 để giảm VRAM)")
    parser.add_argument('--record-replay', type=str, default=None,
                        help="ghi replay perception nén vào thư mục này")
    parser.add_argument('--duration', type=float, default=None,
                        help="tự dừng sau N giây mô phỏng")
    parser.add_argument('--no-display', action='store_true',
                        help="không mở cửa sổ dashboard")
    parser.add_argument('--run-report', type=str, default=None,
                        help="xuất JSON acceptance cho finite runtime/soak")
    parser.add_argument('--crash-journal', type=str, default=None,
                        help="opt-in diagnostic JSONL: NEW file in an existing directory; bounded async writer")
    parser.add_argument('--performance-profile', type=str, default='low-memory',
                        choices=sorted(PERFORMANCE_PROFILES),
                        help="low-memory (mặc định) | scale-85 | balanced | quality")
    parser.add_argument('--runtime-mode', type=str, default='async-stable',
                        choices=['async-stable', 'synchronous'],
                        help="async-stable (mặc định cho Windows GPU) | synchronous")
    parser.add_argument('--inference-device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help="auto=CPU an toàn trên Windows async-stable | cpu | cuda")
    return parser
