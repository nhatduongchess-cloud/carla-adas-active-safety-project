"""Finite-run acceptance report builder for the CARLA ADAS runtime."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RuntimeReportData:
    world: Any
    weather: str
    seed: Any
    vehicles_requested: int
    vehicles_spawned: int
    duration_s: Optional[float]
    frame_count: int
    fps_ema: float
    no_display: bool
    effective_performance: Dict[str, Any]
    stop_reason: str
    run_error: Optional[str]
    loop_wall_started: Optional[float]
    max_speed_kmh: float
    distance_travelled_m: float
    start_location: Any
    end_location: Any
    collision_sensor: Any
    sensor_sync_stats: Any
    radar_sync_stats: Any
    health_monitor: Any
    pipeline_metrics: Any
    perception_runtime: Any
    lane_source_counts: Dict[str, int]
    telemetry: Any
    aeb_triggered: bool = False
    decision_trace_summary: Optional[Dict[str, Any]] = None
    decision_trace_path: Optional[str] = None
    cleanup_summary: Optional[Dict[str, Any]] = None
    town_name: Optional[str] = None


def write_runtime_report(path: str, cfg: Any,
                         data: RuntimeReportData) -> bool:
    """Write one report and return whether every acceptance gate passed."""
    pipeline_summary = data.pipeline_metrics.summary()
    scheduler_summary = data.perception_runtime.scheduler_stats()
    lane_scheduler = scheduler_summary.get("lane")
    pipeline_summary["inference_scheduler"] = scheduler_summary
    pipeline_summary["measurement_semantics"] = {
        "stages": "Control-loop stages; neural stages here cover unique consumed products only (legacy acceptance).",
        "inference_scheduler.telemetry": "All finished requests in the explicit runtime window, including stale/errors/unpolled; shutdown completions counted separately.",
        "inference_scheduler.lane": "Independent lane frames/timing with the same age guard; idle-only admission skips busy requests without a lane backlog. Null means no lane backend. Object neural_inference latency no longer includes waiting for lane.",
        "first_poll_outcomes": "One outcome per completed request's first poll, not per loop. A future-frame result may be consumed later; eligible is not a consumption acknowledgement.",
        "unique_frames": "Strictly increasing frame high-water marks per scheduler/episode; duplicate/out-of-order completions flagged, never counted as new FPS.",
        "latency": "Monotonic queue/submission timing; worker service excludes queue. Submission excludes upstream camera transport/copy; not full sensor-to-control latency.",
        "stage_samples": "samples/percentiles/window_deadline_misses describe retained window; total_samples/deadline_misses are lifetime valid counts. Invalid samples excluded and counted.",
        "inference_age_p95_ms": "Legacy age of last consumed object product sampled per loop, including stale periods after semantics are cleared.",
    }
    stages = pipeline_summary.get("stages", {})
    safety_p99 = stages.get("safety_control", {}).get("p99_ms")
    perception_p95 = stages.get("neural_inference", {}).get("p95_ms")
    health_summary = data.health_monitor.summary()
    radar_enabled = health_summary.get("enabled", {}).get("radar", True)
    radar_availability = health_summary.get("availability", {}).get("radar")
    collisions = (data.collision_sensor.count
                  if data.collision_sensor is not None else 0)
    frame_errors = (data.sensor_sync_stats.frame_errors
                    if data.sensor_sync_stats is not None else 0)
    target_frames = (
        max(1, int(round(float(data.duration_s) * cfg.FPS)))
        if data.duration_s is not None else None)

    criteria = {
        "completed_duration": (
            data.frame_count > 0
            and (target_frames is None
                 or (data.stop_reason == "duration_reached"
                     and data.frame_count >= target_frames))),
        "no_runtime_error": data.run_error is None,
        "cleanup_verified": (data.cleanup_summary or {}).get("verified") is True,
        "collisions_zero": data.collision_sensor is not None and collisions == 0,
        "sensor_frame_errors_zero": (
            data.sensor_sync_stats is not None and frame_errors == 0),
        # None when radar is disabled: a sensor that is not fitted cannot pass
        # an availability criterion, and it previously did, at 100%.
        "radar_availability_ge_99_5pct": (
            None if not radar_enabled
            else radar_availability is not None and float(radar_availability) >= 0.995),
        "real_time_loop_ge_38hz": float(data.fps_ema) >= cfg.FPS * 0.95,
        "safety_p99_le_25ms": _within_limit(safety_p99, cfg.SAFETY_DEADLINE_MS),
        "perception_p95_le_50ms": _within_limit(
            perception_p95, cfg.PERCEPTION_DEADLINE_MS),
        "inference_age_p95_le_150ms": _within_limit(
            pipeline_summary.get("inference_age_p95_ms"),
            cfg.MAX_ASYNC_RESULT_AGE_MS),
        "neural_inference_errors_zero": (
            int(scheduler_summary.get("failed", 0)) == 0
            and int((lane_scheduler or {}).get("failed", 0)) == 0),
        "neural_workers_stopped": (
            not scheduler_summary.get("thread_alive", False)
            and not (lane_scheduler or {}).get("thread_alive", False)),
        "torch_gpu_peak_le_limit": _within_limit(pipeline_summary.get("gpu_peak_mb"), cfg.VRAM_LIMIT_MB),
        "device_gpu_peak_le_limit": _within_limit(pipeline_summary.get("gpu_device_peak_used_mb"), cfg.VRAM_LIMIT_MB),
    }
    # Only a criterion that is explicitly not applicable may be skipped. Every
    # other None stays a failure - a missing measurement is not a pass.
    not_applicable = sorted(
        name for name in ("radar_availability_ge_99_5pct",)
        if criteria.get(name) is None and not radar_enabled)
    run_pass = all(value for name, value in criteria.items() if name not in not_applicable)
    report = {
        "title": "CARLA finite runtime / soak report",
        "status": "PASS" if run_pass else "FAIL",
        "stop_reason": data.stop_reason,
        "error": data.run_error,
        "cleanup": data.cleanup_summary,
        "run": {
            "town": data.town_name or _town_name(data.world),
            "weather": data.weather,
            "seed": data.seed,
            "vehicles_requested": data.vehicles_requested,
            "vehicles_spawned": data.vehicles_spawned,
            "target_duration_s": data.duration_s,
            "simulated_duration_s": round(data.frame_count / cfg.FPS, 3),
            "wall_duration_s": (
                round(time.perf_counter() - data.loop_wall_started, 3)
                if data.loop_wall_started is not None else None),
            "frames": data.frame_count,
            "fps_ema": round(data.fps_ema, 3),
            "headless": bool(data.no_display),
            "performance_profile": data.effective_performance,
        },
        "driving": {
            "collisions": collisions,
            "collision_history": (
                data.collision_sensor.history
                if data.collision_sensor is not None else []),
            "max_speed_kmh": round(data.max_speed_kmh, 3),
            "distance_travelled_m": round(data.distance_travelled_m, 3),
            "start_location": _location(data.start_location),
            "end_location": _location(data.end_location),
        },
        "safety": {
            "aeb_triggered": bool(data.aeb_triggered),
            "decision_trace": data.decision_trace_summary,
            "decision_trace_path": data.decision_trace_path,
        },
        "sensors": {
            "camera_lidar_frame_errors": frame_errors,
            "radar_frame_errors": (
                data.radar_sync_stats.frame_errors
                if data.radar_sync_stats is not None else 0),
            "sync": (data.sensor_sync_stats.summary()
                     if hasattr(data.sensor_sync_stats, "summary") else None),
            "health": health_summary,
        },
        "pipeline": pipeline_summary,
        "learned_lane": {
            "enabled": bool(cfg.LEARNED_LANE_ENABLED),
            "ready": bool(data.perception_runtime.lane_adapter.ready),
            "backend_error": data.perception_runtime.lane_backend_error,
            "model_sha256": cfg.LEARNED_LANE_MODEL_SHA256,
            "input_size": [cfg.LEARNED_LANE_INPUT_WIDTH,
                           cfg.LEARNED_LANE_INPUT_HEIGHT],
            "cadence_hz": cfg.FPS / cfg.LANE_EVERY_N,
            "source_frames": data.lane_source_counts,
        },
        "criteria": criteria,
        "criteria_not_applicable": not_applicable,
        "telemetry_csv": (
            data.telemetry.filename if data.telemetry is not None else None),
    }
    report_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(_json_safe(report), report_file, ensure_ascii=False, indent=2)
    print(f"[System] Runtime report: {report_path} -> {report['status']}")
    print(f"[Safety] AEB triggered={str(bool(data.aeb_triggered)).lower()}")
    return run_pass


def _location(location: Any):
    if location is None:
        return None
    return [location.x, location.y, location.z]


def _town_name(world: Any):
    try:
        return world.get_map().name
    except Exception:
        return None


def _within_limit(value, limit):
    try:
        measured = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return not isinstance(value, bool) and math.isfinite(measured) and 0 <= measured <= limit


def _json_safe(obj):
    """Replace non-finite floats with None so the report is always valid JSON."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    return obj
