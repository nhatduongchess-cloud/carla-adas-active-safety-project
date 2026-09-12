"""Bounded, JSON-safe evidence for non-normal safety decisions.

This is observability only.  It does not own thresholds, alter decisions, or
issue CARLA controls; the orchestrator records the decision after safety has
already produced it.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os
import time
from itertools import islice
from typing import Any, Dict, Iterable, Optional


def _number_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nearest_forward_lidar(obstacles: Iterable[Dict[str, Any]]) -> Optional[float]:
    distances = []
    for obstacle in obstacles or ():
        centroid = obstacle.get("centroid", ())
        if not centroid:
            continue
        x = _number_or_none(centroid[0])
        if x is not None and x > 0.0:
            distances.append(x)
    return min(distances) if distances else None


def _vector(value, axes):
    return [_number_or_none(getattr(value, axis, None)) for axis in axes]


def _context(*, radar_targets, sensor_readout, ego_transform, ego_velocity,
             path_points, lane_source, lane_half_width_m, min_closing_speed_ms,
             radar_processor, radar_velocity_validated, inference_frame_id,
             inference_timestamp, inference_age_ms):
    """Detached, capped diagnostics; no RPC, projection, safety filtering or I/O.

    Candidates are the processed CLUSTERS supplied to safety, in input order.
    raw_velocity_ms is a cluster-weighted value, not an archive of raw rays.
    Save the path for offline projection instead of duplicating safety work here.
    """
    radar = getattr(sensor_readout, "radar_measurement", None)
    fields = ("range_m", "x_m", "y_m", "z_m", "distance_m", "lateral_m",
              "raw_velocity_ms", "closing_speed_ms", "azimuth_rad", "altitude_rad",
              "point_count", "range_std_m", "speed_std_ms")
    candidates = [dict(index=index, **{key: _number_or_none(target.get(key))
                                      for key in fields})
                  for index, target in enumerate(islice(radar_targets or (), 64))]
    points = [[_number_or_none(p[0]), _number_or_none(p[1])]
              for p in islice(path_points or (), 64)]
    return {
        "record_monotonic_s": time.monotonic(),
        "loop_read_start_monotonic_s": _number_or_none(
            getattr(sensor_readout, "capture_timestamp", None)),
        "sensors": {
            "radar": {
                "frame_id": _number_or_none(getattr(radar, "frame", None)),
                "simulation_timestamp_s": _number_or_none(getattr(radar, "timestamp", None)),
                "received_monotonic_s": _number_or_none(
                    getattr(sensor_readout, "radar_received_monotonic_s", None)),
            },
            "lidar": {
                "frame_id": _number_or_none(getattr(sensor_readout, "frame_id", None)),
                "simulation_timestamp_s": _number_or_none(
                    getattr(sensor_readout, "lidar_timestamp", None)),
                "received_monotonic_s": None,
            },
            "camera": {
                "frame_id": _number_or_none(getattr(sensor_readout, "camera_frame_id", None)),
                "simulation_timestamp_s": _number_or_none(
                    getattr(sensor_readout, "camera_sensor_timestamp", None)),
                # Legacy camera wall time is read-start, NOT callback arrival.
                "legacy_read_start_monotonic_s": _number_or_none(
                    getattr(sensor_readout, "camera_wall_timestamp", None)),
                "received_monotonic_s": None,
                "fresh": getattr(sensor_readout, "camera_fresh", None),
            },
        },
        "inference": {
            "frame_id": _number_or_none(inference_frame_id),
            "timestamp_monotonic_s": _number_or_none(inference_timestamp),
            "reported_age_ms": _number_or_none(inference_age_ms),
        },
        # These are control-time cached ego reads, not capture-time ground truth.
        "ego": {
            "location_world_m": _vector(getattr(ego_transform, "location", None), "xyz"),
            "rotation_pitch_yaw_roll_deg": _vector(
                getattr(ego_transform, "rotation", None), ("pitch", "yaw", "roll")),
            "velocity_world_ms": _vector(ego_velocity, "xyz"),
        },
        "radar_candidates": candidates,
        "radar_candidates_omitted": max(0, len(radar_targets or ()) - len(candidates)),
        "radar_velocity_validated": radar_velocity_validated,
        "radar_processor": {key: _number_or_none(getattr(radar_processor, key, None))
                            for key in ("sensor_x_m", "sensor_y_m", "sensor_z_m",
                                        "reference_x_m", "velocity_sign", "min_range_m",
                                        "max_range_m", "cluster_gate_m", "min_target_height_m")},
        "path_points_m": points,
        "path_points_omitted": max(0, len(path_points or ()) - len(points)),
        "lane_source": lane_source,
        "lane_half_width_m": _number_or_none(lane_half_width_m),
        "min_closing_speed_ms": _number_or_none(min_closing_speed_ms),
    }


class DecisionTrace:
    """Keep a bounded record of safety interventions for later diagnosis."""

    SCHEMA_VERSION = 2

    def __init__(self, capacity: int = 512):
        if int(capacity) < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.events = []
        self.total_frames = 0
        self.action_counts = Counter()
        self.context_errors = 0

    def record(self, *, frame_id: Any, decision: Any, ego_speed_ms: Any,
               lidar_obstacles: Iterable[Dict[str, Any]],
               radar_targets: Iterable[Dict[str, Any]], tracks: Iterable[Any],
               fused_detections: Iterable[Dict[str, Any]],
               sensor_readout=None, ego_transform=None, ego_velocity=None,
               path_points=None, lane_source=None, lane_half_width_m=None,
               min_closing_speed_ms=None, radar_processor=None,
               radar_velocity_validated=None, inference_frame_id=None,
               inference_timestamp=None, inference_age_ms=None) -> None:
        """Record only a safety intervention, preserving its decision inputs."""
        self.total_frames += 1
        action = str(getattr(decision, "action", "DRIVE"))
        self.action_counts[action] += 1
        if action == "DRIVE":
            return

        threat = dict(getattr(decision, "threat", None) or {})
        event = {
            "frame_id": int(frame_id),
            "action": action,
            "state": str(getattr(decision, "state", "UNKNOWN")),
            "brake": _number_or_none(getattr(decision, "brake", None)),
            "collision_risk": bool(getattr(decision, "collision_risk", False)),
            "ego_speed_ms": _number_or_none(ego_speed_ms),
            "threat": {
                "source": threat.get("source"),
                "distance_m": _number_or_none(threat.get("distance_m")),
                "ttc_s": _number_or_none(threat.get("ttc_s")),
                "closing_ms": _number_or_none(threat.get("closing_ms")),
                "label": threat.get("label"),
                "lidar_distance_m": _number_or_none(threat.get("lidar_distance_m")),
                "tracker_distance_m": _number_or_none(threat.get("tracker_distance_m")),
                "radar_distance_m": _number_or_none(threat.get("radar_distance_m")),
                "radar_closing_ms": _number_or_none(threat.get("radar_closing_ms")),
            },
            "inputs": {
                "lidar_obstacle_count": len(lidar_obstacles or ()),
                "lidar_nearest_forward_x_m": _nearest_forward_lidar(lidar_obstacles),
                "radar_target_count": len(radar_targets or ()),
                "track_count": len(tracks or ()),
                "fused_detection_count": len(fused_detections or ()),
            },
        }
        try:
            event["context"] = _context(
                radar_targets=radar_targets, sensor_readout=sensor_readout,
                ego_transform=ego_transform, ego_velocity=ego_velocity,
                path_points=path_points, lane_source=lane_source,
                lane_half_width_m=lane_half_width_m, min_closing_speed_ms=min_closing_speed_ms,
                radar_processor=radar_processor, radar_velocity_validated=radar_velocity_validated,
                inference_frame_id=inference_frame_id, inference_timestamp=inference_timestamp,
                inference_age_ms=inference_age_ms)
        except Exception as error:
            # New diagnostics must not interrupt delivery of an existing brake.
            self.context_errors += 1
            event["context"] = None
            event["context_error"] = type(error).__name__
        self.events.append(event)
        if len(self.events) > self.capacity:
            del self.events[:len(self.events) - self.capacity]

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "total_frames": self.total_frames,
            "action_counts": dict(sorted(self.action_counts.items())),
            "intervention_events_retained": len(self.events),
            "capacity": self.capacity,
            "context_errors": self.context_errors,
        }

    def write(self, path: str) -> str:
        """Write the bounded trace atomically enough for post-run inspection."""
        report_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as output:
            json.dump({"summary": self.summary(), "events": self.events}, output,
                      ensure_ascii=False, indent=2)
        return report_path
