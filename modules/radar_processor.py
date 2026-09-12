"""CARLA radar conversion and lightweight clustering.

CARLA encodes each radar detection as ``velocity, azimuth, altitude, depth``.
The public output uses the ego/LiDAR tracking convention: x forward, y right,
z up, and *positive closing speed means approaching*.  The velocity sign is an
explicit configuration value so it can be verified with the three controlled
actors required by the validation plan before radar contributes to AEB.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Any, List

import numpy as np


@dataclass
class RadarTarget:
    x_m: float
    y_m: float
    z_m: float
    range_m: float
    closing_speed_ms: float
    raw_velocity_ms: float
    azimuth_rad: float
    altitude_rad: float
    point_count: int = 1
    range_std_m: float = 0.35
    speed_std_ms: float = 0.75

    def to_measurement(self) -> dict:
        data = asdict(self)
        data.update({
            "distance_m": self.x_m,
            "lateral_m": self.y_m,
            "source": "radar",
            "sources": ("radar",),
            "distance_std_m": self.range_std_m,
        })
        return data


class RadarProcessor:
    def __init__(self, sensor_x_m=2.0, sensor_y_m=0.0, sensor_z_m=0.8,
                 reference_x_m=1.5, velocity_sign=1.0, min_range_m=0.5,
                 max_range_m=80.0, cluster_gate_m=1.2,
                 min_target_height_m=0.15):
        self.sensor_x_m = float(sensor_x_m)
        self.sensor_y_m = float(sensor_y_m)
        self.sensor_z_m = float(sensor_z_m)
        self.reference_x_m = float(reference_x_m)
        self.velocity_sign = 1.0 if float(velocity_sign) >= 0.0 else -1.0
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.cluster_gate_m = float(cluster_gate_m)
        # CARLA radar is ray-cast. With a sensor at 0.8 m and a 10-degree
        # vertical FOV, the lowest rays hit a flat road around 9--10 m ahead.
        # Those returns used to look exactly like a stationary obstacle and
        # latch AEB before the ego could move.  Keep only returns whose hit
        # point is measurably above the ego-frame road plane. Low debris is
        # still covered by the denser LiDAR safety path.
        self.min_target_height_m = float(min_target_height_m)
        self.last_stats = {
            "raw_points": 0,
            "range_valid_points": 0,
            "ground_rejected_points": 0,
            "output_targets": 0,
        }

    @staticmethod
    def _rows(measurement: Any) -> np.ndarray:
        if measurement is None:
            return np.empty((0, 4), dtype=np.float32)
        if isinstance(measurement, np.ndarray):
            rows = np.asarray(measurement, dtype=np.float32)
            return rows.reshape((-1, 4)) if rows.size else np.empty((0, 4), np.float32)
        raw = getattr(measurement, "raw_data", None)
        if raw is not None:
            rows = np.frombuffer(raw, dtype=np.float32)
            return rows.reshape((-1, 4)).copy() if rows.size else np.empty((0, 4), np.float32)
        rows = []
        for detection in measurement:
            rows.append((float(detection.velocity), float(detection.azimuth),
                         float(detection.altitude), float(detection.depth)))
        return np.asarray(rows, dtype=np.float32).reshape((-1, 4))

    def process(self, measurement: Any) -> List[dict]:
        rows = self._rows(measurement).astype(np.float64, copy=False)
        self.last_stats = {
            "raw_points": int(len(rows)),
            "range_valid_points": 0,
            "ground_rejected_points": 0,
            "output_targets": 0,
        }
        if not len(rows):
            return []
        velocity, azimuth, altitude, depth = rows.T
        keep = (np.isfinite(velocity) & np.isfinite(azimuth)
                & np.isfinite(altitude) & np.isfinite(depth)
                & (depth >= self.min_range_m) & (depth <= self.max_range_m))
        velocity, azimuth, altitude, depth = (
            values[keep] for values in (velocity, azimuth, altitude, depth))
        self.last_stats["range_valid_points"] = int(len(depth))
        if not len(depth):
            return []
        planar = depth * np.cos(altitude)
        x = planar * np.cos(azimuth) + self.sensor_x_m - self.reference_x_m
        y = planar * np.sin(azimuth) + self.sensor_y_m
        z = depth * np.sin(altitude) + self.sensor_z_m
        closing = self.velocity_sign * velocity

        # Reject road-plane intersections before clustering/tracking/safety.
        # Filtering after clustering is unsafe: a dense ground cluster can pull
        # a real target centroid downward and survive as a false obstacle.
        above_ground = z >= self.min_target_height_m
        self.last_stats["ground_rejected_points"] = int(
            len(z) - np.count_nonzero(above_ground))
        x, y, z, depth, closing, velocity, azimuth, altitude = (
            values[above_ground] for values in
            (x, y, z, depth, closing, velocity, azimuth, altitude))
        if not len(depth):
            return []

        # 2-D grid aggregation is bounded and vectorized; the previous greedy
        # O(N²) clustering produced latency spikes on dense radar frames.
        bx = np.floor(x / self.cluster_gate_m).astype(np.int64)
        by = np.floor(y / self.cluster_gate_m).astype(np.int64)
        keys = bx * 100000 + by
        _unique, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True)
        weights = 1.0 / np.maximum(0.5, depth)
        weight_sum = np.bincount(inverse, weights=weights)

        def weighted(values):
            return np.bincount(inverse, weights=weights * values) / weight_sum

        means = [weighted(values) for values in
                 (x, y, z, depth, closing, velocity, azimuth, altitude)]
        output = []
        for index, count in enumerate(counts):
            target = RadarTarget(
                x_m=means[0][index], y_m=means[1][index], z_m=means[2][index],
                range_m=means[3][index], closing_speed_ms=means[4][index],
                raw_velocity_ms=means[5][index], azimuth_rad=means[6][index],
                altitude_rad=means[7][index], point_count=int(count),
                range_std_m=max(0.15, 0.35 / math.sqrt(int(count))),
                speed_std_ms=max(0.3, 0.75 / math.sqrt(int(count))))
            output.append(target.to_measurement())
        output.sort(key=lambda target: target["range_m"])
        self.last_stats["output_targets"] = int(len(output))
        return output

    @staticmethod
    def velocity_validation_passed(path, velocity_sign):
        """Trust radar velocity for TTC only after controlled CARLA validation."""
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as stream:
                report = json.load(stream)
            return bool(report.get("pass")) and float(report.get("velocity_sign")) == float(
                1.0 if float(velocity_sign) >= 0.0 else -1.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _cluster(self, points: List[RadarTarget]) -> List[RadarTarget]:
        """Deterministic greedy clustering; avoids adding a runtime dependency."""
        clusters: List[List[RadarTarget]] = []
        for point in sorted(points, key=lambda p: p.range_m):
            chosen = None
            for cluster in clusters:
                cx = sum(p.x_m for p in cluster) / len(cluster)
                cy = sum(p.y_m for p in cluster) / len(cluster)
                if math.hypot(point.x_m - cx, point.y_m - cy) <= self.cluster_gate_m:
                    chosen = cluster
                    break
            if chosen is None:
                clusters.append([point])
            else:
                chosen.append(point)

        output = []
        for cluster in clusters:
            count = len(cluster)
            weights = [1.0 / max(0.5, p.range_m) for p in cluster]
            weight_sum = sum(weights)
            mean = lambda attr: sum(w * getattr(p, attr) for w, p in zip(weights, cluster)) / weight_sum
            output.append(RadarTarget(
                x_m=mean("x_m"), y_m=mean("y_m"), z_m=mean("z_m"),
                range_m=mean("range_m"), closing_speed_ms=mean("closing_speed_ms"),
                raw_velocity_ms=mean("raw_velocity_ms"), azimuth_rad=mean("azimuth_rad"),
                altitude_rad=mean("altitude_rad"), point_count=count,
                range_std_m=max(0.15, 0.35 / math.sqrt(count)),
                speed_std_ms=max(0.3, 0.75 / math.sqrt(count))))
        return output
