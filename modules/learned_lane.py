"""Safe learned-lane adapter and camera-to-ground geometry.

No model is downloaded and no arbitrary Python repository is imported here.
An audited backend must be injected explicitly.  Until then the adapter returns
an invalid estimate, which makes :mod:`lane_source` select CARLA map geometry.
"""

import math
import os
from bisect import bisect_left
from typing import Callable, Optional

try:
    from perception_contracts import LaneEstimate
except ImportError:
    from modules.perception_contracts import LaneEstimate


class GroundProjector:
    """Flat-ground IPM for a forward-facing CARLA camera (zero roll/pitch/yaw)."""

    def __init__(self, fx, fy, cx, cy, camera_height_m=2.4,
                 camera_x_m=1.5, reference_x_m=1.5, max_range_m=80.0):
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.height = float(camera_height_m)
        self.x_offset = float(camera_x_m) - float(reference_x_m)
        self.max_range = float(max_range_m)

    def pixel_to_ground(self, u, v):
        dz = -(float(v) - self.cy) / self.fy
        if dz >= -1e-6:
            return None
        scale = self.height / -dz
        x = scale + self.x_offset
        y = scale * (float(u) - self.cx) / self.fx
        if not 0.0 < x <= self.max_range or not math.isfinite(y):
            return None
        return float(x), float(y)

    def polyline_to_ground(self, pixels):
        points = [self.pixel_to_ground(u, v) for u, v in pixels]
        return [point for point in points if point is not None]


class LearnedLaneAdapter:
    def __init__(self, projector: GroundProjector, backend: Optional[Callable] = None,
                 model_path="", input_size=(800, 320)):
        self.projector = projector
        self.backend = backend
        self.model_path = os.path.abspath(model_path) if model_path else ""
        self.input_size = tuple(input_size)
        self.provenance_error = None
        if self.model_path and not os.path.isfile(self.model_path):
            self.provenance_error = f"lane weight not found: {self.model_path}"

    @property
    def ready(self):
        return self.backend is not None and self.provenance_error is None

    def infer(self, frame, frame_id, timestamp) -> LaneEstimate:
        if not self.ready:
            return LaneEstimate(
                int(frame_id), float(timestamp), "learned", valid=False,
                reason=self.provenance_error or "audited learned-lane backend not configured")
        try:
            output = self.backend(frame, self.input_size)
            confidence = float(output.get("confidence", 0.0))
            metric_lanes = [self.projector.polyline_to_ground(line)
                            for line in output.get("lane_pixels", [])]
            metric_lanes = [self._normalize_line(line) for line in metric_lanes]
            metric_lanes = [line for line in metric_lanes if len(line) >= 2]
            centerline, lane_width, offset, curvature = self._geometry(metric_lanes)
            return LaneEstimate(
                int(frame_id), float(timestamp), "learned", confidence,
                metric_lanes, centerline, offset, curvature, lane_width,
                valid=bool(centerline), reason=None if centerline else "insufficient lane geometry")
        except Exception as exc:
            return LaneEstimate(
                int(frame_id), float(timestamp), "learned", valid=False,
                reason=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _normalize_line(line):
        points = [(float(x), float(y)) for x, y in line
                  if math.isfinite(x) and math.isfinite(y) and x > 0.0]
        points.sort(key=lambda point: point[0])
        normalized = []
        for point in points:
            if normalized and abs(point[0] - normalized[-1][0]) < 1e-4:
                normalized[-1] = point
            else:
                normalized.append(point)
        return normalized

    @staticmethod
    def _interpolate_y(line, x):
        xs = [point[0] for point in line]
        index = bisect_left(xs, x)
        if index <= 0:
            return line[0][1]
        if index >= len(line):
            return line[-1][1]
        x0, y0 = line[index - 1]
        x1, y1 = line[index]
        if abs(x1 - x0) < 1e-6:
            return y1
        ratio = (x - x0) / (x1 - x0)
        return y0 + ratio * (y1 - y0)

    @staticmethod
    def _geometry(lanes):
        if len(lanes) < 2:
            return [], None, None, None
        # Pick the nearest left/right boundaries around y=0.
        candidates = sorted(
            lanes, key=lambda line: sum(point[1] for point in line[:3]) / min(3, len(line)))
        left = [line for line in candidates
                if sum(point[1] for point in line[:3]) / min(3, len(line)) < 0.0]
        right = [line for line in candidates
                 if sum(point[1] for point in line[:3]) / min(3, len(line)) >= 0.0]
        if not left or not right:
            return [], None, None, None
        left_line, right_line = left[-1], right[0]
        start_x = max(left_line[0][0], right_line[0][0])
        end_x = min(left_line[-1][0], right_line[-1][0])
        if end_x - start_x < 0.5:
            return [], None, None, None
        count = min(32, max(3, int((end_x - start_x) / 1.5) + 1))
        sample_x = [start_x + (end_x - start_x) * i / (count - 1)
                    for i in range(count)]
        left_y = [LearnedLaneAdapter._interpolate_y(left_line, x) for x in sample_x]
        right_y = [LearnedLaneAdapter._interpolate_y(right_line, x) for x in sample_x]
        center = [(x, (ly + ry) * 0.5)
                  for x, ly, ry in zip(sample_x, left_y, right_y)]
        widths = [abs(ry - ly) for ly, ry in zip(left_y, right_y)]
        width = sum(widths) / len(widths)
        offset = -center[0][1]
        curvature = None
        if len(center) >= 3:
            a, b, c = center[0], center[len(center) // 2], center[-1]
            area2 = abs((b[0] - a[0]) * (c[1] - a[1])
                        - (b[1] - a[1]) * (c[0] - a[0]))
            ab, bc, ca = (math.dist(a, b), math.dist(b, c), math.dist(c, a))
            if ab * bc * ca > 1e-6:
                curvature = 2.0 * area2 / (ab * bc * ca)
        return center, width, offset, curvature
