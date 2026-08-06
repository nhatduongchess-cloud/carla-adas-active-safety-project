"""Sensor fusion: gán khoảng cách mét (LiDAR) vào các bounding box của camera.

Thay thế heuristic cũ (640 + centroid_y*100, giả định cứng 1280x720) bằng phép
CHIẾU HÌNH HỌC đúng: đưa tâm cụm LiDAR 3D vào mặt phẳng ảnh qua ma trận nội tại
(intrinsics) của camera pinhole trong CARLA, rồi khớp với box mà nó rơi vào.

Ưu tiên:
  1) LiDAR  -> khoảng cách mét thật (nguồn chính, source='lidar').
  2) Hình học bbox (chiều cao thực đã biết) -> ước lượng đơn mắt (source='mono').
"""
# fmt: off
# isort: skip_file
from typing import List, Dict, Any, Optional
import math
import numpy as np


class SensorFusion:
    """Khớp cụm khoảng cách LiDAR vào bounding box 2D bằng phép chiếu pinhole."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fov: float = 90.0,
        lidar_y_sign: float = 1.0,
        real_heights_m: Optional[Dict[str, float]] = None,
        default_height_m: float = 1.5,
        depth_min_m: float = 1.0,
        depth_max_m: float = 60.0,
        bbox_margin_px: float = 12.0,
    ) -> None:
        f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        self.fx = self.fy = f
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.lidar_y_sign = lidar_y_sign
        self.real_heights_m = real_heights_m or {}
        self.default_height_m = default_height_m
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.bbox_margin_px = bbox_margin_px

    # --------------------------------------------------------------------- #
    def _project_cluster(self, centroid):
        """Chiếu tâm cụm LiDAR (x_fwd, y_lat, z_up) -> (u, v, forward_dist).

        Khung LiDAR CARLA: x tiến, y ngang, z lên.
        Khung quang camera: X phải, Y xuống, Z tiến.
          X_cam = sign * y_lat ; Y_cam = -z_up ; Z_cam = x_fwd
        """
        x_fwd, y_lat, z_up = centroid[0], centroid[1], centroid[2]
        if x_fwd <= 0.1:
            return None  # phía sau / quá gần, bỏ qua
        u = self.fx * (self.lidar_y_sign * y_lat) / x_fwd + self.cx
        v = self.fy * (-z_up) / x_fwd + self.cy
        return u, v, x_fwd, y_lat

    def _mono_distance(self, det: Dict[str, Any]) -> float:
        """Ước lượng khoảng cách từ chiều cao pixel của box (pinhole đơn mắt)."""
        x1, y1, x2, y2 = det["bbox"]
        h_px = max(1.0, float(y2 - y1))
        real_h = self.real_heights_m.get(det.get("class"), self.default_height_m)
        dist = self.fy * real_h / h_px
        return float(np.clip(dist, self.depth_min_m, self.depth_max_m))

    # --------------------------------------------------------------------- #
    def fuse(
        self,
        detections_2d: List[Dict[str, Any]],
        lidar_obstacles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Trả về danh sách detection đã bổ sung 'distance_m' và 'distance_source'."""
        # Chiếu trước toàn bộ cụm LiDAR sang tọa độ ảnh (u, v, forward_dist).
        projected = []
        for obs in lidar_obstacles:
            p = self._project_cluster(obs["centroid"])
            if p is not None:
                projected.append(p)

        fused: List[Dict[str, Any]] = []
        for det in detections_2d:
            x1, y1, x2, y2 = det["bbox"]
            m = self.bbox_margin_px

            # Tìm cụm LiDAR gần nhất mà điểm chiếu rơi vào (mở rộng biên) box.
            best_dist = None
            best_lat = None
            for u, v, fwd, lat in projected:
                if (x1 - m) <= u <= (x2 + m) and (y1 - m) <= v <= (y2 + m):
                    if best_dist is None or fwd < best_dist:
                        best_dist = fwd
                        best_lat = lat

            enriched = det.copy()
            if best_dist is not None:
                enriched["distance_m"] = round(float(best_dist), 2)
                enriched["lateral_m"] = round(float(best_lat), 2)
                enriched["distance_source"] = "lidar"
            else:
                dist = self._mono_distance(det)
                # Ước lượng lệch ngang từ tâm box qua phép chiếu ngược pinhole.
                u_center = (x1 + x2) / 2.0
                lateral = self.lidar_y_sign * (u_center - self.cx) / self.fx * dist
                enriched["distance_m"] = round(float(dist), 2)
                enriched["lateral_m"] = round(float(lateral), 2)
                enriched["distance_source"] = "mono"
            fused.append(enriched)

        return fused

    # Tương thích ngược: gọi kiểu cũ SensorFusion.fuse_data(...) vẫn chạy được,
    # nhưng nên chuyển sang tạo instance rồi gọi .fuse().
    @staticmethod
    def fuse_data(detections_2d, lidar_obstacles):
        return SensorFusion().fuse(detections_2d, lidar_obstacles)
