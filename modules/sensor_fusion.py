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
        camera_pose=None,
        lidar_pose=None,
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
        self.camera_pose = tuple(camera_pose or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.lidar_pose = tuple(lidar_pose or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self._r_vehicle_camera = self._rotation_matrix(*self.camera_pose[3:])
        self._r_vehicle_lidar = self._rotation_matrix(*self.lidar_pose[3:])
        self._t_camera = np.asarray(self.camera_pose[:3], dtype=float)
        self._t_lidar = np.asarray(self.lidar_pose[:3], dtype=float)

    @staticmethod
    def _rotation_matrix(roll_deg, pitch_deg, yaw_deg):
        roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        return rz @ ry @ rx

    # --------------------------------------------------------------------- #
    def _project_cluster(self, centroid):
        """Chiếu tâm cụm LiDAR (x_fwd, y_lat, z_up) -> (u, v, forward_dist).

        Khung LiDAR CARLA: x tiến, y ngang, z lên.
        Khung quang camera: X phải, Y xuống, Z tiến.
          X_cam = sign * y_lat ; Y_cam = -z_up ; Z_cam = x_fwd
        """
        lidar_point = np.asarray(centroid[:3], dtype=float)
        vehicle_point = self._r_vehicle_lidar @ lidar_point + self._t_lidar
        camera_point = self._r_vehicle_camera.T @ (vehicle_point - self._t_camera)
        x_cam, y_cam, z_cam = camera_point
        if x_cam <= 0.1:
            return None  # phía sau / quá gần, bỏ qua
        u = self.fx * (self.lidar_y_sign * y_cam) / x_cam + self.cx
        v = self.fy * (-z_cam) / x_cam + self.cy
        # Range/lateral vẫn ở LiDAR frame để khớp tracker + swept path hiện tại.
        return u, v, float(lidar_point[0]), float(lidar_point[1])

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
        for obs_index, obs in enumerate(lidar_obstacles):
            p = self._project_cluster(obs["centroid"])
            if p is not None:
                projected.append((obs_index, *p))

        # Ghép one-to-one toàn cục theo khoảng cách chuẩn hóa tới tâm bbox. Một
        # cluster không còn bị gán đồng thời cho nhiều detection chồng lấn.
        candidates = []
        for det_index, det in enumerate(detections_2d):
            x1, y1, x2, y2 = det["bbox"]
            m = self.bbox_margin_px
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for obs_index, u, v, fwd, lat in projected:
                if (x1 - m) <= u <= (x2 + m) and (y1 - m) <= v <= (y2 + m):
                    score = ((u - uc) / bw) ** 2 + ((v - vc) / bh) ** 2
                    candidates.append((score, det_index, obs_index, fwd, lat))
        candidates.sort(key=lambda item: item[0])
        assignments = {}
        used_obstacles = set()
        for _score, det_index, obs_index, fwd, lat in candidates:
            if det_index not in assignments and obs_index not in used_obstacles:
                assignments[det_index] = (fwd, lat)
                used_obstacles.add(obs_index)

        fused: List[Dict[str, Any]] = []
        for det_index, det in enumerate(detections_2d):
            x1, y1, x2, y2 = det["bbox"]

            enriched = det.copy()
            if det_index in assignments:
                best_dist, best_lat = assignments[det_index]
                enriched["distance_m"] = round(float(best_dist), 2)
                enriched["lateral_m"] = round(float(best_lat), 2)
                enriched["distance_source"] = "lidar"
                enriched["distance_std_m"] = round(max(0.15, 0.015 * best_dist), 2)
            else:
                dist = self._mono_distance(det)
                # Ước lượng lệch ngang từ tâm box qua phép chiếu ngược pinhole.
                u_center = (x1 + x2) / 2.0
                lateral = self.lidar_y_sign * (u_center - self.cx) / self.fx * dist
                enriched["distance_m"] = round(float(dist), 2)
                enriched["lateral_m"] = round(float(lateral), 2)
                enriched["distance_source"] = "mono"
                enriched["distance_std_m"] = round(max(1.0, 0.20 * dist), 2)
            fused.append(enriched)

        return fused

    # Tương thích ngược: gọi kiểu cũ SensorFusion.fuse_data(...) vẫn chạy được,
    # nhưng nên chuyển sang tạo instance rồi gọi .fuse().
    @staticmethod
    def fuse_data(detections_2d, lidar_obstacles):
        return SensorFusion().fuse(detections_2d, lidar_obstacles)
