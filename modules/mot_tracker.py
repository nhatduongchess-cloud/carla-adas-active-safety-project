"""Bám đa vật thể (Multi-Object Tracking) + dự đoán quỹ đạo ngắn hạn.

Hoạt động trong MẶT PHẲNG MẶT ĐẤT của xe ego (x tiến, y ngang, mét), lấy từ
sensor_fusion (distance_m, lateral_m). Mỗi track dùng bộ lọc Kalman vận tốc
không đổi [x, y, vx, vy] -> cho ID ổn định + vận tốc thật (m/s) + dự đoán vị trí
tương lai (constant velocity).

Lưu ý khung quy chiếu: đo trong khung ego TỨC THỜI mỗi bước, nên vx/vy là vận tốc
TƯƠNG ĐỐI so với ego (đúng thứ cần cho va chạm/TTC). Chỉ dùng numpy -> test được
mà không cần CARLA.
"""
# fmt: off
# isort: skip_file
from typing import List, Dict, Any, Optional
import math
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # CI tối thiểu chỉ cài NumPy
    linear_sum_assignment = None


class _KalmanCV:
    """Kalman vận tốc không đổi 2D. Trạng thái: [x, y, vx, vy]."""

    def __init__(self, x, y, dt, meas_var=0.5, proc_var=1.0):
        self.dt = dt
        self.s = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 100.0, 100.0])  # vận tốc ban đầu bất định lớn
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * meas_var
        self._q = proc_var
        self._I = np.eye(4)

    def _F(self, dt):
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    def _Q(self, dt):
        # Nhiễu quá trình gia tốc trắng (white-acceleration model).
        q = self._q
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        return q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                             [0, dt4 / 4, 0, dt3 / 2],
                             [dt3 / 2, 0, dt2, 0],
                             [0, dt3 / 2, 0, dt2]], dtype=float)

    def predict(self, dt=None):
        dt = self.dt if dt is None else dt
        F = self._F(dt)
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + self._Q(dt)

    def innovation_distance(self, z, meas_var=None):
        z = np.asarray(z, dtype=float)
        R = self.R if meas_var is None else np.eye(2) * float(meas_var)
        innovation = z - self.H @ self.s
        S = self.H @ self.P @ self.H.T + R
        return float(innovation.T @ np.linalg.inv(S) @ innovation)

    def update(self, z, meas_var=None):
        z = np.asarray(z, dtype=float)
        R = self.R if meas_var is None else np.eye(2) * float(meas_var)
        y = z - self.H @ self.s
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (self._I - K @ self.H) @ self.P

    def update_radial_velocity(self, unit_xy, radial_velocity_ms, variance=1.0):
        """Scalar Kalman update for velocity projected onto the radar ray."""
        ux, uy = float(unit_xy[0]), float(unit_xy[1])
        H = np.array([[0.0, 0.0, ux, uy]], dtype=float)
        z = np.array([float(radial_velocity_ms)], dtype=float)
        innovation = z - H @ self.s
        S = H @ self.P @ H.T + np.array([[max(1e-4, float(variance))]])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + (K @ innovation).reshape(4)
        self.P = (self._I - K @ H) @ self.P

    def compensate_ego_motion(self, dx, dy, dyaw):
        """Đổi state từ ego frame cũ sang ego frame mới."""
        c, s = np.cos(dyaw), np.sin(dyaw)
        rotation = np.array([[c, s], [-s, c]], dtype=float)
        self.s[:2] = rotation @ (self.s[:2] - np.array([dx, dy], dtype=float))
        self.s[2:] = rotation @ self.s[2:]
        transform = np.zeros((4, 4), dtype=float)
        transform[:2, :2] = rotation
        transform[2:, 2:] = rotation
        self.P = transform @ self.P @ transform.T

    @property
    def pos(self):
        return float(self.s[0]), float(self.s[1])

    @property
    def vel(self):
        return float(self.s[2]), float(self.s[3])


class Track:
    _next_id = 1

    def __init__(self, x, y, dt, class_name, bbox, meas_var=0.5,
                 ego_compensated=False):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = _KalmanCV(x, y, dt, meas_var=meas_var)
        self.class_name = class_name
        self.bbox = bbox
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False
        self.ego_compensated = bool(ego_compensated)
        self.ego_speed_ms = 0.0
        source = "camera"
        self.sources = {source}
        self.last_measurement_frame = None

    @property
    def pos(self):
        return self.kf.pos

    @property
    def vel(self):
        return self.kf.vel

    def speed(self):
        vx, vy = self.vel
        # Khi có ego compensation, Kalman velocity là vận tốc vật trong world
        # (biểu diễn theo trục ego), không phải closing speed.
        return (vx * vx + vy * vy) ** 0.5

    def predict_path(self, horizon_s=2.0, step_s=0.5):
        """Danh sách (x, y) dự đoán theo mô hình vận tốc không đổi."""
        x, y = self.pos
        vx, vy = self.vel
        if self.ego_compensated:
            vx -= self.ego_speed_ms
        pts = []
        t = step_s
        while t <= horizon_s + 1e-9:
            pts.append((x + vx * t, y + vy * t))
            t += step_s
        return pts


class MultiObjectTracker:
    def __init__(self, dt=0.05, gate_m=4.0, min_hits=3, max_age=5):
        self.dt = dt
        self.gate = gate_m
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict[str, Any]], dt: Optional[float] = None,
               ego_motion=None, ego_speed_ms: float = 0.0,
               radar_measurements=None, frame_id=None,
               radar_velocity_validated=True) -> List[Track]:
        """detections: mỗi phần tử cần 'distance_m' (x) và 'lateral_m' (y)."""
        dt = self.dt if dt is None else dt

        motion_valid = bool(ego_motion and ego_motion.get("valid"))
        # 1) Bù chuyển động ego rồi dự đoán tất cả track.
        for tr in self.tracks:
            if motion_valid:
                tr.kf.compensate_ego_motion(
                    ego_motion.get("dx", 0.0), ego_motion.get("dy", 0.0),
                    ego_motion.get("dyaw", 0.0))
                tr.ego_compensated = True
            tr.ego_speed_ms = float(ego_speed_ms)
            tr.kf.predict(dt)
            tr.age += 1
            tr.time_since_update += 1

        # 2) Chuẩn bị phép đo (x, y).
        meas = []
        for d in detections:
            if "distance_m" in d and "lateral_m" in d:
                std = max(0.15, float(d.get("distance_std_m", 0.7)))
                meas.append((float(d["distance_m"]), float(d["lateral_m"]),
                             std * std, d))

        # 3) Mahalanobis gating + Hungarian (fallback greedy nếu CI không có SciPy).
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_meas = set(range(len(meas)))
        pairs = []
        for ti, tr in enumerate(self.tracks):
            tx, ty = tr.pos
            for mi, (mx, my, meas_var, d) in enumerate(meas):
                dist = ((tx - mx) ** 2 + (ty - my) ** 2) ** 0.5
                mahal = tr.kf.innovation_distance((mx, my), meas_var)
                class_penalty = 0.0 if d.get("class") == tr.class_name else 1.5
                if dist <= max(self.gate, 3.0 * meas_var ** 0.5) and mahal <= 16.0:
                    pairs.append((mahal + class_penalty, ti, mi))

        if linear_sum_assignment is not None and self.tracks and meas:
            cost = np.full((len(self.tracks), len(meas)), 1e6, dtype=float)
            for value, ti, mi in pairs:
                cost[ti, mi] = value
            rows, cols = linear_sum_assignment(cost)
            pairs = [(cost[ti, mi], int(ti), int(mi))
                     for ti, mi in zip(rows, cols) if cost[ti, mi] < 1e5]
        else:
            pairs.sort(key=lambda p: p[0])

        for dist, ti, mi in pairs:
            if ti in unmatched_tracks and mi in unmatched_meas:
                mx, my, meas_var, d = meas[mi]
                tr = self.tracks[ti]
                tr.kf.update((mx, my), meas_var)
                tr.hits += 1
                tr.time_since_update = 0
                tr.bbox = d.get("bbox", tr.bbox)
                tr.class_name = d.get("class", tr.class_name)
                tr.sources.add("camera")
                tr.sources.add(d.get("distance_source", d.get("source", "mono")))
                tr.last_measurement_frame = frame_id
                if tr.hits >= self.min_hits:
                    tr.confirmed = True
                unmatched_tracks.discard(ti)
                unmatched_meas.discard(mi)

        # 4) Track mới cho phép đo chưa khớp.
        for mi in unmatched_meas:
            mx, my, meas_var, d = meas[mi]
            track = Track(mx, my, dt, d.get("class", "vehicle"), d.get("bbox"),
                          meas_var=meas_var, ego_compensated=motion_valid)
            track.ego_speed_ms = float(ego_speed_ms)
            track.sources.add(d.get("distance_source", d.get("source", "mono")))
            track.last_measurement_frame = frame_id
            self.tracks.append(track)

        # 5) Radar range + radial-velocity update. Radar cannot change class and
        # does not create a track by itself; it only adds geometric redundancy to
        # camera/LiDAR tracks. Association uses Mahalanobis gating.
        self._update_radar(radar_measurements or [], ego_speed_ms, frame_id,
                           update_velocity=radar_velocity_validated)

        # 6) Xóa track quá hạn.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.confirmed]

    def _update_radar(self, measurements, ego_speed_ms, frame_id, update_velocity=True):
        candidates = []
        for ri, radar in enumerate(measurements):
            if "distance_m" not in radar or "lateral_m" not in radar:
                continue
            rx, ry = float(radar["distance_m"]), float(radar["lateral_m"])
            variance = max(0.05, float(radar.get("range_std_m", 0.35)) ** 2)
            for ti, track in enumerate(self.tracks):
                tx, ty = track.pos
                if math.hypot(tx - rx, ty - ry) > max(self.gate, 3.0 * variance ** 0.5):
                    continue
                mahal = track.kf.innovation_distance((rx, ry), variance)
                if mahal <= 16.0:
                    candidates.append((mahal, ti, ri, rx, ry, variance))
        candidates.sort(key=lambda item: item[0])
        used_tracks, used_radar = set(), set()
        for _mahal, ti, ri, rx, ry, variance in candidates:
            if ti in used_tracks or ri in used_radar:
                continue
            track, radar = self.tracks[ti], measurements[ri]
            track.kf.update((rx, ry), variance)
            distance = max(1e-6, (rx * rx + ry * ry) ** 0.5)
            ux, uy = rx / distance, ry / distance
            if update_velocity:
                closing = float(radar.get("closing_speed_ms", 0.0))
                # State velocity is world-relative after ego-motion compensation;
                # radar velocity is relative to ego. Convert before the scalar update.
                radial_state_velocity = -closing
                if track.ego_compensated:
                    radial_state_velocity += float(ego_speed_ms) * ux
                speed_var = max(0.09, float(radar.get("speed_std_ms", 0.75)) ** 2)
                track.kf.update_radial_velocity((ux, uy), radial_state_velocity, speed_var)
            track.sources.add("radar")
            track.last_measurement_frame = frame_id
            used_tracks.add(ti)
            used_radar.add(ri)
