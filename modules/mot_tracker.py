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
import numpy as np


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

    def update(self, z):
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (self._I - K @ self.H) @ self.P

    @property
    def pos(self):
        return float(self.s[0]), float(self.s[1])

    @property
    def vel(self):
        return float(self.s[2]), float(self.s[3])


class Track:
    _next_id = 1

    def __init__(self, x, y, dt, class_name, bbox):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = _KalmanCV(x, y, dt)
        self.class_name = class_name
        self.bbox = bbox
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False

    @property
    def pos(self):
        return self.kf.pos

    @property
    def vel(self):
        return self.kf.vel

    def speed(self):
        vx, vy = self.vel
        return (vx * vx + vy * vy) ** 0.5

    def predict_path(self, horizon_s=2.0, step_s=0.5):
        """Danh sách (x, y) dự đoán theo mô hình vận tốc không đổi."""
        x, y = self.pos
        vx, vy = self.vel
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

    def update(self, detections: List[Dict[str, Any]], dt: Optional[float] = None) -> List[Track]:
        """detections: mỗi phần tử cần 'distance_m' (x) và 'lateral_m' (y)."""
        dt = self.dt if dt is None else dt

        # 1) Dự đoán tất cả track.
        for tr in self.tracks:
            tr.kf.predict(dt)
            tr.age += 1
            tr.time_since_update += 1

        # 2) Chuẩn bị phép đo (x, y).
        meas = []
        for d in detections:
            if "distance_m" in d and "lateral_m" in d:
                meas.append((float(d["distance_m"]), float(d["lateral_m"]), d))

        # 3) Kết hợp tham lam theo khoảng cách gần nhất (greedy NN + gating).
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_meas = set(range(len(meas)))
        pairs = []
        for ti, tr in enumerate(self.tracks):
            tx, ty = tr.pos
            for mi, (mx, my, _d) in enumerate(meas):
                dist = ((tx - mx) ** 2 + (ty - my) ** 2) ** 0.5
                if dist <= self.gate:
                    pairs.append((dist, ti, mi))
        pairs.sort(key=lambda p: p[0])
        for dist, ti, mi in pairs:
            if ti in unmatched_tracks and mi in unmatched_meas:
                mx, my, d = meas[mi]
                tr = self.tracks[ti]
                tr.kf.update((mx, my))
                tr.hits += 1
                tr.time_since_update = 0
                tr.bbox = d.get("bbox", tr.bbox)
                tr.class_name = d.get("class", tr.class_name)
                if tr.hits >= self.min_hits:
                    tr.confirmed = True
                unmatched_tracks.discard(ti)
                unmatched_meas.discard(mi)

        # 4) Track mới cho phép đo chưa khớp.
        for mi in unmatched_meas:
            mx, my, d = meas[mi]
            self.tracks.append(
                Track(mx, my, dt, d.get("class", "vehicle"), d.get("bbox")))

        # 5) Xóa track quá hạn.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.confirmed]
