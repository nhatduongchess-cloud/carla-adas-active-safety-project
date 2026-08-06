"""Bộ ghi & tổng hợp chỉ số đánh giá (KPI) cho từng kịch bản kiểm thử.

Đây là các chỉ số mà đội ADAS/AD thực sự dùng để nghiệm thu:
  - collisions        : số va chạm (0 = đạt)
  - min_distance_m    : khoảng cách nhỏ nhất tới vật trong làn
  - min_ttc_s         : thời gian-va-chạm nhỏ nhất
  - max_decel_ms2     : phanh mạnh nhất (an toàn vs. thoải mái)
  - max_jerk_ms3      : giật lớn nhất (chỉ số ÊM ÁI / comfort)
  - pct_time_state    : % thời gian ở mỗi trạng thái FSM
  - success           : không va chạm

Thuần stdlib -> test được không cần CARLA.
"""
# fmt: off
# isort: skip_file
import csv
import math


class KpiRecorder:
    def __init__(self, dt: float, scenario: str = "scenario"):
        self.dt = dt
        self.scenario = scenario
        self.t, self.speed, self.dist, self.ttc, self.state = [], [], [], [], []
        self.decel, self.jerk = [], []
        self.collisions = 0
        self._prev_speed = None
        self._prev_decel = None

    def add(self, t, speed_ms, distance_m=None, ttc_s=None, state="NORMAL", collisions=0):
        self.t.append(t)
        self.speed.append(float(speed_ms))
        self.dist.append(distance_m)
        self.ttc.append(ttc_s)
        self.state.append(state)
        self.collisions = max(self.collisions, int(collisions))

        dec = (self._prev_speed - speed_ms) / self.dt if self._prev_speed is not None else 0.0
        jrk = (dec - self._prev_decel) / self.dt if self._prev_decel is not None else 0.0
        self.decel.append(dec)
        self.jerk.append(jrk)
        self._prev_speed = speed_ms
        self._prev_decel = dec

    def summary(self) -> dict:
        n = len(self.speed)
        finite_dist = [d for d in self.dist if d is not None and math.isfinite(d)]
        finite_ttc = [c for c in self.ttc if c is not None and math.isfinite(c) and c > 0]

        state_counts = {}
        for s in self.state:
            state_counts[s] = state_counts.get(s, 0) + 1
        pct = {k: round(100.0 * v / max(1, n), 1) for k, v in state_counts.items()}

        return {
            "scenario": self.scenario,
            "duration_s": round(n * self.dt, 2),
            "frames": n,
            "collisions": self.collisions,
            "min_distance_m": round(min(finite_dist), 2) if finite_dist else None,
            "min_ttc_s": round(min(finite_ttc), 2) if finite_ttc else None,
            "max_decel_ms2": round(max(self.decel), 2) if self.decel else 0.0,
            "max_jerk_ms3": round(max(abs(j) for j in self.jerk), 2) if self.jerk else 0.0,
            "mean_speed_kmh": round(3.6 * sum(self.speed) / max(1, n), 1),
            "max_speed_kmh": round(3.6 * max(self.speed), 1) if self.speed else 0.0,
            "pct_time_state": pct,
            "success": self.collisions == 0,
        }

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "speed_ms", "distance_m", "ttc_s", "decel_ms2", "jerk_ms3", "state"])
            for i in range(len(self.t)):
                w.writerow([
                    round(self.t[i], 3), round(self.speed[i], 3),
                    self.dist[i], self.ttc[i],
                    round(self.decel[i], 3), round(self.jerk[i], 3), self.state[i],
                ])
