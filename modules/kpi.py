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
    # Trần giảm tốc/giật hợp lệ (m/s², m/s³). Phanh thật hiếm khi vượt ~1g; những
    # gai vượt ngưỡng này là ARTIFACT VẬT LÝ lúc VA CHẠM (tốc độ tụt trong 1 tick),
    # không phải phanh có điều khiển -> loại khỏi KPI êm ái để báo cáo phản ánh đúng.
    DECEL_CAP_MS2 = 12.0     # ~1.2 g
    JERK_CAP_MS3 = 60.0

    def __init__(self, dt: float, scenario: str = "scenario",
                 decel_cap_ms2: float = DECEL_CAP_MS2, jerk_cap_ms3: float = JERK_CAP_MS3):
        self.dt = dt
        self.scenario = scenario
        self.decel_cap = decel_cap_ms2
        self.jerk_cap = jerk_cap_ms3
        self.t, self.speed, self.dist, self.ttc, self.state = [], [], [], [], []
        self.decel, self.jerk = [], []
        self.pipeline_latency_ms = []
        self.safety_latency_ms = []
        self.perception_latency_ms = []
        self.inference_age_ms = []
        self.gpu_memory_mb = []
        self.sensor_availability = {}
        self.collisions = 0
        self._prev_speed = None
        self._prev_decel = None

    def add(self, t, speed_ms, distance_m=None, ttc_s=None, state="NORMAL", collisions=0,
            pipeline_latency_ms=None, safety_latency_ms=None,
            perception_latency_ms=None, inference_age_ms=None,
            gpu_memory_mb=None, sensor_availability=None):
        self.t.append(t)
        self.speed.append(float(speed_ms))
        self.dist.append(distance_m)
        self.ttc.append(ttc_s)
        self.state.append(state)
        self.collisions = max(self.collisions, int(collisions))
        valid_latency = (pipeline_latency_ms is not None
                         and math.isfinite(float(pipeline_latency_ms)))
        self.pipeline_latency_ms.append(float(pipeline_latency_ms) if valid_latency else None)
        for values, value in (
                (self.safety_latency_ms, safety_latency_ms),
                (self.perception_latency_ms, perception_latency_ms),
                (self.inference_age_ms, inference_age_ms),
                (self.gpu_memory_mb, gpu_memory_mb)):
            valid = value is not None and math.isfinite(float(value))
            values.append(float(value) if valid else None)
        if sensor_availability:
            self.sensor_availability = dict(sensor_availability)

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

        # Chỉ tính giảm tốc/giật CÓ ĐIỀU KHIỂN (dưới trần vật lý); gai vượt trần là
        # do va chạm, đếm riêng như một chỉ số CHẤT LƯỢNG DỮ LIỆU.
        controlled_decel = [d for d in self.decel if 0.0 <= d <= self.decel_cap]
        controlled_jerk = [abs(j) for j in self.jerk if abs(j) <= self.jerk_cap]
        impact_spikes = sum(1 for d in self.decel if d > self.decel_cap)

        lat = sorted(x for x in self.pipeline_latency_ms if x is not None)
        safety_lat = sorted(x for x in self.safety_latency_ms if x is not None)
        perception_lat = sorted(x for x in self.perception_latency_ms if x is not None)
        inference_age = sorted(x for x in self.inference_age_ms if x is not None)
        gpu_memory = [x for x in self.gpu_memory_mb if x is not None]

        def _pct(values, q):
            if not values:
                return None
            idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
            return round(values[idx], 2)

        return {
            "scenario": self.scenario,
            "duration_s": round(n * self.dt, 2),
            "frames": n,
            "collisions": self.collisions,
            "min_distance_m": round(min(finite_dist), 2) if finite_dist else None,
            "min_ttc_s": round(min(finite_ttc), 2) if finite_ttc else None,
            "max_decel_ms2": round(max(controlled_decel), 2) if controlled_decel else 0.0,
            "max_jerk_ms3": round(max(controlled_jerk), 2) if controlled_jerk else 0.0,
            "impact_decel_spikes": impact_spikes,   # số khung gai giảm tốc do va chạm
            "pipeline_latency_p50_ms": _pct(lat, 0.50),
            "pipeline_latency_p95_ms": _pct(lat, 0.95),
            "pipeline_latency_p99_ms": _pct(lat, 0.99),
            "safety_latency_p99_ms": _pct(safety_lat, 0.99),
            "perception_latency_p95_ms": _pct(perception_lat, 0.95),
            "inference_age_p95_ms": _pct(inference_age, 0.95),
            "gpu_memory_peak_mb": round(max(gpu_memory), 2) if gpu_memory else None,
            "sensor_availability": self.sensor_availability,
            "mean_speed_kmh": round(3.6 * sum(self.speed) / max(1, n), 1),
            "max_speed_kmh": round(3.6 * max(self.speed), 1) if self.speed else 0.0,
            "pct_time_state": pct,
            "success": self.collisions == 0,
        }

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "speed_ms", "distance_m", "ttc_s", "decel_ms2",
                        "jerk_ms3", "pipeline_latency_ms", "safety_latency_ms",
                        "perception_latency_ms", "inference_age_ms", "gpu_memory_mb",
                        "state"])
            for i in range(len(self.t)):
                w.writerow([
                    round(self.t[i], 3), round(self.speed[i], 3),
                    self.dist[i], self.ttc[i],
                    round(self.decel[i], 3), round(self.jerk[i], 3),
                    round(self.pipeline_latency_ms[i], 3)
                    if self.pipeline_latency_ms[i] is not None else None,
                    round(self.safety_latency_ms[i], 3)
                    if self.safety_latency_ms[i] is not None else None,
                    round(self.perception_latency_ms[i], 3)
                    if self.perception_latency_ms[i] is not None else None,
                    round(self.inference_age_ms[i], 3)
                    if self.inference_age_ms[i] is not None else None,
                    round(self.gpu_memory_mb[i], 3)
                    if self.gpu_memory_mb[i] is not None else None,
                    self.state[i],
                ])
