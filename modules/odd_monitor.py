"""Module E — Giám sát ranh giới ODD (Operational Design Domain).

Phân loại điều kiện vận hành theo tầm nhìn / ma sát / SNR thành 3 mức, đúng theo
nguyên tắc L3 (DRIVE PILOT chỉ được phép tự lái trong ODD đã định nghĩa):

    NORMAL     : visibility > 50m  và  μ >= 0.6
    DEGRADED   : 20m <= visibility <= 50m  hoặc  0.3 <= μ < 0.6  -> giảm tốc, tăng gap
    VIOLATION  : visibility < 20m  hoặc  μ < 0.3  (hoặc SNR quá thấp) -> kích Fallback L3
"""
# fmt: off
# isort: skip_file


class ODDMonitor:
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    VIOLATION = "VIOLATION"

    def __init__(self, vis_normal=50.0, vis_violation=20.0,
                 mu_normal=0.6, mu_violation=0.3, snr_violation=0.25,
                 degraded_speed_cap_kmh=40.0):
        self.vis_normal = vis_normal
        self.vis_violation = vis_violation
        self.mu_normal = mu_normal
        self.mu_violation = mu_violation
        self.snr_violation = snr_violation
        self.degraded_speed_cap = degraded_speed_cap_kmh

    def classify(self, conditions: dict, sensor_health=None) -> dict:
        v = conditions["visibility_m"]
        mu = conditions["mu"]
        snr = conditions.get("snr", 1.0)

        range_redundancy_lost = bool(
            sensor_health and sensor_health.get("range_redundancy_lost"))

        if (range_redundancy_lost or v < self.vis_violation
                or mu < self.mu_violation or snr < self.snr_violation):
            state = self.VIOLATION
            speed_cap = 0.0
            gap_multiplier = 2.0
        elif v <= self.vis_normal or mu < self.mu_normal:
            state = self.DEGRADED
            speed_cap = self.degraded_speed_cap
            gap_multiplier = 1.5
        else:
            state = self.NORMAL
            speed_cap = None  # không giới hạn thêm
            gap_multiplier = 1.0

        return {
            "state": state,
            "speed_cap_kmh": speed_cap,
            "gap_multiplier": gap_multiplier,
            # "critical" = điều kiện tụt rất nhanh -> bỏ qua chờ TOR, vào MRM ngay.
            "critical": range_redundancy_lost or v < 10.0 or mu < 0.2,
            "reason": "lidar_and_radar_unavailable" if range_redundancy_lost else None,
            "conditions": conditions,
        }
