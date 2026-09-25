"""Module E — Giám sát ranh giới ODD (Operational Design Domain).

Phân loại điều kiện vận hành theo tầm nhìn / ma sát / SNR thành 3 mức, đúng theo
nguyên tắc L3 (DRIVE PILOT chỉ được phép tự lái trong ODD đã định nghĩa):

    NORMAL     : visibility > 50m  và  μ >= 0.6
    DEGRADED   : 20m <= visibility <= 50m  hoặc  0.3 <= μ < 0.6  -> giảm tốc, tăng gap
    VIOLATION  : visibility < 20m  hoặc  μ < 0.3  (hoặc SNR quá thấp) -> kích Fallback L3
    VIOLATION  : any of the three inputs not a finite number - an ODD that cannot be
                 measured is not treated as satisfied (reason "odd_unmeasurable:...")

Note on friction: weather_model.estimate_conditions produces μ in [0.4, 0.9], so the
μ < 0.3 VIOLATION and μ < 0.2 critical branches are unreachable from weather in this
simulation - CARLA models no ice or standing water. They stay because the thresholds
are right for a real vehicle, and are exercised by unit tests with injected μ only.
"""
# fmt: off
# isort: skip_file


import math
def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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

        # An ODD that cannot be measured is not an ODD that is satisfied. Every
        # comparison against NaN is False, so without this check an all-NaN
        # estimate fell straight through to NORMAL - the monitor reported the
        # vehicle inside its operating domain precisely when it had no idea.
        unmeasurable = [name for name, value in
                        (("visibility_m", v), ("mu", mu), ("snr", snr))
                        if not _finite(value)]

        if (range_redundancy_lost or unmeasurable or v < self.vis_violation
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
            # Unmeasurable is not critical on its own: it asks the driver to
            # take over (TOR) rather than skipping straight to an MRM.
            "critical": range_redundancy_lost or v < 10.0 or mu < 0.2,
            "reason": ("lidar_and_radar_unavailable" if range_redundancy_lost
                       else "odd_unmeasurable:" + ",".join(unmeasurable) if unmeasurable
                       else None),
            "conditions": conditions,
        }
