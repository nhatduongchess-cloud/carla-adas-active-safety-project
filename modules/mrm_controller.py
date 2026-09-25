r"""Module F — Máy trạng thái Fallback L3 + Minimum Risk Maneuver (MRM).

Chuỗi trạng thái lái tự động mức 3:
    L3_ACTIVE  --(DEGRADED)-->  DEGRADED  --(NORMAL)-->  L3_ACTIVE
        |  \                          |
        |   (VIOLATION)               (VIOLATION)
        v                             v
    TAKEOVER_REQUEST (bật hazard, đếm 10s chờ tài xế tiếp quản)
        --(takeover)--> DRIVER_CONTROL --(ODD NORMAL)--> L3_ACTIVE
        | không tiếp quản trong 10s HOẶC điều kiện tụt rất nhanh (critical)
        v
    MRM_EXECUTING (giảm tốc êm theo μ, bật hazard) --(v≈0)--> SAFE_STOP

Logic thuần (không import CARLA) nên test được. Orchestrator sẽ dịch
'override'/'target_decel_ms2'/'hazard' thành lệnh xe thật.
"""
# fmt: off
# isort: skip_file

G = 9.81

# Hằng trạng thái ODD (khớp với odd_monitor) — khai báo cục bộ để tránh import vòng.
ODD_NORMAL = "NORMAL"
ODD_DEGRADED = "DEGRADED"
ODD_VIOLATION = "VIOLATION"


class L3StateMachine:
    L3_ACTIVE = "L3_ACTIVE"
    DEGRADED = "DEGRADED"
    TOR = "TAKEOVER_REQUEST"
    MRM = "MRM_EXECUTING"
    SAFE_STOP = "SAFE_STOP"
    #: The driver accepted a takeover request and is driving. Automation is not
    #: available again until the ODD returns to NORMAL.
    DRIVER = "DRIVER_CONTROL"

    def __init__(self, tor_window_s=10.0, mrm_decel_frac=0.35):
        self.state = self.L3_ACTIVE
        self.tor_window = tor_window_s
        self.mrm_decel_frac = mrm_decel_frac
        self.tor_elapsed = 0.0

    def _escalate(self, critical):
        """Từ vùng lái tự động bước sang xử lý ODD violation."""
        self.tor_elapsed = 0.0
        # Điều kiện tụt rất nhanh -> vào MRM ngay, không chờ tài xế.
        self.state = self.MRM if critical else self.TOR

    def update(self, odd_state, driver_takeover, ego_speed_ms, mu, dt, critical=False):
        # --- 1) Chuyển trạng thái ---
        s = self.state
        if s == self.L3_ACTIVE:
            if odd_state == ODD_VIOLATION:
                self._escalate(critical)
            elif odd_state == ODD_DEGRADED:
                self.state = self.DEGRADED

        elif s == self.DEGRADED:
            if odd_state == ODD_NORMAL:
                self.state = self.L3_ACTIVE
            elif odd_state == ODD_VIOLATION:
                self._escalate(critical)

        elif s == self.TOR:
            self.tor_elapsed += dt
            if driver_takeover:
                # The driver has taken over: hand control to them. This used to
                # return to L3_ACTIVE, which with the ODD still violated issued
                # a fresh takeover request on the very next tick - the state
                # oscillated every frame, reported automation active inside a
                # violated ODD, and inflated the TOR count. With a one-shot
                # takeover it was worse: the re-issued request timed out and
                # started an MRM while the driver was already driving.
                self.state = self.DRIVER
                self.tor_elapsed = 0.0
            elif critical or self.tor_elapsed >= self.tor_window:
                self.state = self.MRM

        elif s == self.DRIVER:
            # Re-engagement is modelled as available as soon as the ODD is
            # NORMAL again. A real system would also wait for the driver to
            # request it; the simulation has no driver to ask.
            if odd_state == ODD_NORMAL:
                self.state = self.L3_ACTIVE

        elif s == self.MRM:
            if ego_speed_ms <= 0.3:
                self.state = self.SAFE_STOP
        # SAFE_STOP: giữ nguyên cho tới khi điều kiện phục hồi (do orchestrator quyết định)

        # --- 2) Đầu ra suy từ TRẠNG THÁI KẾT QUẢ (kể cả tick chuyển tiếp) ---
        hazard = self.state in (self.TOR, self.MRM, self.SAFE_STOP)
        target_decel = 0.0
        if self.state in (self.MRM, self.SAFE_STOP):
            target_decel = self.mrm_decel_frac * max(0.1, mu) * G

        return {
            "state": self.state,
            "hazard": hazard,
            "target_decel_ms2": round(target_decel, 2),
            "override": self.state in (self.MRM, self.SAFE_STOP),
            "tor_elapsed_s": round(self.tor_elapsed, 1),
        }
