r"""Module F — Máy trạng thái Fallback L3 + Minimum Risk Maneuver (MRM).

Chuỗi trạng thái lái tự động mức 3:
    L3_ACTIVE  --(DEGRADED)-->  DEGRADED  --(NORMAL)-->  L3_ACTIVE
        |  \                          |
        |   (VIOLATION)               (VIOLATION)
        v                             v
    TAKEOVER_REQUEST (bật hazard, đếm 10s chờ tài xế tiếp quản)
        --(takeover ACK)--> DRIVER_CONTROL --(engage request AND ODD NORMAL)--> L3_ACTIVE
        | không tiếp quản trong 10s HOẶC điều kiện dưới ngưỡng critical tuyệt đối
        v
    MRM_EXECUTING (giảm tốc êm theo μ, bật hazard) --(v≈0)--> SAFE_STOP

Logic thuần (không import CARLA) nên test được. Orchestrator sẽ dịch
'override'/'target_decel_ms2'/'hazard' thành lệnh xe thật.

Scope, stated so it is not over-read: this is a simulation prototype of ODD /
TOR / MRM state transitions. The 10 s TOR window is a project parameter, not a
value required by a standard. The takeover input is a SIMULATED acknowledgement
(`--driver-takeover`); there is no real driver or manual input device, so every
output carries ``human_takeover_verified: False``. After the ACK the configured
stand-in controller keeps driving and the AEB stays active (it is independent of
who owns the vehicle). Clocks: the TOR window advances by the ``dt`` passed in,
which callers supply as simulation time.

Validation policy: ``dt == 0`` (a repeated frame) is no progress; a ``dt`` that
is negative or not finite is treated as the TOR window having expired (fail toward the MRM, never toward waiting
forever); a non-finite speed never counts as stopped, so the MRM is not left
for SAFE_STOP on bad data.
"""
import math
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
    #: The driver accepted a takeover request and is driving. Automation comes
    #: back only on an explicit engage request with the ODD NORMAL.
    DRIVER = "DRIVER_CONTROL"

    def __init__(self, tor_window_s=10.0, mrm_decel_frac=0.35):
        self.state = self.L3_ACTIVE
        self.tor_window = tor_window_s
        self.mrm_decel_frac = mrm_decel_frac
        self.tor_elapsed = 0.0
        # Events are counted on transitions, never per frame.
        self.tor_events = 0
        self.takeover_events = 0
        self.mrm_events = 0
        self.engage_events = 0

    def _escalate(self, critical):
        """Từ vùng lái tự động bước sang xử lý ODD violation."""
        self.tor_elapsed = 0.0
        # Dưới ngưỡng critical tuyệt đối -> vào MRM ngay, không chờ tài xế.
        self.state = self.MRM if critical else self.TOR

    def update(self, odd_state, driver_takeover, ego_speed_ms, mu, dt, critical=False,
               engage_request=False):
        previous = self.state
        dt_valid = (isinstance(dt, (int, float)) and not isinstance(dt, bool)
                    and math.isfinite(dt) and dt >= 0.0)
        speed_valid = (isinstance(ego_speed_ms, (int, float))
                       and math.isfinite(ego_speed_ms) and ego_speed_ms >= 0.0)
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
            if dt_valid:
                self.tor_elapsed += dt
            else:
                self.tor_elapsed = self.tor_window   # fail toward the MRM
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
            # Re-engagement needs an explicit request AND a NORMAL ODD. It used
            # to happen on NORMAL alone, i.e. automation re-engaged itself.
            # Repeated takeover flags here are ignored: the ACK was consumed.
            if engage_request and odd_state == ODD_NORMAL:
                self.state = self.L3_ACTIVE

        elif s == self.MRM:
            if speed_valid and ego_speed_ms <= 0.3:
                self.state = self.SAFE_STOP
        # SAFE_STOP: giữ nguyên cho tới khi điều kiện phục hồi (do orchestrator quyết định)

        # --- 2) Đầu ra suy từ TRẠNG THÁI KẾT QUẢ (kể cả tick chuyển tiếp) ---
        if self.state != previous:
            if self.state == self.TOR:
                self.tor_events += 1
            elif self.state == self.DRIVER:
                self.takeover_events += 1
            elif self.state == self.MRM:
                self.mrm_events += 1
            elif self.state == self.L3_ACTIVE and previous == self.DRIVER:
                self.engage_events += 1

        hazard = self.state in (self.TOR, self.MRM, self.SAFE_STOP)
        target_decel = 0.0
        if self.state in (self.MRM, self.SAFE_STOP):
            # Unknown friction uses the 0.1 floor: the gentlest MRM, which is
            # the one least likely to exceed the grip that is actually there.
            mu_used = mu if (isinstance(mu, (int, float)) and math.isfinite(mu)) else 0.1
            target_decel = self.mrm_decel_frac * max(0.1, mu_used) * G

        return {
            "state": self.state,
            "hazard": hazard,
            "target_decel_ms2": round(target_decel, 2),
            "override": self.state in (self.MRM, self.SAFE_STOP),
            "tor_elapsed_s": round(self.tor_elapsed, 1),
            "control_owner": "driver" if self.state == self.DRIVER else "system",
            "autonomy_enabled": self.state in (self.L3_ACTIVE, self.DEGRADED),
            "tor_events": self.tor_events,
            "takeover_events": self.takeover_events,
            "mrm_events": self.mrm_events,
            # The takeover input is a simulated flag; nothing verifies a human.
            "human_takeover_verified": False,
        }
