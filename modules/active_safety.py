"""Hệ thống an toàn chủ động (Active Safety) — máy trạng thái AEB + tránh né.

Thiết kế theo kiến trúc "autopilot nền + lớp an toàn ghi đè":
  NORMAL          -> để Traffic Manager tự lái.
  FOLLOW          -> có vật cản phía trước nhưng còn xa: giảm tốc (qua TM).
  EVADE_LEFT/RIGHT-> vật cản chậm/đứng yên, còn quá gần để bám theo nhưng làn
                     bên trống: yêu cầu chuyển làn (qua TM.force_lane_change).
  EMERGENCY_BRAKE -> trong khoảng cách phanh hoặc TTC tới hạn: phanh gấp.

Điểm mấu chốt về ĐỘ TIN CẬY: tín hiệu phanh gấp lấy TRỰC TIẾP từ hành lang phía
trước của LiDAR (khoảng cách mét thật, không cần calib camera), nên nó phanh cho
BẤT KỲ vật cản nào trong làn — kể cả vật YOLO không phân loại (người đi bộ, mảnh
vỡ...). Camera/bbox chỉ dùng để gán nhãn hiển thị.

Lớp này là LOGIC THUẦN (không import carla): nhận số liệu, trả về một quyết định.
chinh.py mới là nơi dịch quyết định thành lệnh CARLA/Traffic Manager.
"""
# fmt: off
# isort: skip_file
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    from friction import stopping_distance
except ImportError:  # khi import kiểu package (modules.active_safety)
    from modules.friction import stopping_distance


@dataclass
class SafetyDecision:
    state: str = "NORMAL"
    # DRIVE | SLOW | LANE_CHANGE_LEFT | LANE_CHANGE_RIGHT | BRAKE
    action: str = "DRIVE"
    brake: float = 0.0
    slow_pct: float = 0.0            # % chênh tốc gửi cho Traffic Manager khi SLOW
    collision_risk: bool = False
    threat: Dict[str, Any] = field(default_factory=dict)


class ActiveSafetySystem:
    def __init__(
        self,
        lane_half_width_m: float = 1.75,
        min_cluster_points: int = 8,
        reaction_time_s: float = 1.0,
        min_safe_dist_m: float = 6.0,
        critical_ttc_s: float = 1.6,
        warning_ttc_s: float = 3.0,
        min_closing_speed: float = 0.3,
        enable_evasion: bool = True,
        evade_lookahead_m: float = 25.0,
        lead_slow_ratio: float = 0.55,
        follow_speed_diff: float = 60.0,
    ) -> None:
        self.lane_half = lane_half_width_m
        self.min_cluster_points = min_cluster_points
        self.reaction_time = reaction_time_s
        self.min_safe_dist = min_safe_dist_m
        self.critical_ttc = critical_ttc_s
        self.warning_ttc = warning_ttc_s
        self.min_closing = min_closing_speed
        self.enable_evasion = enable_evasion
        self.evade_lookahead = evade_lookahead_m
        self.lead_slow_ratio = lead_slow_ratio
        self.follow_speed_diff = follow_speed_diff

        # Điều kiện mặt đường/thời tiết (ODD monitor cập nhật động).
        self.mu = 0.8               # hệ số ma sát (dry mặc định)
        self.gap_multiplier = 1.0   # nhân khoảng cách an toàn khi thời tiết xấu

        # Tham số CAM KẾT + TRỄ (chống phân vân brake<->evade).
        self.evade_commit_s = 1.5       # giữ pha né tối thiểu ngần này (giây)
        self.brake_exit_factor = 1.4    # chỉ nhả phanh khi khoảng cách/TTC > ngưỡng × hệ số
        self.evade_min_ttc = 2.0        # chỉ né khi còn ĐỦ thời gian; ít hơn -> phanh

        # Trạng thái nội bộ để chốt/latch quyết định.
        self._brake_latched = False
        self._evade_dir = None          # 'LANE_CHANGE_LEFT' | 'LANE_CHANGE_RIGHT'
        self._evade_frames_left = 0

        self.prev_nearest: float = math.inf  # cho ước lượng tốc độ tiến lại gần

    def set_conditions(self, mu: float, gap_multiplier: float = 1.0) -> None:
        """ODD monitor gọi để cập nhật ma sát + hệ số giãn khoảng cách theo thời tiết."""
        self.mu = mu
        self.gap_multiplier = gap_multiplier

    # --------------------------------------------------------------------- #
    def _scan_corridors(self, lidar_obstacles: List[Dict[str, Any]]):
        """Quét cụm LiDAR: vật gần nhất trong làn + làn trái/phải có trống không.

        Khung LiDAR/xe: x tiến, +y phải, -y trái.
        """
        nearest = math.inf
        nearest_obs: Optional[Dict[str, Any]] = None
        left_clear = True
        right_clear = True

        for obs in lidar_obstacles:
            x, y, _z = obs["centroid"]
            if obs.get("point_count", 0) < self.min_cluster_points:
                continue
            if x <= 0.0:
                continue

            if abs(y) < self.lane_half and x < nearest:
                nearest = x
                nearest_obs = obs

            if 0.0 < x < self.evade_lookahead:
                # Làn phải kế bên: y in [lane_half, 3*lane_half]
                if self.lane_half <= y < 3.0 * self.lane_half:
                    right_clear = False
                # Làn trái kế bên: y in [-3*lane_half, -lane_half]
                elif -3.0 * self.lane_half < y <= -self.lane_half:
                    left_clear = False

        return nearest, nearest_obs, left_clear, right_clear

    def _predictive(self, tracks):
        """Từ tracker: vật gần nhất trong làn + TTC theo vận tốc TƯƠNG ĐỐI ước lượng.

        tracks: đối tượng có .pos -> (x, y) và .vel -> (vx, vy) trong khung ego.
        """
        nearest = math.inf
        ttc = math.inf
        for tr in tracks:
            x, y = tr.pos
            vx, _vy = tr.vel
            if x <= 0.0 or abs(y) >= self.lane_half:
                continue
            nearest = min(nearest, x)
            closing = -vx  # vx < 0 nghĩa là đang tiến lại gần ego
            # Bỏ vận tốc phi lý (nhiễu track lúc mới khởi tạo) -> tránh phanh oan làm kẹt xe.
            if self.min_closing < closing < 40.0:
                ttc = min(ttc, x / closing)
        return nearest, ttc

    @staticmethod
    def _label_for(nearest: float, fused_detections: List[Dict[str, Any]]) -> str:
        """Gán nhãn hiển thị: detection có distance_m gần khớp với vật LiDAR nhất."""
        best_label, best_gap = "obstacle", math.inf
        for det in fused_detections:
            d = det.get("distance_m")
            if d is None:
                continue
            gap = abs(d - nearest)
            if gap < best_gap:
                best_gap, best_label = gap, det.get("class", "obstacle")
        return best_label if best_gap < 4.0 else "obstacle"

    # --------------------------------------------------------------------- #
    def update(
        self,
        ego_speed_ms: float,
        fused_detections: List[Dict[str, Any]],
        lidar_obstacles: List[Dict[str, Any]],
        dt: float,
        tracks=None,
    ) -> SafetyDecision:
        corridor_nearest, _obs, left_clear, right_clear = self._scan_corridors(lidar_obstacles)

        # Tốc độ tiến lại gần (closing speed) từ lịch sử khoảng cách hành lang LiDAR.
        closing = 0.0
        if math.isfinite(corridor_nearest) and math.isfinite(self.prev_nearest) and dt > 1e-6:
            closing = (self.prev_nearest - corridor_nearest) / dt
        self.prev_nearest = corridor_nearest

        corridor_ttc = math.inf
        if closing > self.min_closing and math.isfinite(corridor_nearest):
            corridor_ttc = corridor_nearest / closing

        # Dự đoán từ tracker (tùy chọn): TTC theo vận tốc từng vật -> phanh SỚM hơn.
        pred_nearest, pred_ttc = math.inf, math.inf
        if tracks:
            pred_nearest, pred_ttc = self._predictive(tracks)

        nearest = min(corridor_nearest, pred_nearest)
        ttc = min(corridor_ttc, pred_ttc)

        decision = SafetyDecision()

        if not math.isfinite(nearest):
            # Mất dấu vật cản -> nhả chốt và trở về lái thường.
            self.prev_nearest = math.inf
            self._brake_latched = False
            self._evade_dir = None
            self._evade_frames_left = 0
            return decision  # NORMAL / DRIVE

        # Khoảng cách an toàn động = quãng đường dừng (theo μ) × hệ số thời tiết + đệm.
        brake_dist = stopping_distance(ego_speed_ms, self.mu, self.reaction_time)
        dyn_safe = self.gap_multiplier * brake_dist + self.min_safe_dist
        critical = (nearest < self.min_safe_dist) or (ttc < self.critical_ttc)
        warning = (nearest < dyn_safe) or (ttc < self.warning_ttc)

        decision.threat = {
            "distance_m": round(nearest, 2),
            "ttc_s": round(ttc, 2) if math.isfinite(ttc) else None,
            "closing_ms": round(closing, 2),
            "label": self._label_for(nearest, fused_detections),
        }

        # ================= PHÂN XỬ CÓ CAM KẾT + TRỄ (chống phân vân) =================
        # Ngưỡng thoát (hysteresis): chỉ rời trạng thái phanh khi ĐÃ RÕ an toàn.
        is_clear = (nearest > dyn_safe * self.brake_exit_factor) and \
                   (ttc > self.warning_ttc * self.brake_exit_factor)
        # "Lead chậm/đứng yên": tiến lại gần ~ bằng tốc độ ego (vật gần như đứng im).
        lead_is_slow = ego_speed_ms > 2.0 and closing >= self.lead_slow_ratio * ego_speed_ms

        def _emit_brake(state, level):
            decision.state = state
            decision.action = "BRAKE"
            decision.brake = level
            decision.collision_risk = True

        # 1) ĐANG PHANH -> giữ tới khi rõ an toàn (không nhả sớm gây phân vân).
        if self._brake_latched:
            if is_clear:
                self._brake_latched = False
            else:
                _emit_brake("EMERGENCY_BRAKE" if critical else "BRAKE_HOLD",
                            1.0 if critical else 0.7)
                return decision

        # 2) PHANH KHẨN CẤP luôn thắng và chốt lại.
        if critical:
            self._brake_latched = True
            self._evade_dir = None
            self._evade_frames_left = 0
            _emit_brake("EMERGENCY_BRAKE", 1.0)
            return decision

        # 3) Đang trong pha NÉ đã cam kết -> giữ hướng, trừ khi phải hủy để phanh.
        if self._evade_frames_left > 0 and self._evade_dir is not None:
            self._evade_frames_left -= 1
            getting_dangerous = (ttc < self.critical_ttc * 1.3) or \
                                (nearest < self.min_safe_dist * 1.3)
            if not warning:
                self._evade_dir = None
                self._evade_frames_left = 0
            elif getting_dangerous:
                # Chuyển làn không kịp/không ăn -> CHỐT PHANH ngay.
                self._brake_latched = True
                self._evade_dir = None
                self._evade_frames_left = 0
                _emit_brake("EMERGENCY_BRAKE", 1.0)
                return decision
            else:
                decision.state = "EVADE_RIGHT" if self._evade_dir == "LANE_CHANGE_RIGHT" else "EVADE_LEFT"
                decision.action = self._evade_dir
                decision.collision_risk = True
                return decision

        # 4) Vật CHẬM/ĐỨNG chắn đường trong tầm né -> DỨT KHOÁT: né (nếu đủ điều kiện)
        #    hoặc PHANH. KHÔNG bao giờ chỉ "SLOW" rồi trôi vào vật cản.
        if lead_is_slow and nearest < self.evade_lookahead:
            can_evade = (self.enable_evasion and (left_clear or right_clear)
                         and ttc >= self.evade_min_ttc)
            if can_evade:
                self._evade_dir = "LANE_CHANGE_RIGHT" if right_clear else "LANE_CHANGE_LEFT"
                self._evade_frames_left = max(1, int(self.evade_commit_s / max(dt, 1e-3)))
                decision.state = "EVADE_RIGHT" if right_clear else "EVADE_LEFT"
                decision.action = self._evade_dir
                decision.collision_risk = True
            else:
                self._brake_latched = True
                _emit_brake("BRAKE_TO_STOP", 1.0)
            return decision

        # 5) Lead đang chạy, chỉ trong vùng cảnh báo -> bám & giảm tốc (để TM lo).
        if warning:
            decision.state = "FOLLOW"
            decision.action = "SLOW"
            decision.slow_pct = self.follow_speed_diff
            decision.collision_risk = True
            return decision

        # 6) NORMAL / DRIVE (mặc định)
        return decision
