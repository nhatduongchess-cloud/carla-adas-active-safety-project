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

CHỐNG "BÒ VÀO VẬT CẢN" (v2): vật chắn làn được nhận diện qua TỐC ĐỘ VẬT (ego trừ
closing) chứ không qua tốc độ ego, cộng thêm creep-guard khi ego bò chậm — nên xe
xử lý dứt khoát (né hoặc DỪNG HẲN) với vật TĨNH kể cả ở tốc độ thấp, thay vì hạ về
SLOW rồi trôi tới. Phanh đã chốt còn được GIỮ qua vài khung mất dấu (latch) để vật
thấp/thưa (cọc công trường) cho cụm LiDAR chớp tắt không làm nhả phanh sớm.

Lớp này là LOGIC THUẦN (không import carla): nhận số liệu, trả về một quyết định.
chinh.py mới là nơi dịch quyết định thành lệnh CARLA/Traffic Manager.
"""
# fmt: off
# isort: skip_file
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    from friction import stopping_distance
    from road_geometry import project_to_safety_path, straight_path
except ImportError:  # khi import kiểu package (modules.active_safety)
    from modules.friction import stopping_distance
    from modules.road_geometry import project_to_safety_path, straight_path


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
        vru_lateral_margin_m: float = 0.8,
        vru_prediction_horizon_s: float = 4.0,
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
        self.lane_change_front_gap = 18.0
        self.lane_change_rear_gap = 12.0
        self.lane_change_prediction_s = 3.0
        self.cut_in_prediction_s = 3.0
        self.vru_lateral_margin = float(vru_lateral_margin_m)
        self.vru_prediction_s = float(vru_prediction_horizon_s)

        # Điều kiện mặt đường/thời tiết (ODD monitor cập nhật động).
        self.mu = 0.8               # hệ số ma sát (dry mặc định)
        self.gap_multiplier = 1.0   # nhân khoảng cách an toàn khi thời tiết xấu

        # Tham số CAM KẾT + TRỄ (chống phân vân brake<->evade).
        self.evade_commit_s = 1.5       # giữ pha né tối thiểu ngần này (giây)
        self.brake_exit_factor = 1.4    # chỉ nhả phanh khi khoảng cách/TTC > ngưỡng × hệ số
        self.evade_min_ttc = 2.0        # chỉ né khi còn ĐỦ thời gian; ít hơn -> phanh

        # Chống "bò vào vật cản" (creep-into-obstacle) — nguồn gốc va chạm với vật
        # TĨNH ở tốc độ thấp. Một "lead" chậm hơn ngưỡng này coi là VẬT CHẮN: phải
        # né hoặc DỪNG HẲN, tuyệt đối không hạ về SLOW rồi trôi tới.
        self.follow_min_lead_speed = 3.0   # m/s (~10.8 km/h)
        # Nếu ego đang bò chậm mà vẫn còn vật trong vùng cảnh báo -> bắt buộc dừng
        # (không phụ thuộc ước lượng closing dễ nhiễu ở tốc độ thấp).
        self.creep_speed_thresh = 3.0      # m/s

        # Trạng thái nội bộ để chốt/latch quyết định.
        self._brake_latched = False
        self._evade_dir = None          # 'LANE_CHANGE_LEFT' | 'LANE_CHANGE_RIGHT'
        self._evade_frames_left = 0
        self._evade_origin_lane_id = None
        # Giữ phanh đã chốt qua vài khung MẤT DẤU (vật thấp/thưa như cọc công
        # trường cho cụm LiDAR chớp tắt) -> không nhả phanh rồi lao tới.
        self.lost_frames_tol = 4
        self._lost_frames = 0

        self.prev_nearest: float = math.inf  # cho ước lượng tốc độ tiến lại gần

    def set_conditions(self, mu: float, gap_multiplier: float = 1.0) -> None:
        """ODD monitor gọi để cập nhật ma sát + hệ số giãn khoảng cách theo thời tiết."""
        self.mu = mu
        self.gap_multiplier = gap_multiplier

    # --------------------------------------------------------------------- #
    def _scan_corridors(self, lidar_obstacles: List[Dict[str, Any]], path_points=None):
        """Quét cụm LiDAR: vật gần nhất trong làn + làn trái/phải có trống không.

        Khung LiDAR/xe: x tiến, +y phải, -y trái.
        """
        nearest = math.inf
        nearest_obs: Optional[Dict[str, Any]] = None
        gaps = {
            "left": {"front": math.inf, "rear": math.inf},
            "right": {"front": math.inf, "rear": math.inf},
        }
        path = path_points or straight_path(self.evade_lookahead + 15.0)

        for obs in lidar_obstacles:
            x, y, _z = obs["centroid"]
            reliable = obs.get("point_count", 0) >= self.min_cluster_points
            reliable = reliable or (
                obs.get("safety_critical", False) and obs.get("point_count", 0) >= 2)
            if not reliable:
                continue
            along, lateral, _distance_to_path = project_to_safety_path(x, y, path)

            if x > 0.0 and abs(lateral) < self.lane_half and along < nearest:
                nearest = along
                nearest_obs = obs

            side = None
            if self.lane_half <= lateral < 3.0 * self.lane_half:
                side = "right"
            elif -3.0 * self.lane_half < lateral <= -self.lane_half:
                side = "left"
            if side is not None:
                if x >= 0.0:
                    gaps[side]["front"] = min(gaps[side]["front"], along)
                else:
                    gaps[side]["rear"] = min(gaps[side]["rear"], abs(x))

        return nearest, nearest_obs, gaps

    @staticmethod
    def _track_relative_velocity(track, ego_speed_ms):
        vx, vy = track.vel
        if getattr(track, "ego_compensated", False):
            vx -= ego_speed_ms
        return vx, vy

    def _predictive(self, tracks, path_points=None, ego_speed_ms=0.0):
        """Từ tracker: vật gần nhất trong làn + TTC theo vận tốc TƯƠNG ĐỐI ước lượng.

        tracks: đối tượng có .pos -> (x, y) và .vel -> (vx, vy) trong khung ego.
        """
        nearest = math.inf
        ttc = math.inf
        path = path_points or straight_path(self.evade_lookahead + 15.0)
        for tr in tracks:
            x, y = tr.pos
            vx, _vy = self._track_relative_velocity(tr, ego_speed_ms)
            if x <= 0.0:
                continue
            along, lateral, _ = project_to_safety_path(x, y, path)
            is_vru = str(getattr(tr, "class_name", "")).lower() in {
                "person", "bicycle", "motorcycle"}
            corridor_half = self.lane_half + (self.vru_lateral_margin if is_vru else 0.0)
            if abs(lateral) < corridor_half:
                nearest = min(nearest, along)
                closing = -vx  # vx < 0 nghĩa là đang tiến lại gần ego
                if self.min_closing < closing < 40.0:
                    ttc = min(ttc, along / closing)
                continue

            # Track đang ở làn bên nhưng quỹ đạo ngắn hạn cắt vào swept path.
            step = 0.25
            t = step
            horizon = self.vru_prediction_s if is_vru else self.cut_in_prediction_s
            while t <= horizon + 1e-9:
                px, py = x + vx * t, y + _vy * t
                p_along, p_lat, _ = project_to_safety_path(px, py, path)
                if px > 0.0 and p_along > 0.0 and abs(p_lat) < corridor_half:
                    nearest = min(nearest, p_along)
                    ttc = min(ttc, t)
                    break
                t += step
        return nearest, ttc

    def _radar_threat(self, radar_targets, path_points=None):
        """Closest conservative radar range/TTC in the current swept path."""
        nearest, ttc, closing_at_threat = math.inf, math.inf, 0.0
        path = path_points or straight_path(self.evade_lookahead + 15.0)
        for target in radar_targets or []:
            x = float(target.get("distance_m", target.get("x_m", math.inf)))
            y = float(target.get("lateral_m", target.get("y_m", 0.0)))
            if not math.isfinite(x) or x <= 0.0:
                continue
            along, lateral, _ = project_to_safety_path(x, y, path)
            if along <= 0.0 or abs(lateral) >= self.lane_half:
                continue
            target_closing = max(0.0, float(target.get("closing_speed_ms", 0.0)))
            target_ttc = (along / target_closing
                          if target_closing > self.min_closing else math.inf)
            if along < nearest:
                nearest = along
                closing_at_threat = target_closing
            ttc = min(ttc, target_ttc)
        return nearest, ttc, closing_at_threat

    def _lane_change_clear(self, side, gaps, tracks, path_points, lane_context,
                           ego_speed_ms=0.0):
        ctx = lane_context or {}
        if not ctx.get(f"{side}_exists", True):
            return False
        if not ctx.get(f"{side}_change_allowed", True):
            return False
        if gaps[side]["front"] < self.lane_change_front_gap:
            return False
        if gaps[side]["rear"] < self.lane_change_rear_gap:
            return False

        path = path_points or straight_path(self.evade_lookahead + 15.0)
        for tr in tracks or []:
            x, y = tr.pos
            vx, vy = self._track_relative_velocity(tr, ego_speed_ms)
            for t in (0.0, 1.0, 2.0, self.lane_change_prediction_s):
                px, py = x + vx * t, y + vy * t
                along, lateral, _ = project_to_safety_path(px, py, path)
                in_side = (self.lane_half <= lateral < 3.0 * self.lane_half) \
                    if side == "right" else (-3.0 * self.lane_half < lateral <= -self.lane_half)
                if in_side and (-self.lane_change_rear_gap < px
                                and along < self.lane_change_front_gap):
                    return False
        return True

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
        path_points=None,
        lane_context=None,
        radar_targets=None,
    ) -> SafetyDecision:
        corridor_nearest, _obs, gaps = self._scan_corridors(lidar_obstacles, path_points)
        left_clear = self._lane_change_clear(
            "left", gaps, tracks, path_points, lane_context, ego_speed_ms)
        right_clear = self._lane_change_clear(
            "right", gaps, tracks, path_points, lane_context, ego_speed_ms)

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
            pred_nearest, pred_ttc = self._predictive(
                tracks, path_points, ego_speed_ms)

        radar_nearest, radar_ttc, radar_closing = self._radar_threat(
            radar_targets, path_points)

        # Never average away danger: the most conservative valid geometry/TTC
        # from LiDAR, tracker, or radar wins.
        distance_candidates = {
            "lidar": corridor_nearest,
            "tracker": pred_nearest,
            "radar": radar_nearest,
        }
        finite_distances = {
            source: distance for source, distance in distance_candidates.items()
            if math.isfinite(distance)
        }
        nearest_source = (min(finite_distances, key=finite_distances.get)
                          if finite_distances else None)
        nearest = (finite_distances[nearest_source]
                   if nearest_source is not None else math.inf)
        ttc = min(corridor_ttc, pred_ttc, radar_ttc)
        conservative_closing = max(closing, radar_closing)

        decision = SafetyDecision()

        def _emit_brake(state, level):
            decision.state = state
            decision.action = "BRAKE"
            decision.brake = level
            decision.collision_risk = True

        if not math.isfinite(nearest):
            # MẤT DẤU vật cản. Nếu ĐANG chốt phanh, giữ thêm vài khung phòng khi cụm
            # LiDAR của vật thấp/thưa (cọc công trường) chớp tắt -> KHÔNG nhả phanh
            # rồi lao tới. Chỉ thực sự trở về lái thường khi mất dấu đủ lâu.
            if self._brake_latched and self._lost_frames < self.lost_frames_tol:
                self._lost_frames += 1
                _emit_brake("BRAKE_HOLD", 0.7)
                return decision
            self.prev_nearest = math.inf
            self._brake_latched = False
            self._evade_dir = None
            self._evade_frames_left = 0
            self._evade_origin_lane_id = None
            self._lost_frames = 0
            return decision  # NORMAL / DRIVE
        self._lost_frames = 0

        # Khoảng cách an toàn động = quãng đường dừng (theo μ) × hệ số thời tiết + đệm.
        brake_dist = stopping_distance(ego_speed_ms, self.mu, self.reaction_time)
        dyn_safe = self.gap_multiplier * brake_dist + self.min_safe_dist
        critical = (nearest < self.min_safe_dist) or (ttc < self.critical_ttc)
        warning = (nearest < dyn_safe) or (ttc < self.warning_ttc)

        # Ước lượng TỐC ĐỘ VẬT CẢN (m/s): ego trừ tốc độ tiến lại gần. Vật đứng yên ->
        # closing ≈ ego -> obstacle_speed ≈ 0. Không phụ thuộc ngưỡng tốc độ ego, nên
        # xe vẫn xử lý dứt khoát vật tĩnh KỂ CẢ khi đang bò chậm (sửa lỗi "bò vào vật").
        obstacle_speed = max(0.0, ego_speed_ms - max(conservative_closing, 0.0))
        lead_is_slow = obstacle_speed < self.follow_min_lead_speed
        # Chốt an toàn độc lập với ước lượng closing (dễ nhiễu ở tốc độ thấp): ego bò
        # chậm + vật còn trong vùng cảnh báo -> BẮT BUỘC hành động (dừng/né).
        creeping = (ego_speed_ms < self.creep_speed_thresh) and (nearest < dyn_safe)
        must_act = (nearest < self.evade_lookahead) and (lead_is_slow or creeping)

        decision.threat = {
            "distance_m": round(nearest, 2),
            "source": nearest_source,
            "ttc_s": round(ttc, 2) if math.isfinite(ttc) else None,
            "closing_ms": round(conservative_closing, 2),
            "lidar_distance_m": round(corridor_nearest, 2)
                                if math.isfinite(corridor_nearest) else None,
            "tracker_distance_m": round(pred_nearest, 2)
                                  if math.isfinite(pred_nearest) else None,
            "radar_closing_ms": round(radar_closing, 2) if radar_targets else None,
            "radar_distance_m": round(radar_nearest, 2)
                                if math.isfinite(radar_nearest) else None,
            "obstacle_speed_ms": round(obstacle_speed, 2),
            "label": self._label_for(nearest, fused_detections),
        }

        # ================= PHÂN XỬ CÓ CAM KẾT + TRỄ (chống phân vân) =================
        # Ngưỡng thoát (hysteresis): chỉ rời trạng thái phanh khi ĐÃ RÕ an toàn.
        is_clear = (nearest > dyn_safe * self.brake_exit_factor) and \
                   (ttc > self.warning_ttc * self.brake_exit_factor)

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
            self._evade_origin_lane_id = None
            _emit_brake("EMERGENCY_BRAKE", 1.0)
            return decision

        # 3) Đang trong pha NÉ đã cam kết -> giữ hướng, trừ khi phải hủy để phanh.
        if self._evade_frames_left > 0 and self._evade_dir is not None:
            target_side = "right" if self._evade_dir == "LANE_CHANGE_RIGHT" else "left"
            current_lane_id = (lane_context or {}).get("current_lane_id")
            lane_change_complete = (
                self._evade_origin_lane_id is not None
                and current_lane_id is not None
                and current_lane_id != self._evade_origin_lane_id)
            target_still_clear = right_clear if target_side == "right" else left_clear
            self._evade_frames_left -= 1
            getting_dangerous = (ttc < self.critical_ttc * 1.3) or \
                                (nearest < self.min_safe_dist * 1.3)
            if lane_change_complete or not warning:
                self._evade_dir = None
                self._evade_frames_left = 0
                self._evade_origin_lane_id = None
            elif getting_dangerous or not target_still_clear:
                # Chuyển làn không kịp/không ăn -> CHỐT PHANH ngay.
                self._brake_latched = True
                self._evade_dir = None
                self._evade_frames_left = 0
                self._evade_origin_lane_id = None
                _emit_brake("EMERGENCY_BRAKE", 1.0)
                return decision
            else:
                decision.state = "EVADE_RIGHT" if self._evade_dir == "LANE_CHANGE_RIGHT" else "EVADE_LEFT"
                decision.action = self._evade_dir
                decision.collision_risk = True
                return decision

        # 4) Vật CHẬM/ĐỨNG chắn đường trong tầm né -> DỨT KHOÁT: né (nếu đủ điều kiện)
        #    hoặc PHANH DỪNG HẲN. KHÔNG bao giờ chỉ "SLOW" rồi trôi vào vật cản.
        if must_act:
            can_evade = (self.enable_evasion and (left_clear or right_clear)
                         and ttc >= self.evade_min_ttc
                         and nearest > self.min_safe_dist)
            if can_evade:
                self._evade_dir = "LANE_CHANGE_RIGHT" if right_clear else "LANE_CHANGE_LEFT"
                self._evade_frames_left = max(1, int(self.evade_commit_s / max(dt, 1e-3)))
                self._evade_origin_lane_id = (lane_context or {}).get("current_lane_id")
                decision.state = "EVADE_RIGHT" if right_clear else "EVADE_LEFT"
                decision.action = self._evade_dir
                decision.collision_risk = True
            else:
                # Không né được (hai làn chặn / hết thời gian) -> phanh dừng & CHỐT lại.
                self._brake_latched = True
                _emit_brake("BRAKE_TO_STOP", 1.0)
            return decision

        # 5) Lead ĐANG CHẠY (đủ nhanh để bám), chỉ trong vùng cảnh báo -> giảm tốc bám.
        if warning:
            decision.state = "FOLLOW"
            decision.action = "SLOW"
            decision.slow_pct = self.follow_speed_diff
            decision.collision_risk = True
            return decision

        # 6) NORMAL / DRIVE (mặc định)
        return decision
