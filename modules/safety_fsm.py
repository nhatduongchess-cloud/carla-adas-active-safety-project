"""Máy trạng thái AEB. Mỗi episode giữ riêng lịch sử phanh và cam kết né."""

import math

if __package__:
    from .friction import stopping_distance
    from .safety_contracts import SafetyConfig, SafetyInput, SafetyOutput
else:
    from friction import stopping_distance
    from safety_contracts import SafetyConfig, SafetyInput, SafetyOutput


class SafetyFSM:
    def __init__(self):
        self.reset()

    def reset(self):
        self._brake_latched = False
        self._latched_level = 0.0
        self._clear_observed_s = 0.0
        self._lost_frames = 0
        self._evade_dir = None
        self._evade_frames_left = 0
        self._evade_origin_lane_id = None
        self.last_transition_reason = None

    def step(self, inputs: SafetyInput, cfg: SafetyConfig) -> SafetyOutput:
        nearest = math.inf if inputs.nearest_distance_m is None else inputs.nearest_distance_m
        ttc = math.inf if inputs.ttc_s is None else inputs.ttc_s
        dt, ego_speed_ms = inputs.dt_s, inputs.ego_speed_ms
        lidar_valid = inputs.lidar_valid
        left_clear, right_clear = inputs.left_clear, inputs.right_clear
        conservative_closing = inputs.conservative_closing_ms

        def _emit_brake(state, level):
            if self._brake_latched:
                self._latched_level = level
            return SafetyOutput(state, "BRAKE", level, collision_risk=True)

        if not math.isfinite(nearest):
            # MẤT DẤU vật cản. Nếu ĐANG chốt phanh, giữ phanh cho tới khi đường
            # trống được QUAN SÁT hợp lệ đủ lâu (clear_confirm_s).
            #
            # Two rules from the 2026-09-25 evidence review (open gap G9):
            #  - a frame with no valid LiDAR data is not evidence that the
            #    obstacle left, so it never counts toward release;
            #  - holding does not weaken the brake. It used to drop from 1.0 to
            #    0.7 on the first empty frame after a critical stop.
            if self._brake_latched:
                if not lidar_valid:
                    self._clear_observed_s = 0.0
                    self.last_transition_reason = "hold: no valid lidar observation"
                    return _emit_brake("BRAKE_HOLD_NO_DATA", self._latched_level or 1.0)
                if math.isfinite(dt) and dt > 0.0:
                    self._clear_observed_s += dt
                if self._clear_observed_s <= cfg.clear_confirm_s + 1e-9:
                    self._lost_frames += 1
                    self.last_transition_reason = "hold: confirming clear path"
                    return _emit_brake("BRAKE_HOLD", self._latched_level or 1.0)
                self.last_transition_reason = (
                    f"release: clear path observed for {self._clear_observed_s:.3f} s")
            self._brake_latched = False
            self._latched_level = 0.0
            self._clear_observed_s = 0.0
            self._evade_dir = None
            self._evade_frames_left = 0
            self._evade_origin_lane_id = None
            self._lost_frames = 0
            return SafetyOutput()
        self._lost_frames = 0
        self._clear_observed_s = 0.0

        # Khoảng cách an toàn động = quãng đường dừng (theo μ) × hệ số thời tiết + đệm.
        brake_dist = stopping_distance(ego_speed_ms, inputs.mu, cfg.reaction_time_s)
        dyn_safe = inputs.gap_multiplier * brake_dist + cfg.min_safe_dist_m
        critical = (nearest < cfg.min_safe_dist_m) or (ttc < cfg.critical_ttc_s)
        warning = (nearest < dyn_safe) or (ttc < cfg.warning_ttc_s)

        # Ước lượng TỐC ĐỘ VẬT CẢN (m/s): ego trừ tốc độ tiến lại gần. Vật đứng yên ->
        # closing ≈ ego -> obstacle_speed ≈ 0. Không phụ thuộc ngưỡng tốc độ ego, nên
        # xe vẫn xử lý dứt khoát vật tĩnh KỂ CẢ khi đang bò chậm (sửa lỗi "bò vào vật").
        obstacle_speed = max(0.0, ego_speed_ms - max(conservative_closing, 0.0))
        lead_is_slow = obstacle_speed < cfg.follow_min_lead_speed
        # Chốt an toàn độc lập với ước lượng closing (dễ nhiễu ở tốc độ thấp): ego bò
        # chậm + vật còn trong vùng cảnh báo -> BẮT BUỘC hành động (dừng/né).
        creeping = (ego_speed_ms < cfg.creep_speed_thresh) and (nearest < dyn_safe)
        must_act = (nearest < cfg.evade_lookahead_m) and (lead_is_slow or creeping)

        # ================= PHÂN XỬ CÓ CAM KẾT + TRỄ (chống phân vân) =================
        # Ngưỡng thoát (hysteresis): chỉ rời trạng thái phanh khi ĐÃ RÕ an toàn.
        is_clear = (nearest > dyn_safe * cfg.brake_exit_factor) and \
                   (ttc > cfg.warning_ttc_s * cfg.brake_exit_factor)

        # 1) ĐANG PHANH -> giữ tới khi rõ an toàn (không nhả sớm gây phân vân).
        if self._brake_latched:
            if is_clear:
                self._brake_latched = False
                self._latched_level = 0.0
                self.last_transition_reason = "release: threat beyond exit hysteresis"
            else:
                return _emit_brake("EMERGENCY_BRAKE" if critical else "BRAKE_HOLD",
                            1.0 if critical else 0.7)

        # 2) PHANH KHẨN CẤP luôn thắng và chốt lại.
        if critical:
            self._brake_latched = True
            self._evade_dir = None
            self._evade_frames_left = 0
            self._evade_origin_lane_id = None
            return _emit_brake("EMERGENCY_BRAKE", 1.0)

        # 3) Đang trong pha NÉ đã cam kết -> giữ hướng, trừ khi phải hủy để phanh.
        if self._evade_frames_left > 0 and self._evade_dir is not None:
            target_side = "right" if self._evade_dir == "LANE_CHANGE_RIGHT" else "left"
            current_lane_id = inputs.current_lane_id
            lane_change_complete = (
                self._evade_origin_lane_id is not None
                and current_lane_id is not None
                and current_lane_id != self._evade_origin_lane_id)
            target_still_clear = right_clear if target_side == "right" else left_clear
            self._evade_frames_left -= 1
            getting_dangerous = (ttc < cfg.critical_ttc_s * 1.3) or \
                                (nearest < cfg.min_safe_dist_m * 1.3)
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
                return _emit_brake("EMERGENCY_BRAKE", 1.0)
            else:
                state = "EVADE_RIGHT" if target_side == "right" else "EVADE_LEFT"
                return SafetyOutput(state, self._evade_dir, collision_risk=True)

        # 4) Vật CHẬM/ĐỨNG chắn đường trong tầm né -> DỨT KHOÁT: né (nếu đủ điều kiện)
        #    hoặc PHANH DỪNG HẲN. KHÔNG bao giờ chỉ "SLOW" rồi trôi vào vật cản.
        if must_act:
            can_evade = (cfg.enable_evasion and (left_clear or right_clear)
                         and ttc >= cfg.evade_min_ttc
                         and nearest > cfg.min_safe_dist_m)
            if can_evade:
                self._evade_dir = "LANE_CHANGE_RIGHT" if right_clear else "LANE_CHANGE_LEFT"
                self._evade_frames_left = max(1, int(cfg.evade_commit_s / max(dt, 1e-3)))
                self._evade_origin_lane_id = inputs.current_lane_id
                state = "EVADE_RIGHT" if right_clear else "EVADE_LEFT"
                return SafetyOutput(state, self._evade_dir, collision_risk=True)
            # Làn bị chặn hoặc thời gian không đủ để né.
            self._brake_latched = True
            return _emit_brake("BRAKE_TO_STOP", 1.0)

        if warning:
            return SafetyOutput("FOLLOW", "SLOW", slow_pct=cfg.follow_speed_diff, collision_risk=True)

        return SafetyOutput()
