"""Chủ động chuyển làn theo Ý ĐỊNH RẼ (rule-based, không cần RL).

Khi tới gần giao lộ và có ý định rẽ trái/phải, chủ động chuyển sang làn cùng chiều
phía đó TRƯỚC khi vào giao lộ (qua Traffic Manager.force_lane_change) — giống hành
vi người lái thật. Dùng waypoint lookahead để phát hiện giao lộ phía trước.
"""
# fmt: off
# isort: skip_file


def intent_to_right(intent: str) -> bool:
    """'right' -> True (force_lane_change sang phải), 'left' -> False."""
    return intent == "right"


class TurnIntentPlanner:
    def __init__(self, cfg, lookahead_m=45.0, cooldown_s=6.0):
        self.lookahead = lookahead_m
        self.cooldown_frames = cooldown_s / cfg.FIXED_DELTA
        self._last_frame = -(10 ** 9)

    def _junction_ahead(self, world, ego):
        wp = world.get_map().get_waypoint(ego.get_location())
        if wp is None:
            return None, False
        cur = wp
        dist = 0.0
        while dist < self.lookahead:
            nxt = cur.next(5.0)
            if not nxt:
                break
            cur = nxt[0]
            dist += 5.0
            if cur.is_junction:
                return wp, True
        return wp, False

    def update(self, ego, world, traffic_manager, intent, frame):
        """intent: 'left' | 'right' | None. Trả về hướng đã chuyển (hoặc None)."""
        import carla
        if intent not in ("left", "right"):
            return None
        if frame - self._last_frame < self.cooldown_frames:
            return None

        wp, junction = self._junction_ahead(world, ego)
        if not junction or wp is None:
            return None

        side_lane = wp.get_right_lane() if intent == "right" else wp.get_left_lane()
        # Chỉ chuyển nếu có làn CÙNG CHIỀU ở phía rẽ (lane_id cùng dấu).
        if (side_lane is not None
                and side_lane.lane_type == carla.LaneType.Driving
                and side_lane.lane_id * wp.lane_id > 0):
            try:
                traffic_manager.force_lane_change(ego, intent_to_right(intent))
                self._last_frame = frame
                return intent
            except Exception:
                return None
        return None
