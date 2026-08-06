from collections import defaultdict
import numpy as np


class CollisionTracker:
    """Theo dõi lịch sử khoảng cách và tính toán Thời gian va chạm (TTC)."""

    def __init__(self):
        self.distance_history = defaultdict(list)

    def compute_ttc(self, track_id, current_distance):
        history = self.distance_history[track_id]
        history.append(current_distance)
        if len(history) > 10:
            history.pop(0)

        if len(history) < 3:
            return 99.0, "Safe"

        y = np.array(history)
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)

        closing_speed = -slope
        if closing_speed > 0.1:
            ttc = current_distance / closing_speed
        else:
            ttc = 99.0

        risk = "Safe"
        if current_distance < 15.0 and closing_speed > 0.1:
            risk = "Warning"
        if current_distance < 8.0 or ttc < 1.5:
            risk = "CRITICAL"

        return round(ttc, 1), risk

    def forget_stale(self, active_ids):
        stale_keys = [k for k in self.distance_history.keys()
                      if k not in active_ids]
        for k in stale_keys:
            del self.distance_history[k]
