import cv2
import numpy as np
from collections import Counter, deque


class TrafficLightTemporalVoter:
    """Per-signal majority vote over recent red/yellow/green/unknown crops."""

    def __init__(self, window=5, min_votes=3, stale_frames=40):
        self.window = int(window)
        self.min_votes = int(min_votes)
        self.stale_frames = int(stale_frames)
        self.histories = {}
        self.last_seen = {}

    def update(self, key, state, frame_id):
        history = self.histories.setdefault(key, deque(maxlen=self.window))
        state = str(state).lower()
        history.append(state if state in {"red", "yellow", "green"} else "unknown")
        self.last_seen[key] = int(frame_id)
        for old_key, last in list(self.last_seen.items()):
            if int(frame_id) - last > self.stale_frames:
                self.last_seen.pop(old_key, None)
                self.histories.pop(old_key, None)
        known = [value for value in history if value != "unknown"]
        if len(known) < self.min_votes:
            return "unknown"
        state, votes = Counter(known).most_common(1)[0]
        return state if votes >= self.min_votes else "unknown"

def classify_traffic_light(box_crop):
    """Phân loại đỏ/vàng/xanh bằng HSV, hỗ trợ cả cụm dọc lẫn cụm ngang."""
    if box_crop.size == 0:
        return "unknown"
        
    hsv = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
    if min(hsv.shape[:2]) < 2:
        return "unknown"

    # Không cắt cứng theo 1/3 dọc: CARLA có cả signal head ngang và bbox YOLO
    # thường chứa padding. Hue trên toàn ROI là tín hiệu chính.
    def color_score(name):
        if name == "red":
            mask = cv2.inRange(hsv, (0, 100, 110), (12, 255, 255))
            mask |= cv2.inRange(hsv, (168, 100, 110), (179, 255, 255))
        elif name == "yellow":
            mask = cv2.inRange(hsv, (16, 110, 120), (36, 255, 255))
        else:
            mask = cv2.inRange(hsv, (38, 80, 80), (100, 255, 255))
        return float(np.count_nonzero(mask)) / max(1, mask.size)

    scores = {name: color_score(name) for name in ("red", "yellow", "green")}
    best_color = max(scores, key=scores.get)
    return best_color if scores[best_color] >= 0.01 else "unknown"
