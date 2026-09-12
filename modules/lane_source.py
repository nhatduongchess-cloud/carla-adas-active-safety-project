"""Safety-gated learned/map lane source arbiter; Hough remains shadow-only."""

import math

try:
    from perception_contracts import LaneEstimate
except ImportError:
    from modules.perception_contracts import LaneEstimate


class LaneSourceArbiter:
    def __init__(self, confidence_min=0.75, stable_frames=5, invalid_frames=3,
                 width_min_m=2.8, width_max_m=4.2, max_jump_m=0.6):
        self.confidence_min = float(confidence_min)
        self.stable_frames = int(stable_frames)
        self.invalid_frames = int(invalid_frames)
        self.width_min = float(width_min_m)
        self.width_max = float(width_max_m)
        self.max_jump = float(max_jump_m)
        self._valid_count = 0
        self._invalid_count = 0
        self._learned_active = False
        self._last_learned_offset = None
        self._last_observation_key = None

    def _plausible(self, estimate):
        if estimate is None or not estimate.valid:
            return False, "invalid learned estimate"
        if estimate.confidence < self.confidence_min:
            return False, "low confidence"
        width = estimate.lane_width_m
        if width is None or not self.width_min <= width <= self.width_max:
            return False, "lane width outside ODD"
        if estimate.offset_m is None or not math.isfinite(estimate.offset_m):
            return False, "invalid center offset"
        if (self._last_learned_offset is not None
                and abs(estimate.offset_m - self._last_learned_offset) > self.max_jump):
            return False, "lane geometry jump"
        return True, None

    def select(self, learned, map_estimate, at_junction=False,
               now_s=None, max_age_ms=None):
        if at_junction:
            self._valid_count = 0
            self._invalid_count = 0
            self._learned_active = False
            self._last_learned_offset = None
            self._last_observation_key = (
                ("learned", int(learned.frame_id)) if learned is not None else None)
            return map_estimate

        stale = bool(
            learned is not None and now_s is not None and max_age_ms is not None
            and learned.age_ms(now_s) > float(max_age_ms))
        if stale:
            learned = None
        plausible, reason = self._plausible(learned)
        if stale:
            reason = "stale learned estimate"
        observation_key = (
            ("learned", int(learned.frame_id)) if learned is not None
            else ("missing", int(map_estimate.frame_id)))
        if observation_key != self._last_observation_key:
            self._last_observation_key = observation_key
            if plausible:
                self._valid_count += 1
                self._invalid_count = 0
                self._last_learned_offset = learned.offset_m
                if self._valid_count >= self.stable_frames:
                    self._learned_active = True
            else:
                self._valid_count = 0
                self._invalid_count += 1
                if self._invalid_count >= self.invalid_frames:
                    self._learned_active = False
                    # A new road segment after a dropout may have a legitimately
                    # different offset. Clear the old baseline so it can earn
                    # ownership again through the full stability gate.
                    self._last_learned_offset = None

        if self._learned_active and plausible:
            return learned
        if map_estimate.reason is None and reason:
            map_estimate.reason = f"learned fallback: {reason}"
        return map_estimate


def map_lane_estimate(frame_id, timestamp, centerline, lane_width_m=3.5):
    points = [(float(x), float(y)) for x, y in centerline]
    valid = (len(points) >= 2 and all(math.isfinite(v) for point in points for v in point)
             and any(math.hypot(b[0]-a[0], b[1]-a[1]) > 1e-6 for a,b in zip(points,points[1:])))
    return LaneEstimate(
        int(frame_id), float(timestamp), "map", confidence=1.0 if valid else 0.0,
        centerline_m=points if valid else [], offset_m=(-points[0][1] if valid else None),
        lane_width_m=float(lane_width_m), valid=valid,
        reason=None if valid else 'unresolved map route')
