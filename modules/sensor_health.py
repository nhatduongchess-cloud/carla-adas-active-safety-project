"""Sensor availability monitoring with non-blocking radar degradation policy."""

from dataclasses import replace
from typing import Dict

try:
    from perception_contracts import SensorState
except ImportError:
    from modules.perception_contracts import SensorState


class SensorHealthMonitor:
    def __init__(self, enabled=None, max_range_misses=3):
        enabled = enabled or {"camera": True, "lidar": True, "radar": True}
        self.states: Dict[str, SensorState] = {
            name: SensorState(name=name, enabled=bool(is_enabled))
            for name, is_enabled in enabled.items()
        }
        self.max_range_misses = int(max_range_misses)
        self.frames = 0
        self.available_counts = {name: 0 for name in self.states}
        self.missing_counts = {name: 0 for name in self.states}
        self.range_redundancy_loss_events = 0
        self._range_redundancy_lost_latched = False

    def observe(self, sensor: str, frame_id: int, timestamp: float,
                available: bool, error=None, now_s=None) -> SensorState:
        if sensor not in self.states:
            self.states[sensor] = SensorState(name=sensor)
            self.available_counts[sensor] = 0
            self.missing_counts[sensor] = 0
        state = self.states[sensor]
        if not state.enabled:
            return state
        state.available = bool(available)
        state.error = None if available else (str(error) if error else "missing frame")
        if available:
            state.frame_id = int(frame_id)
            state.timestamp = float(timestamp)
            state.age_ms = max(0.0, ((now_s if now_s is not None else timestamp)
                                     - timestamp) * 1000.0)
            state.consecutive_misses = 0
            self.available_counts[sensor] += 1
        else:
            state.consecutive_misses += 1
            self.missing_counts[sensor] += 1
            if state.timestamp is not None and now_s is not None:
                state.age_ms = max(0.0, (float(now_s) - state.timestamp) * 1000.0)
        range_lost = self.range_redundancy_lost
        if range_lost and not self._range_redundancy_lost_latched:
            self.range_redundancy_loss_events += 1
        self._range_redundancy_lost_latched = range_lost
        return state

    def next_frame(self) -> None:
        self.frames += 1

    @property
    def range_redundancy_lost(self) -> bool:
        lidar = self.states.get("lidar")
        radar = self.states.get("radar")
        if not lidar or not radar or not radar.enabled:
            return False
        return (lidar.consecutive_misses > self.max_range_misses
                and radar.consecutive_misses > self.max_range_misses)

    def availability(self, sensor: str) -> float:
        state = self.states.get(sensor)
        if state is None or not state.enabled:
            return 1.0
        return self.available_counts.get(sensor, 0) / max(1, self.frames)

    def snapshot(self) -> Dict[str, SensorState]:
        return {name: replace(state) for name, state in self.states.items()}

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "availability": {name: round(self.availability(name), 6)
                             for name in self.states},
            "consecutive_misses": {name: state.consecutive_misses
                                   for name, state in self.states.items()},
            "missing_counts": dict(self.missing_counts),
            "range_redundancy_lost": self.range_redundancy_lost,
            "range_redundancy_ever_lost": self.range_redundancy_loss_events > 0,
            "range_redundancy_loss_events": self.range_redundancy_loss_events,
        }
