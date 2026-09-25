"""Sensor availability monitoring with non-blocking radar degradation policy."""

from dataclasses import replace
from typing import Dict

try:
    from perception_contracts import SensorState
except ImportError:
    from modules.perception_contracts import SensorState



#: Sensors that measure metric range. Losing every enabled one is losing range.
RANGE_SENSORS = ("lidar", "radar")

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
    def range_sensors_enabled(self) -> list:
        return [name for name in RANGE_SENSORS
                if name in self.states and self.states[name].enabled]

    @property
    def range_redundancy_lost(self) -> bool:
        """True when no enabled range sensor is delivering.

        The name is historical and kept because the ODD monitor and dashboard
        consume it. With LiDAR and radar both enabled it means exactly what it
        always meant: both lost. With radar disabled, LiDAR alone carries range,
        so losing it is losing range sensing entirely - and the previous rule,
        which returned False whenever radar was disabled, reported that as
        healthy. The 2026-09-25 evidence review reproduced it: five consecutive
        LiDAR misses, radar off, `range_redundancy_lost: false`, ODD NORMAL.

        A monitor with no range sensor enabled at all has no range sensing by
        construction, and says so.
        """
        enabled = self.range_sensors_enabled
        if not enabled:
            return True
        return all(self.states[name].consecutive_misses > self.max_range_misses
                   for name in enabled)

    @property
    def all_enabled_range_unavailable(self) -> bool:
        """Explicit name for `range_redundancy_lost`: no enabled range sensor
        is delivering (or none is enabled). Same value; the old key is kept for
        its existing consumers."""
        return self.range_redundancy_lost

    @property
    def range_redundancy_degraded(self) -> bool:
        """At least one enabled range sensor is lost but another still delivers.

        Informational: the ODD does not change on it. With a single enabled
        range sensor there is no redundancy to degrade, so this is False and
        a loss shows up as `all_enabled_range_unavailable` instead.
        """
        enabled = self.range_sensors_enabled
        lost = [name for name in enabled
                if self.states[name].consecutive_misses > self.max_range_misses]
        return bool(lost) and len(lost) < len(enabled)

    def availability(self, sensor: str):
        """Share of frames the sensor delivered, or None when it is disabled.

        A disabled sensor used to report 1.0 - 100% available - which let a
        disabled radar pass a "radar availability >= 99.5%" criterion. It has
        no availability; it is not there.
        """
        state = self.states.get(sensor)
        if state is None or not state.enabled:
            return None
        return self.available_counts.get(sensor, 0) / max(1, self.frames)

    def snapshot(self) -> Dict[str, SensorState]:
        return {name: replace(state) for name, state in self.states.items()}

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "enabled": {name: state.enabled for name, state in self.states.items()},
            "availability": {name: (None if self.availability(name) is None
                                    else round(self.availability(name), 6))
                             for name in self.states},
            "range_sensors_enabled": self.range_sensors_enabled,
            "consecutive_misses": {name: state.consecutive_misses
                                   for name, state in self.states.items()},
            "missing_counts": dict(self.missing_counts),
            "range_redundancy_lost": self.range_redundancy_lost,
            "all_enabled_range_unavailable": self.all_enabled_range_unavailable,
            "range_redundancy_degraded": self.range_redundancy_degraded,
            # The miss threshold counts loop frames, so its duration depends on
            # the loop rate; recorded so a report can state it.
            "max_range_misses_frames": self.max_range_misses,
            "range_redundancy_ever_lost": self.range_redundancy_loss_events > 0,
            "range_redundancy_loss_events": self.range_redundancy_loss_events,
        }
