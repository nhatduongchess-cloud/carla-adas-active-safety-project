"""Measured control-loop timing, kept apart from configured rates.

Pure Python, no CARLA import.

WHY THIS EXISTS
---------------
The runtime report used to decide "real time" from ``fps_ema``, an exponential
moving average of the last frame intervals that the HUD displays. The published
demo run shows why that cannot be an acceptance number: ``fps_ema`` was 44.9
while frames divided by the reported wall time gave 20.0 - and that wall time
was sampled when the report was written, so it also included teardown. Its
"simulated duration" was frames / 40, which in async mode is not the simulated
time at all.

This module keeps four things separate, each with its own clock:

* ``configured_step_s``      - the CARLA fixed delta the run asked for;
* ``simulated_elapsed_s``    - from sensor simulation timestamps;
* ``active_control_wall_s``  - wall time from the first to the last control
  cycle window, excluding start-up and teardown (``time.perf_counter``);
* ``delivered_control_hz``   - unique control updates / active wall window.

``real_time_factor`` = simulated / wall. EMA FPS is not used anywhere here.

``SimClock`` validates the simulation-time delta handed to the algorithms
(tracker, safety, TOR window) instead of assuming the configured step: in async
mode a slow loop skips world frames, and a fixed 0.025 s then understates the
real step - the TOR window runs slow and closing speeds come out wrong.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _percentile(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


class ControlWindow:
    """Counts control commands sent inside one explicit wall-clock window."""

    #: A cycle interval longer than this many target periods counts as a
    #: missed control deadline (at 40 Hz: an interval above 50 ms).
    DEADLINE_PERIODS = 2.0

    def __init__(self, max_intervals: int = 200_000) -> None:
        self.max_intervals = int(max_intervals)
        self.started_wall_s: Optional[float] = None
        self.stopped_wall_s: Optional[float] = None
        self.updates = 0
        self.duplicate_frames = 0
        self.intervals_s: List[float] = []
        self.intervals_dropped = 0
        self._last_wall: Optional[float] = None
        self._last_frame: Optional[int] = None
        self._first_sim: Optional[float] = None
        self._last_sim: Optional[float] = None
        self.sim_time_regressions = 0

    def start(self, wall_s: float) -> None:
        if self.started_wall_s is None:
            self.started_wall_s = float(wall_s)

    def cycle(self, wall_s: float, sim_time_s: Any = None, frame_id: Any = None) -> None:
        """Record one control command sent for one sensor frame."""
        if self.started_wall_s is None or self.stopped_wall_s is not None:
            return
        if frame_id is not None and self._last_frame is not None and frame_id <= self._last_frame:
            # The same (or an older) world frame again is not a new update.
            self.duplicate_frames += 1
            return
        if frame_id is not None:
            self._last_frame = frame_id
        self.updates += 1
        if self._last_wall is not None:
            if len(self.intervals_s) < self.max_intervals:
                self.intervals_s.append(float(wall_s) - self._last_wall)
            else:
                self.intervals_dropped += 1
        self._last_wall = float(wall_s)
        if _finite(sim_time_s):
            if self._first_sim is None:
                self._first_sim = float(sim_time_s)
            elif self._last_sim is not None and sim_time_s < self._last_sim:
                self.sim_time_regressions += 1
            self._last_sim = float(sim_time_s)

    def stop(self, wall_s: float) -> None:
        """Close the window. The first call wins, so a later call from a
        cleanup path cannot stretch the window over teardown."""
        if self.started_wall_s is not None and self.stopped_wall_s is None:
            self.stopped_wall_s = float(wall_s)

    def summary(self, target_hz: float, configured_step_s: float) -> Dict[str, Any]:
        wall = None
        if self.started_wall_s is not None:
            end = self.stopped_wall_s if self.stopped_wall_s is not None else self._last_wall
            if end is not None:
                wall = max(0.0, end - self.started_wall_s)
        sim = (self._last_sim - self._first_sim
               if self._first_sim is not None and self._last_sim is not None else None)
        measured = self.updates >= 2 and wall is not None and wall > 0.0
        hz = self.updates / wall if measured else None
        intervals = sorted(self.intervals_s)
        deadline_s = (self.DEADLINE_PERIODS / target_hz) if _finite(target_hz) and target_hz > 0 else None
        misses = (sum(1 for i in intervals if i > deadline_s)
                  if deadline_s is not None else None)

        def ms(value):
            return None if value is None else round(value * 1000.0, 3)

        return {
            "status": "MEASURED" if measured else "NOT_EVALUATED",
            "clock": "time.perf_counter for wall; sensor simulation timestamps for sim",
            "window": "first control cycle window start to loop exit; excludes start-up and teardown",
            "target_hz": target_hz,
            "configured_step_s": configured_step_s,
            "unique_control_updates": self.updates,
            "duplicate_frames_skipped": self.duplicate_frames,
            "active_control_wall_s": None if wall is None else round(wall, 6),
            "delivered_control_hz": None if hz is None else round(hz, 3),
            "simulated_elapsed_s": None if sim is None else round(sim, 6),
            "real_time_factor": (round(sim / wall, 4)
                                 if sim is not None and measured else None),
            "sim_time_regressions": self.sim_time_regressions,
            "control_interval_ms": {
                "count": len(intervals),
                "dropped": self.intervals_dropped,
                "p50": ms(_percentile(intervals, 0.50)),
                "p95": ms(_percentile(intervals, 0.95)),
                "p99": ms(_percentile(intervals, 0.99)),
                "max": ms(intervals[-1] if intervals else None),
            },
            "deadline_definition": (f"interval > {self.DEADLINE_PERIODS:g} target periods"
                                    if deadline_s is not None else None),
            "deadline_misses": misses,
        }


class SimClock:
    """Simulation-time delta for the algorithms, validated.

    ``step(sim_time_s)`` returns ``(dt, status)``:

    * ``"first"``   - no previous sample; ``dt`` is the configured step;
    * ``"ok"``      - ``0 <= dt <= max_gap_s``: the real simulated step
      (``0`` for a repeated frame - no time passed);
    * ``"gap"``     - a jump longer than ``max_gap_s`` (skipped world, reload):
      ``dt`` is the real jump; callers must not integrate it as one normal
      step and should reset rate estimates;
    * ``"invalid"`` - non-finite or backwards time: ``dt`` is ``nan`` so a
      consumer with a validation policy (the L3 state machine) applies it.
    """

    def __init__(self, configured_step_s: float, max_gap_s: float = 0.25) -> None:
        self.configured_step_s = float(configured_step_s)
        self.max_gap_s = float(max_gap_s)
        self._last: Optional[float] = None
        self.counts = {"first": 0, "ok": 0, "gap": 0, "invalid": 0}
        self.max_dt_s = 0.0

    def step(self, sim_time_s: Any):
        if not _finite(sim_time_s):
            self.counts["invalid"] += 1
            return math.nan, "invalid"
        t = float(sim_time_s)
        if self._last is None:
            self._last = t
            self.counts["first"] += 1
            return self.configured_step_s, "first"
        dt = t - self._last
        if dt < 0.0:
            self.counts["invalid"] += 1
            self._last = t
            return math.nan, "invalid"
        self._last = t
        self.max_dt_s = max(self.max_dt_s, dt)
        if dt > self.max_gap_s:
            self.counts["gap"] += 1
            return dt, "gap"
        self.counts["ok"] += 1
        return dt, "ok"

    def summary(self) -> Dict[str, Any]:
        return {"configured_step_s": self.configured_step_s,
                "max_gap_s": self.max_gap_s,
                "max_dt_s": round(self.max_dt_s, 6),
                "counts": dict(self.counts)}
