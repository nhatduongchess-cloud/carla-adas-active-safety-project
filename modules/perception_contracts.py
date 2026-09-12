"""Typed data contracts shared by the perception, safety, and reporting layers.

The contracts intentionally contain no CARLA or deep-learning imports.  They can
therefore be serialized for replay and exercised in the lightweight self-test.
All asynchronous products carry a CARLA frame id and timestamp.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple


Point2D = Tuple[float, float]


def _measurement_age_ms(timestamp: float, now_s: float) -> float:
    """Age in a shared clock domain; invalid/future clocks are unusable, not fresh."""
    try:
        captured, now = float(timestamp), float(now_s)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if not (math.isfinite(captured) and math.isfinite(now) and 0.0 <= captured <= now):
        return math.inf
    return (now - captured) * 1000.0


@dataclass
class SensorState:
    name: str
    enabled: bool = True
    available: bool = False
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    age_ms: Optional[float] = None
    consecutive_misses: int = 0
    error: Optional[str] = None


@dataclass
class SensorFrameBundle:
    frame_id: int
    timestamp: float
    rgb: Any = None
    lidar: Any = None
    radar: Any = None
    sensor_states: Dict[str, SensorState] = field(default_factory=dict)

    def available(self, sensor: str) -> bool:
        state = self.sensor_states.get(sensor)
        return bool(state and state.enabled and state.available)


@dataclass
class LaneEstimate:
    frame_id: int
    timestamp: float
    source: str
    confidence: float = 0.0
    lane_polylines_m: List[List[Point2D]] = field(default_factory=list)
    centerline_m: List[Point2D] = field(default_factory=list)
    offset_m: Optional[float] = None
    curvature_1pm: Optional[float] = None
    lane_width_m: Optional[float] = None
    valid: bool = False
    reason: Optional[str] = None

    def age_ms(self, now_s: float) -> float:
        return _measurement_age_ms(self.timestamp, now_s)


@dataclass
class FusedTrack:
    track_id: int
    frame_id: int
    timestamp: float
    class_name: str
    position_m: Point2D
    velocity_ms: Point2D
    covariance: List[List[float]]
    measurement_age_ms: float = 0.0
    sources: Tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass
class SceneEstimate:
    frame_id: int
    timestamp: float
    tracks: List[FusedTrack] = field(default_factory=list)
    lane: Optional[LaneEstimate] = None
    traffic_controls: Dict[str, Any] = field(default_factory=dict)
    sensor_health: Dict[str, SensorState] = field(default_factory=dict)
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    inference_age_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaggedInferenceResult:
    frame_id: int
    timestamp: float
    completed_at: float
    payload: Any
    latency_ms: float
    error: Optional[str] = None

    def age_ms(self, now_s: float) -> float:
        return _measurement_age_ms(self.timestamp, now_s)


class FrameAgeError(ValueError):
    """Admission failure with a stable telemetry reason; still a ValueError."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def validate_frame_age(frame_id: int, timestamp: float, current_frame_id: int,
                       now_s: float, max_age_ms: float = 150.0) -> None:
    """Reject invalid, future or stale products; both timestamps must share a clock."""
    try:
        limit_ms = float(max_age_ms)
    except (TypeError, ValueError, OverflowError) as error:
        raise FrameAgeError("invalid_limit", "max_age_ms must be finite and non-negative") from error
    if not math.isfinite(limit_ms) or limit_ms < 0.0:
        raise FrameAgeError("invalid_limit", "max_age_ms must be finite and non-negative")
    if int(frame_id) > int(current_frame_id):
        raise FrameAgeError("future_frame",
            f"async result frame {frame_id} is newer than current frame {current_frame_id}")
    age_ms = _measurement_age_ms(timestamp, now_s)
    if not math.isfinite(age_ms):
        raise FrameAgeError("invalid_clock", f"async result frame {frame_id} has invalid or future clock data")
    if age_ms > limit_ms:
        raise FrameAgeError("stale",
            f"async result frame {frame_id} is stale ({age_ms:.1f} ms > {max_age_ms:.1f} ms)")
