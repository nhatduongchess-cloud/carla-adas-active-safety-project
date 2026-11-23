"""Hợp đồng thuần Python cho một bước của máy trạng thái AEB."""

from dataclasses import dataclass
from typing import Literal, get_args

SafetyState = Literal["NORMAL", "FOLLOW", "EMERGENCY_BRAKE", "BRAKE_HOLD",
                      "BRAKE_HOLD_NO_DATA", "BRAKE_TO_STOP", "EVADE_LEFT", "EVADE_RIGHT"]
SafetyAction = Literal["DRIVE", "SLOW", "BRAKE", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"]
SAFETY_STATES = get_args(SafetyState)
SAFETY_ACTIONS = get_args(SafetyAction)


@dataclass(frozen=True)
class SafetyConfig:
    reaction_time_s: float = 1.0
    min_safe_dist_m: float = 6.0
    critical_ttc_s: float = 1.6
    warning_ttc_s: float = 3.0
    enable_evasion: bool = True
    evade_lookahead_m: float = 25.0
    follow_speed_diff: float = 60.0
    follow_min_lead_speed: float = 3.0
    creep_speed_thresh: float = 3.0
    evade_min_ttc: float = 2.0
    evade_commit_s: float = 1.5
    brake_exit_factor: float = 1.4
    clear_confirm_s: float = 0.1


@dataclass(frozen=True)
class SafetyInput:
    """Giá trị chưa làm tròn. None biểu thị khoảng cách/TTC chưa xác định."""

    dt_s: float
    ego_speed_ms: float
    nearest_distance_m: float | None
    ttc_s: float | None
    conservative_closing_ms: float
    lidar_valid: bool
    left_clear: bool
    right_clear: bool
    current_lane_id: int | None
    mu: float
    gap_multiplier: float


@dataclass(frozen=True)
class SafetyOutput:
    state: SafetyState = "NORMAL"
    action: SafetyAction = "DRIVE"
    brake: float = 0.0
    slow_pct: float = 0.0
    collision_risk: bool = False
