"""Behavior Planner using a Finite State Machine (FSM) architecture for ADAS.

SCOPE, AND WHAT THIS PLANNER DELIBERATELY DOES NOT DO.

This FSM decides *tactical comfort* behaviour only: follow the lane, follow a
lead vehicle, stop for a signal. It has no collision-avoidance state and no
emergency stop, and that absence is a design decision rather than an omission.

Collision handling lives in exactly one place, `modules/ego_control.py`, which
resolves AEB, the L3 minimal-risk manoeuvre, a latched control fault and this
planner in a fixed priority order. A safety override here would be a second
authority over the brake, and two authorities means the vehicle's behaviour
depends on which one is consulted first - the failure mode that a single
arbitration point exists to prevent.

The planner therefore ignores `collision_risk` entirely. It used to carry an
EMERGENCY_STOP state that was never reachable, because the only caller
(`ego_driving_stack.py`) passed `collision_risk=False` unconditionally; that
dead branch is removed rather than wired, so the code no longer suggests an
authority it does not have. See `docs/ARCHITECTURE.md` section 8.
"""

from enum import Enum
from typing import Dict, Any


class DrivingState(Enum):
    """Tactical comfort states. Safety states belong to ego_control.py."""
    LANE_FOLLOWING = 1
    FOLLOWING_VEHICLE = 2
    STOPPING_AT_LIGHT = 3


class BehaviorPlanner:
    """Manages tactical decision making and state transitions for the autonomous vehicle."""

    def __init__(self) -> None:
        self.current_state: DrivingState = DrivingState.LANE_FOLLOWING

    def update_state(self, perception_metrics: Dict[str, Any]) -> DrivingState:
        """Evaluates perception metrics and determines the next FSM state.

        Transition Table:
            - LANE_FOLLOWING -> FOLLOWING_VEHICLE: if lead vehicle distance < 15.0m
            - LANE_FOLLOWING -> STOPPING_AT_LIGHT: if traffic light is RED
            - FOLLOWING_VEHICLE -> LANE_FOLLOWING: if gap restored beyond 22.0m
            - STOPPING_AT_LIGHT -> LANE_FOLLOWING: if traffic light is GREEN

        There is no transition out of this table on collision risk. Braking is
        not this object's decision; see the module docstring.
        """
        nearest_distance = perception_metrics.get("nearest_distance", 99.0)
        traffic_light_state = perception_metrics.get("traffic_light", "Green")

        if self.current_state == DrivingState.LANE_FOLLOWING:
            if traffic_light_state == "Red" and nearest_distance < 25.0:
                self.current_state = DrivingState.STOPPING_AT_LIGHT
            elif nearest_distance < 15.0:
                self.current_state = DrivingState.FOLLOWING_VEHICLE

        elif self.current_state == DrivingState.FOLLOWING_VEHICLE:
            if nearest_distance > 22.0:
                self.current_state = DrivingState.LANE_FOLLOWING
            elif traffic_light_state == "Red":
                self.current_state = DrivingState.STOPPING_AT_LIGHT

        elif self.current_state == DrivingState.STOPPING_AT_LIGHT:
            if traffic_light_state == "Green":
                self.current_state = DrivingState.LANE_FOLLOWING

        return self.current_state
