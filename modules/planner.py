"""Behavior Planner using a Finite State Machine (FSM) architecture for ADAS."""

from enum import Enum
from typing import Dict, Any


class DrivingState(Enum):
    """Enumeration of all supported behavioral states."""
    LANE_FOLLOWING = 1
    FOLLOWING_VEHICLE = 2
    STOPPING_AT_LIGHT = 3
    OBSTACLE_AVOIDANCE = 4
    EMERGENCY_STOP = 5


class BehaviorPlanner:
    """Manages tactical decision making and state transitions for the autonomous vehicle."""

    def __init__(self) -> None:
        self.current_state: DrivingState = DrivingState.LANE_FOLLOWING

    def update_state(self, perception_metrics: Dict[str, Any]) -> DrivingState:
        """Evaluates perception metrics and determines the next FSM state.

        Transition Table:
            - LANE_FOLLOWING -> FOLLOWING_VEHICLE: if lead vehicle distance < 15.0m
            - LANE_FOLLOWING -> STOPPING_AT_LIGHT: if traffic light is RED
            - ANY_STATE -> EMERGENCY_STOP: if collision_risk or TTC < 1.5s
        """
        collision_risk = perception_metrics.get("collision_risk", False)
        nearest_distance = perception_metrics.get("nearest_distance", 99.0)
        traffic_light_state = perception_metrics.get("traffic_light", "Green")

        # Highest priority safety override
        if collision_risk:
            self.current_state = DrivingState.EMERGENCY_STOP
            return self.current_state

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

        elif self.current_state == DrivingState.EMERGENCY_STOP:
            if not collision_risk and nearest_distance > 10.0:
                self.current_state = DrivingState.LANE_FOLLOWING

        return self.current_state
