"""Visualization module for rendering HUD, bounding boxes, and perception dashboard."""

import cv2
import numpy as np


class Visualizer:
    """Renders real-time telemetry dashboard and sensor overlays using OpenCV."""

    @staticmethod
    def draw_hud(frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draws telemetry HUD and performance indicators on the camera frame."""
        canvas_height, canvas_width = frame.shape[:2]

        # Draw top status banner
        cv2.putText(
            frame, f"EGO SPEED: {metrics.get('speed', 0.0):.1f} km/h",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        state = metrics.get('fsm_state', 'NORMAL')
        state_color = (0, 0, 255) if 'BRAKE' in state else (
            (0, 165, 255) if state not in ('NORMAL', 'AUTOPILOT RUNNING') else (0, 255, 255))
        cv2.putText(
            frame, f"FSM STATE: {state}",
            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2
        )

        # Dòng mối đe dọa: khoảng cách + TTC vật gần nhất trong làn
        threat = metrics.get("threat") or {}
        if threat.get("distance_m") is not None:
            ttc = threat.get("ttc_s")
            ttc_txt = f"{ttc:.1f}s" if ttc is not None else "--"
            cv2.putText(
                frame,
                f"THREAT: {threat.get('label', 'obstacle')} @ {threat['distance_m']:.1f}m  TTC:{ttc_txt}",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
            )

        if metrics.get("collision_risk", False):
            cv2.rectangle(frame, (50, canvas_height - 120),
                          (canvas_width - 50, canvas_height - 40), (0, 0, 255), -1)
            cv2.putText(
                frame, "WARNING: COLLISION RISK DETECTED!",
                (80, canvas_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3
            )

        return frame
