"""Per-episode observations for the pedestrian-crossing validation run."""

import math


def walker_position(ego_transform, actor):
    """Ground truth for evaluation only, in metres forward/right of the ego."""
    if actor is None:
        return None
    location = actor.get_location()
    dx = location.x - ego_transform.location.x
    dy = location.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    return {"actor_id": actor.id, "forward_m": dx * math.cos(yaw) + dy * math.sin(yaw),
            "right_m": -dx * math.sin(yaw) + dy * math.cos(yaw)}


class PedestrianResponse:
    def __init__(self):
        self.events = dict.fromkeys(("yolo_person", "confirmed_person_track", "safety_response",
                                     "aeb_request", "aeb_command_sent"))
        self.lane_ticks = {"learned": 0, "map": 0}

    def observe(self, frame, world_frame, hazard_frame, dt, detections, tracks, image_frame,
                lane, decision, control):
        self.lane_ticks[lane.source] += 1
        # Only new detector products are supplied here. The observation frame
        # and the source camera frame may differ under asynchronous inference.
        persons = [{"confidence": d["confidence"], "bbox": d["bbox"]}
                   for d in detections if d["class"].lower() == "person"]
        person_tracks = [{"id": t.id, "position_m": t.pos, "velocity_m_s": t.vel,
                          "ego_compensated": t.ego_compensated}
                         for t in tracks if t.class_name.lower() == "person"]
        aeb_requested = decision.action == "BRAKE" and decision.brake > 0
        aeb_sent = (aeb_requested and control.get("command_sent") is True
                    and control.get("applied", {}).get("brake", 0) > 0
                    and (control.get("mode") == "aeb_override"
                         or control.get("brake_source") == "aeb"))
        if hazard_frame is not None and frame >= hazard_frame:
            events = {"yolo_person": bool(persons), "confirmed_person_track": bool(person_tracks),
                      "safety_response": decision.state != "NORMAL", "aeb_request": aeb_requested,
                      "aeb_command_sent": aeb_sent}
            for name, happened in events.items():
                if happened and self.events[name] is None:
                    self.events[name] = {"scenario_frame": frame, "world_frame": world_frame,
                                         "delay_s": round((frame - hazard_frame) * dt, 6)}
                    if name == "yolo_person":
                        self.events[name]["image_frame"] = image_frame
        return {"scenario_frame": frame, "hazard_frame": hazard_frame, "image_frame": image_frame,
                "yolo_persons": persons, "confirmed_person_tracks": person_tracks,
                "lane_source": lane.source, "lane_reason": lane.reason,
                "nearest_distance_source": decision.threat.get("source")}
