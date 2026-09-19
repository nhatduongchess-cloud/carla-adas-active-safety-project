"""Offline contract tests for non-normal safety decision tracing."""
from types import SimpleNamespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from modules.decision_trace import DecisionTrace


class DecisionTraceTests(unittest.TestCase):
    def record_context(self, **kwargs):
        trace = DecisionTrace(capacity=2)
        trace.record(
            frame_id=100, decision=SimpleNamespace(action="BRAKE", threat={}),
            ego_speed_ms=9.0, lidar_obstacles=[], tracks=[], fused_detections=[],
            **kwargs)
        return trace

    def test_radar_context_is_bounded_detached_and_preserves_effective_velocity(self):
        targets = [{"raw_velocity_ms": 25.24, "closing_speed_ms": 0.0,
                    "range_m": 70.0, "x_m": 70.5, "y_m": 0.1, "z_m": 0.8,
                    "distance_m": 70.5, "lateral_m": 0.1, "point_count": 2}]
        before = deepcopy(targets)
        path = [(0.0, 0.0), (38.5, 0.0)]
        trace = self.record_context(radar_targets=targets, path_points=path,
                                    lane_half_width_m=1.75,
                                    radar_velocity_validated=False)
        context = trace.events[0]["context"]
        self.assertEqual(targets, before)
        self.assertEqual(context["radar_candidates"][0]["raw_velocity_ms"], 25.24)
        self.assertEqual(context["radar_candidates"][0]["closing_speed_ms"], 0.0)
        self.assertFalse(context["radar_velocity_validated"])
        targets[0]["range_m"] = 0.0
        path[1] = (0.0, 0.0)
        self.assertEqual(context["radar_candidates"][0]["range_m"], 70.0)
        self.assertEqual(context["path_points_m"], [[0.0, 0.0], [38.5, 0.0]])
        huge = self.record_context(radar_targets=before * 1000,
                                  path_points=[(0.0, 0.0)] * 1000).events[0]["context"]
        self.assertEqual(len(huge["radar_candidates"]), 64)
        self.assertEqual(huge["radar_candidates_omitted"], 936)
        self.assertEqual(len(huge["path_points_m"]), 64)
        self.assertEqual(huge["path_points_omitted"], 936)

    def test_sensor_clocks_remain_separate_and_missing_values_are_not_zero(self):
        readout = SimpleNamespace(
            frame_id=100, lidar_timestamp=2.5, capture_timestamp=500.0,
            camera_frame_id=96, camera_sensor_timestamp=2.4,
            camera_wall_timestamp=499.9, camera_fresh=False,
            radar_measurement=SimpleNamespace(frame=99, timestamp=2.475),
            radar_received_monotonic_s=500.01)
        trace = self.record_context(radar_targets=[], sensor_readout=readout,
                                    inference_frame_id=90, inference_timestamp=499.0,
                                    inference_age_ms=1000.0)
        context = trace.events[0]["context"]
        self.assertEqual(context["sensors"]["radar"]["frame_id"], 99)
        self.assertEqual(context["sensors"]["radar"]["simulation_timestamp_s"], 2.475)
        self.assertEqual(context["sensors"]["radar"]["received_monotonic_s"], 500.01)
        self.assertEqual(context["sensors"]["lidar"]["simulation_timestamp_s"], 2.5)
        self.assertEqual(context["loop_read_start_monotonic_s"], 500.0)
        self.assertIsNone(context["sensors"]["camera"]["received_monotonic_s"])
        self.assertEqual(context["sensors"]["camera"]["legacy_read_start_monotonic_s"], 499.9)
        self.assertEqual(context["inference"]["timestamp_monotonic_s"], 499.0)
        readout.radar_measurement = None
        readout.radar_received_monotonic_s = None
        missing = self.record_context(radar_targets=[], sensor_readout=readout).events[0]["context"]
        self.assertIsNone(missing["sensors"]["radar"]["frame_id"])
        self.assertIsNone(missing["sensors"]["radar"]["received_monotonic_s"])

    def test_ego_pose_and_processor_calibration_are_snapshot_values(self):
        transform = SimpleNamespace(location=SimpleNamespace(x=1., y=2., z=3.),
                                    rotation=SimpleNamespace(pitch=4., yaw=5., roll=6.))
        velocity = SimpleNamespace(x=7., y=8., z=9.)
        processor = SimpleNamespace(sensor_x_m=2., reference_x_m=1.5, velocity_sign=-1.)
        context = self.record_context(
            radar_targets=[], ego_transform=transform, ego_velocity=velocity,
            radar_processor=processor, lane_source="map").events[0]["context"]
        self.assertEqual(context["ego"]["location_world_m"], [1., 2., 3.])
        self.assertEqual(context["ego"]["velocity_world_ms"], [7., 8., 9.])
        self.assertEqual(context["radar_processor"]["reference_x_m"], 1.5)
        self.assertEqual(context["radar_processor"]["velocity_sign"], -1.)

    def test_nonfinite_diagnostic_values_are_null_and_json_strict(self):
        for bad in (float("nan"), float("inf"), -float("inf"), 10**1000):
            with self.subTest(value=type(bad).__name__):
                context = self.record_context(
                    radar_targets=[{"raw_velocity_ms": bad}],
                    inference_age_ms=bad).events[0]["context"]
                self.assertIsNone(context["radar_candidates"][0]["raw_velocity_ms"])
                json.dumps(context, allow_nan=False)

    def test_bad_context_cannot_interrupt_safety_and_drive_skips_it(self):
        trace = self.record_context(radar_targets=[], path_points=[None])
        self.assertIsNone(trace.events[0]["context"])
        self.assertEqual(trace.summary()["context_errors"], 1)
        trace.record(frame_id=101, decision=SimpleNamespace(action="DRIVE"),
                     ego_speed_ms=0., lidar_obstacles=[], radar_targets=[], tracks=[],
                     fused_detections=[], path_points=object())
        self.assertEqual(trace.summary()["context_errors"], 1)
        self.assertEqual(trace.summary()["action_counts"], {"BRAKE": 1, "DRIVE": 1})

    def test_real_safety_sequence_unchanged_and_trace_writes_strict_json(self):
        from modules.active_safety import ActiveSafetySystem
        baseline = ActiveSafetySystem(enable_evasion=False)
        observed = ActiveSafetySystem(enable_evasion=False)
        trace = DecisionTrace(capacity=2)
        for distance, closing in ((70., 0.), (12., 10.), (6., 5.), (80., 0.)):
            targets = [{"distance_m": distance, "lateral_m": 0.,
                        "closing_speed_ms": closing, "raw_velocity_ms": closing}]
            original = deepcopy(targets)
            expected = baseline.update(10., [], [], 0.025, radar_targets=targets)
            actual = observed.update(10., [], [], 0.025, radar_targets=targets)
            saved = deepcopy(actual)
            trace.record(frame_id=100, decision=actual, ego_speed_ms=10.,
                         lidar_obstacles=[], radar_targets=targets, tracks=[],
                         fused_detections=[], path_points=[(0., 0.), (40., 0.)])
            self.assertEqual(vars(actual), vars(expected))
            self.assertEqual(vars(actual), vars(saved))
            self.assertEqual(targets, original)
        self.assertGreater(trace.action_counts["BRAKE"], 0)
        with tempfile.TemporaryDirectory() as folder:
            saved = json.loads(Path(trace.write(str(Path(folder)/"trace.json"))).read_text())
        self.assertEqual(saved["summary"]["schema_version"], 2)
        self.assertEqual(saved["summary"]["context_errors"], 0)
        json.dumps(saved, allow_nan=False)

    def test_drive_is_counted_but_not_retained(self):
        trace = DecisionTrace(capacity=2)
        trace.record(
            frame_id=4, decision=SimpleNamespace(action="DRIVE"), ego_speed_ms=3.0,
            lidar_obstacles=[], radar_targets=[], tracks=[], fused_detections=[])
        self.assertEqual(trace.summary()["total_frames"], 1)
        self.assertEqual(trace.summary()["action_counts"], {"DRIVE": 1})
        self.assertEqual(trace.events, [])

    def test_brake_trace_preserves_source_and_geometry_context(self):
        trace = DecisionTrace(capacity=1)
        decision = SimpleNamespace(
            action="BRAKE", state="EMERGENCY_BRAKE", brake=1.0,
            collision_risk=True,
            threat={"source": "lidar", "distance_m": 7.25, "ttc_s": 1.2,
                    "closing_ms": 4.1, "label": "obstacle",
                    "lidar_distance_m": 7.25, "tracker_distance_m": None,
                    "radar_distance_m": 8.0, "radar_closing_ms": 2.0})
        trace.record(
            frame_id=9, decision=decision, ego_speed_ms=5.0,
            lidar_obstacles=[{"centroid": (7.25, 0.1, 0.4)}],
            radar_targets=[{"distance_m": 8.0}], tracks=[object()],
            fused_detections=[])
        event = trace.events[0]
        self.assertEqual(event["action"], "BRAKE")
        self.assertEqual(event["threat"]["source"], "lidar")
        self.assertEqual(event["threat"]["distance_m"], 7.25)
        self.assertEqual(event["inputs"]["lidar_nearest_forward_x_m"], 7.25)
        self.assertEqual(event["inputs"]["radar_target_count"], 1)

    def test_capacity_discards_oldest_intervention(self):
        trace = DecisionTrace(capacity=1)
        for frame_id in (10, 11):
            trace.record(
                frame_id=frame_id,
                decision=SimpleNamespace(action="SLOW", state="FOLLOW", brake=0.0,
                                         collision_risk=True, threat={}),
                ego_speed_ms=2.0, lidar_obstacles=[], radar_targets=[], tracks=[],
                fused_detections=[])
        self.assertEqual([event["frame_id"] for event in trace.events], [11])


if __name__ == "__main__":
    unittest.main()
