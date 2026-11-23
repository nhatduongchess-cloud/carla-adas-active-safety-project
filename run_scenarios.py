# fmt: off
# isort: skip_file
"""
Chạy bộ 15 kịch bản (port từ ScenarioRunner) qua pipeline của bạn để kiểm thử
AEB / tránh né / phanh khẩn cấp, rồi xuất báo cáo logs/scenario_test_report.json.

Cần CARLA server đang mở. Ví dụ:
    .venvCarLa\\Scripts\\python.exe run_scenarios.py
    .venvCarLa\\Scripts\\python.exe run_scenarios.py --scenarios HardBrake,DynamicObjectCrossing --seconds 18
    .venvCarLa\\Scripts\\python.exe run_scenarios.py --category crossing
"""
import os
import sys
import queue
import argparse
import tempfile
import time
import traceback
import numpy as np
import carla

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (CURRENT_DIR, os.path.join(CURRENT_DIR, "modules")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg
from object_tracking import EnsembleVehicleTracker
from lidar_processor import LidarProcessor
from sensor_fusion import SensorFusion
from active_safety import ActiveSafetySystem
from safety_replay import SafetyReplayWriter, finish_capture
from mot_tracker import MultiObjectTracker
from odd_monitor import ODDMonitor
from mrm_controller import L3StateMachine
from weather_model import estimate_conditions
from collision_sensor import CollisionSensor
from kpi import KpiRecorder
from ego_control import EgoController, spawn_ego_safe, destroy_all_actors
from l3_report import case_id, exit_code_for, write_report
from scenario_acceptance import (REACTION_METRIC_VERSION, assess_scenario,
                                 finalize_after_cleanup, reaction_timing)
from fault_acceptance import assess_fault_behavior
from weather_acceptance import assess_weather_behavior
from sensor_sync import (retrieve_exact_frame, SensorSyncStats, warmup_sensor_streams,
                         OptionalFrameReader)
from sensor_setup import (configure_camera_blueprint, configure_lidar_blueprint,
                          configure_radar_blueprint)
from radar_processor import RadarProcessor
from sensor_health import SensorHealthMonitor
from inference_scheduler import LatestFrameScheduler
from lane_source import map_lane_estimate
from pedestrian_response import PedestrianResponse, walker_position
from pipeline_metrics import PipelineMetrics
from weather_config import load_weather_profiles, weather_kwargs
from ego_motion import EgoMotionEstimator
from radar_crossing import radar_reference_pose
from scene_semantics import summarize_traffic_controls
from simulation_guard import (force_traffic_lights_green, restore_traffic_lights,
                              get_or_load_world)
from runtime_config import configure_inference_device
from provenance import carla_versions, git_state, runtime_manifest
import scenario_library as lib

REPORT_PATH = "logs/scenario_test_report.json"

# Bộ nghiệm thu LÕI (curated) — cố định, có kiểm soát, đại diện đủ nhóm rủi ro
# (bám xe / phanh gấp / vật tĩnh / người băng qua / công trường / cắt làn). Báo cáo
# mặc định chạy đúng bộ này để KẾT QUẢ ỔN ĐỊNH, TÁI LẬP — thay vì đổ toàn bộ 15
# kịch bản một cách tùy hứng. Dùng '--scenarios all' nếu muốn chạy hết.
CORE_SUITE = [
    "FollowLeadingVehicle",
    "HardBrake",
    "StationaryObjectCrossing",
    "DynamicObjectCrossing",
    "ConstructionObstacle",
    "CutInFrom_left_Lane",
]
MAX_REPORTED_SCENARIOS = 8   # trần số kịch bản ghi vào 1 báo cáo (giữ báo cáo gọn)


def _rgb(img):
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, :3].copy()


def _pc(meas):
    p = np.frombuffer(meas.raw_data, dtype=np.float32)
    return np.reshape(p, (int(p.shape[0] / 4), 4)).copy()


def run_one(client, world, bp_lib, spec, shared, args, *, replay_dir=None, provenance=None):
    object_tracker, lidar_processor, sensor_fusion = shared
    dt = cfg.FIXED_DELTA
    actors = []
    camera = lidar = radar = collision = None
    scenario = None
    inference_scheduler = None
    focused = args.scenarios == "pedestrian"
    trace_crossing = replay_dir and (spec.name == "DynamicObjectCrossing" or spec.category == "sidewalk")
    crossing = PedestrianResponse() if focused or trace_crossing else None
    lane_scheduler = None
    safety_capture = None
    light_snapshot = []

    safety = ActiveSafetySystem(
        lane_half_width_m=cfg.LANE_HALF_WIDTH_M, min_cluster_points=cfg.MIN_CLUSTER_POINTS,
        reaction_time_s=cfg.REACTION_TIME_S, min_safe_dist_m=cfg.MIN_SAFE_DIST_M,
        critical_ttc_s=cfg.CRITICAL_TTC_S, warning_ttc_s=cfg.WARNING_TTC_S,
        min_closing_speed=cfg.MIN_CLOSING_SPEED, enable_evasion=cfg.ENABLE_EVASION,
        evade_lookahead_m=cfg.EVADE_LOOKAHEAD_M, lead_slow_ratio=cfg.LEAD_SLOW_RATIO,
        follow_speed_diff=cfg.TM_FOLLOW_SPEED_DIFF,
        vru_lateral_margin_m=cfg.VRU_LATERAL_MARGIN_M,
        vru_prediction_horizon_s=cfg.VRU_PREDICTION_HORIZON_S,
        evade_commit_s=cfg.EVADE_COMMIT_S,
        brake_exit_factor=cfg.BRAKE_EXIT_FACTOR)
    mot = MultiObjectTracker(dt=dt)
    ego_motion_tracker = EgoMotionEstimator()
    l3_sm = L3StateMachine(tor_window_s=cfg.TOR_WINDOW_S, mrm_decel_frac=cfg.MRM_DECEL_FRAC)
    kpi = KpiRecorder(dt=dt, scenario=spec.name)
    weather_profiles = load_weather_profiles(cfg.WEATHER_CONFIG_PATH)
    profile = weather_profiles.get(args.weather, {})
    conditions = estimate_conditions(
        precipitation=profile.get('precipitation', 0.0),
        fog_density=profile.get('fog_density', 0.0),
        wetness=profile.get('wetness', 0.0),
        precipitation_deposits=profile.get('precipitation_deposits', 0.0))
    odd_monitor = ODDMonitor()
    odd = odd_monitor.classify(conditions)
    safety.set_conditions(conditions['mu'], odd['gap_multiplier'])
    radar_processor = RadarProcessor(
        cfg.RADAR_X, cfg.RADAR_Y, cfg.RADAR_Z, reference_x_m=cfg.LIDAR_X,
        velocity_sign=cfg.RADAR_VELOCITY_SIGN, max_range_m=cfg.RADAR_RANGE_M,
        min_target_height_m=cfg.RADAR_MIN_TARGET_HEIGHT_M)
    radar_velocity_validated = (
        not cfg.RADAR_REQUIRE_SIGN_VALIDATION
        or RadarProcessor.velocity_validation_passed(
            cfg.RADAR_SIGN_VALIDATION_PATH, cfg.RADAR_VELOCITY_SIGN))
    health = SensorHealthMonitor(
        {"camera": True, "lidar": True, "radar": cfg.ENABLE_RADAR},
        max_range_misses=cfg.RADAR_MAX_MISSING_FRAMES)

    def _neural_worker(payload, inference_frame_id):
        image, same_frame_point_cloud = payload
        _, worker_detections, worker_metrics = object_tracker.process(
            image, None, inference_frame_id)
        same_frame_obstacles = lidar_processor.extract_perception_obstacles(
            same_frame_point_cloud)
        return {"detections": worker_detections,
                "fused": sensor_fusion.fuse(worker_detections, same_frame_obstacles),
                "metrics": worker_metrics}

    inference_scheduler = LatestFrameScheduler(_neural_worker, "scenario-gpu")
    stage_metrics = PipelineMetrics({"safety_loop": cfg.SAFETY_DEADLINE_MS,
                                     "perception": cfg.PERCEPTION_DEADLINE_MS})

    row = None
    reacted = braked = evaded = False
    odd_violation_seen = False
    mrm_seen = False
    odd_states_seen = set()
    first_reaction_frame = None
    first_reaction_after_hazard_frame = None
    ttc_at_reaction = None
    distance_at_reaction = None
    source_at_reaction = None
    sync_stats = SensorSyncStats()
    last_control_status = {}
    warmup_steps = max(0, int(1.5 * cfg.FPS) - 1)
    scenario_steps = int(args.seconds * cfg.FPS)

    try:
        if focused:
            from neural_perception import build_lane_perception
            lane_adapter, lane_arbiter, lane_error = build_lane_perception(cfg)
            if not lane_adapter.ready or not object_tracker.model_provenance:
                raise RuntimeError(f"pedestrian suite requires local YOLO and lane weights: "
                                   f"{lane_error or lane_adapter.provenance_error or object_tracker.provenance_errors}")

            def infer_lane(payload, frame_id):
                image, timestamp = payload
                return lane_adapter.infer(image, frame_id, timestamp)

            lane_scheduler = LatestFrameScheduler(infer_lane, "scenario-lane")
        world.set_weather(carla.WeatherParameters(**weather_kwargs(profile))
                          if profile else carla.WeatherParameters.ClearNoon)
        light_snapshot = force_traffic_lights_green(world)
        ego, _ = spawn_ego_safe(world, bp_lib.find('vehicle.tesla.model3'),
                                preferred_index=args.spawn)
        actors.append(ego)

        tm = client.get_trafficmanager(cfg.TM_PORT)
        tm.set_synchronous_mode(True)
        try:
            tm.set_random_device_seed(args.seed)   # autopilot TÁI LẬP được
        except Exception:
            pass
        tm.distance_to_leading_vehicle(ego, cfg.TM_LEADING_DISTANCE_M)
        ego.set_autopilot(True, tm.get_port())
        controller = EgoController(ego, tm, cfg, world=world)

        cam_bp = configure_camera_blueprint(bp_lib.find('sensor.camera.rgb'), cfg)
        lid_bp = configure_lidar_blueprint(bp_lib.find('sensor.lidar.ray_cast'), cfg)
        rad_bp = (configure_radar_blueprint(bp_lib.find('sensor.other.radar'), cfg)
                  if cfg.ENABLE_RADAR else None)
        camera = world.spawn_actor(cam_bp, carla.Transform(
            carla.Location(x=cfg.CAM_X, y=cfg.CAM_Y, z=cfg.CAM_Z),
            carla.Rotation(roll=cfg.CAM_ROLL, pitch=cfg.CAM_PITCH, yaw=cfg.CAM_YAW)), attach_to=ego)
        lidar = world.spawn_actor(lid_bp, carla.Transform(
            carla.Location(x=cfg.LIDAR_X, y=cfg.LIDAR_Y, z=cfg.LIDAR_Z),
            carla.Rotation(roll=cfg.LIDAR_ROLL, pitch=cfg.LIDAR_PITCH, yaw=cfg.LIDAR_YAW)), attach_to=ego)
        actors.extend([camera, lidar])
        if rad_bp is not None:
            radar = world.spawn_actor(rad_bp, carla.Transform(
                carla.Location(x=cfg.RADAR_X, y=cfg.RADAR_Y, z=cfg.RADAR_Z),
                carla.Rotation(roll=cfg.RADAR_ROLL, pitch=cfg.RADAR_PITCH,
                               yaw=cfg.RADAR_YAW)), attach_to=ego)
            actors.append(radar)
        collision = CollisionSensor(world, ego, fps=cfg.FPS)
        actors.append(collision.sensor)

        iq, lq, rq = queue.Queue(), queue.Queue(), queue.Queue()
        camera.listen(iq.put)
        lidar.listen(lq.put)
        if radar is not None:
            radar.listen(rq.put)
        radar_stats = SensorSyncStats()
        radar_reader = OptionalFrameReader(rq, "radar", radar_stats)

        # GPU sensor có thể chưa phát dữ liệu ở tick đầu. Bootstrap đến khi cả
        # camera/LiDAR cùng có đúng frame rồi mới bắt đầu tính KPI.
        warmup_sensor_streams(world, {"camera": iq, "lidar": lq})
        # Cho ego chạy vài tick rồi mới dựng vật cản (để có tốc độ).
        if replay_dir:
            safety_capture = SafetyReplayWriter(replay_dir, {
                "source": "carla-scenario", "scenario": spec.name,
                "town": world.get_map().name, "seed": args.seed, "weather": args.weather,
                "fault": args.fault, "fault_start_s": args.fault_start,
                "fault_duration_s": args.fault_duration,
                "warmup_steps": warmup_steps, "runtime_mode": "synchronous",
                "seconds_per_scenario": args.seconds, "provenance": provenance,
            })
            safety_capture.start_episode(spec.name, safety.fsm_config())
        for warm_index in range(warmup_steps):
            warm_frame = world.tick()
            retrieve_exact_frame(iq, warm_frame, sensor_name="camera", stats=sync_stats)
            retrieve_exact_frame(lq, warm_frame, sensor_name="lidar", stats=sync_stats)
            if cfg.ENABLE_RADAR:
                radar_reader.get(warm_frame, cfg.RADAR_OPTIONAL_TIMEOUT_S)
            warm_velocity = ego.get_velocity()
            warm_speed = (warm_velocity.x ** 2 + warm_velocity.y ** 2
                          + warm_velocity.z ** 2) ** 0.5
            warm_decision = safety.update(warm_speed, [], [], dt)
            warm_l3 = l3_sm.update(
                odd['state'], False, warm_speed, conditions['mu'], dt,
                critical=odd['critical'])
            warm_control = controller.apply(
                warm_decision, warm_l3, odd['state'], -warm_index - 1,
                target_speed_kmh=30.0, respect_traffic_controls=False)
            if safety_capture:
                safety_capture.record(safety, warm_decision, warm_l3, warm_control,
                                      frame_id=warm_frame, phase="warmup")
        scenario = spec.builder(world, ego, tm)
        last_detections = []
        last_fused = []
        last_inference_frame = None

        last_camera_loss_frame = -1
        last_lane_frame = None
        for frame in range(scenario_steps):
            w_frame = world.tick()
            pipeline_t0 = time.perf_counter()
            sensor_t0 = time.perf_counter()
            rgb = _rgb(retrieve_exact_frame(
                iq, w_frame, sensor_name="camera", stats=sync_stats))
            pc = _pc(retrieve_exact_frame(
                lq, w_frame, sensor_name="lidar", stats=sync_stats))
            radar_data = (radar_reader.get(w_frame, cfg.RADAR_OPTIONAL_TIMEOUT_S)
                          if cfg.ENABLE_RADAR else None)
            stage_metrics.record("sensor_sync", (time.perf_counter() - sensor_t0) * 1000.0)
            safety_t0 = time.perf_counter()

            # Kích hoạt actor trước perception/control của frame này để đo reaction
            # delay từ đúng thời điểm scenario bắt đầu, không phải từ đầu phiên.
            scenario.tick(frame, ego, world)

            fault_active = (args.fault != "none"
                            and frame * dt >= args.fault_start
                            and frame * dt < args.fault_start + args.fault_duration)
            camera_ok = not (fault_active and args.fault == "camera-loss")
            lidar_ok = not (fault_active and args.fault in ("lidar-loss", "lidar-radar-loss"))
            radar_ok = (radar_data is not None
                        and not (fault_active and args.fault in ("radar-loss", "lidar-radar-loss")))
            health.next_frame()
            capture_ts = time.monotonic()
            health.observe("camera", w_frame, capture_ts, camera_ok, now_s=capture_ts)
            health.observe("lidar", w_frame, capture_ts, lidar_ok, now_s=capture_ts)
            health.observe("radar", w_frame, capture_ts, radar_ok, now_s=capture_ts)

            geometry_t0 = time.perf_counter()
            obstacles = (lidar_processor.extract_safety_obstacles(pc) if lidar_ok else [])
            radar_targets = radar_processor.process(radar_data) if radar_ok else []
            radar_safety_targets = (radar_targets if radar_velocity_validated else [
                {**target, "closing_speed_ms": 0.0} for target in radar_targets])
            stage_metrics.record("safety_geometry",
                                 (time.perf_counter() - geometry_t0) * 1000.0)
            detect_frame = frame % cfg.DETECT_EVERY_N == 0
            if not camera_ok:
                last_camera_loss_frame = w_frame
            if lane_scheduler and camera_ok and frame % cfg.LANE_EVERY_N == 0:
                lane_scheduler.submit(w_frame, capture_ts, (rgb, capture_ts), only_if_idle=True)
            if detect_frame and camera_ok:
                # Fault injection must affect both the 40 Hz safety path and
                # the async perception-fusion path.  Otherwise a lidar-loss
                # test would silently keep using the real point cloud inside
                # the worker and produce a false PASS.
                perception_pc = (pc if lidar_ok
                                 else np.empty((0, 4), dtype=np.float32))
                inference_scheduler.submit(
                    w_frame, capture_ts, (rgb.copy(), perception_pc.copy()))
            inference_result = inference_scheduler.latest(
                w_frame, time.monotonic(), cfg.MAX_ASYNC_RESULT_AGE_MS)
            usable = (camera_ok and inference_result is not None and inference_result.error is None
                      and inference_result.frame_id > last_camera_loss_frame)
            fresh_inference = usable and inference_result.frame_id != last_inference_frame
            if fresh_inference:
                if focused:
                    inference_scheduler.record_consumed(inference_result, time.monotonic())
                last_inference_frame = inference_result.frame_id
                last_detections = inference_result.payload["detections"]
                last_fused = inference_result.payload["fused"]
            elif not usable:
                last_detections, last_fused = [], []
            _, fused = last_detections, last_fused
            inference_age_ms = (inference_result.age_ms(time.monotonic())
                                if inference_result and camera_ok else None)
            traffic_control = summarize_traffic_controls(
                fused, cfg.TRAFFIC_LIGHT_RANGE_M, cfg.STOP_SIGN_RANGE_M)
            vel = ego.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            decision_t0 = time.perf_counter()
            ego_transform = ego.get_transform()
            radar_pose = radar_reference_pose(ego_transform, cfg.LIDAR_X)
            ego_motion = ego_motion_tracker.update(ego_transform)
            tracks = mot.update(
                fused if fresh_inference else [], dt,
                ego_motion=ego_motion, ego_speed_ms=speed,
                radar_measurements=radar_targets, frame_id=w_frame,
                radar_velocity_validated=radar_velocity_validated)
            ego_path, lane_context = controller.path_context()
            control_path = ego_path
            if focused:
                now = time.monotonic()
                lane_result = lane_scheduler.latest(w_frame, now, cfg.MAX_ASYNC_RESULT_AGE_MS)
                candidate = (lane_result.payload if camera_ok and lane_result is not None
                             and lane_result.error is None and lane_result.frame_id > last_camera_loss_frame
                             else None)
                if candidate is not None and candidate.frame_id != last_lane_frame:
                    lane_scheduler.record_consumed(lane_result, now)
                    last_lane_frame = candidate.frame_id
                selected_lane = lane_arbiter.select(
                    candidate, map_lane_estimate(w_frame, capture_ts, ego_path),
                    at_junction=lane_context.get("is_junction", False),
                    now_s=now, max_age_ms=cfg.MAX_ASYNC_RESULT_AGE_MS)
                if selected_lane.source == "learned":
                    control_path = selected_lane.centerline_m
            decision = safety.update(
                speed, fused, obstacles, dt, tracks=tracks,
                path_points=control_path, lane_context=lane_context,
                radar_targets=radar_safety_targets,
                radar_pose=radar_pose,
                # An injected LiDAR loss produces [] obstacles; say it is no
                # data, so it cannot release a latched brake as a clear road.
                lidar_valid=lidar_ok)
            odd = odd_monitor.classify(conditions, health.summary())
            # Cùng lý do như chinh.py: ODD đổi thì vùng an toàn phải đổi theo.
            safety.set_conditions(conditions['mu'], odd['gap_multiplier'])
            odd_states_seen.add(odd['state'])
            l3 = l3_sm.update(odd['state'], False, speed, conditions['mu'], dt,
                              critical=odd['critical'])
            odd_violation_seen = odd_violation_seen or odd['state'] == "VIOLATION"
            mrm_seen = mrm_seen or l3['state'] in ("MRM_EXECUTING", "SAFE_STOP")
            stage_metrics.record("tracking_safety",
                                 (time.perf_counter() - decision_t0) * 1000.0)

            control_t0 = time.perf_counter()
            last_control_status = controller.apply(
                decision, l3, odd['state'], frame,
                target_speed_kmh=max(10.0, ego.get_speed_limit()),
                traffic_control=traffic_control, respect_traffic_controls=False)
            stage_metrics.record("control", (time.perf_counter() - control_t0) * 1000.0)
            pipeline_latency_ms = (time.perf_counter() - pipeline_t0) * 1000.0
            safety_latency_ms = (time.perf_counter() - safety_t0) * 1000.0
            stage_metrics.record("safety_loop", safety_latency_ms)
            observations = None
            if crossing is not None:
                if not focused:
                    candidate = None
                    selected_lane = map_lane_estimate(w_frame, capture_ts, ego_path)
                observations = crossing.observe(
                    frame, w_frame, scenario.hazard_frame, dt,
                    last_detections if fresh_inference else [], tracks,
                    inference_result.frame_id if fresh_inference else None,
                    selected_lane, decision, last_control_status)
                observations.update(
                    walker=walker_position(ego_transform, scenario.actors[0] if scenario.actors else None),
                    camera_available=camera_ok, lane_centerline_m=control_path,
                    radar_pose_world=radar_pose,
                    radar_xy_m=[(t["distance_m"], t["lateral_m"]) for t in radar_targets],
                    radar_closing_m_s=[t["closing_speed_ms"] for t in radar_safety_targets],
                    person_track_hits=[(t.id, t.hits, t.time_since_update) for t in mot.tracks
                                       if t.class_name.lower() == "person"],
                    fresh_inference=bool(fresh_inference),
                    lane_image_frame=candidate.frame_id if candidate else None,
                    lane_confidence=candidate.confidence if candidate else None,
                    lane_width_m=candidate.lane_width_m if candidate else None,
                    lane_candidate_reason=candidate.reason if candidate else None,
                    at_junction=lane_context.get("is_junction", False),
                    ttc_by_source_s=safety.last_ttc_by_source)
            if safety_capture:
                safety_capture.record(safety, decision, l3, last_control_status, frame_id=w_frame,
                                      observations=observations)
            if fresh_inference:
                stage_metrics.record("perception", inference_result.latency_ms)

            if decision.state != "NORMAL":
                reacted = True
                if scenario.triggered and first_reaction_frame is None:
                    first_reaction_frame = frame
                if (scenario.hazard_frame is not None and frame >= scenario.hazard_frame
                        and first_reaction_after_hazard_frame is None):
                    first_reaction_after_hazard_frame = frame
                    ttc_at_reaction = (decision.threat or {}).get('ttc_s')
                    distance_at_reaction = (decision.threat or {}).get('distance_m')
                    source_at_reaction = (decision.threat or {}).get('source')
            if decision.action == "BRAKE":
                braked = True
            if "EVADE" in decision.state:
                evaded = True
            if spec.category == "sidewalk" and not scenario.sidewalk_metrics["passed_walker"]:
                scenario.sidewalk_metrics["braked_on_approach"] |= decision.action == "BRAKE"
                scenario.sidewalk_metrics["evaded_on_approach"] |= "EVADE" in decision.state

            threat = decision.threat or {}
            kpi.add(t=frame * dt, speed_ms=speed, distance_m=threat.get('distance_m'),
                    ttc_s=threat.get('ttc_s'), state=decision.state,
                    collisions=collision.count, pipeline_latency_ms=pipeline_latency_ms,
                    safety_latency_ms=safety_latency_ms,
                    perception_latency_ms=(inference_result.latency_ms
                                           if fresh_inference else None),
                    inference_age_ms=inference_age_ms,
                    sensor_availability=health.summary()["availability"])

        summary = kpi.summary()
        os.makedirs("logs", exist_ok=True)
        kpi.write_csv(os.path.join(
            "logs", f"kpi_scn_{spec.name}_{args.weather}_seed{args.seed}_{args.fault}.csv"))

        collisions = summary["collisions"]
        # Surface-to-surface clearance, not centre distance: the old value was
        # centre-to-centre, which could not fall below the 0.25 m criterion for
        # a vehicle target. Centre distance is still reported, under its name.
        actor_clearance = scenario.min_clearance_m
        if not np.isfinite(actor_clearance):
            actor_clearance = None
        centre_distance = scenario.min_actor_distance_m
        if not np.isfinite(centre_distance):
            centre_distance = None
        # Reaction latency v2: measured to the first reaction at/after the
        # hazard, never clamped (see scenario_acceptance.reaction_timing).
        timing = reaction_timing(scenario.hazard_frame, first_reaction_frame,
                                 first_reaction_after_hazard_frame, dt)
        reaction_delay_s = timing["reaction_delay_s"]
        acceptance_kpi = dict(summary)
        acceptance_kpi["min_distance_m"] = actor_clearance
        scenario_errors = list(scenario.errors)
        if spec.category == "sidewalk":
            metrics = scenario.sidewalk_metrics
            moving = spec.name == "ParallelPedestrian"
            if (not metrics["passed_walker"] or (metrics["min_lateral_m"] or 0.) <= cfg.LANE_HALF_WIDTH_M
                    or (moving and metrics["travel_m"] < 1.) or (not moving and metrics["travel_m"] > .3)):
                scenario_errors.append(f"sidewalk control exposure incomplete: {metrics}")
        if scenario.error_count > len(scenario.errors):
            scenario_errors.append(f"... {scenario.error_count - len(scenario.errors)} more")
        if scenario.measurement_errors:
            scenario_errors.append(
                f"actor state unreadable on {scenario.measurement_errors} measurement(s); "
                "clearance/trigger distance incomplete")
        verdict = assess_scenario(
            spec.name, spec.category, acceptance_kpi,
            triggered=scenario.hazard_frame is not None,
            reacted=timing["reacted_to_hazard"],
            reaction_delay_s=reaction_delay_s,
            braked=(metrics["braked_on_approach"] if spec.category == "sidewalk" else braked),
            evaded=(metrics["evaded_on_approach"] if spec.category == "sidewalk" else evaded),
            sensor_frame_errors=sync_stats.frame_errors,
            scenario_errors=scenario_errors)
        passed, reasons = verdict["pass"], verdict["reasons"]
        status = verdict["status"]
        fault_verdict = assess_fault_behavior(
            args.fault, health.summary(), odd_violation_seen, mrm_seen)
        if not fault_verdict["pass"]:
            passed = False
            status = "FAIL"
            reasons = list(reasons) + fault_verdict["reasons"]
        # Weather and sensor-fault acceptance are orthogonal.  Confirmed loss
        # of both range sensors must cause ODD VIOLATION/MRM even in clear
        # weather, and must not be labeled as a false *weather* violation.
        health_summary = health.summary()
        external_violation_expected = bool(
            args.fault == "lidar-radar-loss"
            and health_summary.get("range_redundancy_ever_lost"))
        weather_verdict = assess_weather_behavior(
            profile.get("expect_odd"), odd_states_seen, mrm_seen,
            external_violation_expected=external_violation_expected)
        if not weather_verdict["pass"]:
            passed = False
            status = "FAIL"
            reasons = list(reasons) + weather_verdict["reasons"]

        if focused:
            verdict["criteria"]["post_hazard_aeb_command_required"] = True
            if scenario.hazard_frame is not None and crossing.events["aeb_command_sent"] is None:
                passed, status = False, "FAIL"
                reasons = list(reasons) + ["No AEB brake command sent at or after the crossing command"]

        print(f"  [{spec.name}] collisions={collisions} reacted={reacted} "
              f"aeb={braked} braked={braked} evaded={evaded} minTTC={summary['min_ttc_s']} "
              f"minDist={summary['min_distance_m']} -> {'PASS' if passed else 'FAIL'}")
        row = {
            "profile": spec.name, "name": spec.name, "category": spec.category,
            "tests": spec.tests, "collisions": collisions, "reacted": reacted,
            "braked": braked, "aeb_triggered": braked, "evaded": evaded,
            "min_distance_m": summary["min_distance_m"],
            "scenario_min_clearance_m": (round(actor_clearance, 2)
                                          if actor_clearance is not None else None),
            "clearance_basis": scenario.clearance_basis,
            "scenario_min_center_distance_m": (round(centre_distance, 2)
                                               if centre_distance is not None else None),
            "min_ttc_s": summary["min_ttc_s"],
            "max_decel_ms2": summary["max_decel_ms2"], "kpi": summary,
            "triggered": scenario.triggered,
            "trigger_frame": scenario.trigger_frame,
            "hazard_frame": scenario.hazard_frame,
            "first_reaction_frame": first_reaction_frame,
            "reaction_delay_s": reaction_delay_s,
            "reaction_metric_version": timing["reaction_metric_version"],
            "preemptive_response": timing["preemptive_response"],
            "first_reaction_minus_hazard_s": timing["first_reaction_minus_hazard_s"],
            "first_reaction_after_hazard_frame": first_reaction_after_hazard_frame,
            "reacted_any_time": reacted,
            "ttc_at_reaction_s": ttc_at_reaction,
            "distance_at_reaction_m": distance_at_reaction,
            "source_at_reaction": source_at_reaction,
            "sensor_sync": sync_stats.summary(),
            "sensor_frame_errors": sync_stats.frame_errors,
            "sensor_health": health.summary(),
            "radar_sync": radar_stats.summary(),
            "radar_availability": health.availability("radar"),
            "radar_velocity_validated": radar_velocity_validated,
            "weather": args.weather,
            "fault": args.fault,
            "fault_acceptance": fault_verdict,
            "weather_acceptance": weather_verdict,
            "odd_violation_seen": odd_violation_seen,
            "odd_states_seen": sorted(odd_states_seen),
            "mrm_seen": mrm_seen,
            "seed": args.seed,
            "last_control_status": last_control_status,
            "stage_metrics": stage_metrics.summary(),
            "criteria": verdict["criteria"],
            "status": status,
            "invalid_reasons": verdict["invalid_reasons"],
            "scenario_errors": scenario_errors,
            "pass": passed, "reasons": reasons,
        }
        if spec.category == "sidewalk":
            row["sidewalk_exposure"] = scenario.sidewalk_metrics
        if trace_crossing and not focused:
            row["events_after_actor_command"] = crossing.events
        if focused:
            row["pedestrian_response"] = {
                "events_after_crossing_command": crossing.events, "lane_ticks": crossing.lane_ticks,
                "lane_model_sha256": cfg.LEARNED_LANE_MODEL_SHA256}
        return row

    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"  [{spec.name}] LỖI: {reason}")
        traceback.print_exc()
        # An exception is an ERROR, not a failed ADAS case and not a collision:
        # collisions are None (unknown), never a sentinel like -1.
        row = {"profile": spec.name, "name": spec.name, "category": spec.category,
               "tests": spec.tests, "collisions": None, "pass": False,
               "status": "ERROR", "seed": args.seed, "weather": args.weather,
               "fault": args.fault, "reasons": [reason]}
        return row
    finally:
        # The verdict is final only after cleanup. Every failure below is
        # collected and attached to the returned row (same dict object), so a
        # teardown failure after a passing scenario fails the case instead of
        # being a console warning.
        cleanup_errors = []

        def _cleanup(what, action):
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(f"{what}: {type(exc).__name__}: {exc}")
                print(f"  [{spec.name}] Cleanup {what} failed: {exc}")

        _cleanup("restore traffic lights", lambda: restore_traffic_lights(light_snapshot))
        if inference_scheduler is not None:
            _cleanup("inference scheduler", inference_scheduler.close)
        if lane_scheduler is not None:
            _cleanup("lane scheduler", lane_scheduler.close)
        if row is not None and "pedestrian_response" in row:
            for key, scheduler in (("yolo_worker", inference_scheduler), ("lane_worker", lane_scheduler)):
                stats = scheduler.stats()
                row["pedestrian_response"][key] = stats
                if stats["thread_alive"]:
                    cleanup_errors.append(f"{key} still active after shutdown")
                if stats["failed"]:
                    row.update(status="ERROR", **{"pass": False})
                    row["reasons"].append(f"{key} failed on {stats['failed']} inference request(s)")
        if scenario is not None:
            _cleanup("scenario actors", lambda: scenario.destroy(client))
        for s in (camera, lidar, radar):
            if s is not None:
                _cleanup(f"stop {getattr(s, 'type_id', 'sensor')}",
                         lambda s=s: s.stop() if s.is_alive else None)
        if collision is not None:
            _cleanup("stop collision sensor", collision.stop)
        if actors:
            def _destroy_owned():
                responses = client.apply_batch_sync(
                    [carla.command.DestroyActor(a) for a in actors], False)
                failed = [r.error for r in responses if getattr(r, "error", None)]
                # Vài tick để CARLA thực sự hủy actor trước khi kịch bản sau spawn lại.
                for _ in range(4):
                    world.tick()
                if failed:
                    raise RuntimeError(f"{len(failed)} destroy command(s) failed: {failed[:3]}")
            _cleanup("destroy owned actors", _destroy_owned)
        if row is not None:
            finalize_after_cleanup(row, cleanup_errors)
        if safety_capture:
            capture = finish_capture(safety_capture, warmup_steps + scenario_steps,
                                     safety.steps == warmup_steps + scenario_steps)
            if row is not None:
                row["safety_replay"] = capture


def main():
    parser = argparse.ArgumentParser(description="Ported ScenarioRunner test suite")
    parser.add_argument('--scenarios', type=str, default='core',
                        help="'pedestrian' (learned lane and AEB trace) | 'pedestrian-controls' "
                             "(standing and parallel walkers) | 'core' (default) | 'all' | scenario names")
    parser.add_argument('--category', type=str, default=None,
                        help="lọc theo nhóm: lead|crossing|cutin|junction|oncoming")
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--spawn', type=int, default=0, help="chỉ số spawn point cho ego")
    parser.add_argument('--town', type=str, default=None,
                        help="nạp map benchmark, khuyến nghị Town02 cho RTX 4070")
    parser.add_argument('--seed', type=int, default=42,
                        help="hạt giống ngẫu nhiên -> kết quả TÁI LẬP")
    parser.add_argument('--seeds', type=str, default=None,
                        help="ma trận seed, ví dụ 42,1337,2026 (ghi đè --seed)")
    parser.add_argument('--weather', type=str, default='clear')
    parser.add_argument('--weathers', type=str, default=None,
                        help="ma trận weather, ví dụ clear,light_rain,heavy_rain,fog,storm")
    parser.add_argument('--fault', default='none', choices=[
        'none', 'camera-loss', 'radar-loss', 'lidar-loss', 'lidar-radar-loss'])
    parser.add_argument('--fault-start', type=float, default=5.0)
    parser.add_argument('--fault-duration', type=float, default=4.0)
    parser.add_argument('--limit', type=int, default=0,
                        help="trần scenario; 0 = không giới hạn")
    parser.add_argument('--report', type=str, default=None,
                        help="đường dẫn JSON report; dùng tên riêng để không ghi đè matrix khác")
    parser.add_argument('--safety-replay-out', type=str, default=None,
                        help="ghi replay quyết định AEB, một thư mục mới cho mỗi ca")
    parser.add_argument('--clean', action='store_true',
                        help="dọn mọi xe/người/cảm biến còn sót trước khi chạy")
    parser.add_argument('--inference-device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help="thiết bị suy luận; auto dùng CPU an toàn khi chạy CARLA trên Windows")
    args = parser.parse_args()
    if args.report is None:
        args.report = {"pedestrian": "logs/pedestrian_crossing_report.json",
                       "pedestrian-controls": "logs/pedestrian_controls_report.json"}.get(
                           args.scenarios, REPORT_PATH)
    if args.safety_replay_out:
        try:
            os.makedirs(args.safety_replay_out, exist_ok=False)
        except OSError as exc:
            parser.error(f"--safety-replay-out: {exc}")
    elif args.scenarios == "pedestrian":
        report_dir = os.path.dirname(os.path.abspath(args.report))
        os.makedirs(report_dir, exist_ok=True)
        args.safety_replay_out = tempfile.mkdtemp(prefix="pedestrian-trace-", dir=report_dir)

    # Resolve the device before constructing the shared tracker.  In ``auto``
    # mode this selects CPU on Windows to avoid CUDA/D3D11 contention with CARLA.
    device_info = configure_inference_device(cfg, args.inference_device,
                                              "async-stable")

    seeds = ([int(x.strip()) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else [args.seed])
    weathers = ([x.strip() for x in args.weathers.split(',') if x.strip()]
                if args.weathers else [args.weather])

    if args.scenarios == 'all':
        specs = list(lib.CATALOG)
    elif args.scenarios == 'pedestrian-controls':
        specs = lib.SIDEWALK_CONTROLS
    elif args.scenarios == 'pedestrian':
        specs = [s for s in lib.CATALOG if s.name == "DynamicObjectCrossing"]
    elif args.scenarios == 'core':
        order = {n: i for i, n in enumerate(CORE_SUITE)}
        specs = sorted([s for s in lib.CATALOG if s.name in order], key=lambda s: order[s.name])
    else:
        want = [x.strip() for x in args.scenarios.split(',')]
        specs = [s for s in lib.CATALOG if s.name in want]
    if args.category:
        specs = [s for s in specs if s.category == args.category]
    if not specs:
        sys.exit(f"[Scn] Không có kịch bản khớp. Có: {lib.list_names()}")
    # Giới hạn số kịch bản ghi vào báo cáo (giữ báo cáo gọn, có kiểm soát).
    if args.limit and args.limit > 0 and len(specs) > args.limit:
        print(f"[Scn] Giới hạn {len(specs)} -> {args.limit} kịch bản ghi báo cáo.")
        specs = specs[:args.limit]

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world, map_reloaded = get_or_load_world(client, args.town)
    if args.town:
        action = "đã nạp" if map_reloaded else "tái sử dụng"
        print(f"[Scn] {action} map {world.get_map().name}.")
    bp_lib = world.get_blueprint_library()

    if args.clean:
        n = destroy_all_actors(world, client)
        print(f"[Scn] --clean: đã dọn {n} actor còn sót.")

    print(f"[Scn] Nạp mô hình nhận thức (dùng chung). Sẽ chạy {len(specs)} kịch bản.")
    shared = (
        EnsembleVehicleTracker(device=cfg.YOLO_DEVICE, use_ensemble=cfg.USE_ENSEMBLE,
                               model_a=cfg.YOLO_MODEL_A, model_b=cfg.YOLO_MODEL_B,
                               imgsz=cfg.YOLO_IMGSZ, half=cfg.YOLO_HALF,
                               use_optimized=cfg.YOLO_USE_OPTIMIZED,
                               target_classes=cfg.DETECTED_CLASSES,
                               class_names=cfg.CLASS_NAMES,
                               conf_threshold=cfg.CONF_THRES,
                               nms_threshold=cfg.IOU_THRES,
                               traffic_light_model_path=cfg.TRAFFIC_LIGHT_MODEL_PATH),
        LidarProcessor(eps=0.6, min_samples=5, max_points=cfg.LIDAR_MAX_POINTS),
        SensorFusion(width=cfg.CAM_WIDTH, height=cfg.CAM_HEIGHT, fov=cfg.CAM_FOV,
                     lidar_y_sign=cfg.LIDAR_Y_SIGN, real_heights_m=cfg.REAL_HEIGHTS_M,
                     default_height_m=cfg.DEFAULT_HEIGHT_M,
                     depth_min_m=cfg.DEPTH_MIN_M, depth_max_m=cfg.DEPTH_MAX_M,
                     camera_pose=(cfg.CAM_X, cfg.CAM_Y, cfg.CAM_Z,
                                  cfg.CAM_ROLL, cfg.CAM_PITCH, cfg.CAM_YAW),
                     lidar_pose=(cfg.LIDAR_X, cfg.LIDAR_Y, cfg.LIDAR_Z,
                                 cfg.LIDAR_ROLL, cfg.LIDAR_PITCH, cfg.LIDAR_YAW)),
    )
    shared[0].warmup(cfg.CAM_WIDTH, cfg.CAM_HEIGHT, runs=2)

    original = world.get_settings()
    sync = world.get_settings()
    sync.synchronous_mode = True
    sync.fixed_delta_seconds = cfg.FIXED_DELTA
    try:
        world.apply_settings(sync, 10.0)
    except RuntimeError as apply_error:
        # CARLA can acknowledge the server-side transition but time out on the
        # client RPC.  A bounded tick lets the game thread make progress before
        # retrying; if the server is really gone, restore best-effort and expose
        # a clear error instead of entering the scenario loop half-synchronized.
        try:
            world.tick(10.0)
            world.apply_settings(sync, 10.0)
        except Exception as recovery_error:
            try:
                world.apply_settings(original, 10.0)
            except Exception:
                pass
            raise RuntimeError(
                "CARLA synchronous-mode setup failed after recovery tick: "
                f"apply={apply_error}; recovery={recovery_error}") from apply_error

    # The planned matrix is fixed BEFORE anything runs, so a case that never
    # ran is reported as NOT_RUN rather than silently missing.
    planned_case_ids = [case_id({"name": spec.name, "seed": run_seed,
                                 "weather": run_weather, "fault": args.fault})
                        for run_seed in seeds for run_weather in weathers for spec in specs]
    # Provenance, captured once before the matrix. Unknown fields stay None
    # with a reason; nothing is filled in from the current machine later.
    provenance = runtime_manifest(
        detector=shared[0], carla_version=carla_versions(client))
    provenance.update(git_state(os.path.dirname(os.path.abspath(__file__))))
    try:
        applied = world.get_settings()      # read back, not what was requested
        provenance["world_settings_read_back"] = {
            "synchronous_mode": bool(applied.synchronous_mode),
            "fixed_delta_seconds": applied.fixed_delta_seconds,
            "substepping": bool(getattr(applied, "substepping", False)),
            "max_substep_delta_time": getattr(applied, "max_substep_delta_time", None),
            "max_substeps": getattr(applied, "max_substeps", None)}
    except Exception as exc:
        provenance["world_settings_read_back"] = {"unknown_reason": f"{type(exc).__name__}: {exc}"}
    meta = {
        "provenance": provenance,
        "seeds": seeds,
        "weathers": weathers,
        "fault": args.fault,
        "inference_device": device_info["effective"],
        "inference_device_requested": device_info["requested"],
        "inference_device_reason": device_info["reason"],
        "town": args.town or world.get_map().name,
        "core_scenarios": CORE_SUITE,
        "seconds_per_scenario": args.seconds,
        "fps": cfg.FPS,
        "suite": args.scenarios,
        # The 45-case catalog keeps the project's 18/18 core + 43/45 catalog
        # policy; every other matrix must pass every planned case.
        "suite_policy": ("core_plus_catalog" if args.scenarios == "all" and not args.category
                         and not args.limit else "all_planned_cases"),
        "planned_case_ids": planned_case_ids,
        "reaction_metric_version": REACTION_METRIC_VERSION,
        "clearance_metric": "surface-to-surface 2D footprint (modules/clearance.py)",
        "thresholds": {
            "critical_ttc_s": cfg.CRITICAL_TTC_S,
            "warning_ttc_s": cfg.WARNING_TTC_S,
            "min_safe_dist_m": cfg.MIN_SAFE_DIST_M,
            "lane_half_width_m": cfg.LANE_HALF_WIDTH_M,
            "evade_lookahead_m": cfg.EVADE_LOOKAHEAD_M,
        },
        "acceptance_criteria": {
            "scenario_triggered": "== true",
            "collisions": "== 0",
            "reacted": "== true at or after the hazard frame",
            "reaction_delay_s": "<= 1.0 (metric v2: first reaction at/after hazard, not clamped)",
            "min_clearance_m": ">= 0.25 when measurable (surface-to-surface)",
            "sensor_frame_errors": "== 0",
            "cutin": "must command BRAKE or EVADE",
            "cleanup": "verified per case",
        },
    }
    if args.scenarios == 'pedestrian-controls':
        criteria = meta["acceptance_criteria"]
        criteria.pop("reacted")
        criteria.pop("reaction_delay_s")
        criteria["sidewalk"] = "ego passes walker outside corridor without braking or evasion"
    title = "CARLA ScenarioRunner Port — AEB/Avoidance Test Report (simulation)"

    results = []
    matrix_error = None
    try:
        for run_seed in seeds:
            args.seed = run_seed
            np.random.seed(run_seed)
            for run_weather in weathers:
                args.weather = run_weather
                for spec in specs:
                    print(f"[Scn] === {spec.name} ({spec.category}) | "
                          f"seed={run_seed} weather={run_weather} fault={args.fault} ===")
                    replay_dir = (os.path.join(args.safety_replay_out, f"case-{len(results) + 1:03d}")
                                  if args.safety_replay_out else None)
                    results.append(run_one(client, world, bp_lib, spec, shared, args,
                                           replay_dir=replay_dir, provenance=provenance))
                    # Checkpoint after every case (atomic replace): if the
                    # server dies, the finished cases survive and the rest
                    # show up as NOT_RUN.
                    write_report(args.report, results, title=title,
                                 meta=dict(meta, partial=True))
    except Exception as exc:
        # Stop the matrix in a controlled way. The finished cases are kept;
        # nothing is retried and no remaining case is counted as passed.
        matrix_error = f"{type(exc).__name__}: {exc}"
        print(f"[Scn] Matrix stopped: {matrix_error}")
        traceback.print_exc()
    finally:
        restore_errors = []
        try:
            client.get_trafficmanager(cfg.TM_PORT).set_synchronous_mode(False)
        except Exception as exc:
            restore_errors.append(f"traffic manager sync off: {exc}")
        try:
            world.apply_settings(original)
        except Exception as exc:
            restore_errors.append(f"restore world settings: {exc}")

    final_meta = dict(meta, partial=matrix_error is not None,
                      matrix_error=matrix_error, suite_restore_errors=restore_errors)
    try:
        report = write_report(args.report, results, title=title, meta=final_meta)
    except Exception as exc:
        print(f"[Scn] REPORT NOT WRITTEN ({args.report}): {type(exc).__name__}: {exc}")
        return 4
    code = exit_code_for(report)
    if args.safety_replay_out and any(
        row.get("safety_replay", {}).get("status") != "COMPLETE" for row in results
    ):
        code = max(code, 2)
        print("[Scn] Safety replay incomplete or unavailable. See the per-case report.")
    if matrix_error is not None or restore_errors:
        code = max(code, 2)
    suite = report["suite"]
    print("\n============== SCENARIO TEST SUMMARY ==============")
    print(f"  Planned: {suite['planned']} | attempted: {suite['attempted']} | "
          f"PASS: {suite['passed']} | FAIL: {suite['failed']} | INVALID: {suite['invalid']} "
          f"| ERROR: {suite['error']} | NOT_RUN: {suite['not_run']}")
    print(f"  Gate ({report['acceptance_gate']['policy']}): "
          f"{report['acceptance_gate']['status']} | exit code {code}")
    print(f"  Báo cáo: {args.report}")
    print("==================================================")
    return code


if __name__ == '__main__':
    # Exit 0 only when the suite's acceptance gate PASSED. 1 FAIL, 2 INVALID
    # (or the matrix/restore broke), 3 NOT_EVALUATED, 4 report not written.
    sys.exit(main())
