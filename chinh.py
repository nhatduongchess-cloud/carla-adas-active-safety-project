# fmt: off
# isort: skip_file

"""
Hệ thống điều phối trung tâm (Main Orchestrator) cho CARLA ADAS Pipeline.
Kiến trúc: custom ego control + lớp an toàn ghi đè (AEB + tránh né); Traffic
Manager chỉ điều phối NPC, hoặc ego khi người dùng opt-in rõ ràng qua config.

Luồng mỗi khung hình:
    tick -> RGB + LiDAR -> phát hiện (YOLO ensemble) -> gộp khoảng cách (LiDAR)
    -> quyết định an toàn (máy trạng thái) -> áp lệnh (phanh / giảm tốc / chuyển
       làn) -> HUD + telemetry.
"""
import time
import traceback
import os
import sys
from contextlib import ExitStack, nullcontext
import cv2
import carla

# Console Windows có thể mặc định cp1252; bảo đảm log tiếng Việt không làm tiến
# trình chết trước khi kết nối CARLA.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("\n[System] Khởi chạy Pipeline ADAS...")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Tự động dò tìm thư mục chứa các module nhận thức
target_module_dir = CURRENT_DIR
for root, dirs, files in os.walk(CURRENT_DIR):
    if "midas_estimator.py" in files:
        target_module_dir = root
        break

if target_module_dir not in sys.path:
    sys.path.insert(0, target_module_dir)

import config as cfg

# ==============================================================================
# KHAI BÁO CÁC MODULE XỬ LÝ (PERCEPTION & SAFETY)
# ==============================================================================
try:
    from object_tracking import draw_detections
    from active_safety import ActiveSafetySystem
    from mot_tracker import MultiObjectTracker
    from odd_monitor import ODDMonitor
    from mrm_controller import L3StateMachine
    from weather_model import estimate_conditions
    from weather_config import load_weather_profiles, weather_kwargs
    from ego_control import EgoController, spawn_ego_safe
    from rl_speed_controller import RLSpeedController
    from turn_intent import TurnIntentPlanner
    from data_logger import TelemetryLogger
    from traffic_spawner import TrafficSpawner
    from dashboard_view import DashboardView
    from radar_processor import RadarProcessor
    from sensor_health import SensorHealthMonitor
    from pipeline_metrics import PipelineMetrics
    from lane_source import map_lane_estimate
    from perception_contracts import SensorFrameBundle
    from replay_io import ReplayWriter
    from ego_motion import EgoMotionEstimator
    from scene_semantics import summarize_traffic_controls
    from simulation_guard import (SynchronousWorldSession,
                                  get_or_load_world)
    from runtime_config import (build_argument_parser, configure_inference_device,
                                configure_runtime)
    from sensor_runtime import SensorRig
    from neural_perception import NeuralPerceptionRuntime
    from runtime_report import RuntimeReportData, write_runtime_report
    from runtime_cleanup import cleanup_runtime
    from decision_trace import DecisionTrace
    from probe_journal import RuntimeCrashJournal
    print("[System] ✅ Đã nạp thành công các module Nhận thức và An toàn!")
except ModuleNotFoundError as e:
    print(f"[System] ❌ Lỗi nạp module: {e}. Hãy kiểm tra lại thư mục!")
    sys.exit(1)


# ==============================================================================
# ORCHESTRATOR - LUỒNG ĐIỀU PHỐI CHÍNH
# ==============================================================================
def main(num_vehicles: int = 30, hazard: bool = False, seed=None,
         weather: str = "clear", driver_takeover: bool = False, turn=None, town=None,
         replay_dir=None, duration_s=None, no_display: bool = False,
         run_report=None, performance_profile: str = "low-memory",
         runtime_mode: str = "async-stable", inference_device: str = "auto",
         crash_journal=None, spawn_index: int = 0) -> bool:
    stable_async, effective_performance = configure_runtime(
        cfg, performance_profile, runtime_mode)
    device_info = configure_inference_device(cfg, inference_device, runtime_mode)
    effective_performance["inference_device"] = device_info["effective"]
    effective_performance["inference_device_requested"] = device_info["requested"]
    effective_performance["inference_device_reason"] = device_info["reason"]
    print(f"[Performance] {performance_profile}: {effective_performance}")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    # Chỉ reload khi thật sự đổi map. Reload cùng Town02 giữa các lần demo từng
    # làm CARLA/Unreal chết RPC trên GPU 8 GB dù lượt trước đã cleanup sạch.
    map_transition = {}
    world, map_reloaded = get_or_load_world(
        client, town, recovery_timeout_s=10.0, client_timeout_s=30.0,
        diagnostics=map_transition)
    if map_transition:
        print(f"[System] Map load timeout retained; late completion verified: {map_transition}")
    if town and map_reloaded:
        print(f"[System] Đã nạp bản đồ '{town}'.")
    elif town:
        print(f"[System] Tái sử dụng map hiện tại '{world.get_map().name}' (không reload).")
    else:
        print(f"[System] Dùng map hiện tại '{world.get_map().name}'.")
    blueprint_library = world.get_blueprint_library()
    original_world_settings = world.get_settings()
    report_town_name = world.get_map().name
    actor_list = []

    sensor_rig = None
    collision_sensor = None
    telemetry = None
    traffic_spawner = None
    # Traffic Manager must be created while the freshly-loaded world is still
    # asynchronous. Creating its server after the world is already frozen can
    # leave the first synchronous tick waiting forever.
    traffic_manager = client.get_trafficmanager(cfg.TM_PORT)
    perception_runtime = None
    replay_writer = None
    ego_vehicle = None
    sensor_sync_stats = None
    radar_sync_stats = None
    frame_count = 0
    fps_ema = 0.0
    spawned_traffic = 0
    max_speed_kmh = 0.0
    distance_travelled_m = 0.0
    previous_location = None
    start_location = None
    end_location = None
    loop_wall_started = None
    stop_reason = "not_started"
    run_error = None
    run_pass = True
    lane_source_counts = {"learned": 0, "map": 0}
    # Explicit observability: set only when the safety controller commands BRAKE.
    aeb_triggered = False
    decision_trace = DecisionTrace()
    decision_trace_path = None
    crash_log = RuntimeCrashJournal()

    sync_session = SynchronousWorldSession(
        world, fps=cfg.FPS, timeout_s=10.0,
        traffic_manager=traffic_manager)
    try:
        sync_session.__enter__()
    except Exception as exc:
        print(f"[System] ❌ Không thể khóa CARLA ở {cfg.FPS} Hz: {exc}")
        return False
    if stable_async:
        print("[System] Tạm khóa CARLA/TM trong khi warm-up model GPU.")
    else:
        print(f"[System] CARLA/TM đã khóa synchronous {cfg.FPS} Hz trước khi nạp model.")
    if sync_session.recovered_transition:
        print("[System] ⚠️ Đã phục hồi CARLA sau timeout khi bật synchronous mode.")

    print("[System] Đang nạp các mạng Neural...")
    try:
        perception_runtime = NeuralPerceptionRuntime(cfg)
    except Exception as exc:
        # Model/weight failures used to escape before the main finally block,
        # leaving CARLA synchronous and causing the next run to time out.
        run_error = f"{type(exc).__name__}: {exc}"
        print(f"[System] ❌ Không thể khởi tạo perception runtime: {run_error}")
        try:
            sync_session.close()
        except Exception as cleanup_error:
            print(f"[System] Cảnh báo khôi phục CARLA sau lỗi model: {cleanup_error}")
        return False
    try:
        lidar_processor = perception_runtime.lidar_processor
        lane_detector = perception_runtime.lane_detector
        lane_arbiter = perception_runtime.lane_arbiter
        active_safety = ActiveSafetySystem(
            lane_half_width_m=cfg.LANE_HALF_WIDTH_M,
            min_cluster_points=cfg.MIN_CLUSTER_POINTS,
            reaction_time_s=cfg.REACTION_TIME_S,
            min_safe_dist_m=cfg.MIN_SAFE_DIST_M,
            critical_ttc_s=cfg.CRITICAL_TTC_S,
            warning_ttc_s=cfg.WARNING_TTC_S,
            min_closing_speed=cfg.MIN_CLOSING_SPEED,
            enable_evasion=cfg.ENABLE_EVASION,
            evade_lookahead_m=cfg.EVADE_LOOKAHEAD_M,
            lead_slow_ratio=cfg.LEAD_SLOW_RATIO,
            follow_speed_diff=cfg.TM_FOLLOW_SPEED_DIFF,
            vru_lateral_margin_m=cfg.VRU_LATERAL_MARGIN_M,
            vru_prediction_horizon_s=cfg.VRU_PREDICTION_HORIZON_S,
        )
        # Bám đa vật thể + dự đoán quỹ đạo -> phanh dự báo (predictive AEB).
        object_mot = MultiObjectTracker(dt=cfg.FIXED_DELTA)
        ego_motion_tracker = EgoMotionEstimator()
        # Giám sát ODD + máy trạng thái Fallback L3.
        odd_monitor = ODDMonitor()
        l3_sm = L3StateMachine(tor_window_s=cfg.TOR_WINDOW_S, mrm_decel_frac=cfg.MRM_DECEL_FRAC)
        # RL: điều chỉnh tốc độ tuần hành theo mật độ + chuyển làn theo ý định rẽ.
        rl_speed = RLSpeedController(cfg.RL_POLICY_PATH) if cfg.USE_RL_SPEED else None
        turn_planner = TurnIntentPlanner(cfg)

        dashboard = DashboardView(lane_half_m=cfg.LANE_HALF_WIDTH_M)
        telemetry = TelemetryLogger(flush_every=cfg.TELEMETRY_FLUSH_EVERY)
        radar_processor = RadarProcessor(
            cfg.RADAR_X, cfg.RADAR_Y, cfg.RADAR_Z,
            reference_x_m=cfg.LIDAR_X, velocity_sign=cfg.RADAR_VELOCITY_SIGN,
            max_range_m=cfg.RADAR_RANGE_M,
            min_target_height_m=cfg.RADAR_MIN_TARGET_HEIGHT_M)
        radar_velocity_validated = (
            not cfg.RADAR_REQUIRE_SIGN_VALIDATION
            or RadarProcessor.velocity_validation_passed(
                cfg.RADAR_SIGN_VALIDATION_PATH, cfg.RADAR_VELOCITY_SIGN))
        if cfg.ENABLE_RADAR and not radar_velocity_validated:
            print("[Radar] Range enabled; radial velocity excluded from TTC until "
                  f"{cfg.RADAR_SIGN_VALIDATION_PATH} passes controlled validation.")
        health_monitor = SensorHealthMonitor(
            {"camera": True, "lidar": True, "radar": cfg.ENABLE_RADAR},
            max_range_misses=cfg.RADAR_MAX_MISSING_FRAMES)
        metric_capacity = max(
            20000,
            int(float(duration_s) * cfg.FPS) + 100 if duration_s is not None else 20000)
        pipeline_metrics = PipelineMetrics({
            "safety_control": cfg.SAFETY_DEADLINE_MS,
            "neural_inference": cfg.PERCEPTION_DEADLINE_MS}, max_samples=metric_capacity)
        if stable_async:
            # No GPU sensor exists yet, so the transition back to async cannot race
            # an RGB render. Camera/LiDAR/radar are spawned only after this point.
            sync_session.close()
            print("[System] Runtime async-stable: safety geometry 40 Hz, RGB 10 Hz.")

    except Exception as exc:
        cleanup_summary = cleanup_runtime(
            client=client, world=world, actors=actor_list, session=sync_session,
            traffic_manager=traffic_manager, original_settings=original_world_settings,
            perception=perception_runtime, telemetry=telemetry,
            primary_error=f'{type(exc).__name__}: {exc}')
        print(f"[System] Startup failed: {exc}; cleanup={cleanup_summary}")
        return False

    try:
        crash_log = RuntimeCrashJournal(crash_journal)
        crash_log.emit('runtime', 'start', config=effective_performance,
                       requested_town=town, seed=seed)
        if replay_dir:
            replay_writer = ReplayWriter(replay_dir, metadata={
                "fps": cfg.FPS, "town": town, "weather": weather,
                "camera": [cfg.CAM_WIDTH, cfg.CAM_HEIGHT, cfg.CAM_FOV]})
        # 1. Khởi tạo Ego Vehicle
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            sys.exit("[Error] Không tìm thấy Spawn Points trên bản đồ.")

        ego_vehicle, ego_spawn_index = spawn_ego_safe(
            world, blueprint_library.find('vehicle.tesla.model3'), spawn_points,
            preferred_index=spawn_index)
        print(f"[System] Ego spawn point index = {ego_spawn_index} "
              f"(yêu cầu {spawn_index}; đổi --spawn-index nếu phía trước có vật cản).")
        actor_list.append(ego_vehicle)
        start_location = ego_vehicle.get_location()
        previous_location = start_location

        # 2. EgoController sở hữu ego control. Không pre-register ego với TM:
        # custom-control fault phải dừng an toàn, không chuyển ngầm sang autopilot.
        ego_controller = EgoController(
            ego_vehicle, traffic_manager, cfg, world=world)
        print(f"[Control] Ego mode: {ego_controller.mode}")

        # 2b. Sinh giao thông NPC ngẫu nhiên (+ tùy chọn xe chướng ngại) để kiểm thử
        traffic_spawner = TrafficSpawner(world, traffic_manager, seed=seed)
        if num_vehicles > 0:
            spawned_traffic = traffic_spawner.spawn_traffic(num_vehicles)
        if hazard:
            traffic_spawner.spawn_hazard_ahead(ego_vehicle, distance_m=cfg.HAZARD_DISTANCE_M)

        # 2c. Thời tiết + điều kiện ODD (Module E) — áp một lần cho cả phiên chạy.
        profiles = {}
        try:
            profiles = load_weather_profiles(cfg.WEATHER_CONFIG_PATH)
        except Exception as e:
            print(f"[Weather] Không nạp được hồ sơ thời tiết: {e}")
        prof = profiles.get(weather, {})
        if prof:
            world.set_weather(carla.WeatherParameters(**weather_kwargs(prof)))
            print(f"[Weather] Áp hồ sơ '{weather}'.")
        else:
            print(f"[Weather] Không có hồ sơ '{weather}', giữ thời tiết hiện tại.")
        conditions = estimate_conditions(
            precipitation=prof.get('precipitation', 0.0),
            fog_density=prof.get('fog_density', 0.0),
            wetness=prof.get('wetness', 0.0),
            precipitation_deposits=prof.get('precipitation_deposits', 0.0))
        odd = odd_monitor.classify(conditions)
        active_safety.set_conditions(conditions['mu'], odd['gap_multiplier'])
        print(f"[ODD] {odd['state']} | vis={conditions['visibility_m']}m "
              f"mu={conditions['mu']} snr={conditions['snr']}")

        # 3. SensorRig là chủ sở hữu duy nhất của actors/queues cảm biến.
        sensor_rig = SensorRig(world, ego_vehicle, cfg, stable_async)
        sensor_rig.spawn()
        collision_sensor = sensor_rig.collision
        sensor_sync_stats = sensor_rig.sync_stats
        radar_sync_stats = sensor_rig.radar_sync_stats

        print("[System] Sẵn sàng vòng lặp xử lý nhận thức (Perception Loop). Nhấn 'q' để thoát.")

        runtime_context = nullcontext() if stable_async else sync_session
        with runtime_context, ExitStack() as inference_window:
            sensor_rig.warmup(time.monotonic())
            frame_count = 0
            fps_ema = 0.0
            rl_target_kmh = 30.0
            density = 0.0
            last_detections = []
            last_fused_detections = []
            last_inference_frame = None
            last_inference_error_frame = None
            last_object_timestamp = None
            last_object_frame_id = None
            last_learned_lane = None
            last_lane_frame = None
            # Learned lane is advisory while the vehicle is moving. During an
            # AEB brake manoeuvre, reserve the GPU/CPU budget for the safety
            # path; map lane remains available and learned inference resumes
            # automatically when the brake action is released.
            lane_inference_allowed = True
            metrics = {'detected_vehicles': 0}
            lane_overlay = None
            prev_t = time.perf_counter()
            loop_wall_started = time.perf_counter()
            perception_runtime.start_measurement()
            # Freeze before runtime_context restores synchronous world settings.
            inference_window.callback(perception_runtime.stop_measurement)
            # The actor transform is only authoritative after CARLA has ticked;
            # sampling immediately after spawn may return the zero transform.
            start_location = ego_vehicle.get_location()
            previous_location = start_location
            end_location = start_location
            stop_reason = "running"
            target_frames = (max(1, int(round(float(duration_s) * cfg.FPS)))
                             if duration_s is not None else None)

            control_status = {"mode": ego_controller.mode}
            while True:
                if target_frames is not None and frame_count >= target_frames:
                    stop_reason = "duration_reached"
                    break
                pipeline_t0 = time.perf_counter()
                capture_timestamp = time.monotonic()
                crash_log.record_runtime('sensor.before', world=world, ego=ego_vehicle,
                                         carla_map=carla_map, loop_index=frame_count,
                                         controller_mode=ego_controller.mode)
                sensor_frame = sensor_rig.read(capture_timestamp)
                crash_log.record_runtime('sensor.after', world=world, ego=ego_vehicle,
                                         carla_map=carla_map, loop_index=frame_count,
                                         frame_id=sensor_frame.frame_id,
                                         lidar_time_s=sensor_frame.lidar_timestamp,
                                         camera_frame_id=sensor_frame.camera_frame_id)
                w_frame = sensor_frame.frame_id
                rgb_frame = sensor_frame.rgb
                point_cloud = sensor_frame.point_cloud
                radar_data = sensor_frame.radar_measurement
                camera_fresh = sensor_frame.camera_fresh
                lidar_obstacles = lidar_processor.extract_safety_obstacles(point_cloud)
                radar_targets = radar_processor.process(radar_data)
                radar_safety_targets = (radar_targets if radar_velocity_validated else [
                    {**target, "closing_speed_ms": 0.0} for target in radar_targets])

                health_monitor.next_frame()
                # Async camera and LiDAR timestamps use different clocks.  Use
                # monotonic wall timestamps for freshness, otherwise a healthy
                # async camera can be marked stale indefinitely.
                camera_wall_timestamp = sensor_frame.camera_wall_timestamp
                camera_age_ms = (
                    max(0.0, capture_timestamp - camera_wall_timestamp) * 1000.0
                    if camera_wall_timestamp is not None else float("inf"))
                camera_available = camera_age_ms <= cfg.MAX_ASYNC_RESULT_AGE_MS
                health_monitor.observe(
                    "camera",
                    (sensor_frame.camera_frame_id
                     if sensor_frame.camera_frame_id is not None else w_frame),
                    sensor_frame.camera_wall_timestamp or capture_timestamp,
                    camera_available, error="camera semantics stale",
                    now_s=capture_timestamp)
                health_monitor.observe("lidar", w_frame, capture_timestamp, True,
                                       now_s=capture_timestamp)
                health_monitor.observe("radar", w_frame, capture_timestamp,
                                       radar_data is not None,
                                       error="optional radar frame missing",
                                       now_s=capture_timestamp)
                frame_bundle = SensorFrameBundle(
                    w_frame, capture_timestamp, rgb_frame, point_cloud, radar_targets,
                    health_monitor.snapshot())
                if replay_writer is not None:
                    replay_writer.write(
                        w_frame, capture_timestamp, rgb_frame, point_cloud,
                        RadarProcessor._rows(radar_data),
                        labels={"detections": last_detections},
                        sensor_states={name: vars(state)
                                       for name, state in frame_bundle.sensor_states.items()})

                # One pending GPU slot: newer captures replace queued work. Fusion
                # happens inside the worker with LiDAR from the SAME CARLA frame.
                detect_frame = (camera_fresh if stable_async
                                else frame_count % cfg.DETECT_EVERY_N == 0)
                if detect_frame:
                    inference_frame_id = (
                        sensor_frame.camera_frame_id if stable_async else w_frame)
                    inference_timestamp = (
                        sensor_frame.camera_wall_timestamp if stable_async
                        else capture_timestamp)
                    perception_runtime.submit(
                        inference_frame_id, inference_timestamp,
                        rgb_frame, point_cloud,
                        (camera_fresh if stable_async
                         else frame_count % cfg.LANE_EVERY_N == 0)
                        and lane_inference_allowed)
                inference_result, inference_product = perception_runtime.poll(
                    w_frame, time.monotonic(), cfg.MAX_ASYNC_RESULT_AGE_MS)
                fresh_inference = bool(
                    inference_product is not None
                    and inference_result.frame_id != last_inference_frame)
                if (inference_result is not None and inference_result.error is not None
                        and inference_result.frame_id != last_inference_error_frame):
                    last_inference_error_frame = inference_result.frame_id
                    print(f"[Perception] Frame {inference_result.frame_id} failed: "
                          f"{inference_result.error}")
                if fresh_inference:
                    perception_runtime.record_consumed(inference_result, time.monotonic())
                    product = inference_product
                    last_inference_frame = inference_result.frame_id
                    if product.get("detections") is not None:
                        last_detections = product["detections"]
                        last_fused_detections = product["fused"]
                        metrics = product["metrics"]
                        last_object_timestamp = inference_result.timestamp
                        last_object_frame_id = inference_result.frame_id
                    for stage, latency_ms in product.get("stage_latency_ms", {}).items():
                        if latency_ms is not None:
                            pipeline_metrics.record(stage, latency_ms)
                    pipeline_metrics.record("neural_inference", inference_result.latency_ms)
                lane_result, lane_estimate = perception_runtime.poll_lane(
                    w_frame, time.monotonic(), cfg.MAX_ASYNC_RESULT_AGE_MS)
                if lane_estimate is not None and lane_result.frame_id != last_lane_frame:
                    last_lane_frame = lane_result.frame_id
                    last_learned_lane = lane_estimate
                    perception_runtime.record_lane_consumed(lane_result, time.monotonic())
                    if lane_result.error is not None:
                        print(f"[Lane] Frame {lane_result.frame_id} failed: {lane_result.error}")
                object_age_ms = (
                    (capture_timestamp - last_object_timestamp) * 1000.0
                    if last_object_timestamp is not None else None)
                if (object_age_ms is not None
                        and object_age_ms <= cfg.MAX_ASYNC_RESULT_AGE_MS):
                    detections = last_detections
                    fused_detections = last_fused_detections
                else:
                    # Never feed expired/error camera semantics into fusion or
                    # traffic-control decisions. Current LiDAR/radar geometry
                    # continues through the independent safety path below.
                    last_detections = []
                    last_fused_detections = []
                    detections = []
                    fused_detections = []
                    metrics = {'detected_vehicles': 0}
                annotated_frame = rgb_frame.copy()
                draw_detections(annotated_frame, detections)
                pipeline_metrics.record_inference_age(object_age_ms)
                traffic_control = summarize_traffic_controls(
                    fused_detections, cfg.TRAFFIC_LIGHT_RANGE_M, cfg.STOP_SIGN_RANGE_M)

                # --- Bám vật + dự đoán + quyết định an toàn ---
                velocity = ego_vehicle.get_velocity()
                ego_speed_ms = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
                max_speed_kmh = max(max_speed_kmh, ego_speed_ms * 3.6)
                current_location = ego_vehicle.get_location()
                if previous_location is not None:
                    distance_travelled_m += current_location.distance(previous_location)
                previous_location = current_location
                end_location = current_location
                ego_path, lane_context = ego_controller.path_context(turn_intent=turn)
                map_lane = map_lane_estimate(w_frame, capture_timestamp, ego_path)
                selected_lane = lane_arbiter.select(
                    last_learned_lane, map_lane,
                    at_junction=lane_context.get("is_junction", False),
                    now_s=capture_timestamp,
                    max_age_ms=cfg.MAX_ASYNC_RESULT_AGE_MS)
                lane_source_counts[selected_lane.source] = (
                    lane_source_counts.get(selected_lane.source, 0) + 1)
                control_path = (selected_lane.centerline_m
                                if selected_lane.source == "learned" else ego_path)
                # Cập nhật tracker khi có phát hiện; khung giữa chỉ dự đoán (không đo lại).
                ego_transform = ego_vehicle.get_transform()
                ego_motion = ego_motion_tracker.update(ego_transform)
                tracks = object_mot.update(
                    fused_detections if fresh_inference else [], cfg.FIXED_DELTA,
                    ego_motion=ego_motion, ego_speed_ms=ego_speed_ms,
                    radar_measurements=radar_targets, frame_id=w_frame,
                    radar_velocity_validated=radar_velocity_validated)
                decision = active_safety.update(
                    ego_speed_ms, fused_detections, lidar_obstacles, cfg.FIXED_DELTA,
                    tracks=tracks, path_points=control_path, lane_context=lane_context,
                    radar_targets=radar_safety_targets)
                decision_trace.record(
                    frame_id=w_frame, decision=decision, ego_speed_ms=ego_speed_ms,
                    lidar_obstacles=lidar_obstacles, radar_targets=radar_safety_targets,
                    tracks=tracks, fused_detections=fused_detections,
                    sensor_readout=sensor_frame, ego_transform=ego_transform, ego_velocity=velocity,
                    path_points=control_path, lane_source=selected_lane.source,
                    lane_half_width_m=active_safety.lane_half,
                    min_closing_speed_ms=active_safety.min_closing,
                    radar_processor=radar_processor, radar_velocity_validated=radar_velocity_validated,
                    inference_frame_id=last_object_frame_id,
                    inference_timestamp=last_object_timestamp, inference_age_ms=object_age_ms)
                lane_inference_allowed = decision.action != "BRAKE"
                if decision.action == "BRAKE":
                    aeb_triggered = True

                # --- Fallback L3 (ODD): TOR -> MRM -> SAFE_STOP ---
                odd = odd_monitor.classify(conditions, health_monitor.summary())
                l3 = l3_sm.update(odd['state'], driver_takeover, ego_speed_ms,
                                  conditions['mu'], cfg.FIXED_DELTA, critical=odd['critical'])

                # --- RL: tốc độ tuần hành theo MẬT ĐỘ giao thông (đường vắng nhanh, đông chậm) ---
                if rl_speed is not None and frame_count % cfg.RL_SPEED_EVERY_N == 0:
                    # Mật độ = SỐ PHƯƠNG TIỆN thực phía trước (KHÔNG tính vật tĩnh ven
                    # đường như tường/cột) -> tránh "đông giả" khiến RL ghì tốc độ xe.
                    raw_density = min(1.0, sum(
                        1 for d in fused_detections
                        if d.get('distance_m', 999.0) < cfg.DENSITY_RANGE_M
                    ) / cfg.DENSITY_NORM)
                    # Làm mượt (EMA) -> mật độ không nhảy khi phát hiện chập chờn -> đỡ giật tốc độ.
                    density = (1.0 - cfg.DENSITY_EMA) * density + cfg.DENSITY_EMA * raw_density
                    th = decision.threat or {}
                    new_target = rl_speed.desired_speed_kmh(
                        ego_speed_ms, density, th.get('distance_m'), th.get('ttc_s'))
                    # Mượt tốc độ mục tiêu -> chuyển mức 15->35 km/h ÊM thay vì bậc thang.
                    rl_target_kmh = (1.0 - cfg.RL_TARGET_EMA) * rl_target_kmh + cfg.RL_TARGET_EMA * new_target

                # Một controller dùng chung cho runtime + scenario. Custom stack điều
                # khiển trajectory/PID; AEB và L3 vẫn luôn có quyền ghi đè cao nhất.
                target_kmh = rl_target_kmh if rl_speed is not None else max(
                    10.0, ego_vehicle.get_speed_limit())
                target_kmh = max(cfg.RL_MIN_CRUISE_KMH, target_kmh)
                if turn and ego_controller.uses_traffic_manager_control:
                    turn_planner.update(
                        ego_vehicle, world, traffic_manager, turn, frame_count)
                crash_log.record_runtime('control.before', world=world, ego=ego_vehicle,
                                         carla_map=carla_map, frame_id=w_frame,
                                         controller_mode=ego_controller.mode,
                                         decision_action=decision.action, decision_state=decision.state,
                                         requested_aeb_brake=decision.brake, l3=l3, odd_state=odd['state'])
                control_status = ego_controller.apply(
                    decision, l3, odd['state'], frame_count,
                    target_speed_kmh=target_kmh, traffic_control=traffic_control,
                    turn_intent=turn)
                crash_log.record_runtime('control.after', world=world, ego=ego_vehicle,
                                         carla_map=carla_map, frame_id=w_frame,
                                         controller_mode=ego_controller.mode,
                                         control_status=control_status, read_control=True)
                pipeline_latency_ms = (time.perf_counter() - pipeline_t0) * 1000.0
                pipeline_metrics.record("safety_control", pipeline_latency_ms)
                pipeline_metrics.sample_gpu_memory()
                # --- Telemetry (ghi thưa) ---
                if frame_count % cfg.LOG_EVERY_N == 0:
                    telemetry.log(ego_vehicle, len(fused_detections))

                # --- Dashboard + hiển thị (mỗi RENDER_EVERY_N khung -> giảm giật, tăng FPS) ---
                if frame_count % cfg.RENDER_EVERY_N == 0:
                    metrics['speed'] = ego_speed_ms * 3.6
                    metrics['fsm_state'] = decision.state
                    metrics['collision_risk'] = decision.collision_risk
                    metrics['threat'] = decision.threat
                    metrics['vehicles'] = len(detections)
                    metrics['collisions'] = collision_sensor.count
                    metrics['fps'] = fps_ema
                    metrics['pipeline_latency_ms'] = pipeline_latency_ms
                    metrics['inference_age_ms'] = object_age_ms
                    metrics['frame_budget_ms'] = cfg.FIXED_DELTA * 1000.0
                    metrics['sensor_frame_errors'] = sensor_sync_stats.frame_errors
                    metrics['radar_availability'] = health_monitor.availability('radar')
                    metrics['radar_frame_errors'] = radar_sync_stats.frame_errors
                    metrics['sensor_health'] = health_monitor.summary()
                    metrics['radar_closing_ms'] = (decision.threat or {}).get(
                        'radar_closing_ms')
                    metrics['radar_velocity_validated'] = radar_velocity_validated
                    metrics['gpu_peak_mb'] = pipeline_metrics.gpu_peak_mb
                    metrics['control_mode'] = control_status.get('mode')
                    metrics['behavior_state'] = control_status.get('behavior_state')
                    metrics['detected_vrus'] = sum(
                        d.get('class') in ('Person', 'Bicycle') for d in detections)
                    metrics['traffic_light_state'] = traffic_control['traffic_light_state']
                    metrics['stop_sign_active'] = traffic_control['stop_sign']
                    metrics['l3_state'] = l3['state']
                    metrics['odd_state'] = odd['state']
                    metrics['mu'] = conditions['mu']
                    metrics['visibility_m'] = conditions['visibility_m']
                    if rl_speed is not None:
                        metrics['rl_target_kmh'] = rl_target_kmh
                        metrics['density'] = density

                    if frame_count % cfg.LANE_EVERY_N == 0:
                        lane_overlay = lane_detector.line_overlay(annotated_frame)
                    lane_frame = cv2.addWeighted(annotated_frame, 0.85, lane_overlay, 0.85, 0) \
                        if lane_overlay is not None else annotated_frame
                    # Độ lệch tâm làn (từ lane detector) -> HUD cảnh báo chệch làn.
                    metrics['lane_offset_m'] = selected_lane.offset_m
                    metrics['lane_source'] = selected_lane.source
                    metrics['lane_confidence'] = selected_lane.confidence
                    metrics['hough_lane_offset_m'] = (lane_detector.lane_offset_m
                                                       if lane_detector.lane_detected else None)
                    hud_frame = dashboard.render(lane_frame, metrics, lidar_obstacles, tracks=tracks)
                    if not no_display:
                        cv2.imshow("ADAS Portfolio - Central Dashboard", hud_frame)

                # waitKey mỗi khung để cửa sổ phản hồi + bắt phím 'q' (rất rẻ).
                if not no_display and cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_reason = "user_quit"
                    break
                # Delivered loop rate including dashboard work.
                now = time.perf_counter()
                inst_fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
                prev_t = now
                fps_ema = (inst_fps if fps_ema == 0.0
                           else 0.9 * fps_ema + 0.1 * inst_fps)
                frame_count += 1

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\n[System] Dừng đột ngột bởi người dùng.")
    except Exception as e:
        stop_reason = "error"
        run_error = f"{type(e).__name__}: {e}"
        print(f"\n[System] ❌ Ngoại lệ nghiêm trọng hệ thống: {e}")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)
    finally:
        cleanup_summary = cleanup_runtime(
            client=client, world=world, actors=actor_list, sensor_rig=sensor_rig,
            traffic_spawner=traffic_spawner, session=sync_session,
            traffic_manager=traffic_manager, original_settings=original_world_settings,
            perception=perception_runtime, telemetry=telemetry, replay=replay_writer,
            journal=crash_log, close_windows=cv2.destroyAllWindows,
            primary_error=run_error, stop_reason=stop_reason)
        if cleanup_summary['verified']:
            print("[System] Cleanup verified: owned actors removed; world progress restored.")
        else:
            run_pass = False
            run_error = run_error or 'Runtime cleanup failed; see cleanup outcomes'
            print(f"[System] Cleanup NOT verified: {cleanup_summary['steps']}")

        if run_report:
            try:
                decision_trace_path = decision_trace.write(os.path.join(
                    os.path.dirname(os.path.abspath(run_report)), "decision_trace.json"))
            except Exception as trace_error:
                # Observability must never hide the actual runtime result.
                print(f"[System] Decision trace write warning: {trace_error}")
            run_pass = write_runtime_report(
                run_report,
                cfg,
                RuntimeReportData(
                    world=world,
                    weather=weather,
                    seed=seed,
                    vehicles_requested=num_vehicles,
                    vehicles_spawned=spawned_traffic,
                    duration_s=duration_s,
                    frame_count=frame_count,
                    fps_ema=fps_ema,
                    no_display=no_display,
                    effective_performance=effective_performance,
                    stop_reason=stop_reason,
                    run_error=run_error,
                    loop_wall_started=loop_wall_started,
                    max_speed_kmh=max_speed_kmh,
                    distance_travelled_m=distance_travelled_m,
                    start_location=start_location,
                    end_location=end_location,
                    collision_sensor=collision_sensor,
                    sensor_sync_stats=sensor_sync_stats,
                    radar_sync_stats=radar_sync_stats,
                    health_monitor=health_monitor,
                    pipeline_metrics=pipeline_metrics,
                    perception_runtime=perception_runtime,
                    lane_source_counts=lane_source_counts,
                    telemetry=telemetry,
                    aeb_triggered=aeb_triggered,
                    decision_trace_summary=decision_trace.summary(),
                    decision_trace_path=decision_trace_path,
                    cleanup_summary=cleanup_summary,
                    town_name=report_town_name,
                ),
            )

    return run_pass and run_error is None


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = build_argument_parser(cfg)
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error('--duration phải > 0')
    success = main(num_vehicles=args.vehicles, hazard=args.hazard, seed=args.seed,
                   weather=args.weather, driver_takeover=args.driver_takeover,
                   turn=args.turn, town=args.town, replay_dir=args.record_replay,
                   duration_s=args.duration, no_display=args.no_display,
                   run_report=args.run_report,
                   performance_profile=args.performance_profile,
                   runtime_mode=args.runtime_mode,
                   inference_device=args.inference_device,
                   crash_journal=args.crash_journal,
                   spawn_index=args.spawn_index)
    if not success:
        sys.exit(1)
