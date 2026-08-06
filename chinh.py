# fmt: off
# isort: skip_file

"""
Hệ thống điều phối trung tâm (Main Orchestrator) cho CARLA ADAS Pipeline.
Kiến trúc: Autopilot nền (Traffic Manager) + Lớp an toàn ghi đè (AEB + tránh né).

Luồng mỗi khung hình:
    tick -> RGB + LiDAR -> phát hiện (YOLO ensemble) -> gộp khoảng cách (LiDAR)
    -> quyết định an toàn (máy trạng thái) -> áp lệnh (phanh / giảm tốc / chuyển
       làn) -> HUD + telemetry.
"""
import argparse
import time
import traceback
import os
import sys
import queue
import numpy as np
import cv2
import carla

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
    from midas_estimator import DepthEstimator
    from object_tracking import EnsembleVehicleTracker, draw_detections
    from lidar_processor import LidarProcessor
    from lane_detection import LaneDetector
    from sensor_fusion import SensorFusion
    from active_safety import ActiveSafetySystem
    from mot_tracker import MultiObjectTracker
    from odd_monitor import ODDMonitor
    from mrm_controller import L3StateMachine
    from weather_model import estimate_conditions
    from weather_config import load_weather_profiles, weather_kwargs
    from ego_control import spawn_ego_safe
    from rl_speed_controller import RLSpeedController
    from turn_intent import TurnIntentPlanner
    from data_logger import TelemetryLogger
    from traffic_spawner import TrafficSpawner
    from collision_sensor import CollisionSensor
    from dashboard_view import DashboardView
    print("[System] ✅ Đã nạp thành công các module Nhận thức và An toàn!")
except ModuleNotFoundError as e:
    print(f"[System] ❌ Lỗi nạp module: {e}. Hãy kiểm tra lại thư mục!")
    sys.exit(1)


class CarlaSyncManager:
    """Quản lý trạng thái đồng bộ hóa của môi trường CARLA."""
    def __init__(self, world: carla.World, fps: int = 20) -> None:
        self.world = world
        self.original_settings = world.get_settings()
        self.sync_settings = world.get_settings()
        self.sync_settings.synchronous_mode = True
        self.sync_settings.fixed_delta_seconds = 1.0 / fps

    def __enter__(self):
        self.world.apply_settings(self.sync_settings)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.world.apply_settings(self.original_settings)
        print("[System] Khôi phục CARLA về chế độ bất đồng bộ.")


def process_rgb_image(carla_image: carla.Image) -> np.ndarray:
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


def process_lidar_data(carla_lidar: carla.LidarMeasurement) -> np.ndarray:
    points = np.frombuffer(carla_lidar.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points.copy()


def retrieve_sensor(sensor_queue, frame_id, timeout=2.0):
    """Lấy đúng gói cảm biến KHỚP frame vừa tick (mẫu chuẩn của CARLA sync mode).

    Bỏ các gói cũ tồn trong hàng đợi -> chống lệch khung/dữ liệu cũ (nguyên nhân
    'đơ camera' + rubberbanding do desync). data.frame < frame_id là gói cũ -> bỏ.
    """
    while True:
        data = sensor_queue.get(timeout=timeout)
        if data.frame >= frame_id:
            return data


def set_hazard_lights(vehicle, on: bool) -> None:
    """Bật/tắt đèn khẩn cấp (hazard) — dùng khi MRM/Fallback L3."""
    try:
        if on:
            state = carla.VehicleLightState(
                carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker)
        else:
            state = carla.VehicleLightState.NONE
        vehicle.set_light_state(state)
    except Exception:
        pass


# ==============================================================================
# ORCHESTRATOR - LUỒNG ĐIỀU PHỐI CHÍNH
# ==============================================================================
def main(num_vehicles: int = 30, hazard: bool = False, seed=None,
         weather: str = "clear", driver_takeover: bool = False, turn=None, town=None) -> None:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    # Đổi sang bản đồ nhẹ hơn (vd Town02) giúp tăng FPS đáng kể so với Town10HD_Opt.
    if town:
        print(f"[System] Đang nạp bản đồ '{town}' (nhẹ hơn -> FPS cao hơn)...")
        world = client.load_world(town)
    else:
        world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    actor_list = []

    camera_sensor = None
    lidar_sensor = None
    collision_sensor = None
    telemetry = None
    traffic_spawner = None

    print("[System] Đang nạp các mạng Neural...")
    depth_estimator = DepthEstimator() if cfg.USE_MIDAS else None
    object_tracker = EnsembleVehicleTracker(
        device=cfg.YOLO_DEVICE, use_ensemble=cfg.USE_ENSEMBLE,
        model_a=cfg.YOLO_MODEL_A, model_b=cfg.YOLO_MODEL_B,
        imgsz=cfg.YOLO_IMGSZ, half=cfg.YOLO_HALF, use_optimized=cfg.YOLO_USE_OPTIMIZED)
    lidar_processor = LidarProcessor(eps=0.6, min_samples=5, max_points=cfg.LIDAR_MAX_POINTS)
    lane_detector = LaneDetector()

    # Fusion & Safety đều lấy hình học từ config -> MỘT nguồn chân lý duy nhất.
    sensor_fusion = SensorFusion(
        width=cfg.CAM_WIDTH, height=cfg.CAM_HEIGHT, fov=cfg.CAM_FOV,
        lidar_y_sign=cfg.LIDAR_Y_SIGN, real_heights_m=cfg.REAL_HEIGHTS_M,
        default_height_m=cfg.DEFAULT_HEIGHT_M,
        depth_min_m=cfg.DEPTH_MIN_M, depth_max_m=cfg.DEPTH_MAX_M,
    )
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
    )
    # Bám đa vật thể + dự đoán quỹ đạo -> phanh dự báo (predictive AEB).
    object_mot = MultiObjectTracker(dt=cfg.FIXED_DELTA)
    # Giám sát ODD + máy trạng thái Fallback L3.
    odd_monitor = ODDMonitor()
    l3_sm = L3StateMachine(tor_window_s=cfg.TOR_WINDOW_S, mrm_decel_frac=cfg.MRM_DECEL_FRAC)
    # RL: điều chỉnh tốc độ tuần hành theo mật độ + chuyển làn theo ý định rẽ.
    rl_speed = RLSpeedController(cfg.RL_POLICY_PATH) if cfg.USE_RL_SPEED else None
    turn_planner = TurnIntentPlanner(cfg)

    dashboard = DashboardView(lane_half_m=cfg.LANE_HALF_WIDTH_M)
    telemetry = TelemetryLogger(flush_every=cfg.TELEMETRY_FLUSH_EVERY)

    try:
        # 1. Khởi tạo Ego Vehicle
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            sys.exit("[Error] Không tìm thấy Spawn Points trên bản đồ.")

        ego_vehicle, _ = spawn_ego_safe(
            world, blueprint_library.find('vehicle.tesla.model3'), spawn_points)
        actor_list.append(ego_vehicle)

        # 2. Traffic Manager đảm nhiệm autopilot nền
        traffic_manager = client.get_trafficmanager(cfg.TM_PORT)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.distance_to_leading_vehicle(ego_vehicle, cfg.TM_LEADING_DISTANCE_M)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        autopilot_engaged = True
        last_lane_change_frame = -(10 ** 9)
        cooldown_frames = cfg.LANE_CHANGE_COOLDOWN_S / cfg.FIXED_DELTA

        # 2b. Sinh giao thông NPC ngẫu nhiên (+ tùy chọn xe chướng ngại) để kiểm thử
        traffic_spawner = TrafficSpawner(world, traffic_manager, seed=seed)
        if num_vehicles > 0:
            traffic_spawner.spawn_traffic(num_vehicles)
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

        # 3. Cảm biến — độ phân giải/FOV lấy từ config
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(cfg.CAM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(cfg.CAM_HEIGHT))
        camera_bp.set_attribute('fov', str(cfg.CAM_FOV))
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '60')
        lidar_bp.set_attribute('rotation_frequency', str(cfg.FPS))
        lidar_bp.set_attribute('points_per_second', str(cfg.LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute('channels', '32')

        camera_sensor = world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=cfg.CAM_X, z=cfg.CAM_Z)),
            attach_to=ego_vehicle)
        lidar_sensor = world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=cfg.LIDAR_X, z=cfg.LIDAR_Z)),
            attach_to=ego_vehicle)
        actor_list.extend([camera_sensor, lidar_sensor])

        # Cảm biến va chạm để chấm điểm khách quan (số lần va chạm).
        collision_sensor = CollisionSensor(world, ego_vehicle, fps=cfg.FPS)
        actor_list.append(collision_sensor.sensor)

        image_queue: "queue.Queue" = queue.Queue()
        lidar_queue: "queue.Queue" = queue.Queue()
        camera_sensor.listen(image_queue.put)
        lidar_sensor.listen(lidar_queue.put)

        print("[System] Sẵn sàng vòng lặp xử lý nhận thức (Perception Loop). Nhấn 'q' để thoát.")

        with CarlaSyncManager(world, fps=cfg.FPS):
            frame_count = 0
            fps_ema = 0.0
            rl_target_kmh = 30.0
            density = 0.0
            last_detections = []
            lane_overlay = None
            prev_t = time.perf_counter()

            last_sent_speed_diff = None
            last_sent_rl_target = None
            while True:
                w_frame = world.tick()
                rgb_frame = process_rgb_image(retrieve_sensor(image_queue, w_frame))
                point_cloud = process_lidar_data(retrieve_sensor(lidar_queue, w_frame))

                # --- Perception: chạy YOLO ensemble mỗi N khung; giữa các khung để
                #     MultiObjectTracker (Kalman) NỘI SUY -> giữ nguyên mô hình, tăng FPS ---
                detect_frame = (frame_count % cfg.DETECT_EVERY_N == 0)
                if detect_frame:
                    annotated_frame, detections, metrics = object_tracker.process(
                        rgb_frame, None, frame_count)
                    last_detections = detections
                else:
                    detections = last_detections
                    annotated_frame = rgb_frame.copy()
                    draw_detections(annotated_frame, detections)
                    metrics = {'detected_vehicles': len(detections)}

                # LiDAR + fusion chạy MỖI KHUNG (an toàn AEB không đổi).
                lidar_obstacles = lidar_processor.extract_obstacles(point_cloud)
                fused_detections = sensor_fusion.fuse(detections, lidar_obstacles)

                # --- Bám vật + dự đoán + quyết định an toàn ---
                velocity = ego_vehicle.get_velocity()
                ego_speed_ms = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
                # Cập nhật tracker khi có phát hiện; khung giữa chỉ dự đoán (không đo lại).
                tracks = object_mot.update(fused_detections if detect_frame else [], cfg.FIXED_DELTA)
                decision = active_safety.update(
                    ego_speed_ms, fused_detections, lidar_obstacles, cfg.FIXED_DELTA, tracks=tracks)

                # --- Fallback L3 (ODD): TOR -> MRM -> SAFE_STOP ---
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

                # --- Áp lệnh điều khiển (ưu tiên: L3 override > AEB > lái thường) ---
                if l3['override']:
                    # MRM/SAFE_STOP: giảm tốc êm + đèn khẩn cấp, ngoài ODD của L3.
                    if autopilot_engaged:
                        ego_vehicle.set_autopilot(False)
                        autopilot_engaged = False
                    brake_cmd = 1.0 if l3['state'] == 'SAFE_STOP' else min(1.0, l3['target_decel_ms2'] / 6.0)
                    ego_vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=brake_cmd,
                        hand_brake=(l3['state'] == 'SAFE_STOP')))
                    set_hazard_lights(ego_vehicle, True)
                elif decision.action == "BRAKE":
                    set_hazard_lights(ego_vehicle, l3['hazard'])
                    if autopilot_engaged:
                        ego_vehicle.set_autopilot(False)
                        autopilot_engaged = False
                    ego_vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=decision.brake, hand_brake=False))
                else:
                    set_hazard_lights(ego_vehicle, l3['hazard'])
                    if not autopilot_engaged:
                        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
                        autopilot_engaged = True

                    # Chủ động chuyển làn theo Ý ĐỊNH RẼ (--turn) trước giao lộ.
                    if turn:
                        turn_planner.update(ego_vehicle, world, traffic_manager, turn, frame_count)

                    # DEGRADED -> giảm tốc độ tối đa (giữ trong ODD).
                    drive_diff = 40.0 if odd['state'] == "DEGRADED" else cfg.TM_DEFAULT_SPEED_DIFF
                    if decision.action == "SLOW":
                        target_diff = max(decision.slow_pct, drive_diff)
                        if last_sent_speed_diff != target_diff:
                            traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, target_diff)
                            last_sent_speed_diff = target_diff
                    elif decision.action in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
                        if last_sent_speed_diff != drive_diff:
                            traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, drive_diff)
                            last_sent_speed_diff = drive_diff

                        if frame_count - last_lane_change_frame > cooldown_frames:
                            traffic_manager.force_lane_change(
                                ego_vehicle, decision.action == "LANE_CHANGE_RIGHT")
                            last_lane_change_frame = frame_count
                    else:  # DRIVE — tốc độ tuần hành do RL đặt theo mật độ giao thông
                        if rl_speed is not None:
                            tgt = min(rl_target_kmh, 40.0) if odd['state'] == "DEGRADED" else rl_target_kmh
                            tgt = max(cfg.RL_MIN_CRUISE_KMH, tgt)  # sàn: luôn di chuyển khi đường thoáng

                            # Chỉ gửi lệnh xuống server khi tốc độ RL thay đổi lệch đáng kể (> 0.5 km/h)
                            if last_sent_rl_target is None or abs(last_sent_rl_target - tgt) > 0.5:
                                try:
                                    traffic_manager.set_desired_speed(ego_vehicle, tgt)
                                    last_sent_rl_target = tgt
                                except Exception:
                                    if last_sent_speed_diff != drive_diff:
                                        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, drive_diff)
                                        last_sent_speed_diff = drive_diff
                        else:
                            if last_sent_speed_diff != drive_diff:
                                traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, drive_diff)
                                last_sent_speed_diff = drive_diff
                # --- FPS (EMA) ---
                now = time.perf_counter()
                inst_fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
                prev_t = now
                fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps

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
                    hud_frame = dashboard.render(lane_frame, metrics, lidar_obstacles, tracks=tracks)
                    cv2.imshow("ADAS Portfolio - Central Dashboard", hud_frame)

                # waitKey mỗi khung để cửa sổ phản hồi + bắt phím 'q' (rất rẻ).
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_count += 1

    except KeyboardInterrupt:
        print("\n[System] Dừng đột ngột bởi người dùng.")
    except Exception as e:
        print(f"\n[System] ❌ Ngoại lệ nghiêm trọng hệ thống: {e}")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)
    finally:
        # Dọn dẹp cho MỌI đường thoát (kể cả nhấn 'q'), tránh treo server ở sync mode.
        if camera_sensor is not None and camera_sensor.is_alive:
            camera_sensor.stop()
        if lidar_sensor is not None and lidar_sensor.is_alive:
            lidar_sensor.stop()
        if collision_sensor is not None:
            collision_sensor.stop()
        if telemetry is not None:
            telemetry.close()
        cv2.destroyAllWindows()
        if traffic_spawner is not None:
            traffic_spawner.destroy(client)
        if actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            world.tick()
        print("[System] ✅ Đã giải phóng bộ nhớ và tiêu hủy Actors triệt để.")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CARLA ADAS pipeline (autopilot + AEB/avoidance)")
    parser.add_argument('--vehicles', type=int, default=cfg.NUM_NPC_VEHICLES,
                        help="số xe NPC ngẫu nhiên (mặc định %(default)s; 0 = không sinh)")
    parser.add_argument('--hazard', action='store_true',
                        help="đặt một xe ĐỨNG YÊN ngay trước ego để test phanh gấp/tránh né")
    parser.add_argument('--seed', type=int, default=None,
                        help="seed ngẫu nhiên để tái lập kịch bản giao thông")
    parser.add_argument('--weather', type=str, default='clear',
                        help="hồ sơ thời tiết: clear | light_rain | heavy_rain | fog | storm")
    parser.add_argument('--driver-takeover', action='store_true',
                        help="mô phỏng tài xế tiếp quản khi có Takeover Request (mặc định: không)")
    parser.add_argument('--turn', type=str, default=None, choices=['left', 'right'],
                        help="ý định rẽ -> chủ động chuyển làn phía đó trước giao lộ")
    parser.add_argument('--town', type=str, default=None,
                        help="nạp bản đồ nhẹ hơn để tăng FPS, vd: Town02, Town01, Town05")
    args = parser.parse_args()
    main(num_vehicles=args.vehicles, hazard=args.hazard, seed=args.seed,
         weather=args.weather, driver_takeover=args.driver_takeover, turn=args.turn, town=args.town)
