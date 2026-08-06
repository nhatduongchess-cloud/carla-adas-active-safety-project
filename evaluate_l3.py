# fmt: off
# isort: skip_file
"""
Bộ kiểm thử tự động DRIVE PILOT-style: chạy pipeline qua các HỒ SƠ THỜI TIẾT trong
config/weather_config.yaml, đo KPI + hành vi ODD/MRM, rồi xuất báo cáo tuân thủ L3
ra logs/mercedes_l3_validation_report.json.

Cách chạy (cần CARLA server đang mở):
    .venvCarLa\\Scripts\\python.exe evaluate_l3.py
    .venvCarLa\\Scripts\\python.exe evaluate_l3.py --profiles clear,heavy_rain,fog --seconds 15
"""
import os
import sys
import time
import queue
import argparse
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
from mot_tracker import MultiObjectTracker
from odd_monitor import ODDMonitor
from mrm_controller import L3StateMachine
from weather_model import estimate_conditions
from weather_config import load_weather_profiles, weather_kwargs
from collision_sensor import CollisionSensor
from kpi import KpiRecorder
from ego_control import spawn_ego_safe
from l3_report import assess_profile, write_report


def _rgb(img):
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, :3].copy()


def _lidar(meas):
    p = np.frombuffer(meas.raw_data, dtype=np.float32)
    return np.reshape(p, (int(p.shape[0] / 4), 4)).copy()


def _set_hazard(vehicle, on):
    try:
        s = carla.VehicleLightState(
            carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker) \
            if on else carla.VehicleLightState.NONE
        vehicle.set_light_state(s)
    except Exception:
        pass


def run_scenario(client, world, bp_lib, name, profile, shared, args):
    """Chạy một kịch bản thời tiết, trả về dict đánh giá."""
    object_tracker, lidar_processor, sensor_fusion = shared
    actors = []
    camera = lidar = collision = None
    dt = cfg.FIXED_DELTA

    # Thành phần CÓ TRẠNG THÁI -> tạo mới mỗi kịch bản.
    safety = ActiveSafetySystem(
        lane_half_width_m=cfg.LANE_HALF_WIDTH_M, min_cluster_points=cfg.MIN_CLUSTER_POINTS,
        reaction_time_s=cfg.REACTION_TIME_S, min_safe_dist_m=cfg.MIN_SAFE_DIST_M,
        critical_ttc_s=cfg.CRITICAL_TTC_S, warning_ttc_s=cfg.WARNING_TTC_S,
        min_closing_speed=cfg.MIN_CLOSING_SPEED, enable_evasion=cfg.ENABLE_EVASION,
        evade_lookahead_m=cfg.EVADE_LOOKAHEAD_M, lead_slow_ratio=cfg.LEAD_SLOW_RATIO,
        follow_speed_diff=cfg.TM_FOLLOW_SPEED_DIFF)
    mot = MultiObjectTracker(dt=dt)
    odd_monitor = ODDMonitor()
    l3_sm = L3StateMachine(tor_window_s=cfg.TOR_WINDOW_S, mrm_decel_frac=cfg.MRM_DECEL_FRAC)
    kpi = KpiRecorder(dt=dt, scenario=name)

    # Điều kiện ODD từ hồ sơ thời tiết.
    conditions = estimate_conditions(
        precipitation=profile.get('precipitation', 0.0),
        fog_density=profile.get('fog_density', 0.0),
        wetness=profile.get('wetness', 0.0),
        precipitation_deposits=profile.get('precipitation_deposits', 0.0))
    odd = odd_monitor.classify(conditions)
    safety.set_conditions(conditions['mu'], odd['gap_multiplier'])
    expect_odd = profile.get('expect_odd', odd['state'])

    # Sự kiện theo dõi.
    mrm_triggered = False
    tor_count = 0
    false_diseng = 0
    late_braking = 0
    prev_l3 = 'L3_ACTIVE'
    prev_brake = False

    try:
        world.set_weather(carla.WeatherParameters(**weather_kwargs(profile)))

        ego, _ = spawn_ego_safe(world, bp_lib.find('vehicle.tesla.model3'))
        actors.append(ego)

        tm = client.get_trafficmanager(cfg.TM_PORT)
        tm.set_synchronous_mode(True)
        tm.distance_to_leading_vehicle(ego, cfg.TM_LEADING_DISTANCE_M)
        ego.set_autopilot(True, tm.get_port())
        engaged = True
        last_lc = -(10 ** 9)
        cooldown = cfg.LANE_CHANGE_COOLDOWN_S / dt

        # Xe chướng ngại đứng yên phía trước (để kiểm AEB dưới thời tiết).
        if not args.no_hazard:
            wp = world.get_map().get_waypoint(ego.get_location()).next(cfg.HAZARD_DISTANCE_M)
            if wp:
                tf = wp[0].transform
                tf.location.z += 0.3
                hz = world.try_spawn_actor(bp_lib.find('vehicle.audi.etron'), tf)
                if hz:
                    hz.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                    actors.append(hz)

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(cfg.CAM_WIDTH))
        cam_bp.set_attribute('image_size_y', str(cfg.CAM_HEIGHT))
        cam_bp.set_attribute('fov', str(cfg.CAM_FOV))
        lid_bp = bp_lib.find('sensor.lidar.ray_cast')
        lid_bp.set_attribute('range', '60')
        lid_bp.set_attribute('rotation_frequency', str(cfg.FPS))
        lid_bp.set_attribute('points_per_second', '200000')
        lid_bp.set_attribute('channels', '32')

        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=cfg.CAM_X, z=cfg.CAM_Z)), attach_to=ego)
        lidar = world.spawn_actor(lid_bp, carla.Transform(carla.Location(x=cfg.LIDAR_X, z=cfg.LIDAR_Z)), attach_to=ego)
        actors.extend([camera, lidar])
        collision = CollisionSensor(world, ego, fps=cfg.FPS)
        actors.append(collision.sensor)

        iq, lq = queue.Queue(), queue.Queue()
        camera.listen(iq.put)
        lidar.listen(lq.put)

        n_frames = int(args.seconds * cfg.FPS)
        for frame in range(n_frames):
            world.tick()
            rgb = _rgb(iq.get(timeout=2.0))
            pc = _lidar(lq.get(timeout=2.0))

            _, detections, _ = object_tracker.process(rgb, None, frame)
            obstacles = lidar_processor.extract_obstacles(pc)
            fused = sensor_fusion.fuse(detections, obstacles)

            vel = ego.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            tracks = mot.update(fused, dt)
            decision = safety.update(speed, fused, obstacles, dt, tracks=tracks)
            l3 = l3_sm.update(odd['state'], args.driver_takeover, speed, conditions['mu'], dt,
                              critical=odd['critical'])

            # Áp lệnh (ưu tiên: L3 override > AEB > lái thường) — giống chinh.py.
            if l3['override']:
                if engaged:
                    ego.set_autopilot(False); engaged = False
                bcmd = 1.0 if l3['state'] == 'SAFE_STOP' else min(1.0, l3['target_decel_ms2'] / 6.0)
                ego.apply_control(carla.VehicleControl(brake=bcmd, hand_brake=(l3['state'] == 'SAFE_STOP')))
                _set_hazard(ego, True)
            elif decision.action == "BRAKE":
                _set_hazard(ego, l3['hazard'])
                if engaged:
                    ego.set_autopilot(False); engaged = False
                ego.apply_control(carla.VehicleControl(brake=decision.brake))
            else:
                _set_hazard(ego, l3['hazard'])
                if not engaged:
                    ego.set_autopilot(True, tm.get_port()); engaged = True
                drive_diff = 40.0 if odd['state'] == "DEGRADED" else cfg.TM_DEFAULT_SPEED_DIFF
                if decision.action == "SLOW":
                    tm.vehicle_percentage_speed_difference(ego, max(decision.slow_pct, drive_diff))
                elif decision.action in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
                    tm.vehicle_percentage_speed_difference(ego, drive_diff)
                    if frame - last_lc > cooldown:
                        tm.force_lane_change(ego, decision.action == "LANE_CHANGE_RIGHT")
                        last_lc = frame
                else:
                    tm.vehicle_percentage_speed_difference(ego, drive_diff)

            # --- Ghi KPI + sự kiện ---
            threat = decision.threat or {}
            kpi.add(t=frame * dt, speed_ms=speed, distance_m=threat.get('distance_m'),
                    ttc_s=threat.get('ttc_s'), state=decision.state, collisions=collision.count)

            if l3['state'] in ('MRM_EXECUTING', 'SAFE_STOP'):
                mrm_triggered = True
            if l3['state'] == 'TAKEOVER_REQUEST' and prev_l3 != 'TAKEOVER_REQUEST':
                tor_count += 1
            if odd['state'] == 'NORMAL' and l3['state'] != 'L3_ACTIVE' and prev_l3 == 'L3_ACTIVE':
                false_diseng += 1
            is_brake = decision.action == "BRAKE"
            if is_brake and not prev_brake:
                ttc = threat.get('ttc_s')
                if ttc is not None and ttc < 0.8:
                    late_braking += 1
            prev_l3, prev_brake = l3['state'], is_brake

        summary = kpi.summary()
        result = assess_profile(name, expect_odd, odd['state'], summary,
                                mrm_triggered, tor_count, false_diseng, late_braking)
        os.makedirs("logs", exist_ok=True)
        kpi.write_csv(os.path.join("logs", f"kpi_{name}.csv"))
        verdict = "PASS" if result["pass"] else "FAIL"
        print(f"  [{name}] ODD={odd['state']} MRM={mrm_triggered} "
              f"collisions={summary['collisions']} minTTC={summary['min_ttc_s']} -> {verdict}")
        return result

    finally:
        for s in (camera, lidar):
            if s is not None and s.is_alive:
                s.stop()
        if collision is not None:
            collision.stop()
        if actors:
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        world.tick()


def main():
    parser = argparse.ArgumentParser(description="Mercedes L3 validation harness")
    parser.add_argument('--profiles', type=str, default='all',
                        help="danh sách hồ sơ ngăn cách dấu phẩy, hoặc 'all'")
    parser.add_argument('--seconds', type=float, default=15.0, help="thời lượng mỗi kịch bản")
    parser.add_argument('--no-hazard', action='store_true', help="không đặt xe chướng ngại")
    parser.add_argument('--driver-takeover', action='store_true',
                        help="mô phỏng tài xế tiếp quản khi có TOR")
    args = parser.parse_args()

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    profiles = load_weather_profiles(cfg.WEATHER_CONFIG_PATH)
    names = list(profiles.keys()) if args.profiles == 'all' else \
        [p.strip() for p in args.profiles.split(',') if p.strip() in profiles]
    if not names:
        sys.exit(f"[Eval] Không có hồ sơ hợp lệ. Có: {list(profiles.keys())}")

    print("[Eval] Nạp mô hình nhận thức (dùng chung cho mọi kịch bản)...")
    shared = (
        EnsembleVehicleTracker(device=cfg.YOLO_DEVICE, use_ensemble=cfg.USE_ENSEMBLE,
                               model_a=cfg.YOLO_MODEL_A, model_b=cfg.YOLO_MODEL_B,
                               imgsz=cfg.YOLO_IMGSZ, half=cfg.YOLO_HALF,
                               use_optimized=cfg.YOLO_USE_OPTIMIZED),
        LidarProcessor(eps=0.6, min_samples=5, max_points=cfg.LIDAR_MAX_POINTS),
        SensorFusion(width=cfg.CAM_WIDTH, height=cfg.CAM_HEIGHT, fov=cfg.CAM_FOV,
                     lidar_y_sign=cfg.LIDAR_Y_SIGN, real_heights_m=cfg.REAL_HEIGHTS_M,
                     default_height_m=cfg.DEFAULT_HEIGHT_M,
                     depth_min_m=cfg.DEPTH_MIN_M, depth_max_m=cfg.DEPTH_MAX_M),
    )

    original = world.get_settings()
    sync = world.get_settings()
    sync.synchronous_mode = True
    sync.fixed_delta_seconds = cfg.FIXED_DELTA
    world.apply_settings(sync)

    results = []
    try:
        for name in names:
            print(f"[Eval] === Kịch bản: {name} ===")
            try:
                results.append(run_scenario(client, world, bp_lib, name, profiles[name], shared, args))
            except Exception as e:
                print(f"  [{name}] LỖI: {e}")
                results.append(assess_profile(name, profiles[name].get('expect_odd', '?'),
                                              'ERROR', {"collisions": -1}, False, 0, 0, 0))
    finally:
        world.apply_settings(original)

    report = write_report(cfg.L3_REPORT_PATH, results)
    s = report["summary"]
    print("\n================ L3 VALIDATION SUMMARY ================")
    print(f"  Kịch bản: {s['scenarios']} | ĐẠT: {s['passed']} | TRƯỢT: {s['failed']} "
          f"| Tỉ lệ: {s['pass_rate_pct']}%")
    print(f"  Báo cáo: {cfg.L3_REPORT_PATH}")
    print("======================================================")


if __name__ == '__main__':
    main()
