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
from collision_sensor import CollisionSensor
from kpi import KpiRecorder
from ego_control import EgoController, spawn_ego_safe, destroy_all_actors
from l3_report import write_report
import scenario_library as lib

REPORT_PATH = "logs/scenario_test_report.json"


def _rgb(img):
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, :3].copy()


def _pc(meas):
    p = np.frombuffer(meas.raw_data, dtype=np.float32)
    return np.reshape(p, (int(p.shape[0] / 4), 4)).copy()


def run_one(client, world, bp_lib, spec, shared, args):
    object_tracker, lidar_processor, sensor_fusion = shared
    dt = cfg.FIXED_DELTA
    actors = []
    camera = lidar = collision = None
    scenario = None

    safety = ActiveSafetySystem(
        lane_half_width_m=cfg.LANE_HALF_WIDTH_M, min_cluster_points=cfg.MIN_CLUSTER_POINTS,
        reaction_time_s=cfg.REACTION_TIME_S, min_safe_dist_m=cfg.MIN_SAFE_DIST_M,
        critical_ttc_s=cfg.CRITICAL_TTC_S, warning_ttc_s=cfg.WARNING_TTC_S,
        min_closing_speed=cfg.MIN_CLOSING_SPEED, enable_evasion=cfg.ENABLE_EVASION,
        evade_lookahead_m=cfg.EVADE_LOOKAHEAD_M, lead_slow_ratio=cfg.LEAD_SLOW_RATIO,
        follow_speed_diff=cfg.TM_FOLLOW_SPEED_DIFF)
    mot = MultiObjectTracker(dt=dt)
    l3_sm = L3StateMachine(tor_window_s=cfg.TOR_WINDOW_S, mrm_decel_frac=cfg.MRM_DECEL_FRAC)
    kpi = KpiRecorder(dt=dt, scenario=spec.name)
    odd = ODDMonitor().classify(estimate_conditions())  # trời quang -> NORMAL
    safety.set_conditions(estimate_conditions()['mu'], odd['gap_multiplier'])

    reacted = braked = evaded = False

    try:
        world.set_weather(carla.WeatherParameters.ClearNoon)
        ego, _ = spawn_ego_safe(world, bp_lib.find('vehicle.tesla.model3'),
                                preferred_index=args.spawn)
        actors.append(ego)

        tm = client.get_trafficmanager(cfg.TM_PORT)
        tm.set_synchronous_mode(True)
        tm.distance_to_leading_vehicle(ego, cfg.TM_LEADING_DISTANCE_M)
        ego.set_autopilot(True, tm.get_port())
        controller = EgoController(ego, tm, cfg)

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(cfg.CAM_WIDTH))
        cam_bp.set_attribute('image_size_y', str(cfg.CAM_HEIGHT))
        cam_bp.set_attribute('fov', str(cfg.CAM_FOV))
        lid_bp = bp_lib.find('sensor.lidar.ray_cast')
        for k, v in (('range', '60'), ('rotation_frequency', str(cfg.FPS)),
                     ('points_per_second', '200000'), ('channels', '32')):
            lid_bp.set_attribute(k, v)
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=cfg.CAM_X, z=cfg.CAM_Z)), attach_to=ego)
        lidar = world.spawn_actor(lid_bp, carla.Transform(carla.Location(x=cfg.LIDAR_X, z=cfg.LIDAR_Z)), attach_to=ego)
        actors.extend([camera, lidar])
        collision = CollisionSensor(world, ego, fps=cfg.FPS)
        actors.append(collision.sensor)

        iq, lq = queue.Queue(), queue.Queue()
        camera.listen(iq.put)
        lidar.listen(lq.put)

        # Cho ego chạy vài tick rồi mới dựng vật cản (để có tốc độ).
        for _ in range(int(1.5 * cfg.FPS)):
            world.tick(); iq.get(timeout=2.0); lq.get(timeout=2.0)
        scenario = spec.builder(world, ego, tm)

        for frame in range(int(args.seconds * cfg.FPS)):
            world.tick()
            rgb = _rgb(iq.get(timeout=2.0))
            pc = _pc(lq.get(timeout=2.0))

            _, detections, _ = object_tracker.process(rgb, None, frame)
            obstacles = lidar_processor.extract_obstacles(pc)
            fused = sensor_fusion.fuse(detections, obstacles)
            vel = ego.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            tracks = mot.update(fused, dt)
            decision = safety.update(speed, fused, obstacles, dt, tracks=tracks)
            l3 = l3_sm.update(odd['state'], False, speed, estimate_conditions()['mu'], dt, critical=odd['critical'])

            controller.apply(decision, l3, odd['state'], frame)
            scenario.tick(frame, ego, world)

            if decision.state != "NORMAL":
                reacted = True
            if decision.action == "BRAKE":
                braked = True
            if "EVADE" in decision.state:
                evaded = True

            threat = decision.threat or {}
            kpi.add(t=frame * dt, speed_ms=speed, distance_m=threat.get('distance_m'),
                    ttc_s=threat.get('ttc_s'), state=decision.state, collisions=collision.count)

        summary = kpi.summary()
        os.makedirs("logs", exist_ok=True)
        kpi.write_csv(os.path.join("logs", f"kpi_scn_{spec.name}.csv"))

        collisions = summary["collisions"]
        reasons = []
        if collisions > 0:
            reasons.append(f"{collisions} va chạm")
        if not reacted:
            reasons.append("không phản ứng (không phát hiện vật cản?)")
        passed = collisions == 0 and reacted

        print(f"  [{spec.name}] collisions={collisions} reacted={reacted} "
              f"braked={braked} evaded={evaded} minTTC={summary['min_ttc_s']} "
              f"minDist={summary['min_distance_m']} -> {'PASS' if passed else 'FAIL'}")
        return {
            "profile": spec.name, "name": spec.name, "category": spec.category,
            "tests": spec.tests, "collisions": collisions, "reacted": reacted,
            "braked": braked, "evaded": evaded,
            "min_distance_m": summary["min_distance_m"], "min_ttc_s": summary["min_ttc_s"],
            "max_decel_ms2": summary["max_decel_ms2"], "kpi": summary,
            "pass": passed, "reasons": reasons,
        }

    except Exception as e:
        print(f"  [{spec.name}] LỖI: {e}")
        return {"profile": spec.name, "name": spec.name, "category": spec.category,
                "tests": spec.tests, "collisions": -1, "pass": False, "reasons": [str(e)]}
    finally:
        if scenario is not None:
            scenario.destroy(client)
        for s in (camera, lidar):
            if s is not None and s.is_alive:
                s.stop()
        if collision is not None:
            collision.stop()
        if actors:
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        # Vài tick để CARLA thực sự hủy actor trước khi kịch bản sau spawn lại.
        for _ in range(4):
            world.tick()


def main():
    parser = argparse.ArgumentParser(description="Ported ScenarioRunner test suite")
    parser.add_argument('--scenarios', type=str, default='all', help="danh sách tên, hoặc 'all'")
    parser.add_argument('--category', type=str, default=None,
                        help="lọc theo nhóm: lead|crossing|cutin|junction|oncoming")
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--spawn', type=int, default=0, help="chỉ số spawn point cho ego")
    parser.add_argument('--clean', action='store_true',
                        help="dọn mọi xe/người/cảm biến còn sót trước khi chạy")
    args = parser.parse_args()

    specs = lib.CATALOG
    if args.category:
        specs = [s for s in specs if s.category == args.category]
    if args.scenarios != 'all':
        want = [x.strip() for x in args.scenarios.split(',')]
        specs = [s for s in specs if s.name in want]
    if not specs:
        sys.exit(f"[Scn] Không có kịch bản khớp. Có: {lib.list_names()}")

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    if args.clean:
        n = destroy_all_actors(world, client)
        print(f"[Scn] --clean: đã dọn {n} actor còn sót.")

    print(f"[Scn] Nạp mô hình nhận thức (dùng chung). Sẽ chạy {len(specs)} kịch bản.")
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
        for spec in specs:
            print(f"[Scn] === {spec.name} ({spec.category}) — {spec.tests} ===")
            results.append(run_one(client, world, bp_lib, spec, shared, args))
    finally:
        world.apply_settings(original)

    report = write_report(REPORT_PATH, results,
                          title="CARLA ScenarioRunner Port — AEB/Avoidance Test Report")
    s = report["summary"]
    print("\n============== SCENARIO TEST SUMMARY ==============")
    print(f"  Kịch bản: {s['scenarios']} | ĐẠT: {s['passed']} | TRƯỢT: {s['failed']} "
          f"| Tỉ lệ: {s['pass_rate_pct']}%")
    print(f"  Báo cáo: {REPORT_PATH}")
    print("==================================================")


if __name__ == '__main__':
    main()
