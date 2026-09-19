# -*- coding: utf-8 -*-
"""
Self-test KHÔNG cần CARLA cho lớp hình học + an toàn.

Chạy:  python selftest.py
Chỉ phụ thuộc numpy (cho sensor_fusion) và stdlib (active_safety). KHÔNG import
carla / torch / cv2 / ultralytics, nên kiểm được phần "toán" của pipeline trước
khi mở server CARLA.
"""
import os
import sys
import time
import numpy as np

# Ép UTF-8 để in tiếng Việt được trên mọi console (tránh lỗi cp1252 trên Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from sensor_fusion import SensorFusion          # noqa: E402
from active_safety import ActiveSafetySystem    # noqa: E402
from mot_tracker import MultiObjectTracker      # noqa: E402
from kpi import KpiRecorder                      # noqa: E402
from friction import stopping_distance          # noqa: E402
from weather_model import estimate_conditions   # noqa: E402
from weather_config import load_weather_profiles  # noqa: E402
from odd_monitor import ODDMonitor              # noqa: E402
from mrm_controller import L3StateMachine       # noqa: E402
from sensor_fusion_eval import SensorFusionEval  # noqa: E402
from l3_report import assess_profile, build_report, write_report  # noqa: E402
import scenario_library as scnlib  # noqa: E402
from rl_experiment import SpeedExperiment  # noqa: E402
from rl_env import SpeedControlEnv  # noqa: E402
from rl_speed_controller import RLSpeedController  # noqa: E402
from turn_intent import intent_to_right  # noqa: E402
from road_geometry import project_to_path, _select_next_waypoint  # noqa: E402
from scenario_acceptance import assess_scenario  # noqa: E402
from sensor_sync import (LatestFrameReader, OptionalFrameReader,
                         SensorFrameError, SensorSyncStats, put_latest,
                         retrieve_exact_frame, warmup_sensor_streams)  # noqa: E402
from local_planner import LocalPlanner  # noqa: E402
from planner import BehaviorPlanner, DrivingState  # noqa: E402
from lateral_controller import LateralController  # noqa: E402
from longitudinal_controller import LongitudinalController  # noqa: E402
from ego_motion import EgoMotionEstimator  # noqa: E402
from scene_semantics import summarize_traffic_controls, StopSignState  # noqa: E402
from sensor_setup import (configure_camera_blueprint, configure_lidar_blueprint,
                          configure_radar_blueprint)  # noqa: E402
from radar_processor import RadarProcessor  # noqa: E402
from sensor_health import SensorHealthMonitor  # noqa: E402
from inference_scheduler import (LatestFrameScheduler,
                                 usable_inference_payload)  # noqa: E402
from perception_contracts import (LaneEstimate, TaggedInferenceResult,
                                  validate_frame_age)  # noqa: E402
from lane_source import LaneSourceArbiter, map_lane_estimate  # noqa: E402
from learned_lane import GroundProjector, LearnedLaneAdapter  # noqa: E402
from replay_io import ReplayWriter, ReplayReader  # noqa: E402
from pipeline_metrics import PipelineMetrics  # noqa: E402
from traffic_light import TrafficLightTemporalVoter  # noqa: E402
from lidar_processor import LidarProcessor  # noqa: E402
from fault_acceptance import assess_fault_behavior  # noqa: E402
from weather_acceptance import assess_weather_behavior  # noqa: E402
from simulation_guard import (SynchronousWorldSession,
                              enter_synchronous_mode, get_or_load_world,
                              restore_world_settings, same_carla_map)  # noqa: E402
from audit_week2_dataset import evaluate_gates  # noqa: E402
from collect_carla_dataset import (parse_target_classes,
                                   prioritize_static_stop_sign_spawns,
                                   traffic_light_spawn_score)  # noqa: E402
from collect_targeted_supplement import build_targeted_plan  # noqa: E402
from collect_stopsign_supplement import build_stop_sign_plan  # noqa: E402
from audit_dataset_archive import normalized_member_name  # noqa: E402
from audit_object_sources import evaluate_source_gates  # noqa: E402
from prepare_stopsign_training_source import (deduplicate_records,
                                              parse_yolo_row)  # noqa: E402
from build_yolo_adas_dataset import xyxy_to_yolo  # noqa: E402
from training_utils import deterministic_subset  # noqa: E402
from wait_for_carla import wait_until_ready  # noqa: E402
import config as runtime_cfg  # noqa: E402
import queue as _queue  # noqa: E402


def obs(x, y, z=0.0, n=20):
    return {"centroid": [x, y, z], "point_count": n}


def det(bbox, cls="Car"):
    return {"bbox": bbox, "class": cls, "confidence": 0.9}


passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


# ---------------------------------------------------------------------------
# 1) SensorFusion: phép chiếu LiDAR -> ảnh và khớp bbox
# ---------------------------------------------------------------------------
print("[1] SensorFusion (1280x720, fov90 -> fx=640, cx=640, cy=360)")
fus = SensorFusion(width=1280, height=720, fov=90.0)

# Cụm ngay trước mặt @10m -> chiếu về tâm (640,360); box ở giữa phải khớp.
fused = fus.fuse([det((600, 300, 680, 420))], [obs(10.0, 0.0)])
check("cluster thang truoc gan bbox trung tam -> distance_m ~ 10 (lidar)",
      abs(fused[0]["distance_m"] - 10.0) < 0.1 and fused[0]["distance_source"] == "lidar")

# Cụm lệch phải (y=+2 @10m) -> u=768; box bên phải phải khớp, box giữa thì không.
fused = fus.fuse([det((740, 300, 800, 420))], [obs(10.0, 2.0)])
check("cluster lech phai -> khop box ben phai (lidar)",
      fused[0]["distance_source"] == "lidar" and abs(fused[0]["distance_m"] - 10.0) < 0.1)

# Không cụm LiDAR nào khớp -> mono fallback theo chiều cao box.
# Car cao 1.5m, box cao 96px -> d = 640*1.5/96 = 10.0m
fused = fus.fuse([det((600, 300, 680, 396))], [])
check("khong co lidar -> mono fallback ~ 10m",
      fused[0]["distance_source"] == "mono" and abs(fused[0]["distance_m"] - 10.0) < 0.3)

# Pose 6-DoF: LiDAR cao hơn camera 0.1m -> projection dịch lên trên tâm ảnh.
fus_pose = SensorFusion(
    width=1280, height=720, fov=90.0,
    camera_pose=(1.5, 0.0, 2.4, 0.0, 0.0, 0.0),
    lidar_pose=(1.5, 0.0, 2.5, 0.0, 0.0, 0.0))
projected = fus_pose._project_cluster((10.0, 0.0, 0.0))
check("extrinsic z lidar-camera duoc ap dung vao projection",
      projected is not None and 352.0 < projected[1] < 355.0)

# Một cluster không được cấp cho hai bbox chồng nhau.
fused = fus.fuse([
    det((590, 290, 690, 420), "Car"),
    det((600, 300, 680, 410), "Truck"),
], [obs(10.0, 0.0)])
check("fusion one-to-one: mot cluster chi gan mot detection",
      sum(d["distance_source"] == "lidar" for d in fused) == 1)
check("fusion xuat uncertainty: lidar chac hon mono",
      min(d["distance_std_m"] for d in fused if d["distance_source"] == "lidar") <
      min(d["distance_std_m"] for d in fused if d["distance_source"] == "mono"))

# ---------------------------------------------------------------------------
# 2) ActiveSafetySystem: máy trạng thái
# ---------------------------------------------------------------------------
print("[2] ActiveSafetySystem FSM")
dt = 0.05

# a) Không vật cản -> NORMAL
s = ActiveSafetySystem()
d = s.update(8.0, [], [], dt)
check("khong vat can -> NORMAL/DRIVE", d.state == "NORMAL" and d.action == "DRIVE")

# b) Vật cản rất gần trong làn (4m < min_safe 6m) -> EMERGENCY_BRAKE
s = ActiveSafetySystem()
d = s.update(8.0, [], [obs(4.0, 0.0)], dt)
check("vat can @4m trong lan -> EMERGENCY_BRAKE", d.state == "EMERGENCY_BRAKE" and d.brake == 1.0)

# c) Vật ngoài làn (lệch phải, không có gì thẳng trước) -> NORMAL
s = ActiveSafetySystem()
d = s.update(8.0, [], [obs(10.0, 3.0)], dt)
check("vat ngoai lan -> NORMAL", d.state == "NORMAL")

# d) Bám theo: trong khoảng an toàn động nhưng lead gần giữ tốc -> FOLLOW/SLOW
s = ActiveSafetySystem()
s.update(10.0, [], [obs(12.0, 0.0)], dt)               # khởi tạo lịch sử
d = s.update(10.0, [], [obs(11.9, 0.0)], dt)           # closing ~2 m/s (chậm)
check("lead giu toc, trong safe-dist -> FOLLOW/SLOW",
      d.state == "FOLLOW" and d.action == "SLOW" and d.slow_pct > 0)

# e) Tránh né: lead chậm/đứng yên, hai làn bên trống -> EVADE_RIGHT
s = ActiveSafetySystem()
s.update(10.0, [], [obs(25.0, 0.0)], dt)               # prev = 25
d = s.update(10.0, [], [obs(24.5, 0.0)], dt)           # closing ~10 m/s (ego lao toi)
check("lead cham + lan trong -> EVADE_RIGHT",
      d.state == "EVADE_RIGHT" and d.action == "LANE_CHANGE_RIGHT")

# f) Tránh né khi làn phải bị chặn -> EVADE_LEFT
s = ActiveSafetySystem()
s.update(10.0, [], [obs(25.0, 0.0), obs(10.0, 3.5)], dt)
d = s.update(10.0, [], [obs(24.5, 0.0), obs(10.0, 3.5)], dt)  # y=3.5 chặn làn phải
check("lan phai bi chan -> EVADE_LEFT",
      d.state == "EVADE_LEFT" and d.action == "LANE_CHANGE_LEFT")

# ---------------------------------------------------------------------------
# 2b) HỒI QUY: vật TĨNH ở tốc độ THẤP — sửa lỗi "xe bò thẳng vào vật gây va chạm"
#     (bug ConstructionObstacle: braked=false, evaded=false, 18 va chạm).
# ---------------------------------------------------------------------------
print("[2b] Regression: low-speed stationary obstacle (no creep-into-collision)")

# a) Ego BÒ CHẬM (1.5 m/s), vật tĩnh chắn làn, HAI làn bên chặn -> phải PHANH DỪNG,
#    KHÔNG được hạ về SLOW rồi bò tới (đây chính là hành vi cũ gây va chạm).
s = ActiveSafetySystem()
s.update(1.5, [], [obs(9.2, 0.0), obs(9.0, 3.5), obs(9.0, -3.5)], dt)
d = s.update(1.5, [], [obs(9.0, 0.0), obs(9.0, 3.5), obs(9.0, -3.5)], dt)
check("bo cham + vat tinh chan lan -> BRAKE (khong creep/SLOW)", d.action == "BRAKE")

# b) Ego bò chậm, vật tĩnh nhưng làn bên TRỐNG -> NÉ (tránh) thay vì bò tới.
s = ActiveSafetySystem()
s.update(1.5, [], [obs(20.2, 0.0)], dt)
d = s.update(1.5, [], [obs(20.0, 0.0)], dt)
check("bo cham + vat tinh, lan trong -> EVADE (tranh, khong bo toi)", "EVADE" in d.state)

# c) LATCH bền vững qua khung MẤT DẤU: đang phanh mà cụm LiDAR chớp tắt 1 khung
#    (vật thấp/thưa như cọc) -> VẪN giữ phanh, không lao tới.
s = ActiveSafetySystem()
s.update(10.0, [], [obs(5.0, 0.0)], dt)          # nearest 5 < min_safe -> emergency + latch
d1 = s.update(10.0, [], [], dt)                  # mất dấu 1 khung
check("mat dau 1 khung khi dang phanh -> van BRAKE (latch ben)", d1.action == "BRAKE")
for _ in range(6):                               # mất dấu đủ lâu -> mới nhả
    dN = s.update(10.0, [], [], dt)
check("mat dau du lau (> tol) -> nha phanh ve NORMAL", dN.state == "NORMAL")

# d) KHÔNG hồi quy phần bám xe: lead ĐANG CHẠY (~8 m/s) trong vùng cảnh báo -> FOLLOW,
#    không phanh oan (chỉ vật CHẬM/ĐỨNG mới bị xử lý dứt khoát).
s = ActiveSafetySystem()
s.update(12.0, [], [obs(14.0, 0.0)], dt)
d = s.update(12.0, [], [obs(13.8, 0.0)], dt)     # closing ~4 -> obstacle_speed ~8 m/s
check("lead dang chay trong warning -> FOLLOW (khong phanh oan)", d.state == "FOLLOW")

# ---------------------------------------------------------------------------
# 3) MultiObjectTracker: ID ổn định + ước lượng vận tốc tương đối
# ---------------------------------------------------------------------------
print("[3] MultiObjectTracker (vat tien lai gan 5 m/s)")
mot = MultiObjectTracker(dt=0.05, gate_m=4.0, min_hits=3, max_age=5)
x = 30.0
tracks = []
for _ in range(25):
    x -= 0.25  # 5 m/s * 0.05 s
    tracks = mot.update([{"distance_m": x, "lateral_m": 0.0,
                          "class": "Car", "bbox": (0, 0, 10, 10)}], dt=0.05)
check("tao duoc 1 track on dinh", len(tracks) == 1)
if tracks:
    vx, vy = tracks[0].vel
    check("uoc luong vx ~ -5 m/s (dang tien lai gan)", -6.5 < vx < -3.5)
    path = tracks[0].predict_path(horizon_s=2.0, step_s=0.5)
    check("du doan quy dao tien ve phia ego (x giam dan)",
          len(path) == 4 and path[-1][0] < tracks[0].pos[0])

# ---------------------------------------------------------------------------
# 4) Predictive AEB: phanh SỚM dựa trên vận tốc track (corridor rỗng)
# ---------------------------------------------------------------------------
print("[4] Predictive AEB")


class _StubTrack:
    def __init__(self, x, y, vx, vy, class_name="Car"):
        self.pos = (x, y)
        self.vel = (vx, vy)
        self.class_name = class_name


s = ActiveSafetySystem()
d = s.update(12.0, [], [], 0.05, tracks=[_StubTrack(20.0, 0.0, -15.0, 0.0)])
check("track lao toi nhanh (TTC~1.3s) -> EMERGENCY_BRAKE", d.state == "EMERGENCY_BRAKE")

s = ActiveSafetySystem()
d = s.update(8.0, [], [], 0.05, tracks=[_StubTrack(20.0, 0.0, -3.0, 0.0)])
check("track tien cham (TTC~6.7s), con xa -> NORMAL", d.state == "NORMAL")

s = ActiveSafetySystem()
d = s.update(12.0, [], [], 0.05, tracks=[_StubTrack(20.0, 3.0, -15.0, 0.0)])
check("track ngoai lan -> NORMAL du toc do cao", d.state == "NORMAL")

# VRU được mở rộng hành lang và horizon vì quỹ đạo người/xe đạp khó đoán hơn.
s = ActiveSafetySystem(vru_lateral_margin_m=0.8)
d_vru = s.update(8.0, [], [], 0.05,
                 tracks=[_StubTrack(16.0, 2.2, -8.0, 0.0, "Person")])
s = ActiveSafetySystem(vru_lateral_margin_m=0.8)
d_car = s.update(8.0, [], [], 0.05,
                 tracks=[_StubTrack(16.0, 2.2, -8.0, 0.0, "Car")])
check("VRU sat mep lan kich hoat som hon xe cung vi tri",
      d_vru.state != "NORMAL" and d_car.state == "NORMAL")

# Ego compensation: vật đứng yên trong world không bị hiểu nhầm là đang chạy.
mot_comp = MultiObjectTracker(dt=0.05, gate_m=4.0, min_hits=3, max_age=5)
x = 30.0
tracks_comp = []
for i in range(30):
    if i:
        x -= 0.25
    tracks_comp = mot_comp.update(
        [{"distance_m": x, "lateral_m": 0.0, "distance_std_m": 0.2,
          "class": "Car", "bbox": (0, 0, 10, 10)}], dt=0.05,
        ego_motion={"valid": i > 0, "dx": 0.25, "dy": 0.0, "dyaw": 0.0},
        ego_speed_ms=5.0)
check("ego-motion compensation: vat tinh co world-speed gan 0",
      len(tracks_comp) == 1 and tracks_comp[0].ego_compensated and
      abs(tracks_comp[0].vel[0]) < 0.6)
if tracks_comp:
    relative_path = tracks_comp[0].predict_path(horizon_s=1.0, step_s=1.0)
    check("du doan relative van tru ego-speed de tinh collision",
          relative_path[0][0] < tracks_comp[0].pos[0] - 4.0)

# ---------------------------------------------------------------------------
# 5) KpiRecorder: tổng hợp chỉ số đánh giá
# ---------------------------------------------------------------------------
print("[5] KpiRecorder")
kpi = KpiRecorder(dt=0.1)
# Giả lập: giảm tốc từ 10 m/s về 0, khoảng cách thu hẹp, không va chạm.
speed = 10.0
dist = 25.0
for i in range(40):
    speed = max(0.0, speed - 0.4)
    dist = max(3.0, dist - 0.6)
    ttc = dist / speed if speed > 0.1 else 99.0
    kpi.add(t=i * 0.1, speed_ms=speed, distance_m=dist, ttc_s=ttc,
            state="EMERGENCY_BRAKE" if speed < 4 else "FOLLOW", collisions=0)
summary = kpi.summary()
check("min_distance_m duoc ghi nhan (~3m)", 2.5 <= summary["min_distance_m"] <= 3.5)
check("khong va cham -> success True", summary["collisions"] == 0 and summary["success"] is True)
check("max_decel > 0 (co phanh)", summary["max_decel_ms2"] > 0.0)

# ---------------------------------------------------------------------------
# 5b) Arbiter: cam kết + trễ (chống phân vân brake<->evade) — sửa lỗi tông vật cản
# ---------------------------------------------------------------------------
print("[5b] Decision arbiter (commitment + hysteresis)")

# a) Vật đứng chắn đường, HAI làn bên đều bị chặn -> PHANH dứt khoát (không SLOW).
s = ActiveSafetySystem()
s.update(10.0, [], [obs(20.4, 0.0), obs(10.0, 3.5), obs(10.0, -3.5)], dt)
d = s.update(10.0, [], [obs(20.0, 0.0), obs(10.0, 3.5), obs(10.0, -3.5)], dt)
check("vat dung, khong lan trong -> BRAKE_TO_STOP", d.state == "BRAKE_TO_STOP" and d.action == "BRAKE")

# b) Phanh khẩn cấp CHỐT LẠI: vật lùi ra một chút vẫn giữ phanh (hysteresis).
s = ActiveSafetySystem()
s.update(10.0, [], [obs(5.0, 0.0)], dt)                 # nearest 5 < min_safe -> emergency + latch
d = s.update(10.0, [], [obs(9.0, 0.0)], dt)            # lui ra 9m, chua ro an toan
check("phanh da chot -> giu phanh (khong nha som)", "BRAKE" in d.state and d.action == "BRAKE")

# c) Đã cam kết NÉ -> giữ nguyên hướng dù làn phải nhấp nháy "bị chặn" 1 frame.
s = ActiveSafetySystem()
n = 24.5
d = None
for _ in range(3):
    n -= 0.5
    d = s.update(10.0, [], [obs(n, 0.0)], dt)          # ~10 m/s -> EVADE_RIGHT (cam ket)
check("bat dau -> EVADE_RIGHT", d.state == "EVADE_RIGHT")
n -= 0.5
d2 = s.update(10.0, [], [obs(n, 0.0), obs(12.0, 3.5)], dt)  # lan phai nhap nhay bi chan
check("lan dich bi chan khi dang ne -> abort sang BRAKE", d2.action == "BRAKE")

# d) Vật phía SAU trong làn bên phải phải chặn quyết định đổi làn phải.
s = ActiveSafetySystem()
s.update(10.0, [], [obs(25.0, 0.0), obs(-6.0, 3.5)], dt)
d = s.update(10.0, [], [obs(24.5, 0.0), obs(-6.0, 3.5)], dt)
check("blind spot/rear gap chan lan phai -> chi ne trai", d.state == "EVADE_LEFT")

# e) Topology không cho đổi làn -> brake, dù occupancy bên cạnh đang trống.
s = ActiveSafetySystem()
ctx_no_lane = {
    "current_lane_id": 1,
    "left_exists": False, "right_exists": False,
    "left_change_allowed": False, "right_change_allowed": False,
}
s.update(10.0, [], [obs(25.0, 0.0)], dt, lane_context=ctx_no_lane)
d = s.update(10.0, [], [obs(24.5, 0.0)], dt, lane_context=ctx_no_lane)
check("khong co lan hop le de ne -> BRAKE_TO_STOP", d.action == "BRAKE")

# f) Cụm LiDAR nhỏ nhưng được nhánh sparse-safety xác nhận vẫn phải kích AEB.
s = ActiveSafetySystem()
small = obs(4.0, 0.0, n=2)
small["safety_critical"] = True
d = s.update(8.0, [], [small], dt)
check("sparse safety cluster 2 diem @4m -> EMERGENCY_BRAKE", d.action == "BRAKE")

# g) Track từ làn bên có vận tốc ngang cắt vào swept path -> predictive AEB.
s = ActiveSafetySystem()
d = s.update(10.0, [], [], dt, tracks=[_StubTrack(14.0, -3.2, -1.0, 1.5)])
check("track cut-in du kien vao lan <3s -> phan ung predictive", d.action == "BRAKE")

# h) Swept path cong: vật nằm trên centerline cong phải được thấy; vật chỉ nằm
# trong corridor thẳng cũ nhưng ngoài đường cong phải bị loại.
curve = [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0), (30.0, 10.0)]
along, lateral, _ = project_to_path(16.0, 3.0, curve)
check("project swept-path cong -> lateral gan 0", along > 10.0 and abs(lateral) < 0.5)
s = ActiveSafetySystem(enable_evasion=False)
d_on = s.update(8.0, [], [obs(16.0, 3.0)], dt, path_points=curve)
check("vat tren duong cong -> safety phan ung", d_on.state != "NORMAL")
s = ActiveSafetySystem(enable_evasion=False)
d_off = s.update(8.0, [], [obs(18.0, 0.0)], dt, path_points=curve)
check("vat ngoai swept-path cong -> khong bao gia", d_off.state == "NORMAL")

# ---------------------------------------------------------------------------
# 6) Friction-based stopping distance
# ---------------------------------------------------------------------------
print("[6] Friction stopping distance")
check("v=0 -> quang duong dung = 0", stopping_distance(0.0, 0.8) == 0.0)
check("v=10,mu=0.8 -> ~16.4m", abs(stopping_distance(10.0, 0.8, 1.0) - 16.37) < 0.3)
check("mu thap hon -> quang duong dai hon (duong uot)",
      stopping_distance(10.0, 0.4) > stopping_distance(10.0, 0.8))

# ---------------------------------------------------------------------------
# 7) Weather model + ODD monitor (Module E)
# ---------------------------------------------------------------------------
print("[7] Weather model + ODD monitor")
clear = estimate_conditions(0, 0, 0)
check("troi quang -> visibility 200, mu 0.9, snr 1.0",
      clear["visibility_m"] == 200.0 and clear["mu"] == 0.9 and clear["snr"] == 1.0)
storm = estimate_conditions(precipitation=100, fog_density=80, wetness=100)
check("bao -> visibility < 20m, mu ~0.4", storm["visibility_m"] < 20 and storm["mu"] < 0.45)

odd = ODDMonitor()
check("quang -> NORMAL", odd.classify(clear)["state"] == "NORMAL")
check("suong mu vua (vis 30) -> DEGRADED",
      odd.classify({"visibility_m": 30, "mu": 0.8, "snr": 0.8})["state"] == "DEGRADED")
check("ma sat thap (mu 0.5) -> DEGRADED",
      odd.classify({"visibility_m": 100, "mu": 0.5, "snr": 0.9})["state"] == "DEGRADED")
vio = odd.classify({"visibility_m": 8, "mu": 0.4, "snr": 0.5})
check("tam nhin < 20m -> VIOLATION + critical", vio["state"] == "VIOLATION" and vio["critical"] is True)

# Keep every configured weather expectation aligned with the deterministic
# engineering model used by the runtime and scenario harness.  This catches a
# profile/config drift before an expensive CARLA weather matrix is started.
_weather_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "weather_config.yaml")
_weather_profiles = load_weather_profiles(_weather_path)
for _profile_name, _profile in _weather_profiles.items():
    _profile_conditions = estimate_conditions(
        precipitation=_profile.get("precipitation", 0.0),
        fog_density=_profile.get("fog_density", 0.0),
        wetness=_profile.get("wetness", 0.0),
        precipitation_deposits=_profile.get("precipitation_deposits", 0.0))
    _observed_odd = odd.classify(_profile_conditions)["state"]
    check(f"weather profile {_profile_name}: expect_odd khop mo hinh",
          _observed_odd == str(_profile.get("expect_odd", "")).upper())

# ---------------------------------------------------------------------------
# 8) L3 MRM state machine (Module F)
# ---------------------------------------------------------------------------
print("[8] L3 MRM state machine")
sm = L3StateMachine(tor_window_s=1.0)
r = sm.update("VIOLATION", False, 10.0, 0.4, 0.1)          # L3_ACTIVE -> TAKEOVER_REQUEST
check("VIOLATION -> TAKEOVER_REQUEST", r["state"] == "TAKEOVER_REQUEST" and r["hazard"] is True)
for _ in range(12):
    r = sm.update("VIOLATION", False, 10.0, 0.4, 0.1)       # het 10s, khong tiep quan
check("het cua so 10s -> MRM_EXECUTING", r["state"] == "MRM_EXECUTING" and r["override"] is True)
r = sm.update("VIOLATION", False, 0.0, 0.4, 0.1)           # xe da dung
check("v~0 -> SAFE_STOP", r["state"] == "SAFE_STOP")

sm2 = L3StateMachine()
r = sm2.update("VIOLATION", False, 10.0, 0.4, 0.1, critical=True)
check("dieu kien tut nhanh (critical) -> MRM ngay", r["state"] == "MRM_EXECUTING")

sm3 = L3StateMachine()
sm3.update("VIOLATION", False, 10.0, 0.4, 0.1)             # -> TOR
r = sm3.update("VIOLATION", True, 10.0, 0.4, 0.1)          # tai xe tiep quan
check("tai xe tiep quan -> ve L3_ACTIVE", r["state"] == "L3_ACTIVE")

# ---------------------------------------------------------------------------
# 9) Noise-aware sensor fusion (Module G)
# ---------------------------------------------------------------------------
print("[9] Noise-aware sensor fusion")
sfe = SensorFusionEval()
w_clear = sfe.weights({"snr": 1.0, "visibility_m": 200})
check("troi quang -> radar khong lan at (radar <= camera)", w_clear["radar"] <= w_clear["camera"])
w_fog = sfe.weights({"snr": 0.1, "visibility_m": 8})
check("suong mu day -> radar lan at (> 0.6)", w_fog["radar"] > 0.6)
fused = sfe.fuse_range(None, 20.0, 10.0, {"snr": 0.1, "visibility_m": 8})
check("mu day, camera mat -> ket qua nghieng ve radar (<14m)", fused is not None and fused < 14.0)

# ---------------------------------------------------------------------------
# 10) L3 validation report (chấm điểm tuân thủ + xuất JSON)
# ---------------------------------------------------------------------------
print("[10] L3 validation report")
r_clear = assess_profile("clear", "NORMAL", "NORMAL",
                         {"collisions": 0, "min_ttc_s": None}, False, 0, 0, 0)
check("clear NORMAL, khong MRM, khong va cham -> PASS", r_clear["pass"] is True)

r_fog = assess_profile("fog", "VIOLATION", "VIOLATION",
                       {"collisions": 0, "min_ttc_s": 2.0}, True, 1, 0, 0)
check("fog VIOLATION + co MRM + khong va cham -> PASS", r_fog["pass"] is True)

r_crash = assess_profile("heavy_rain", "DEGRADED", "DEGRADED",
                         {"collisions": 1}, False, 0, 0, 0)
check("co va cham -> FAIL", r_crash["pass"] is False)

r_false = assess_profile("clear", "NORMAL", "NORMAL",
                         {"collisions": 0}, True, 1, 1, 0)
check("ngat nham khi ODD NORMAL -> FAIL", r_false["pass"] is False)

rep = build_report([r_clear, r_fog, r_crash, r_false])
check("bao cao: pass_rate = 50%", rep["summary"]["pass_rate_pct"] == 50.0)

_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "_selftest_report.json")
write_report(_tmp, [r_clear, r_fog])
import json as _json  # noqa: E402
with open(_tmp, encoding="utf-8") as _f:
    _loaded = _json.load(_f)
check("write_report tao file JSON hop le", _loaded["summary"]["scenarios"] == 2)
os.remove(_tmp)

# ---------------------------------------------------------------------------
# 11) Scenario library (port ScenarioRunner) — kiem tra CATALOG khong can CARLA
# ---------------------------------------------------------------------------
print("[11] Scenario library catalog")
check("co dung 15 kich ban", len(scnlib.CATALOG) == 15)
check("ten kich ban khong trung", len(set(scnlib.list_names())) == 15)
_cats = {s.category for s in scnlib.CATALOG}
check("du 5 nhom kich ban", _cats == {"lead", "crossing", "cutin", "junction", "oncoming"})
check("moi spec co builder goi duoc", all(callable(s.builder) for s in scnlib.CATALOG))
check("moi spec co mo ta + chuc nang", all(s.description and s.tests for s in scnlib.CATALOG))
check("get() tra dung spec", scnlib.get("HardBrake").category == "lead")


class _ScenarioLoc:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y


class _ScenarioActor:
    def __init__(self, x=5.0, y=0.0):
        self.is_alive = True
        self._location = _ScenarioLoc(x, y)

    def get_location(self):
        return self._location


_ego_for_scenario = _ScenarioActor(0.0, 0.0)
_hazard_active = {"value": False}
_running = scnlib.RunningScenario(
    [_ScenarioActor(5.0, 0.0)], trigger_distance=10.0,
    reaction_condition=lambda *_args: _hazard_active["value"])
_running.tick(0, _ego_for_scenario, None)
check("scenario tach action-trigger khoi hazard reaction timer",
      _running.trigger_frame == 0 and _running.hazard_frame is None)
_hazard_active["value"] = True
_running.tick(7, _ego_for_scenario, None)
check("scenario hazard timer bat dau khi ground-truth vao corridor",
      _running.hazard_frame == 7)

# ---------------------------------------------------------------------------
# 12) RL: SpeedExperiment (obs/action/reward) — soi gương rllib-integration
# ---------------------------------------------------------------------------
print("[12] RL speed experiment/env/controller")
exp = SpeedExperiment()
check("6 action toc do", len(exp.get_actions()) == 6)
check("compute_action(3) = 35 km/h", abs(exp.compute_action(3) - 35 / 3.6) < 1e-6)
o = exp.obs_from_measured(10.0, 0.5, 20.0, 3.0, 8.0)
check("obs 5 chieu, chuan hoa [0,1]", len(o) == 5 and all(0.0 <= x <= 1.0 for x in o))


def _st(v_kmh, dens, lead=False, dist=60.0):
    v = v_kmh / 3.6
    return {"v_ms": v, "density": dens, "lead_present": lead, "lead_dist": dist,
            "lead_speed": v, "prev_target_ms": v, "accel_ms2": 0.0,
            "target_ms": v, "collided": False}


check("duong vang: chay 35 co reward > dung yen",
      exp.compute_reward(_st(35, 0.1)) > exp.compute_reward(_st(0, 0.1)))
check("dong + lead sat: bi phat an toan (reward am hon)",
      exp.compute_reward(_st(45, 0.9, lead=True, dist=5.0)) <
      exp.compute_reward(_st(15, 0.9, lead=True, dist=25.0)))

env = SpeedControlEnv(seed=0)
obs0 = env.reset()
o1, r1, d1, i1 = env.step(3)
check("env.reset -> obs 5 chieu", len(obs0) == 5)
check("env.step -> (obs, reward, done, info)",
      len(o1) == 5 and isinstance(r1, float) and isinstance(d1, bool) and "v_kmh" in i1)

# Controller: khong co torch/policy trong base python -> tu dong dung heuristic
ctrl = RLSpeedController("weights/_khong_ton_tai.pt")
check("chua co policy -> heuristic", ctrl.policy is None)
check("heuristic: duong vang -> >30 km/h", ctrl.desired_speed_kmh(5.0, 0.1, 60.0, 99.0) > 30.0)
check("heuristic: duong dong -> <30 km/h", ctrl.desired_speed_kmh(5.0, 0.9, 10.0, 2.0) < 30.0)
# Trần an toàn: đường "vắng" nhưng vật cản GẦN -> tốc độ tuần hành bị ghì mạnh.
check("safety-cap: vat can gan 8m -> tuan hanh < 25 km/h",
      ctrl.desired_speed_kmh(5.0, 0.1, 8.0, 3.0) < 25.0)
check("safety-cap: khong vat can -> khong bi ghi (>30)",
      ctrl.desired_speed_kmh(5.0, 0.1, None, None) > 30.0)

# SAN (min_cruise) PHAI NAM TRONG TRAN, khong duoc nang nguoc muc tieu da bi ha.
# Loi cu: chinh.py ap san 18 km/h SAU khi trandda kep -> vat can 8m van bi day
# len 18 km/h. Bay gio san ap TRUOC tran nen tran luon thang.
_cap_8m = RLSpeedController.safety_cap_kmh(8.0, 3.0)
check("san khong bao gio vuot tran: vat can 8m + san 18 km/h",
      ctrl.desired_speed_kmh(5.0, 0.1, 8.0, 3.0, min_cruise_kmh=18.0) <= _cap_8m + 1e-9)
check("tran 8m/ttc3 that su thap hon san 18 km/h (neu khong, phep thu tren vo nghia)",
      _cap_8m < 18.0)
check("san van co tac dung khi duong thoang",
      abs(ctrl.desired_speed_kmh(5.0, 0.9, None, None, min_cruise_kmh=40.0) - 40.0) < 1e-9)
check("san mac dinh 0 -> khong doi hanh vi cu",
      abs(ctrl.desired_speed_kmh(5.0, 0.9, 10.0, 2.0)
          - ctrl.desired_speed_kmh(5.0, 0.9, 10.0, 2.0, min_cruise_kmh=0.0)) < 1e-9)
check("safety_cap_kmh la API cong khai, khong vat can -> vo cuc",
      RLSpeedController.safety_cap_kmh(None, None) == float("inf"))

# Turn intent
check("intent_to_right('right') True", intent_to_right("right") is True)
check("intent_to_right('left') False", intent_to_right("left") is False)

# ---------------------------------------------------------------------------
# 13) Sensor frame integrity
# ---------------------------------------------------------------------------
print("[13] Sensor frame integrity")


class _Frame:
    def __init__(self, frame):
        self.frame = frame


q = _queue.Queue()
q.put(_Frame(9)); q.put(_Frame(10))
sync_stats = SensorSyncStats()
got = retrieve_exact_frame(q, 10, timeout=0.1, sensor_name="test", stats=sync_stats)
check("bo frame cu va lay dung frame yeu cau", got.frame == 10 and sync_stats.stale_discarded == 1)
q = _queue.Queue(); q.put(_Frame(12))
future_rejected = False
try:
    retrieve_exact_frame(q, 11, timeout=0.1, sensor_name="test")
except SensorFrameError:
    future_rejected = True
check("khong ghep frame tuong lai voi frame hien tai", future_rejected)
q = _queue.Queue()
timeout_descriptive = False
try:
    retrieve_exact_frame(q, 20, timeout=0.01, sensor_name="camera")
except TimeoutError as exc:
    timeout_descriptive = "camera" in str(exc) and "20" in str(exc)
check("sensor timeout co ten sensor + frame de debug", timeout_descriptive)


class _WarmWorld:
    def __init__(self, queues):
        self.frame = 30
        self.queues = queues

    def tick(self):
        self.frame += 1
        if self.frame >= 32:  # mô phỏng GPU sensor bỏ tick đầu
            for sensor_queue in self.queues.values():
                sensor_queue.put(_Frame(self.frame))
        return self.frame


warm_queues = {"camera": _queue.Queue(), "lidar": _queue.Queue()}
warm_frame = warmup_sensor_streams(
    _WarmWorld(warm_queues), warm_queues, attempts=3, timeout=0.01)
check("sensor warm-up vuot qua tick dau chua co du lieu", warm_frame == 32)

latest_q = _queue.Queue(maxsize=2)
put_latest(latest_q, _Frame(40))
put_latest(latest_q, _Frame(41))
latest_dropped = put_latest(latest_q, _Frame(42))
latest_reader = LatestFrameReader(latest_q, "camera")
check("async sensor queue bounded va lay frame moi nhat",
      latest_dropped and latest_reader.get_latest().frame == 42)
future_q = _queue.Queue()
future_q.put(_Frame(50)); future_q.put(_Frame(52))
future_reader = LatestFrameReader(future_q, "camera")
first_ready = future_reader.get_at_or_before(51)
second_ready = future_reader.get_at_or_before(52)
check("async reader khong ghep frame tuong lai va giu lai cho tick sau",
      first_ready.frame == 50 and second_ready.frame == 52)


class _Blueprint:
    def __init__(self):
        self.values = {}

    def set_attribute(self, name, value):
        self.values[name] = value

    def has_attribute(self, name):
        return name == "sensor_tick"


class _SensorCfg:
    CAM_WIDTH, CAM_HEIGHT, CAM_FOV = 960, 540, 90.0
    FIXED_DELTA, FPS, LIDAR_POINTS_PER_SECOND = 0.025, 40, 560000
    RADAR_RANGE_M, RADAR_POINTS_PER_SECOND = 80.0, 4000
    RADAR_HORIZONTAL_FOV, RADAR_VERTICAL_FOV = 35.0, 10.0


cam_bp, lidar_bp, radar_bp = _Blueprint(), _Blueprint(), _Blueprint()
configure_camera_blueprint(cam_bp, _SensorCfg)
configure_lidar_blueprint(lidar_bp, _SensorCfg)
configure_radar_blueprint(radar_bp, _SensorCfg)
check("camera/lidar phat moi world tick trong synchronous mode",
      cam_bp.values["sensor_tick"] == "0.0" and
      lidar_bp.values["sensor_tick"] == "0.0")
check("radar 80m/35deg/4000pps va phat moi tick",
      radar_bp.values["range"] == "80.0" and
      radar_bp.values["horizontal_fov"] == "35.0" and
      radar_bp.values["points_per_second"] == "4000" and
      radar_bp.values["sensor_tick"] == "0.0")


class _AsyncSensorCfg(_SensorCfg):
    CAMERA_SENSOR_TICK_S = 0.1
    CAMERA_POSTPROCESS = False
    LIDAR_SENSOR_TICK_S = 0.025
    RADAR_SENSOR_TICK_S = 0.025


async_cam_bp, async_lidar_bp, async_radar_bp = (
    _Blueprint(), _Blueprint(), _Blueprint())
configure_camera_blueprint(async_cam_bp, _AsyncSensorCfg)
configure_lidar_blueprint(async_lidar_bp, _AsyncSensorCfg)
configure_radar_blueprint(async_radar_bp, _AsyncSensorCfg)
check("async-stable giu geometry 40Hz va gioi han camera 10Hz",
      async_cam_bp.values["sensor_tick"] == "0.1"
      and async_lidar_bp.values["sensor_tick"] == "0.025"
      and async_radar_bp.values["sensor_tick"] == "0.025")

# ---------------------------------------------------------------------------
# 14) Scenario acceptance
# ---------------------------------------------------------------------------
print("[14] Scenario acceptance")
base_kpi = {"collisions": 0, "min_distance_m": 2.0}
v = assess_scenario("CutIn", "cutin", base_kpi, triggered=True, reacted=True,
                    reaction_delay_s=0.25, braked=True, evaded=False)
check("scenario hop le + phan ung dung han -> PASS", v["pass"] is True)
v = assess_scenario("Construction", "crossing", {"collisions": 1, "min_distance_m": 0.0},
                    triggered=True, reacted=True, reaction_delay_s=0.1,
                    braked=True, evaded=False)
check("co collision -> FAIL", v["pass"] is False)
v = assess_scenario("CutIn", "cutin", base_kpi, triggered=True, reacted=True,
                    reaction_delay_s=0.2, braked=False, evaded=False)
check("cut-in co state nhung khong brake/evade -> FAIL", v["pass"] is False)
v = assess_scenario("Missing", "crossing", base_kpi, triggered=False, reacted=False,
                    reaction_delay_s=None, braked=False, evaded=False)
check("scenario khong trigger -> FAIL", v["pass"] is False)

# ---------------------------------------------------------------------------
# 15) Custom local planning + closed-loop controller math
# ---------------------------------------------------------------------------
print("[15] Custom planning/control")
base_path = [(float(x), 0.0) for x in range(0, 31, 3)]
lane_change = LocalPlanner.lane_change_trajectory(base_path, 3.5, transition_m=21.0)
offsets = [point[1] for point in lane_change]
check("lane-change quintic bat dau 0 va ket thuc dung offset",
      abs(offsets[0]) < 1e-9 and abs(offsets[-1] - 3.5) < 1e-9)
check("lane-change offset don dieu, khong giat nguoc",
      all(b >= a for a, b in zip(offsets, offsets[1:])))

lat = LateralController()
check("pure-pursuit: path thang -> steer gan 0",
      abs(lat.compute_path_steering(base_path, 8.0)) < 1e-9)
check("pure-pursuit: target ben phai -> steer duong",
      lat.compute_path_steering([(0.0, 0.0), (8.0, 2.0), (20.0, 3.0)], 8.0) > 0.0)


class _Waypoint:
    def __init__(self, yaw):
        self.transform = type("Transform", (), {
            "rotation": type("Rotation", (), {"yaw": yaw})()})()


branches = [_Waypoint(-45.0), _Waypoint(0.0), _Waypoint(50.0)]
check("route command left chon nhanh trai",
      _select_next_waypoint(branches, 0.0, "left").transform.rotation.yaw == -45.0)
check("route command right chon nhanh phai",
      _select_next_waypoint(branches, 0.0, "right").transform.rotation.yaw == 50.0)

# BehaviorPlanner CHI lo hanh vi chien thuat. An toan thuoc ve ego_control.py
# (mot diem trong tai duy nhat). Truoc day planner co nhanh EMERGENCY_STOP nhung
# khong bao gio chay duoc vi ego_driving_stack luon truyen collision_risk=False.
_states = {s.name for s in DrivingState}
check("DrivingState khong con trang thai an toan",
      _states == {"LANE_FOLLOWING", "FOLLOWING_VEHICLE", "STOPPING_AT_LIGHT"})
_bp = BehaviorPlanner()
check("planner bo qua collision_risk (phanh khong phai viec cua no)",
      _bp.update_state({"collision_risk": True, "nearest_distance": 99.0})
      == DrivingState.LANE_FOLLOWING)
check("planner: xe truoc 10m -> FOLLOWING_VEHICLE",
      _bp.update_state({"nearest_distance": 10.0}) == DrivingState.FOLLOWING_VEHICLE)
check("planner: gian cach phuc hoi 30m -> LANE_FOLLOWING",
      _bp.update_state({"nearest_distance": 30.0}) == DrivingState.LANE_FOLLOWING)
check("planner: den do -> STOPPING_AT_LIGHT",
      _bp.update_state({"nearest_distance": 20.0, "traffic_light": "Red"})
      == DrivingState.STOPPING_AT_LIGHT)
check("planner: den xanh -> LANE_FOLLOWING",
      _bp.update_state({"nearest_distance": 20.0, "traffic_light": "Green"})
      == DrivingState.LANE_FOLLOWING)

lon = LongitudinalController(kp=0.4, ki=0.05, kd=0.05)
throttle, brake = lon.update(10.0, 0.0, 0.05)
check("PID tang toc chi ra throttle", throttle > 0.0 and brake == 0.0)
throttle, brake = lon.update(0.0, 10.0, 0.05)
check("PID giam toc chi ra brake", throttle == 0.0 and brake > 0.0)


class _Location:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Rotation:
    def __init__(self, yaw):
        self.yaw = yaw


class _Transform:
    def __init__(self, x, y, yaw):
        self.location = _Location(x, y)
        self.rotation = _Rotation(yaw)


ego_motion_est = EgoMotionEstimator()
m0 = ego_motion_est.update(_Transform(0.0, 0.0, 90.0))
m1 = ego_motion_est.update(_Transform(0.0, 2.0, 100.0))
check("ego-motion doi world delta sang ego local frame",
      not m0["valid"] and m1["valid"] and abs(m1["dx"] - 2.0) < 1e-6 and
      abs(m1["dy"]) < 1e-6 and abs(m1["dyaw"] - 0.17453) < 1e-3)

# ---------------------------------------------------------------------------
# 16) Scene semantics
# ---------------------------------------------------------------------------
print("[16] Traffic semantics")
traffic = summarize_traffic_controls([
    {"class": "TrafficLight", "distance_m": 30.0, "traffic_light_state": "green"},
    {"class": "TrafficLight", "distance_m": 12.0, "traffic_light_state": "red"},
    {"class": "StopSign", "distance_m": 10.0},
])
check("chon den giao thong gan nhat", traffic["traffic_light_state"] == "red")
check("nhan dien stop-sign trong tam xu ly", traffic["stop_sign"] is True)

stop_state = StopSignState(hold_s=0.15, cooldown_s=0.30)
check("stop-sign detection tao lenh dung",
      stop_state.update(True, speed_ms=2.0, dt=0.05) is True)
for _ in range(3):
    pending = stop_state.update(True, speed_ms=0.0, dt=0.05)
check("stop-sign nha sau khi dung du hold-time", pending is False)
check("stop-sign cooldown ngan kich hoat lap tuc",
      stop_state.update(True, speed_ms=0.0, dt=0.05) is False)

# ---------------------------------------------------------------------------
# 17) Radar, sensor-health, bounded async, and lane-source contracts
# ---------------------------------------------------------------------------
print("[17] Radar/health/async/lane contracts")
radar_processor = RadarProcessor(
    sensor_x_m=2.0, sensor_z_m=0.8, reference_x_m=1.5,
    velocity_sign=1.0, cluster_gate_m=0.5, min_target_height_m=0.15)
radar_rows = np.array([
    [8.0, 0.0, 0.0, 20.0],
    [8.2, 0.01, 0.0, 20.1],
    [-3.0, 0.30, 0.0, 30.0],
], dtype=np.float32)
radar_targets = radar_processor.process(radar_rows)
check("radar polar -> ego va cluster diem gan nhau",
      len(radar_targets) == 2 and abs(radar_targets[0]["distance_m"] - 20.55) < 0.2)
check("radar closing speed chuan hoa duong khi tien gan",
      8.0 < radar_targets[0]["closing_speed_ms"] < 8.2)

# Tia dưới của radar cao 0.8m chạm mặt đường ở khoảng 9--10m. Ground return
# phải bị loại trước khi vào tracker/AEB, còn hit trên thân xe phải được giữ.
ground_depths = np.array([9.0, 10.0, 12.0], dtype=np.float32)
ground_rows = np.column_stack((
    np.zeros(3, dtype=np.float32),
    np.zeros(3, dtype=np.float32),
    np.arcsin(-0.8 / ground_depths),
    ground_depths,
)).astype(np.float32)
ground_targets = radar_processor.process(ground_rows)
check("radar loai tia cham mat duong 9-12m", not ground_targets and
      radar_processor.last_stats["ground_rejected_points"] == 3)

vehicle_depth = 15.0
vehicle_height = 0.55
vehicle_rows = np.array([[
    5.0, 0.0, np.arcsin((vehicle_height - 0.8) / vehicle_depth), vehicle_depth,
]], dtype=np.float32)
vehicle_targets = radar_processor.process(vehicle_rows)
check("radar giu hit than xe @15m cao hon mat duong",
      len(vehicle_targets) == 1 and vehicle_targets[0]["z_m"] > 0.5)

empty_road_safety = ActiveSafetySystem(enable_evasion=False)
empty_road_decision = empty_road_safety.update(
    0.0, [], [], 0.025, radar_targets=ground_targets)
check("duong trong sau ground-filter -> NORMAL/DRIVE",
      empty_road_decision.state == "NORMAL" and
      empty_road_decision.action == "DRIVE")

vehicle_safety = ActiveSafetySystem(enable_evasion=False)
vehicle_decision = vehicle_safety.update(
    5.0, [], [], 0.025, radar_targets=vehicle_targets)
check("vat can radar that @15m -> BRAKE va ghi dung source",
      vehicle_decision.action == "BRAKE" and
      vehicle_decision.threat.get("source") == "radar")

optional_q = _queue.Queue()
optional_q.put(_Frame(12))
optional_reader = OptionalFrameReader(optional_q, "radar")
check("optional radar khong ghep frame tuong lai", optional_reader.get(11, 0.001) is None)
check("optional radar giu frame tuong lai cho tick ke", optional_reader.get(12, 0.001).frame == 12)

health = SensorHealthMonitor({"camera": True, "lidar": True, "radar": True}, 3)
for frame in range(4):
    health.next_frame()
    health.observe("camera", frame, float(frame), True)
    health.observe("lidar", frame, float(frame), False)
    health.observe("radar", frame, float(frame), False)
check("mat lidar+radar >3 frame -> range redundancy lost", health.range_redundancy_lost)
odd_sensor = ODDMonitor().classify(estimate_conditions(), health.summary())
check("range redundancy lost -> ODD VIOLATION critical",
      odd_sensor["state"] == "VIOLATION" and odd_sensor["critical"])
health_off = SensorHealthMonitor({"camera": True, "lidar": True, "radar": False}, 3)
for frame in range(5):
    health_off.next_frame(); health_off.observe("lidar", frame, float(frame), False)
check("radar-off khong tao regression MRM moi", not health_off.range_redundancy_lost)
check("sensor health ghi lai lich su dual-range loss",
      health.range_redundancy_loss_events == 1 and
      health.summary()["range_redundancy_ever_lost"] is True)

fault_single = assess_fault_behavior(
    "radar-loss", {"availability": {"radar": 0.8},
                   "range_redundancy_ever_lost": False})
fault_dual = assess_fault_behavior(
    "lidar-radar-loss",
    {"availability": {"lidar": 0.8, "radar": 0.8},
     "range_redundancy_ever_lost": True}, True, True)
check("fault acceptance: mat radar don le van fallback", fault_single["pass"])
check("fault acceptance: dual range bat buoc ODD+MRM", fault_dual["pass"])
check("weather acceptance: NORMAL khong false violation",
      assess_weather_behavior("NORMAL", {"NORMAL"}, False)["pass"])
check("weather acceptance: VIOLATION bat buoc MRM",
      not assess_weather_behavior("VIOLATION", {"VIOLATION"}, False)["pass"] and
      assess_weather_behavior("VIOLATION", {"VIOLATION"}, True)["pass"])
check("weather acceptance: VIOLATION bat ngo trong NORMAL van FAIL",
      not assess_weather_behavior(
          "NORMAL", {"NORMAL", "VIOLATION"}, True)["pass"])
check("weather acceptance: dual-range VIOLATION khong bi cham nham la weather",
      assess_weather_behavior(
          "NORMAL", {"NORMAL", "VIOLATION"}, True,
          external_violation_expected=True)["pass"])

mot_radar = MultiObjectTracker(dt=0.05, min_hits=1)
mot_radar.update([{"distance_m": 20.0, "lateral_m": 0.0, "class": "Car"}], 0.05)
mot_radar.update([], 0.05, radar_measurements=[{
    "distance_m": 19.8, "lateral_m": 0.0, "closing_speed_ms": 5.0,
    "range_std_m": 0.3, "speed_std_ms": 0.4}], frame_id=2)
check("radar Mahalanobis update radial velocity vao track",
      mot_radar.tracks[0].vel[0] < -4.0 and "radar" in mot_radar.tracks[0].sources)

radar_safety = ActiveSafetySystem(enable_evasion=False)
radar_decision = radar_safety.update(
    10.0, [], [], 0.025,
    radar_targets=[{"distance_m": 12.0, "lateral_m": 0.0,
                    "closing_speed_ms": 10.0}])
check("radar TTC bao thu hon -> AEB", radar_decision.action == "BRAKE")

scheduler = LatestFrameScheduler(lambda payload, frame_id: payload * 2, "selftest-gpu")
scheduler.submit(10, time.monotonic(), 7)
deadline = time.monotonic() + 1.0
scheduled = None
while time.monotonic() < deadline and scheduled is None:
    scheduled = scheduler.latest(10, time.monotonic(), 150.0)
    time.sleep(0.001)
scheduler.close()
check("bounded scheduler tra result kem frame id", scheduled is not None and
      scheduled.frame_id == 10 and scheduled.payload == 14)
check("bounded scheduler loai ket qua qua han",
      scheduler.latest(10, scheduled.timestamp + 0.151, 150.0) is None)
failed_product = TaggedInferenceResult(
    11, time.monotonic(), time.perf_counter(), {"unsafe": True}, 1.0,
    error="synthetic failure")
check("async error khong tai su dung camera semantics cu",
      usable_inference_payload(None) is None
      and usable_inference_payload(failed_product) is None
      and usable_inference_payload(scheduled) == 14)

map_lane = map_lane_estimate(1, 1.0, [(0.0, 0.0), (20.0, 0.0)])
arbiter = LaneSourceArbiter(stable_frames=5, invalid_frames=3)
def learned_lane(frame_id, timestamp, offset=-0.1, valid=True):
    return LaneEstimate(
        frame_id, timestamp, "learned", confidence=0.9,
        centerline_m=[(0.0, -offset), (20.0, -offset)], offset_m=offset,
        lane_width_m=3.5, valid=valid)

same_lane = learned_lane(1, 1.0)
same_frame_sources = [arbiter.select(same_lane, map_lane).source for _ in range(5)]
check("learned lane khong dem lap cung frame id",
      same_frame_sources == ["map"] * 5)
confidence_arbiter = LaneSourceArbiter(stable_frames=1)
below_threshold = learned_lane(30, 30.0)
below_threshold.confidence = 0.749
at_threshold = learned_lane(31, 31.0)
at_threshold.confidence = 0.75
check("learned lane chi nam quyen khi confidence >= 0.75",
      confidence_arbiter.select(below_threshold, map_lane).source == "map"
      and confidence_arbiter.select(at_threshold, map_lane).source == "learned")
selected = None
for frame_id in range(2, 7):
    selected = arbiter.select(learned_lane(frame_id, float(frame_id)), map_lane)
check("learned lane can 5 frame hop le moi nam quyen", selected.source == "learned")
for frame_id in range(7, 10):
    selected = arbiter.select(learned_lane(frame_id, float(frame_id), valid=False), map_lane)
check("learned lane loi 3 frame -> map fallback", selected.source == "map")
for frame_id in range(10, 15):
    selected = arbiter.select(
        learned_lane(frame_id, float(frame_id), offset=-0.9), map_lane)
check("learned lane co the tai kich hoat sau dropout va doi hinh hoc",
      selected.source == "learned")
stale_lane = learned_lane(15, 15.0)
for current_frame in range(20, 23):
    current_map = map_lane_estimate(current_frame, 20.0, [(0.0, 0.0), (20.0, 0.0)])
    selected = arbiter.select(
        stale_lane, current_map, now_s=20.0, max_age_ms=150.0)
check("learned lane qua han -> map fallback", selected.source == "map")
check("junction luon uu tien map",
      arbiter.select(learned_lane(23, 23.0), map_lane, True).source == "map")

projector = GroundProjector(480.0, 480.0, 480.0, 270.0, camera_height_m=2.4)
ground = projector.pixel_to_ground(480.0, 390.0)
check("IPM camera pixel duoi horizon -> metric BEV", ground is not None and
      abs(ground[0] - 9.6) < 0.1 and abs(ground[1]) < 1e-6)
left_boundary = LearnedLaneAdapter._normalize_line(
    [(20.0, -1.75), (4.0, -1.75), (10.0, -1.75)])
right_boundary = LearnedLaneAdapter._normalize_line(
    [(4.0, 1.75), (15.0, 1.75), (20.0, 1.75)])
aligned_center, aligned_width, aligned_offset, _ = LearnedLaneAdapter._geometry(
    [left_boundary, right_boundary])
check("learned lane noi suy 2 bien khac sampling theo chieu gan -> xa",
      len(aligned_center) >= 3
      and all(aligned_center[index][0] < aligned_center[index + 1][0]
              for index in range(len(aligned_center) - 1))
      and abs(aligned_width - 3.5) < 1e-6
      and abs(aligned_offset) < 1e-6)
age_rejected = False
try:
    validate_frame_age(10, 1.0, 12, 1.2, max_age_ms=150.0)
except ValueError:
    age_rejected = True
check("async product qua 150ms bi loai", age_rejected)

replay_dir = os.path.join(os.getcwd(), "logs", "selftest_replay")
with ReplayWriter(replay_dir, {"test": True}, append=False) as writer:
    writer.write(7, 1.25, np.zeros((4, 6, 3), np.uint8),
                 np.zeros((2, 4), np.float32), np.zeros((1, 4), np.float32))
replay_rows = list(ReplayReader(replay_dir))
check("replay round-trip khong can CARLA", len(replay_rows) == 1 and
      replay_rows[0][0]["frame_id"] == 7 and replay_rows[0][1]["rgb"].shape == (4, 6, 3))

stage_metrics = PipelineMetrics({"safety": 25.0})
for latency in (10.0, 20.0, 30.0):
    stage_metrics.record("safety", latency)
_pipeline_summary = stage_metrics.summary()
stage_summary = _pipeline_summary["stages"]["safety"]
check("stage metrics p99 + deadline miss", stage_summary["p99_ms"] == 30.0 and
      stage_summary["deadline_misses"] == 1)
check("stage metrics tach PyTorch va device-wide GPU memory",
      _pipeline_summary["gpu_peak_mb"] is None and
      _pipeline_summary["gpu_device_peak_used_mb"] is None and
      _pipeline_summary["gpu_device_total_mb"] is None)
check("CARLA map guard nhan Town02 va full asset path la cung map",
      same_carla_map("Carla/Maps/Town02", "Town02"))
check("CARLA map guard van reload khi doi map",
      not same_carla_map("Carla/Maps/Town02", "Town05"))

class _FakeMap:
    def __init__(self, name):
        self.name = name

class _FakeWorldForMapGuard:
    def __init__(self, name):
        self._map = _FakeMap(name)

    def get_map(self):
        return self._map

class _FakeClientForMapGuard:
    def __init__(self, name):
        self.world = _FakeWorldForMapGuard(name)
        self.load_calls = []

    def get_world(self):
        return self.world

    def load_world(self, name):
        self.load_calls.append(name)
        self.world = _FakeWorldForMapGuard(f"Carla/Maps/{name}")
        return self.world

same_map_client = _FakeClientForMapGuard("Carla/Maps/Town02")
same_world, same_reloaded = get_or_load_world(same_map_client, "Town02")
check("map guard khong goi load_world cho cung map",
      same_world is same_map_client.world and not same_reloaded
      and same_map_client.load_calls == [])
other_map_client = _FakeClientForMapGuard("Carla/Maps/Town02")
other_world, other_reloaded = get_or_load_world(other_map_client, "Town05")
check("map guard chi load_world khi doi map",
      other_world.get_map().name.endswith("Town05") and other_reloaded
      and other_map_client.load_calls == ["Town05"])


class _FakeSettings:
    def __init__(self, synchronous_mode=False, fixed_delta_seconds=None):
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds


class _FakeSyncWorld:
    def __init__(self, timeout_once=False, events=None):
        self.settings = _FakeSettings()
        self.timeout_once = timeout_once
        self.apply_calls = 0
        self.tick_calls = 0
        self.events = events

    def get_settings(self):
        return _FakeSettings(
            self.settings.synchronous_mode, self.settings.fixed_delta_seconds)

    def apply_settings(self, settings, timeout_s):
        self.apply_calls += 1
        if self.events is not None:
            self.events.append(
                "world_sync_on" if settings.synchronous_mode else "world_sync_off")
        self.settings = _FakeSettings(
            settings.synchronous_mode, settings.fixed_delta_seconds)
        if self.timeout_once:
            self.timeout_once = False
            raise RuntimeError("simulated acknowledgement timeout")
        return 100 + self.apply_calls

    def tick(self, timeout_s):
        self.tick_calls += 1
        return 200 + self.tick_calls


normal_sync_world = _FakeSyncWorld()
normal_original, normal_frame, normal_recovered = enter_synchronous_mode(
    normal_sync_world, 40)
check("sync guard ap dung 40Hz va restore cau hinh ban dau",
      normal_frame == 101 and not normal_recovered
      and normal_sync_world.settings.synchronous_mode
      and abs(normal_sync_world.settings.fixed_delta_seconds - 0.025) < 1e-9)
restore_world_settings(normal_sync_world, normal_original)
check("sync guard restore async sau khi ket thuc",
      not normal_sync_world.settings.synchronous_mode)
timeout_sync_world = _FakeSyncWorld(timeout_once=True)
_, recovered_frame, recovered_transition = enter_synchronous_mode(
    timeout_sync_world, 40)
check("sync guard phat recovery tick khi apply timeout mot phan",
      recovered_transition and recovered_frame == 201
      and timeout_sync_world.tick_calls == 1
      and timeout_sync_world.settings.synchronous_mode)


class _FakeTrafficManager:
    def __init__(self, events):
        self.events = events

    def set_synchronous_mode(self, enabled):
        self.events.append("tm_sync_on" if enabled else "tm_sync_off")


sync_order_events = []
ordered_world = _FakeSyncWorld(events=sync_order_events)
ordered_tm = _FakeTrafficManager(sync_order_events)
ordered_original, _, _ = enter_synchronous_mode(
    ordered_world, 40, traffic_manager=ordered_tm)
restore_world_settings(
    ordered_world, ordered_original, traffic_manager=ordered_tm)
check("sync ordering bat world truoc TM va tat TM truoc world",
      sync_order_events == [
          "world_sync_on", "tm_sync_on", "tm_sync_off", "world_sync_off"])

managed_events = []
managed_world = _FakeSyncWorld(events=managed_events)
managed_tm = _FakeTrafficManager(managed_events)
managed_session = SynchronousWorldSession(managed_world, 40)
managed_session.__enter__()
managed_session.attach_traffic_manager(managed_tm)
managed_session.__enter__()  # repeated context entry must not apply twice
check("managed sync khoa world truoc khi gan Traffic Manager va enter idempotent",
      managed_session.active
      and managed_events == ["world_sync_on", "tm_sync_on"])
managed_session.close()
managed_session.close()  # final cleanup may safely close an exited context
check("managed sync close idempotent va restore dung thu tu",
      not managed_session.active
      and managed_events == [
          "world_sync_on", "tm_sync_on", "tm_sync_off", "world_sync_off"])


class _ReadyProbeWorld:
    @staticmethod
    def get_settings():
        return _FakeSettings(synchronous_mode=False)

    @staticmethod
    def wait_for_tick(timeout_s):
        return type("Snapshot", (), {"frame": 321})()

    @staticmethod
    def get_map():
        return type("Map", (), {"name": "Carla/Maps/Town02"})()


class _ReadyProbeClient:
    def __init__(self, host, port):
        self.timeout = None

    def set_timeout(self, timeout_s):
        self.timeout = timeout_s

    @staticmethod
    def get_server_version():
        return "test-version"

    @staticmethod
    def get_world():
        return _ReadyProbeWorld()


ready_probe = wait_until_ready(
    _ReadyProbeClient, "127.0.0.1", 2000, timeout_s=1.0, poll_s=0.0)
check("CARLA readiness probe doi world tick thay vi chi check port",
      ready_probe["frame"] == 321
      and ready_probe["map"].endswith("Town02")
      and ready_probe["attempts"] == 1)

quality_profile = tuple(runtime_cfg._PERFORMANCE_BASELINE.items())
runtime_cfg.apply_performance_overrides({
    "CAM_WIDTH": 640, "CAM_HEIGHT": 360, "RENDER_EVERY_N": 4})
runtime_cfg.apply_performance_overrides({})
check("performance profile khong ro cau hinh tu lan chay truoc",
      tuple((name, getattr(runtime_cfg, name)) for name, _ in quality_profile)
      == quality_profile)

voter = TrafficLightTemporalVoter(window=5, min_votes=3)
states = [voter.update("front", state, i)
          for i, state in enumerate(("red", "unknown", "red", "green", "red"))]
check("traffic-light temporal vote loc unknown/jitter", states[-1] == "red")

fast_lidar = LidarProcessor(max_points=500)
synthetic_cloud = np.vstack([
    np.column_stack((np.full(20, 4.0), np.linspace(-0.1, 0.1, 20),
                     np.linspace(-0.2, 0.3, 20), np.ones(20))),
    np.column_stack((np.linspace(1.0, 10.0, 100), np.zeros(100),
                     np.full(100, -2.5), np.ones(100))),
]).astype(np.float32)
fast_obstacles = fast_lidar.extract_safety_obstacles(synthetic_cloud)
check("LiDAR safety voxel giu vat can va loai ground",
      bool(fast_obstacles) and min(o["centroid"][0] for o in fast_obstacles) > 3.5)

perception_cloud = np.vstack([
    np.column_stack((np.linspace(3.8, 4.2, 24), np.linspace(-0.2, 0.2, 24),
                     np.linspace(-0.3, 0.4, 24), np.ones(24))),
    np.column_stack((np.linspace(9.7, 10.3, 30), np.linspace(1.8, 2.2, 30),
                     np.linspace(-0.2, 0.8, 30), np.ones(30))),
    np.column_stack((np.linspace(1.0, 12.0, 100), np.zeros(100),
                     np.full(100, -2.5), np.ones(100))),
]).astype(np.float32)
perception_obstacles = fast_lidar.extract_perception_obstacles(perception_cloud)
check("LiDAR perception voxel tach 2 vat can va loai ground",
      len(perception_obstacles) == 2 and
      all(o["source"] == "voxel_components" for o in perception_obstacles))
check("LiDAR perception voxel giu centroid/extent/covariance",
      abs(perception_obstacles[0]["centroid"][0] - 4.0) < 0.3 and
      len(perception_obstacles[0]["extent"]) == 3 and
      np.asarray(perception_obstacles[0]["covariance"]).shape == (3, 3))

dataset_gates = evaluate_gates(
    20000, {"train": 14000, "validation": 3000, "test": 3000},
    {split: {"clear", "light_rain", "heavy_rain", "fog", "storm"}
     for split in ("train", "validation", "test")},
    {name: 20000 for name in ("rgb", "road_line", "depth", "lidar", "radar")},
    20000, 0, 0, 30, {"valid": True}, {"Car": 1})
all_dataset_classes = {name: 1 for name in (
    "Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck",
    "TrafficLight", "StopSign")}
ready_dataset_gates = evaluate_gates(
    20000, {"train": 14000, "validation": 3000, "test": 3000},
    {split: {"clear", "light_rain", "heavy_rain", "fog", "storm"}
     for split in ("train", "validation", "test")},
    {name: 20000 for name in ("rgb", "road_line", "depth", "lidar", "radar")},
    20000, 0, 0, 30, {"valid": True}, all_dataset_classes,
    {split: all_dataset_classes for split in ("train", "validation", "test")})
check("dataset audit tach collection PASS khoi 8-class readiness",
      dataset_gates["collection_pass"]
      and not dataset_gates["training_ready_8_class"]
      and "StopSign" in dataset_gates["missing_classes"]
      and ready_dataset_gates["training_ready_8_class"])


class _TargetTransform:
    def __init__(self):
        self.location = type("Location", (), {"x": 0.0, "y": 0.0})()

    @staticmethod
    def get_forward_vector():
        return type("Vector", (), {"x": 1.0, "y": 0.0})()


target_transform = _TargetTransform()
front_light = type("Location", (), {"x": 30.0, "y": 2.0})()
rear_light = type("Location", (), {"x": -20.0, "y": 0.0})()
front_stop = type(
    "StopSign", (), {"transform": type(
        "Transform", (), {"location": front_light})()})()
ordered_stop_spawns, stop_approaches = prioritize_static_stop_sign_spawns(
    [front_stop], [target_transform])
check("targeted capture parse class + uu tien control phia truoc",
      parse_target_classes("car,StopSign,car") == ["Car", "StopSign"]
      and traffic_light_spawn_score(target_transform, front_light) is not None
      and traffic_light_spawn_score(target_transform, rear_light) is None
      and stop_approaches == 1
      and ordered_stop_spawns == [target_transform])
targeted_plan = build_targeted_plan(50)
check("targeted supplement tach Town05/Town10HD va du 5 weather",
      len(targeted_plan) == 10
      and sum(job["frames"] for job in targeted_plan) == 500
      and {job["town"] for job in targeted_plan} == {"Town05", "Town10HD_Opt"}
      and {job["weather"] for job in targeted_plan}
      == {"clear", "light_rain", "heavy_rain", "fog", "storm"})
stop_sign_plan = build_stop_sign_plan(20)
check("stop-sign supplement chi dung validation/test va du 5 weather",
      len(stop_sign_plan) == 10
      and sum(job["frames"] for job in stop_sign_plan) == 200
      and {job["split"] for job in stop_sign_plan} == {"validation", "test"}
      and {job["town"] for job in stop_sign_plan}
      == {"Town05", "Town10HD_Opt"}
      and {job["weather"] for job in stop_sign_plan}
      == {"clear", "light_rain", "heavy_rain", "fog", "storm"})
unsafe_archive_names = []
for archive_name in ("../escape.jpg", "/absolute/file.jpg", "C:/evil.exe"):
    try:
        normalized_member_name(archive_name)
    except ValueError:
        unsafe_archive_names.append(archive_name)
check("dataset archive gate chan path traversal va drive path",
      normalized_member_name("train/images/stop_001.jpg")
      == "train/images/stop_001.jpg"
      and len(unsafe_archive_names) == 3)
combined_counts = {
    split: {name: 1 for name in all_dataset_classes}
    for split in ("train", "validation", "test")
}
combined_counts["train"]["StopSign"] = 0
combined_source_gates = evaluate_source_gates(combined_counts, True)
check("combined object-source audit chi con thieu StopSign train",
      not combined_source_gates["training_ready_8_class"]
      and combined_source_gates["missing_classes_by_split"]["train"]
      == ["StopSign"]
      and not combined_source_gates["missing_classes_by_split"]["validation"]
      and not combined_source_gates["missing_classes_by_split"]["test"]
      and evaluate_source_gates({
          **combined_counts,
          "train": {**combined_counts["train"], "StopSign": 1},
      }, True)["training_ready_8_class"])
valid_stop_row = parse_yolo_row("14 0.5 0.5 0.2 0.3", "memory", 1)
bad_stop_row_rejected = False
try:
    parse_yolo_row("14 0.99 0.5 0.2 0.3", "memory", 1)
except ValueError:
    bad_stop_row_rejected = True
check("StopSign YOLO gate map dung class va chan bbox vuot bien",
      valid_stop_row[0] == 14 and bad_stop_row_rejected)
deduped_records, duplicate_records = deduplicate_records([
    {"stem": "a", "sha256": "same"},
    {"stem": "b", "sha256": "same"},
    {"stem": "c", "sha256": "different"},
])
check("StopSign source loai exact duplicate mot cach xac dinh",
      [record["stem"] for record in deduped_records] == ["a", "c"]
      and duplicate_records == ["b"])
yolo_box = xyxy_to_yolo([20, 10, 60, 50], [100, 100])
check("CARLA bbox xyxy chuyen dung sang YOLO normalized",
      np.allclose(yolo_box, [0.4, 0.3, 0.4, 0.4]))
subset_a = deterministic_subset(list(range(100)), 0.05, 42)
subset_b = deterministic_subset(list(reversed(range(100))), 0.05, 42)
check("training subset giu dung ti le va deterministic theo seed",
      len(subset_a) == 5 and subset_a == subset_b)

# ---------------------------------------------------------------------------
print(f"\nKET QUA: {passed} PASS / {failed} FAIL")
sys.exit(1 if failed else 0)
