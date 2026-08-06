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
from odd_monitor import ODDMonitor              # noqa: E402
from mrm_controller import L3StateMachine       # noqa: E402
from sensor_fusion_eval import SensorFusionEval  # noqa: E402
from l3_report import assess_profile, build_report, write_report  # noqa: E402
import scenario_library as scnlib  # noqa: E402
from rl_experiment import SpeedExperiment  # noqa: E402
from rl_env import SpeedControlEnv  # noqa: E402
from rl_speed_controller import RLSpeedController  # noqa: E402
from turn_intent import intent_to_right  # noqa: E402


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
    def __init__(self, x, y, vx, vy):
        self.pos = (x, y)
        self.vel = (vx, vy)


s = ActiveSafetySystem()
d = s.update(12.0, [], [], 0.05, tracks=[_StubTrack(20.0, 0.0, -15.0, 0.0)])
check("track lao toi nhanh (TTC~1.3s) -> EMERGENCY_BRAKE", d.state == "EMERGENCY_BRAKE")

s = ActiveSafetySystem()
d = s.update(8.0, [], [], 0.05, tracks=[_StubTrack(20.0, 0.0, -3.0, 0.0)])
check("track tien cham (TTC~6.7s), con xa -> NORMAL", d.state == "NORMAL")

s = ActiveSafetySystem()
d = s.update(12.0, [], [], 0.05, tracks=[_StubTrack(20.0, 3.0, -15.0, 0.0)])
check("track ngoai lan -> NORMAL du toc do cao", d.state == "NORMAL")

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
check("da cam ket -> van giu EVADE_RIGHT (khong phan van)", d2.state == "EVADE_RIGHT")

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

# Turn intent
check("intent_to_right('right') True", intent_to_right("right") is True)
check("intent_to_right('left') False", intent_to_right("left") is False)

# ---------------------------------------------------------------------------
print(f"\nKET QUA: {passed} PASS / {failed} FAIL")
sys.exit(1 if failed else 0)
