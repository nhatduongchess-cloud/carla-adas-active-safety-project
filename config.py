# -*- coding: utf-8 -*-
"""
Cấu hình trung tâm (single source of truth) cho toàn bộ pipeline ADAS.

Mọi tham số về độ phân giải, hình học cảm biến và ngưỡng an toàn đều khai báo
tại đây. chinh.py sẽ import module này và truyền các giá trị xuống từng module,
nên KHÔNG được để mỗi file tự "đoán" độ phân giải như trước (800x600 vs 1280x720
vs 1280) — đó chính là nguồn gốc lỗi sai vùng ROI và AEB không kích hoạt.
"""
import math
import os

# ==============================================================================
# 1. VÒNG LẶP MÔ PHỎNG (Simulation loop)
# ==============================================================================
FPS = 40
FIXED_DELTA = 1.0 / FPS  # giây mỗi tick (dùng cho tính TTC)

# ==============================================================================
# 2. CAMERA RGB  — nguồn chân lý duy nhất về độ phân giải & FOV
# ==============================================================================
CAM_WIDTH = 960
CAM_HEIGHT = 540
CAM_FOV = 90.0  # độ
CAMERA_SENSOR_TICK_S = 0.0
CAMERA_POSTPROCESS = True

# Vị trí gắn cảm biến (mét, tương đối tâm xe). Camera và LiDAR đặt gần trùng nhau
# để có thể xấp xỉ chung một khung chiếu (pinhole) mà không cần calib phức tạp.
CAM_X, CAM_Z = 1.5, 2.4
LIDAR_X, LIDAR_Z = 1.5, 2.5
CAM_Y = 0.0
LIDAR_Y = 0.0
CAM_ROLL, CAM_PITCH, CAM_YAW = 0.0, 0.0, 0.0
LIDAR_ROLL, LIDAR_PITCH, LIDAR_YAW = 0.0, 0.0, 0.0

# Dấu trục Y của LiDAR so với "bên phải" của ảnh camera.
#   +1.0 : điểm bên phải xe -> nửa phải khung hình (quy ước phổ biến của CARLA)
#   -1.0 : nếu thấy vật thể trái/phải bị gán khoảng cách ngược -> đổi sang -1.0
LIDAR_Y_SIGN = 1.0

# Radar trước xe là sensor DỰ PHÒNG: thiếu frame không được chặn control. Tọa độ
# đầu ra được đổi về cùng gốc x với LiDAR để tracker/AEB không nhận một bước nhảy
# giả do hai sensor gắn lệch nhau 0.5 m.
ENABLE_RADAR = True
RADAR_X, RADAR_Y, RADAR_Z = 2.0, 0.0, 0.8
RADAR_ROLL, RADAR_PITCH, RADAR_YAW = 0.0, 0.0, 0.0
RADAR_RANGE_M = 80.0
RADAR_HORIZONTAL_FOV = 35.0
RADAR_VERTICAL_FOV = 10.0
RADAR_POINTS_PER_SECOND = 4000
RADAR_SENSOR_TICK_S = 0.0
RADAR_OPTIONAL_TIMEOUT_S = 0.005
# Điểm hit radar phải cao hơn mặt đường ít nhất mức này. Với radar cao 0.8 m và
# vertical FOV 10°, tia thấp chạm road ở ~9-10 m; không lọc sẽ gây AEB giả trên
# mọi map. LiDAR vẫn chịu trách nhiệm với cone/debris thấp sát mặt đường.
RADAR_MIN_TARGET_HEIGHT_M = 0.15
# Đã đo trên CARLA đang dùng bằng validate_radar_sign.py: raw velocity ÂM khi
# actor tiến gần và DƯƠNG khi rời xa. Chuẩn hóa lại thành closing-speed dương.
RADAR_VELOCITY_SIGN = -1.0
RADAR_REQUIRE_SIGN_VALIDATION = True
RADAR_SIGN_VALIDATION_PATH = "logs/radar_sign_validation.json"
RADAR_MAHALANOBIS_GATE = 16.0
RADAR_MAX_MISSING_FRAMES = 3


def camera_intrinsics(width: int = CAM_WIDTH, height: int = CAM_HEIGHT, fov: float = CAM_FOV):
    """Trả về (fx, fy, cx, cy) cho mô hình pinhole của CARLA."""
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return f, f, width / 2.0, height / 2.0


# ==============================================================================
# 3. PHÁT HIỆN VẬT THỂ (YOLO ensemble)
# ==============================================================================
CONF_THRES = 0.40
IOU_THRES = 0.45
# COCO road users + traffic semantics.
DETECTED_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]
CLASS_NAMES = {
    0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle',
    5: 'Bus', 7: 'Truck', 9: 'TrafficLight', 11: 'StopSign',
}

# Chiều cao thực (mét) dùng cho ước lượng khoảng cách đơn mắt (mono fallback)
# khi không có cụm LiDAR nào khớp vào bounding box.
REAL_HEIGHTS_M = {
    'Car': 1.5,
    'Motorcycle': 1.5,
    'Bus': 3.2,
    'Truck': 3.5,
    'Person': 1.7,
    'Bicycle': 1.5,
    'TrafficLight': 0.8,
    'StopSign': 0.75,
}
DEFAULT_HEIGHT_M = 1.5
VRU_LATERAL_MARGIN_M = 0.8
VRU_PREDICTION_HORIZON_S = 4.0
TRAFFIC_LIGHT_RANGE_M = 45.0
STOP_SIGN_RANGE_M = 18.0
STOP_SIGN_HOLD_S = 1.5
STOP_SIGN_COOLDOWN_S = 8.0
TRAFFIC_LIGHT_MODEL_PATH = ""  # local .npz only; empty = HSV + temporal vote fallback

# ==============================================================================
# 4. ƯỚC LƯỢNG ĐỘ SÂU (MiDaS) — tùy chọn
# ==============================================================================
# LiDAR đã cho khoảng cách mét chính xác, nên MiDaS KHÔNG nằm trên đường quyết
# định phanh. Tắt để tăng FPS; bật nếu muốn bản đồ độ sâu để trực quan hóa.
USE_MIDAS = False
DEPTH_EVERY_N_FRAMES = 2

# ==============================================================================
# 5. LỌC KHOẢNG CÁCH HỢP LỆ
# ==============================================================================
DEPTH_MIN_M = 1.0
DEPTH_MAX_M = 60.0

# ==============================================================================
# 6. AN TOÀN CHỦ ĐỘNG (Active safety / AEB / tránh vật)
# ==============================================================================
# Hành lang "trong làn" theo trục ngang LiDAR (mét). ~1.75m mỗi bên ≈ nửa làn.
LANE_HALF_WIDTH_M = 1.75

# Số điểm tối thiểu của một cụm LiDAR để coi là vật thể thật (khử nhiễu).
MIN_CLUSTER_POINTS = 5   # hạ ngưỡng: LiDAR thưa hơn vẫn bắt được vật cản -> AEB ổn định

# Mô hình khoảng cách phanh động: d_safe = v * reaction + margin
REACTION_TIME_S = 1.0
MIN_SAFE_DIST_M = 6.0

# Ngưỡng thời gian-va-chạm (Time-To-Collision, giây)
CRITICAL_TTC_S = 1.6   # dưới mức này -> PHANH GẤP
WARNING_TTC_S = 3.0    # dưới mức này -> giảm tốc / tránh
MIN_CLOSING_SPEED = 0.3  # m/s, dưới mức này coi như không tiến lại gần

# Tránh né (evasion) bằng chuyển làn qua Traffic Manager
ENABLE_EVASION = True
EVADE_LOOKAHEAD_M = 25.0       # tầm nhìn kiểm tra làn bên có trống không
LANE_CHANGE_COOLDOWN_S = 4.0   # chống spam lệnh chuyển làn
# Coi xe dẫn đầu là "chậm/đứng yên" khi tốc độ tiến lại gần ≥ tỉ lệ này của tốc độ ego
LEAD_SLOW_RATIO = 0.55

# ==============================================================================
# 7. TRAFFIC MANAGER (autopilot nền)
# ==============================================================================
TM_PORT = 8000
# % chênh tốc so với giới hạn: âm = nhanh hơn, dương = chậm hơn.
TM_DEFAULT_SPEED_DIFF = 0.0
TM_FOLLOW_SPEED_DIFF = 60.0   # khi FOLLOW: chạy chậm còn ~40% giới hạn
TM_LEADING_DISTANCE_M = 3.0

# Ego dùng stack planning/control của dự án. Nếu controller phát sinh lỗi runtime,
# EgoController phải MRM/safe-stop; không được tự bật Traffic Manager autopilot
# vì đường này đang là nguồn native CARLA crash cần cô lập (W1.2).
EGO_CONTROL_MODE = "custom"          # "custom" | "traffic_manager"
# Cờ legacy chỉ giữ tương thích config cũ; runtime custom không còn dùng nó để
# chuyển ego sang TM. Chỉ EGO_CONTROL_MODE="traffic_manager" mới là opt-in rõ ràng.
CUSTOM_CONTROL_FALLBACK_TM = False
CONTROL_LOOKAHEAD_M = 45.0
CONTROL_WHEELBASE_M = 2.85
CONTROL_MAX_STEER_DEG = 35.0
CONTROL_KP = 0.45
CONTROL_KI = 0.04
CONTROL_KD = 0.08
LANE_CHANGE_TRANSITION_M = 22.0

# ==============================================================================
# 8. KIỂM THỬ (traffic + kịch bản)
# ==============================================================================
# 24 NPC is the measured stable default for the scale-85 profile on the target
# RTX 4070 Laptop. 26 NPC remains available through ``--vehicles 26``, but fell
# below the 38 Hz real-time acceptance gate in the repeatable Town02 soak test.
NUM_NPC_VEHICLES = 24
# khoảng cách đặt xe chướng ngại đứng yên (--hazard)
HAZARD_DISTANCE_M = 20.0

# ==============================================================================
# 9. HIỆU NĂNG / GPU  (giảm lag)
# ==============================================================================
YOLO_DEVICE = 'cuda'           # 'cuda' để chạy trên GPU (RTX 4070)
USE_ENSEMBLE = False           # CHỈ 1 mô hình (yolov8n) -> MƯỢT hơn (đặt True nếu muốn chính xác cao)
YOLO_MODEL_A = 'yolov8n.pt'
YOLO_MODEL_B = 'yolov10n.pt'
YOLO_IMGSZ = 480               # kích thước suy luận nhỏ hơn -> nhanh hơn (cấu hình thấp)
YOLO_HALF = True               # fp16 trên GPU -> nhanh hơn, gần như không giảm độ chính xác
YOLO_USE_OPTIMIZED = False      # onnxruntime hiện là bản CPU -> dùng .pt trên GPU nhanh hơn. Bật True khi có onnxruntime-gpu/TensorRT thật
LIDAR_MAX_POINTS = 2000        # hạ mẫu mạnh hơn trước DBSCAN (cấu hình thấp)
LIDAR_POINTS_PER_SECOND = 60000  # thưa hơn nhưng VẪN đủ để AEB ổn định (đừng để quá thấp như 5000)
LIDAR_RANGE_M = 60.0
LIDAR_CHANNELS = 32
LIDAR_SENSOR_TICK_S = 0.0

# --- Tối ưu FPS mà GIỮ NGUYÊN mô hình (phát hiện thưa + tracker nội suy) ---
# Chạy ensemble YOLO mỗi N khung; giữa các khung, MultiObjectTracker (Kalman) dự
# đoán -> vẫn dùng đúng mô hình, chỉ giảm tần suất. Đặt =1 nếu muốn full-rate.
DETECT_EVERY_N = 2             # 20 Hz tại world tick 40 Hz
LANE_EVERY_N = 4               # learned lane/Hough shadow 10 Hz
RENDER_EVERY_N = 3            # vẽ dashboard + imshow mỗi 3 khung
LOG_EVERY_N = 6              # ghi telemetry mỗi 6 khung
TELEMETRY_FLUSH_EVERY = 15     # ghi đĩa telemetry thưa lại (bớt I/O mỗi khung)
CUDNN_BENCHMARK = True         # bật cudnn.benchmark cho kích thước cố định (nhanh, không đổi độ chính xác)
MAX_ASYNC_RESULT_AGE_MS = 150.0
GPU_QUEUE_SIZE = 1
SAFETY_DEADLINE_MS = 25.0
PERCEPTION_DEADLINE_MS = 50.0
VRAM_LIMIT_MB = 7680.0

# Runtime profiles temporarily override these values. Keep an immutable copy so
# changing profile in the same Python process cannot leak values from the prior
# run (for example low-memory -> quality remaining at 640x360).
_PERFORMANCE_BASELINE = {
    "CAM_WIDTH": CAM_WIDTH,
    "CAM_HEIGHT": CAM_HEIGHT,
    "YOLO_IMGSZ": YOLO_IMGSZ,
    "LIDAR_POINTS_PER_SECOND": LIDAR_POINTS_PER_SECOND,
    "LIDAR_MAX_POINTS": LIDAR_MAX_POINTS,
    "RENDER_EVERY_N": RENDER_EVERY_N,
    "CAMERA_SENSOR_TICK_S": CAMERA_SENSOR_TICK_S,
    "CAMERA_POSTPROCESS": CAMERA_POSTPROCESS,
    "LIDAR_SENSOR_TICK_S": LIDAR_SENSOR_TICK_S,
    "RADAR_SENSOR_TICK_S": RADAR_SENSOR_TICK_S,
}


def apply_performance_overrides(overrides):
    """Restore the quality baseline, then apply one runtime profile atomically."""
    unknown = set(overrides) - set(_PERFORMANCE_BASELINE)
    if unknown:
        raise KeyError(f"unknown performance setting(s): {sorted(unknown)}")
    globals().update(_PERFORMANCE_BASELINE)
    globals().update(dict(overrides))

# Learned lane adapter. Runtime KHÔNG tải model; chỉ nạp artifact local có
# provenance/checksum. Khi trống hoặc lỗi, map path nắm quyền và Hough vẫn shadow.
LEARNED_LANE_ENABLED = True
LEARNED_LANE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights", "ufldv2_tusimple_res18.pth")
LEARNED_LANE_MODEL_SHA256 = (
    "490d8f81995cf6382005e037ca70b5260de6cff7da863c0965b2daeb3f710a4c")
LEARNED_LANE_DEVICE = "cuda"
LEARNED_LANE_FP16 = True
LEARNED_LANE_INPUT_WIDTH = 800
LEARNED_LANE_INPUT_HEIGHT = 320
LEARNED_LANE_CONFIDENCE = 0.75
LEARNED_LANE_STABLE_FRAMES = 5
LEARNED_LANE_INVALID_FRAMES = 3
LANE_WIDTH_MIN_M = 2.8
LANE_WIDTH_MAX_M = 4.2
LANE_GEOMETRY_JUMP_M = 0.6

# Làm mượt tín hiệu điều khiển -> chống giật ga/phanh (rubberbanding).
DENSITY_EMA = 0.3            # hệ số mượt mật độ giao thông (0=đóng băng, 1=không mượt)
RL_TARGET_EMA = 0.3         # hệ số mượt tốc độ mục tiêu RL

# ==============================================================================
# 10. L3 / ODD / MRM (DRIVE PILOT-style)
# ==============================================================================
DRY_FRICTION_MU = 0.85         # μ mặc định đường khô
TOR_WINDOW_S = 10.0            # cửa sổ yêu cầu tài xế tiếp quản (Takeover Request)
MRM_DECEL_FRAC = 0.35          # tỉ lệ μ*g dùng cho giảm tốc êm khi MRM
L3_REPORT_PATH = "logs/mercedes_l3_validation_report.json"
WEATHER_CONFIG_PATH = "config/weather_config.yaml"

# ==============================================================================
# 11. HỌC TĂNG CƯỜNG (RL) — tối ưu tốc độ theo mật độ + chuyển làn theo ý định rẽ
# ==============================================================================
USE_RL_SPEED = True            # đặt tốc độ tuần hành bằng policy RL (đã train)
RL_POLICY_PATH = "weights/rl_speed_policy.pt"
RL_SPEED_EVERY_N = 5           # cập nhật tốc độ mong muốn mỗi N khung hình
DENSITY_RANGE_M = 40.0         # tầm nhìn ước lượng mật độ giao thông phía trước
DENSITY_NORM = 5.0             # số PHƯƠNG TIỆN phía trước coi là "đông" (chuẩn hoá 0..1)
RL_MIN_CRUISE_KMH = 18.0       # sàn tốc độ khi đường thoáng (tránh xe đứng yên do lệnh RL ~0)
