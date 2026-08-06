# -*- coding: utf-8 -*-
"""
Cấu hình trung tâm (single source of truth) cho toàn bộ pipeline ADAS.

Mọi tham số về độ phân giải, hình học cảm biến và ngưỡng an toàn đều khai báo
tại đây. chinh.py sẽ import module này và truyền các giá trị xuống từng module,
nên KHÔNG được để mỗi file tự "đoán" độ phân giải như trước (800x600 vs 1280x720
vs 1280) — đó chính là nguồn gốc lỗi sai vùng ROI và AEB không kích hoạt.
"""
import math

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

# Vị trí gắn cảm biến (mét, tương đối tâm xe). Camera và LiDAR đặt gần trùng nhau
# để có thể xấp xỉ chung một khung chiếu (pinhole) mà không cần calib phức tạp.
CAM_X, CAM_Z = 1.5, 2.4
LIDAR_X, LIDAR_Z = 1.5, 2.5

# Dấu trục Y của LiDAR so với "bên phải" của ảnh camera.
#   +1.0 : điểm bên phải xe -> nửa phải khung hình (quy ước phổ biến của CARLA)
#   -1.0 : nếu thấy vật thể trái/phải bị gán khoảng cách ngược -> đổi sang -1.0
LIDAR_Y_SIGN = 1.0


def camera_intrinsics(width: int = CAM_WIDTH, height: int = CAM_HEIGHT, fov: float = CAM_FOV):
    """Trả về (fx, fy, cx, cy) cho mô hình pinhole của CARLA."""
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return f, f, width / 2.0, height / 2.0


# ==============================================================================
# 3. PHÁT HIỆN VẬT THỂ (YOLO ensemble)
# ==============================================================================
CONF_THRES = 0.40
IOU_THRES = 0.45
# COCO: 2 car, 3 motorcycle, 5 bus, 7 truck  (phương tiện giao thông)
DETECTED_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Chiều cao thực (mét) dùng cho ước lượng khoảng cách đơn mắt (mono fallback)
# khi không có cụm LiDAR nào khớp vào bounding box.
REAL_HEIGHTS_M = {
    'Car': 1.5,
    'Motorcycle': 1.5,
    'Bus': 3.2,
    'Truck': 3.5,
    'Person': 1.7,
}
DEFAULT_HEIGHT_M = 1.5

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

# ==============================================================================
# 8. KIỂM THỬ (traffic + kịch bản)
# ==============================================================================
NUM_NPC_VEHICLES = 30          # số xe NPC mặc định (ghi đè bằng --vehicles)
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

# --- Tối ưu FPS mà GIỮ NGUYÊN mô hình (phát hiện thưa + tracker nội suy) ---
# Chạy ensemble YOLO mỗi N khung; giữa các khung, MultiObjectTracker (Kalman) dự
# đoán -> vẫn dùng đúng mô hình, chỉ giảm tần suất. Đặt =1 nếu muốn full-rate.
DETECT_EVERY_N = 3             # cấu hình thấp: phát hiện YOLO mỗi 3 khung (tracker Kalman nội suy)
LANE_EVERY_N = 5              # dò làn (Hough) thưa hơn
RENDER_EVERY_N = 3            # vẽ dashboard + imshow mỗi 3 khung
LOG_EVERY_N = 6              # ghi telemetry mỗi 6 khung
TELEMETRY_FLUSH_EVERY = 15     # ghi đĩa telemetry thưa lại (bớt I/O mỗi khung)
CUDNN_BENCHMARK = True         # bật cudnn.benchmark cho kích thước cố định (nhanh, không đổi độ chính xác)

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
