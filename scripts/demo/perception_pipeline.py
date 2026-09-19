import carla
import cv2
import numpy as np
from ultralytics import YOLO
import queue


def process_lane_detection(frame):
    """
    Trích xuất và chồng lớp giới hạn làn đường dựa trên biến đổi Hough.
    """
    # 1. Chuyển đổi không gian màu và làm mịn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Nhận diện gradient biên
    edges = cv2.Canny(blur, 50, 150)

    # 3. Phân vùng tính toán (ROI)
    height, width = edges.shape
    # Thiết lập tọa độ đa giác: Góc dưới trái, Góc dưới phải, Đỉnh phải, Đỉnh trái
    polygons = np.array([
        [(0, height), (width, height), (int(width * 0.55),
                                        int(height * 0.6)), (int(width * 0.45), int(height * 0.6))]
    ])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 4. Trích xuất véc-tơ tuyến tính (Hough Lines)
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50,
                            lines=np.array([]), minLineLength=40, maxLineGap=150)

    # 5. Khởi tạo ma trận rỗng (Kênh màu đen) để render đồ họa vạch kẻ đường
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Vẽ đường thẳng nội suy màu Xanh lá (Green), độ dày 5px
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # 6. Dung hợp (Alpha Blending) mặt nạ vạch kẻ lên khung hình gốc
    blended_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return blended_frame


# [KHỐI 1] Khởi tạo hạt nhân AI (Sử dụng YOLOv8 nano để tối ưu Real-time)
print("Đang tải mô hình Neural Network...")
model = YOLO("yolov8n.pt")

# [KHỐI 2] Thiết lập cầu nối đến CARLA Server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# [KHỐI 3] Khởi tạo Phương tiện & Tích hợp Cảm biến
# Sinh một chiếc Tesla Model 3 ngẫu nhiên trên bản đồ
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
# Kích hoạt chế độ tự lái của giả lập để tạo dữ liệu động
vehicle.set_autopilot(True)

# Định nghĩa thông số RGB Camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera_bp.set_attribute('sensor_tick', '0.03')  # Giới hạn ~30 FPS

# Gắn camera lên mui xe (Tọa độ tương đối so với trung tâm xe)
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# [KHỐI 4] Cấu hình Luồng Dữ liệu (Data Stream)
image_queue = queue.Queue()
camera.listen(image_queue.put)

try:
    print("Khởi chạy luồng Perception thành công. Nhấn 'q' tại cửa sổ Camera để thoát.")
    while True:
        # Rút trích khung hình từ hàng đợi
        image = image_queue.get()

        # Tiền xử lý: Ép kiểu dữ liệu raw của CARLA thành ma trận numpy BGR
        img_array = np.reshape(np.copy(image.raw_data),
                               (image.height, image.width, 4))
        img_bgr = img_array[:, :, :3]

        # [KHỐI 5] Suy luận Computer Vision (Inference)
        # Truyền ma trận ảnh qua mạng nơ-ron
        results = model(img_bgr, verbose=False)

        # Trích xuất khung hình đã được AI dán nhãn (Bounding Boxes & Labels)
        annotated_frame = results[0].plot()

        # [KHỐI 6] Khởi tạo Dashboard hiển thị
        cv2.imshow("Smart Mobility Perception - YOLOv8", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # [KHỐI 5] Suy luận Computer Vision (Inference)

        # Bước A: Nội suy Làn đường (Lane Detection)
        lane_frame = process_lane_detection(img_bgr)

        # Bước B: Truyền ma trận đã xử lý làn đường qua mạng nơ-ron YOLO để nhận diện vật thể
        results = model(lane_frame, verbose=False)

        # Bước C: Trích xuất khung hình cuối cùng với đầy đủ Bounding Boxes và Lane Lines
        final_frame = results[0].plot()

        # [KHỐI 6] Khởi tạo Dashboard hiển thị
        cv2.imshow(
            "Smart Mobility Perception - YOLO & Lane Detection", final_frame)

finally:
    # Cơ chế tự động giải phóng tài nguyên hệ thống
    print("Đang hủy kết nối và giải phóng bộ nhớ...")
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    cv2.destroyAllWindows()
