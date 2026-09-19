import carla
import cv2
import numpy as np
from ultralytics import YOLO
import queue
import time
import csv
import math
from datetime import datetime
from collections import defaultdict


def process_lane_detection(frame):
    """
    Trích xuất và chồng lớp giới hạn làn đường dựa trên biến đổi Hough.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    polygons = np.array([
        [(0, height), (width, height), (int(width * 0.55),
                                        int(height * 0.6)), (int(width * 0.45), int(height * 0.6))]
    ])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50,
                            minLineLength=40, maxLineGap=150)

    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    blended_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return blended_frame


def main():
    print("[1/3] Đang tải mô hình AI và cấu hình ByteTrack...")
    model = YOLO("yolov8n.pt")

    # [NÂNG CẤP] Khởi tạo bộ nhớ tạm để lưu lịch sử quỹ đạo (Trajectory)
    track_history = defaultdict(lambda: [])
    MAX_TAIL_LENGTH = 30  # Tối ưu hóa: Chỉ giữ lại 30 điểm tọa độ gần nhất để không tràn RAM

    print("[2/3] Đang kết nối tới CARLA Simulator (Cổng 2000)...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    print("[3/3] Đang thiết lập phương tiện và camera hành trình...")
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.03')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"telemetry_log_{timestamp_str}.csv"
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ['Timestamp', 'Location_X', 'Location_Y', 'Speed_kmh', 'Objects_Tracked'])

    print(
        f"\n>>> KHỞI CHẠY THÀNH CÔNG! Dữ liệu đang được ghi vào: {csv_filename} <<<")

    time.sleep(2)

    try:
        while True:
            # 1. Thu thập hình ảnh
            image = image_queue.get()
            img_array = np.reshape(
                np.copy(image.raw_data), (image.height, image.width, 4))
            img_bgr = img_array[:, :, :3]

            # 2. Xử lý Làn đường
            lane_frame = process_lane_detection(img_bgr)

            # 3. [NÂNG CẤP] Chạy luồng Tracking với cấu hình ByteTrack (persit=True để duy trì ID)
            results = model.track(lane_frame, persist=True,
                                  tracker="bytetrack.yaml", verbose=False)

            # Trích xuất khung hình cơ bản đã vẽ ID
            final_frame = results[0].plot()

            # 4. [NÂNG CẤP] Trực quan hóa Quỹ đạo (Đồ họa vệt đuôi xe)
            objects_tracked = 0
            if results[0].boxes.id is not None:
                # Lấy tọa độ tâm x, y và danh sách ID
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                objects_tracked = len(track_ids)

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # Cập nhật điểm tâm mới

                    if len(track) > MAX_TAIL_LENGTH:
                        # Xóa điểm cũ nhất để tạo hiệu ứng đuôi dịch chuyển
                        track.pop(0)

                    # Dựng đường polyline màu cam thể hiện quỹ đạo
                    points = np.hstack(track).astype(
                        np.int32).reshape((-1, 1, 2))
                    cv2.polylines(final_frame, [points], isClosed=False, color=(
                        0, 140, 255), thickness=3)

            # 5. Ghi Dữ liệu Động lực học
            current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            speed_kmh = 3.6 * \
                math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            csv_writer.writerow([current_time, round(location.x, 2), round(
                location.y, 2), round(speed_kmh, 2), objects_tracked])

            # 6. Hiển thị Dashboard
            cv2.putText(final_frame, f"Speed: {speed_kmh:.1f} km/h",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(final_frame, f"Tracked: {objects_tracked}", (
                20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)

            cv2.imshow("Smart Mobility Perception - MOT System", final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\nĐang đóng tệp dữ liệu và dọn dẹp bộ nhớ...")
        csv_file.close()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("Đã hoàn tất!")


if __name__ == '__main__':
    main()
