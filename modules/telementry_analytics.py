
# fmt: off
# isort: skip_file
import carla
import time
import csv
import math

class TelemetryAnalytics:
    def __init__(self, vehicle, output_file="vehicle_telemetry.csv"):
        """
        Khởi tạo module phân tích dữ liệu cho Ego Vehicle.
        """
        self.vehicle = vehicle
        self.output_file = output_file
        self.prev_velocity = carla.Vector3D(0, 0, 0)
        self.prev_time = time.time()

        # Khởi tạo file CSV và ghi Header
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Speed_kmh", "Acceleration_ms2", "Steering_Angle", "Tire_Wear_Index"])

    def compute_wear_index(self, speed_ms, acceleration_ms2, steering):
        """
        Tính toán chỉ số mài mòn dựa trên động lực học của xe.
        """
        k_coefficient = 0.005 # Hệ số hao mòn giả định
        lateral_force_proxy = speed_ms * abs(steering)
        wear_index = k_coefficient * (acceleration_ms2**2 + lateral_force_proxy**2)
        return wear_index

    def log_data(self):
        """
        Hàm này cần được gọi liên tục trong vòng lặp (game loop) của CARLA.
        """
        current_time = time.time()
        dt = current_time - self.prev_time

        # Tránh lỗi chia cho 0 trong vòng lặp đầu tiên
        if dt == 0: 
            return

        # 1. Trích xuất Vận tốc
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6

        # 2. Trích xuất Góc đánh lái
        control = self.vehicle.get_control()
        steering = control.steer

        # 3. Tính toán Gia tốc (dv/dt)
        accel_x = (velocity.x - self.prev_velocity.x) / dt
        accel_y = (velocity.y - self.prev_velocity.y) / dt
        accel_z = (velocity.z - self.prev_velocity.z) / dt
        acceleration_ms2 = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # 4. Đánh giá Mài mòn
        wear_idx = self.compute_wear_index(speed_ms, acceleration_ms2, steering)

        # 5. Ghi dữ liệu ra CSV
        with open(self.output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                round(current_time, 2), 
                round(speed_kmh, 2), 
                round(acceleration_ms2, 2), 
                round(steering, 4), 
                round(wear_idx, 5)
            ])

        # Cập nhật trạng thái cho chu kỳ tiếp theo
        self.prev_velocity = velocity
        self.prev_time = current_time

# ==========================================
# CÁCH TÍCH HỢP VÀO SCRIPT YOLO HIỆN TẠI
# ==========================================
def main():
    # 1. Kết nối với CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # Tìm Ego Vehicle hiện tại trong thế giới (hoặc spawn xe mới)
    ego_vehicle = None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'hero' or actor.attributes.get('role_name') == 'ego_vehicle':
            ego_vehicle = actor
            break

    # Dự phòng: Nếu không gán role_name, lấy chiếc xe đầu tiên tìm thấy
    if not ego_vehicle:
        ego_vehicle = world.get_actors().filter('vehicle.*')[0]

    print(f"Bắt đầu thu thập dữ liệu cho xe: {ego_vehicle.type_id}")

    # 2. Khởi tạo Object Telemetry
    analytics = TelemetryAnalytics(ego_vehicle)

    try:
        # Vòng lặp mô phỏng (Có thể đặt song song với vòng lặp YOLO inference)
        while True:
            analytics.log_data()
            time.sleep(0.1) # Tốc độ lấy mẫu: 10Hz

    except KeyboardInterrupt:
        print("\nĐã dừng thu thập dữ liệu. File vehicle_telemetry.csv đã sẵn sàng.")

if __name__ == '__main__':
    main()