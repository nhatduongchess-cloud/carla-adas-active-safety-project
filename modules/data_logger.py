import csv
import math
from datetime import datetime


class TelemetryLogger:
    """Mô-đun đo lường và ghi nhận dữ liệu định lượng thời gian thực cho ADAS."""

    def __init__(self, flush_every=15):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = f"telemetry_log_{timestamp}.csv"
        self.file = open(self.filename, mode='w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'Timestamp', 'Location_X', 'Location_Y', 'Speed_kmh', 'Objects_Tracked'
        ])
        self._flush_every = max(1, flush_every)
        self._writes = 0

    def log(self, vehicle, objects_tracked):
        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * \
            math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        self.writer.writerow([
            current_time,
            round(location.x, 2),
            round(location.y, 2),
            round(speed_kmh, 2),
            objects_tracked
        ])
        # Ghi đĩa thưa lại để bớt I/O mỗi khung (flush theo chu kỳ).
        self._writes += 1
        if self._writes % self._flush_every == 0:
            self.file.flush()
        return speed_kmh

    def close(self):
        if self.file:
            self.file.close()
