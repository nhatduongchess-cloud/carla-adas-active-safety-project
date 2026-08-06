"""Module G — Dung hợp cảm biến dư thừa (Camera + LiDAR + Radar) theo nhiễu.

Kiến trúc dư thừa: khi mưa/sương mù nặng, camera và LiDAR suy giảm mạnh còn RADAR
(doppler/range) bền vững -> hạ trọng số quang, tăng trọng số radar. Cung cấp:
    - weights(conditions)          : trọng số động 3 cảm biến (tổng = 1)
    - fuse_range(cam,lidar,radar)  : ước lượng khoảng cách hợp nhất có trọng số
    - classify_event(...)          : đếm 'late_braking' và 'false_disengagement'
"""
# fmt: off
# isort: skip_file


class SensorFusionEval:
    def __init__(self, base_camera=0.4, base_lidar=0.4, base_radar=0.2):
        self.base = {"camera": base_camera, "lidar": base_lidar, "radar": base_radar}

    def weights(self, conditions: dict) -> dict:
        snr = conditions.get("snr", 1.0)
        vis = conditions.get("visibility_m", 200.0)
        # 'optical' = độ khả dụng của cảm biến quang, 0..1.
        optical = max(0.0, min(1.0, snr * min(1.0, vis / 50.0)))

        cam = self.base["camera"] * optical
        lidar = self.base["lidar"] * (0.4 + 0.6 * optical)   # LiDAR bền hơn camera đôi chút
        radar = self.base["radar"] + (1.0 - optical) * 0.5    # radar gánh khi quang kém

        total = cam + lidar + radar
        if total <= 0:
            return {"camera": 0.0, "lidar": 0.0, "radar": 1.0}
        return {
            "camera": round(cam / total, 3),
            "lidar": round(lidar / total, 3),
            "radar": round(radar / total, 3),
        }

    def fuse_range(self, cam_d, lidar_d, radar_d, conditions: dict):
        """Khoảng cách hợp nhất (mét) theo trọng số động; bỏ qua cảm biến None."""
        w = self.weights(conditions)
        num = den = 0.0
        for key, d in (("camera", cam_d), ("lidar", lidar_d), ("radar", radar_d)):
            if d is not None:
                num += w[key] * d
                den += w[key]
        return round(num / den, 2) if den > 0 else None

    @staticmethod
    def classify_event(ttc_at_brake=None, odd_state_at_disengage=None,
                       late_ttc_thresh=0.8) -> dict:
        """Phân loại sự kiện đánh giá an toàn.

        - late_braking: bắt đầu phanh khi TTC đã quá thấp (đáng lẽ phanh sớm hơn).
        - false_disengagement: ngắt/kích fallback khi ODD vẫn NORMAL (báo động giả).
        """
        late = ttc_at_brake is not None and ttc_at_brake < late_ttc_thresh
        false_diseng = odd_state_at_disengage == "NORMAL"
        return {"late_braking": bool(late), "false_disengagement": bool(false_diseng)}
