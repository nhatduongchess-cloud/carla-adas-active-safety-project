import cv2
import numpy as np


def draw_bev(objects_data, width=300, height=300):
    """Vẽ sơ đồ góc nhìn từ trên xuống (BEV) dựa trên khoảng cách và độ lệch ngang."""
    bev_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    bev_canvas[:] = (30, 30, 30)  # Nền tối xám

    ego_x, ego_y = width // 2, height - 20
    cv2.circle(bev_canvas, (ego_x, ego_y), 6, (0, 255, 255), -1)

    for obj in objects_data:
        lat_offset = obj.get('lateral_offset', 0.0)
        distance = obj.get('distance', 10.0)
        risk = obj.get('risk', 'Safe')

        px = int(ego_x + lat_offset * 15)
        py = int(ego_y - distance * 5)

        if 0 <= px < width and 0 <= py < height:
            color = (0, 0, 255) if risk == "CRITICAL" else (
                0, 165, 255) if risk == "Warning" else (0, 255, 0)
            cv2.circle(bev_canvas, (px, py), 5, color, -1)

    return bev_canvas
