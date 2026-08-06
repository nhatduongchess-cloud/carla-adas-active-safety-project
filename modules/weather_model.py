"""Mô hình hóa điều kiện môi trường từ tham số thời tiết CARLA.

CARLA cho các đại lượng thời tiết (precipitation, fog_density, wetness...) nhưng
KHÔNG cho trực tiếp tầm nhìn/ma sát/SNR. Module này ước lượng ba đại lượng mà
Module E (ODD monitor) cần:
    - visibility_m : tầm nhìn (mét)
    - mu           : hệ số ma sát đường (dry ~0.9 -> soaked ~0.4)
    - snr          : tỉ lệ tín hiệu/nhiễu cảm biến quang (camera/LiDAR), 0..1

Đây là mô hình kỹ thuật có thể tinh chỉnh, không phải giá trị vật lý tuyệt đối.
"""
# fmt: off
# isort: skip_file
import math


def estimate_conditions(precipitation=0.0, fog_density=0.0, wetness=0.0,
                        precipitation_deposits=0.0) -> dict:
    # Tầm nhìn: sương mù chi phối theo hàm mũ; mưa cũng làm giảm.
    vis_fog = 200.0 * math.exp(-fog_density / 25.0)
    vis_rain = 200.0 * math.exp(-precipitation / 120.0)
    visibility = min(vis_fog, vis_rain)

    # Ma sát: mặt đường càng ướt μ càng thấp.
    wet = max(wetness, precipitation, precipitation_deposits) / 100.0
    mu = 0.9 - 0.5 * min(1.0, wet)

    # SNR cảm biến quang: giảm theo sương mù + mưa.
    noise = min(1.0, (fog_density + precipitation) / 200.0)
    snr = max(0.0, 1.0 - noise)

    return {
        "visibility_m": round(visibility, 1),
        "mu": round(mu, 3),
        "snr": round(snr, 3),
    }
