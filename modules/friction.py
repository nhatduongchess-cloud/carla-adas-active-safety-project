"""Động lực học phanh phụ thuộc hệ số ma sát đường μ.

Công thức quãng đường dừng (physics-based):
    d_stop = v * t_response + v^2 / (2 * μ * g)
trong đó phần đầu là quãng đường trong thời gian phản ứng, phần sau là quãng
đường phanh vật lý. Khi đường ướt/băng (μ nhỏ) -> d_stop tăng mạnh -> hệ thống
phải giữ khoảng cách xa hơn và phanh sớm hơn.
"""
# fmt: off
# isort: skip_file

G = 9.81  # m/s^2


def stopping_distance(v_ms: float, mu: float, t_response: float = 1.0) -> float:
    """Quãng đường dừng an toàn (mét) theo tốc độ v và ma sát μ."""
    mu = max(0.05, mu)
    v = max(0.0, v_ms)
    return v * t_response + (v * v) / (2.0 * mu * G)


def max_comfortable_decel(mu: float, comfort_frac: float = 1.0) -> float:
    """Gia tốc phanh tối đa khả dụng (m/s^2) trên mặt đường có ma sát μ."""
    return comfort_frac * max(0.05, mu) * G
