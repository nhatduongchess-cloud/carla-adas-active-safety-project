"""Nạp hồ sơ thời tiết từ config/weather_config.yaml (chỉ phụ thuộc PyYAML).

Tách riêng khỏi CARLA: hàm này chỉ trả về dict tham số; chinh.py / evaluate_l3.py
sẽ dựng carla.WeatherParameters từ dict đó.
"""
# fmt: off
# isort: skip_file
import os
import yaml


def load_weather_profiles(path="config/weather_config.yaml") -> dict:
    if not os.path.isabs(path):
        # Cho phép chạy từ thư mục gốc dự án.
        path = os.path.join(os.getcwd(), path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("profiles", {})


def weather_kwargs(profile: dict) -> dict:
    """Lọc các khóa hợp lệ để dựng carla.WeatherParameters (bỏ 'expect_odd')."""
    keys = ("cloudiness", "precipitation", "precipitation_deposits", "wetness",
            "fog_density", "sun_altitude_angle", "wind_intensity",
            "fog_distance", "fog_falloff", "scattering_intensity")
    return {k: float(v) for k, v in profile.items() if k in keys}
