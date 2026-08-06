"""Interface Experiment cho RL — SOI GƯƠNG kiến trúc CARLA rllib-integration.

rllib-integration tách biệt: CarlaEnv(gym.Env) chỉ điều phối, còn BaseExperiment
định nghĩa observation/action/reward. Ta tái hiện đúng interface đó nhưng KHÔNG
phụ thuộc gym/ray (chỉ numpy) để chạy được trên máy bạn.

SpeedExperiment: bài toán TỐI ƯU TỐC ĐỘ theo mật độ giao thông cho xe điện —
đường vắng chạy >30 km/h, đường đông chạy <30 km/h, đồng thời tiết kiệm năng lượng.
Cùng một định nghĩa obs/reward được DÙNG CHUNG cho:
  - môi trường thay thế nhanh (rl_env.SpeedControlEnv) để huấn luyện offline, và
  - triển khai thật trong CARLA (rl_speed_controller).
"""
# fmt: off
# isort: skip_file
import numpy as np


# --- Shim không gian (thay cho gym.spaces để khỏi phụ thuộc gym) ---
class Discrete:
    def __init__(self, n):
        self.n = int(n)

    def __repr__(self):
        return f"Discrete({self.n})"


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype

    def __repr__(self):
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"


class BaseExperiment:
    """Cùng bộ phương thức như rllib_integration/base_experiment.py::BaseExperiment."""

    def __init__(self, config=None):
        self.config = config or {}

    def reset(self):
        pass

    def get_action_space(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError

    def compute_action(self, action):
        raise NotImplementedError

    def get_observation(self, state):
        raise NotImplementedError

    def get_done_status(self, state):
        raise NotImplementedError

    def compute_reward(self, state):
        raise NotImplementedError


class SpeedExperiment(BaseExperiment):
    """Định nghĩa obs/action/reward cho tối ưu tốc độ theo mật độ (xe điện)."""

    ACTIONS_KMH = [0, 15, 25, 35, 45, 55]
    V_MAX_KMH = 60.0
    OBS_DIM = 5

    def __init__(self, config=None):
        super().__init__(config)
        self.v_max = self.V_MAX_KMH / 3.6            # m/s
        self.accel_max = 2.5
        self.t_headway = 1.5
        self.d0 = 5.0
        self.d_crit = 4.0
        # trọng số reward (đã tinh chỉnh để đường vắng ~35 km/h, đường đông <30)
        self.w_progress = 1.2
        self.w_energy = 1.0
        self.w_energy_accel = 0.3
        self.w_safety = 1.5
        self.w_collision = 20.0
        self.w_comfort = 0.2
        self.w_shape = 0.15

    # -- không gian --
    def get_action_space(self):
        return Discrete(len(self.ACTIONS_KMH))

    def get_observation_space(self):
        return Box(0.0, 1.0, (self.OBS_DIM,))

    def get_actions(self):
        return {i: kmh for i, kmh in enumerate(self.ACTIONS_KMH)}

    def compute_action(self, action):
        """Policy -> tốc độ mục tiêu (m/s). (Triển khai CARLA sẽ đưa vào TM.)"""
        return self.ACTIONS_KMH[int(action)] / 3.6

    # -- quan sát --
    def get_observation(self, state):
        v = state["v_ms"]
        lead_dist = state.get("lead_dist", 60.0)
        lead_speed = state.get("lead_speed", v)
        prev_target = state.get("prev_target_ms", v)
        ttc = lead_dist / max(0.1, v - lead_speed) if v > lead_speed else 99.0
        obs = np.array([
            v / self.v_max,
            state["density"],
            min(lead_dist, 60.0) / 60.0,
            min(ttc, 10.0) / 10.0,
            prev_target / self.v_max,
        ], dtype=np.float32)
        return obs, {}

    def obs_from_measured(self, v_ms, density, nearest_m, ttc_s, prev_target_ms):
        """Xây observation từ SỐ ĐO THẬT (khi triển khai CARLA) — cùng chuẩn hoá."""
        nearest_m = 60.0 if nearest_m is None else nearest_m
        ttc_s = 99.0 if ttc_s is None else ttc_s
        return np.array([
            min(v_ms, self.v_max) / self.v_max,
            float(np.clip(density, 0.0, 1.0)),
            min(nearest_m, 60.0) / 60.0,
            min(ttc_s, 10.0) / 10.0,
            min(prev_target_ms, self.v_max) / self.v_max,
        ], dtype=np.float32)

    def get_done_status(self, state):
        return bool(state.get("collided", False))

    # -- phần thưởng (phi tuyến: năng lượng ~ v^2 + a^2) --
    def compute_reward(self, state):
        v = state["v_ms"]
        a = state.get("accel_ms2", 0.0)
        target = state.get("target_ms", v)
        prev_target = state.get("prev_target_ms", v)
        density = state["density"]
        lead_present = state.get("lead_present", False)
        lead_dist = state.get("lead_dist", 60.0)
        collided = state.get("collided", False)

        v_n = v / self.v_max
        progress = self.w_progress * v_n
        energy = self.w_energy * (v_n ** 2 + self.w_energy_accel * (a / self.accel_max) ** 2)

        safety = 0.0
        if collided:
            safety -= self.w_collision
        elif lead_present:
            desired_gap = v * self.t_headway + self.d0
            if lead_dist < desired_gap:
                safety -= self.w_safety * (desired_gap - lead_dist) / desired_gap

        comfort = -self.w_comfort * ((target - prev_target) / self.v_max) ** 2

        v_kmh = v * 3.6
        shape = 0.0
        if density < 0.3 and v_kmh > 30.0:
            shape += self.w_shape
        if density > 0.6 and v_kmh < 30.0:
            shape += self.w_shape

        return float(progress - energy + safety + comfort + shape)
