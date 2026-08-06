"""Môi trường thay thế (surrogate) để HUẤN LUYỆN offline — mirror CarlaEnv.

Giống rllib-integration: lớp Env chỉ lo ĐỘNG HỌC + tick; còn observation/reward do
SpeedExperiment định nghĩa (dùng chung với triển khai CARLA thật). Ở đây "core" là
một mô hình động học nhanh (numpy) mô phỏng mật độ giao thông thay đổi + xe dẫn
đầu, nên train được trong vài phút trên CPU mà KHÔNG cần CARLA.
"""
# fmt: off
# isort: skip_file
import numpy as np

try:
    from rl_experiment import SpeedExperiment
except ImportError:  # khi import kiểu package
    from modules.rl_experiment import SpeedExperiment

# Tương thích các import cũ
ACTIONS_KMH = SpeedExperiment.ACTIONS_KMH
V_MAX_KMH = SpeedExperiment.V_MAX_KMH
OBS_DIM = SpeedExperiment.OBS_DIM


class SpeedControlEnv:
    def __init__(self, seed=None, horizon=200, dt=0.5, experiment=None):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.dt = dt
        self.exp = experiment or SpeedExperiment()
        self.v_max = self.exp.v_max
        self.accel_max = self.exp.accel_max
        self._fixed = False
        self.reset()

    @property
    def n_actions(self):
        return self.exp.get_action_space().n

    # ------------------------------------------------------------------ #
    def reset(self):
        self._fixed = False
        self.t = 0
        self.v = float(self.rng.uniform(0, self.v_max * 0.5))
        self.density = float(self.rng.uniform(0, 1))
        self._density_target = self.density
        self.prev_target = self.v
        self._spawn_lead()
        return self.exp.get_observation(self._state())[0]

    def set_density(self, d):
        """Cố định mật độ (dùng khi ĐÁNH GIÁ đường vắng/đông riêng biệt)."""
        self.density = float(d)
        self._density_target = float(d)
        self._fixed = True
        self._spawn_lead()

    def _spawn_lead(self):
        if self.density > 0.35 and self.rng.random() < 0.9:
            self.lead = True
            self.lead_speed = self.v_max * (1.0 - 0.85 * self.density)
            self.lead_dist = float(self.rng.uniform(15, 40))
        else:
            self.lead = False
            self.lead_speed = self.v
            self.lead_dist = 60.0

    def _evolve_density(self):
        if self._fixed:
            return
        if self.rng.random() < 0.02:
            self._density_target = float(self.rng.choice([0.1, 0.5, 0.9]))
        self.density += 0.05 * (self._density_target - self.density)
        self.density = float(np.clip(self.density, 0.0, 1.0))
        if self.rng.random() < 0.03:
            self._spawn_lead()

    def _state(self, accel=0.0, target=None, collided=False):
        return {
            "v_ms": self.v,
            "density": self.density,
            "lead_present": self.lead,
            "lead_dist": self.lead_dist,
            "lead_speed": self.lead_speed,
            "prev_target_ms": self.prev_target,
            "accel_ms2": accel,
            "target_ms": self.v if target is None else target,
            "collided": collided,
        }

    # ------------------------------------------------------------------ #
    def step(self, action_idx):
        target = self.exp.compute_action(action_idx)
        dv = float(np.clip(target - self.v, -self.accel_max * self.dt, self.accel_max * self.dt))
        a = dv / self.dt
        self.v = max(0.0, self.v + dv)

        if self.lead:
            self.lead_dist += (self.lead_speed - self.v) * self.dt

        self._evolve_density()

        collided = self.lead and self.lead_dist < self.exp.d_crit
        state = self._state(accel=a, target=target, collided=collided)
        reward = self.exp.compute_reward(state)
        obs, info = self.exp.get_observation(state)
        info = {"v_kmh": self.v * 3.6, "density": self.density, "collided": collided}

        self.prev_target = target
        self.t += 1
        done = collided or self.t >= self.horizon
        return obs, reward, done, info
