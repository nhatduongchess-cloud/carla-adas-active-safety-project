"""Triển khai policy RL vào CARLA: quan sát -> tốc độ mong muốn (km/h).

Dùng cùng SpeedExperiment (định nghĩa observation) như lúc train. Nếu chưa có
checkpoint (chưa train) hoặc thiếu torch -> tự động dùng HEURISTIC theo mật độ,
nên chinh.py luôn chạy được kể cả trước khi huấn luyện.

Lưu ý an toàn: controller CHỈ đề xuất tốc độ tuần hành (cruise). Lớp AEB/MRM vẫn
ghi đè phanh khẩn cấp — RL không bao giờ vô hiệu hoá an toàn.
"""
# fmt: off
# isort: skip_file
try:
    from rl_experiment import SpeedExperiment
except ImportError:
    from modules.rl_experiment import SpeedExperiment


class RLSpeedController:
    def __init__(self, policy_path="weights/rl_speed_policy.pt"):
        self.exp = SpeedExperiment()
        self.actions_kmh = self.exp.ACTIONS_KMH
        self.prev_target_ms = 0.0
        self.policy = None
        self._torch = None
        try:
            import torch
            try:
                from rl_agent import QNet
            except ImportError:
                from modules.rl_agent import QNet
            ckpt = torch.load(policy_path, map_location="cpu")
            net = QNet(ckpt["obs_dim"], ckpt["n_actions"])
            net.load_state_dict(ckpt["model"])
            net.eval()
            self.policy = net
            self._torch = torch
            print(f"[RL] Đã nạp policy tốc độ: {policy_path}")
        except Exception as e:
            print(f"[RL] Chưa có policy ({e}) -> dùng heuristic theo mật độ.")

    def _heuristic_kmh(self, density):
        if density < 0.3:
            return 40.0
        if density > 0.6:
            return 20.0
        return 30.0

    def desired_speed_kmh(self, ego_speed_ms, density, nearest_dist_m, ttc_s):
        """Trả tốc độ mong muốn (km/h) theo tình huống hiện tại."""
        if self.policy is None:
            target = self._heuristic_kmh(density)
        else:
            obs = self.exp.obs_from_measured(
                ego_speed_ms, density, nearest_dist_m, ttc_s, self.prev_target_ms)
            t = self._torch.as_tensor(obs, dtype=self._torch.float32).unsqueeze(0)
            with self._torch.no_grad():
                a = int(self.policy(t).argmax(1).item())
            target = float(self.actions_kmh[a])
        self.prev_target_ms = target / 3.6
        return target
