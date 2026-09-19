"""Triển khai policy RL vào CARLA: quan sát -> tốc độ mong muốn (km/h).

Dùng cùng SpeedExperiment (định nghĩa observation) như lúc train. Nếu chưa có
checkpoint (chưa train) hoặc thiếu torch -> tự động dùng HEURISTIC theo mật độ,
nên chinh.py luôn chạy được kể cả trước khi huấn luyện.

Lưu ý an toàn: controller CHỈ đề xuất tốc độ tuần hành (cruise). Lớp AEB/MRM vẫn
ghi đè phanh khẩn cấp — RL không bao giờ vô hiệu hoá an toàn.
"""
# fmt: off
# isort: skip_file
import math
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
            ckpt = torch.load(policy_path, map_location="cpu", weights_only=True)
            net = QNet(ckpt["obs_dim"], ckpt["n_actions"])
            net.load_state_dict(ckpt["model"])
            net.eval()
            self.policy = net
            self._torch = torch
            print(f"[RL] Loaded speed policy: {policy_path}")
        except Exception as e:
            print(f"[RL] Policy unavailable ({e}); using density heuristic.")

    def _heuristic_kmh(self, density):
        if density < 0.3:
            return 40.0
        if density > 0.6:
            return 20.0
        return 30.0

    @staticmethod
    def safety_cap_kmh(nearest_dist_m, ttc_s,
                       buffer_m=5.0, a_comfort=2.5, time_gap_s=1.8):
        """TRẦN TỐC ĐỘ AN TOÀN (km/h) cho tốc độ tuần hành.

        RL/heuristic chỉ tối ưu THÔNG LƯỢNG; trần này bảo đảm tốc độ luôn nằm trong
        bao an toàn: (1) dừng ÊM được trong khoảng trống phía trước, và (2) giữ
        time-gap tối thiểu tới vật gần nhất. Đây là mức tuần hành — lớp AEB/MRM vẫn
        ghi đè phanh khẩn cấp khi cần.

        PUBLIC: chinh.py gọi lại hàm này MỖI KHUNG HÌNH sau khi làm mượt (EMA),
        vì khoảng trống phía trước thay đổi từng khung trong khi policy chỉ chạy
        mỗi RL_SPEED_EVERY_N khung. Trần an toàn phải là TIẾNG NÓI CUỐI CÙNG và
        phải được tính trên khoảng cách HIỆN TẠI.
        """
        if nearest_dist_m is None or nearest_dist_m >= 900.0:
            return float("inf")
        usable = max(0.0, nearest_dist_m - buffer_m)
        v_stop = math.sqrt(2.0 * a_comfort * usable)   # v để dừng êm trong 'usable' mét
        v_gap = nearest_dist_m / max(time_gap_s, 1e-3)  # giữ time-gap
        cap_ms = min(v_stop, v_gap)
        if ttc_s is not None and 0.0 < ttc_s < 6.0:     # TTC ngắn -> ghì thêm
            cap_ms *= max(0.3, ttc_s / 6.0)
        return cap_ms * 3.6

    def desired_speed_kmh(self, ego_speed_ms, density, nearest_dist_m, ttc_s,
                          min_cruise_kmh=0.0):
        """Trả tốc độ mong muốn (km/h), LUÔN nằm dưới trần an toàn.

        THỨ TỰ SÀN → TRẦN LÀ QUAN TRỌNG, KHÔNG PHẢI TÙY Ý.

        `min_cruise_kmh` là sàn chống "xe đứng yên" khi policy trả ~0 trên đường
        thoáng. Trước đây sàn này được áp ở chinh.py SAU khi trần an toàn đã kẹp,
        nên một sàn 18 km/h có thể NÂNG NGƯỢC một mục tiêu vừa bị hạ có chủ đích
        vì phía trước có vật cản gần. Sàn phải nằm TRONG trần:

            target = min( max(raw, san), tran )

        Vật cản gần -> tran < san -> kết quả là tran. Đường thoáng -> tran = vô
        cực -> kết quả là max(raw, san). Sàn chỉ còn tác dụng ở đúng tình huống
        nó được sinh ra để xử lý.
        """
        if self.policy is None:
            target = self._heuristic_kmh(density)
        else:
            obs = self.exp.obs_from_measured(
                ego_speed_ms, density, nearest_dist_m, ttc_s, self.prev_target_ms)
            t = self._torch.as_tensor(obs, dtype=self._torch.float32).unsqueeze(0)
            with self._torch.no_grad():
                a = int(self.policy(t).argmax(1).item())
            target = float(self.actions_kmh[a])
        # Sàn trước, rồi trần — trần luôn có tiếng nói cuối cùng.
        target = max(target, max(0.0, float(min_cruise_kmh)))
        target = min(target, self.safety_cap_kmh(nearest_dist_m, ttc_s))
        self.prev_target_ms = target / 3.6
        return target
