# fmt: off
# isort: skip_file
"""
Huấn luyện tác tử DQN tối ưu tốc độ theo mật độ giao thông (offline, KHÔNG cần CARLA).

    .venvCarLa\\Scripts\\python.exe scripts\\training\\train_rl.py --episodes 300

Sau khi train: lưu policy -> weights/rl_speed_policy.pt, đường cong reward ->
weights/rl_train_curve.png, và ĐÁNH GIÁ trên đường vắng vs đông để xác nhận
>30 km/h khi vắng / <30 km/h khi đông.
"""
import os
import sys
import argparse
import numpy as np

# Ép UTF-8 để in tiếng Việt được trên mọi console (tránh lỗi cp1252 trên Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CURRENT_DIR, "modules"))

from rl_env import SpeedControlEnv                 # noqa: E402
from rl_agent import DQNAgent, ReplayBuffer        # noqa: E402
from rl_experiment import SpeedExperiment          # noqa: E402


def evaluate(agent, density, episodes=6, horizon=200):
    """Chạy greedy ở mật độ cố định, trả tốc độ trung bình (km/h) ở nửa sau tập."""
    speeds = []
    for e in range(episodes):
        env = SpeedControlEnv(seed=10_000 + e, horizon=horizon)
        env.set_density(density)
        obs = env.exp.get_observation(env._state())[0]
        vs = []
        for t in range(horizon):
            a = agent.act(obs, eps=0.0)
            obs, _, done, info = env.step(a)
            if t > horizon // 2:
                vs.append(info["v_kmh"])
            if done:
                break
        if vs:
            speeds.append(np.mean(vs))
    return float(np.mean(speeds)) if speeds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--out", type=str, default="weights/rl_speed_policy.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp = SpeedExperiment()
    env = SpeedControlEnv(seed=args.seed, horizon=args.horizon, experiment=exp)
    obs_dim = exp.OBS_DIM
    agent = DQNAgent(obs_dim, env.n_actions, seed=args.seed)
    buffer = ReplayBuffer(50_000, obs_dim)

    eps, eps_min, eps_decay = 1.0, 0.05, 0.99
    rewards = []
    print(f"[RL] Bắt đầu huấn luyện: {args.episodes} tập, {env.n_actions} action "
          f"({exp.ACTIONS_KMH} km/h)")

    for ep in range(args.episodes):
        obs = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            a = agent.act(obs, eps)
            nobs, r, done, _ = env.step(a)
            buffer.push(obs, a, r, nobs, float(done))
            agent.learn(buffer, bs=64)
            obs = nobs
            ep_r += r
        eps = max(eps_min, eps * eps_decay)
        if ep % 10 == 0:
            agent.update_target()
        rewards.append(ep_r)
        if ep % 25 == 0 or ep == args.episodes - 1:
            print(f"  ep {ep:4d} | reward {ep_r:8.1f} | eps {eps:.2f}")

    agent.save(args.out)
    print(f"[RL] Đã lưu policy -> {args.out}")

    # Đường cong reward
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3))
        plt.plot(rewards, lw=1)
        plt.xlabel("episode"); plt.ylabel("reward"); plt.title("DQN speed-optimization training")
        plt.tight_layout(); plt.savefig("weights/rl_train_curve.png", dpi=110)
        print("[RL] Đường cong reward -> weights/rl_train_curve.png")
    except Exception as e:
        print(f"[RL] (bỏ qua vẽ đồ thị: {e})")

    # ĐÁNH GIÁ: đường vắng vs đông
    empty = evaluate(agent, density=0.1)
    busy = evaluate(agent, density=0.9)
    print("\n============== ĐÁNH GIÁ POLICY ==============")
    print(f"  Đường VẮNG (density 0.1): tốc độ TB = {empty:5.1f} km/h  (mục tiêu > 30)")
    print(f"  Đường ĐÔNG (density 0.9): tốc độ TB = {busy:5.1f} km/h  (mục tiêu < 30)")
    ok = empty > 30.0 and busy < 30.0
    print(f"  KẾT QUẢ: {'ĐẠT ✅ (vắng>30, đông<30)' if ok else 'CHƯA ĐẠT — cần train thêm/tinh chỉnh'}")
    print("=============================================")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
