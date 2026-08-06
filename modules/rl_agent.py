"""DQN thuần PyTorch (không cần ray/stable-baselines) cho SpeedExperiment.

Q-network MLP nhỏ + replay buffer + target network + epsilon-greedy. Đủ nhẹ để
huấn luyện trên CPU trong vài phút với môi trường thay thế.
"""
# fmt: off
# isort: skip_file
import os
import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap = capacity
        self.i = 0
        self.full = False
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.ns = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros(capacity, np.float32)

    def push(self, s, a, r, ns, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, d
        self.i = (i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, bs, rng):
        idx = rng.integers(0, len(self), size=bs)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99, device="cpu", seed=0):
        self.device = device
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        torch.manual_seed(seed)
        self.q = QNet(obs_dim, n_actions).to(device)
        self.tgt = QNet(obs_dim, n_actions).to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(t).argmax(1).item())

    def learn(self, buffer, bs=64):
        if len(buffer) < bs:
            return None
        s, a, r, ns, d = buffer.sample(bs, self.rng)
        s = torch.as_tensor(s, device=self.device)
        ns = torch.as_tensor(ns, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.tgt(ns).max(1).values
            target = r + self.gamma * (1.0 - d) * nq
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.tgt.load_state_dict(self.q.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.q.state_dict(),
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["model"])
        self.update_target()
