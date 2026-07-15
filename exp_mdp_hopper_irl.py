"""
v2 rewrite of the imitation comparison in advanced_experiments.py
(advanced_experiments_irl.png).

Imitation learners recover behavior/reward from a demonstrator. When the
demonstrator is sycophantic (lazy: y = latent - 2*v_x), imitation reproduces the
biased objective and the recovered policy stops hopping. ESA instead audits the
source against the reference and keeps the latent objective. We compare:

  * IRL / GAIL       : PPO on the sycophantic reward directly (recovers & optimizes
                       the biased reward -- inverse-RL / adversarial-imitation proxy).
  * Behavior Cloning : supervised MLP fit to (state -> action) on demonstrations
                       rolled out from the biased expert; no reward, pure imitation.
  * ESA (ours)       : PPO on the region-trust ESA reward (esa_mdp.ContextualTrust).

Every trained policy is evaluated on the CLEAN latent Hopper reward and bar-charted.
Both imitation baselines reproduce the biased low-velocity behavior; only ESA
recovers forward hopping. MuJoCo + SB3 + a small torch BC head.
"""
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from exp_mdp_hopper import StrategicHopperWrapper


class SycophanticRewardWrapper(gym.Wrapper):
    """The demonstrator's (biased) reward the imitation learners see."""
    def step(self, action):
        obs, latent, term, trunc, info = self.env.step(action)
        xv = info.get("x_velocity", latent)
        return obs, latent - 2.0 * xv, term, trunc, info


def train_ppo(kind, timesteps, seed):
    def make():
        e = gym.make("Hopper-v4", render_mode=None)
        if kind == "irl":
            return SycophanticRewardWrapper(e)
        return StrategicHopperWrapper(e, {"rewards": [], "liar_weight": [],
                                          "bias_mag": [], "audits": []}, method="esa")
    env = VecMonitor(DummyVecEnv([make]))
    model = PPO("MlpPolicy", env, verbose=0, device="auto", seed=seed)
    model.learn(total_timesteps=timesteps)
    return model


def collect_demos(model, n=8000, seed=7):
    env = gym.make("Hopper-v4", render_mode=None)
    obs, _ = env.reset(seed=seed)
    O, A = [], []
    for _ in range(n):
        a, _ = model.predict(obs, deterministic=True)
        O.append(np.asarray(obs, dtype=np.float32)); A.append(np.asarray(a, dtype=np.float32))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(O, dtype=np.float32), np.array(A, dtype=np.float32)


class BCAgent:
    """Behavior cloning: MLP fit to (state -> action) demonstrations; SB3-style predict."""
    def __init__(self, obs_dim, act_dim):
        import torch.nn as nn
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, act_dim))

    def fit(self, O, A, epochs=300, lr=1e-3, batch=256):
        import torch
        Ot, At = torch.tensor(O), torch.tensor(A)
        opt = torch.optim.Adam(self.net.parameters(), lr)
        n = len(O)
        for _ in range(epochs):
            idx = torch.randperm(n)
            for b in range(0, n, batch):
                j = idx[b:b + batch]
                opt.zero_grad()
                loss = ((self.net(Ot[j]) - At[j]) ** 2).mean()
                loss.backward(); opt.step()
        return self

    def predict(self, obs, deterministic=True):
        import torch
        with torch.no_grad():
            a = self.net(torch.tensor(np.asarray(obs, dtype=np.float32))).numpy()
        return a, None


def train_bc(timesteps, seed):
    expert = train_ppo("irl", timesteps, seed)          # biased (sycophantic) expert
    O, A = collect_demos(expert)
    return BCAgent(O.shape[1], A.shape[1]).fit(O, A)


def eval_latent(model, steps=3000, seed=123):
    env = gym.make("Hopper-v4", render_mode=None)
    obs, _ = env.reset(seed=seed)
    rs = []
    for _ in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        rs.append(r)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return float(np.mean(rs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kinds = [("irl", "IRL / GAIL\n(mimics majority)", "#7F8C8D"),
             ("bc", "Behavior Cloning\n(clones biased demos)", "#F39C12"),
             ("esa", "ESA region-trust\n(ours)", "#6C3483")]
    scores = {}
    for k, _, _ in kinds:
        print(f"training {k} ...", flush=True)
        m = train_bc(args.timesteps, args.seed) if k == "bc" else train_ppo(k, args.timesteps, args.seed)
        scores[k] = eval_latent(m)
        print(f"  {k}: latent eval reward/step = {scores[k]:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [lab for _, lab, _ in kinds]
    vals = [scores[k] for k, _, _ in kinds]
    cols = [c for _, _, c in kinds]
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("latent (true) reward per step, clean eval")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    import os; os.makedirs("paper/figures", exist_ok=True)
    fig.savefig("paper/figures/q1_mdp_hopper_irl.png", dpi=200)
    print("saved paper/figures/q1_mdp_hopper_irl.png")
