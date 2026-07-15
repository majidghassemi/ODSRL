"""
v2 rewrite of the IRL/GAIL comparison in advanced_experiments.py
(advanced_experiments_irl.png).

An IRL/GAIL agent recovers a reward by imitating the demonstrator/majority. When
the majority is sycophantic (lazy: y = latent - 2*v_x), imitation reproduces the
biased objective and the recovered policy stops hopping. ESA instead audits the
source against the reference and keeps the latent objective. We train:

  * IRL/GAIL proxy : PPO on the sycophantic reward directly (mimics the biased
                     majority) -- the imitation baseline.
  * Standard PPO   : PPO on the naive mean of the 80% biased panel.
  * ESA (ours)     : PPO on the region-trust ESA reward (esa_mdp.ContextualTrust).

then evaluate every trained policy on the CLEAN latent Hopper reward and bar-chart
the mean latent reward per step. MuJoCo + SB3.
"""
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from exp_mdp_hopper import StrategicHopperWrapper


class SycophanticRewardWrapper(gym.Wrapper):
    """The demonstrator's (biased) reward the IRL/GAIL agent imitates."""
    def step(self, action):
        obs, latent, term, trunc, info = self.env.step(action)
        xv = info.get("x_velocity", latent)
        return obs, latent - 2.0 * xv, term, trunc, info


def train(kind, timesteps, seed):
    def make():
        e = gym.make("Hopper-v4", render_mode=None)
        if kind == "irl":
            return SycophanticRewardWrapper(e)
        return StrategicHopperWrapper(e, {"rewards": [], "liar_weight": [],
                                          "bias_mag": [], "audits": []},
                                      method="standard" if kind == "standard" else "esa")
    env = VecMonitor(DummyVecEnv([make]))
    model = PPO("MlpPolicy", env, verbose=0, device="auto", seed=seed)
    model.learn(total_timesteps=timesteps)
    return model


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
             ("standard", "Standard PPO\n(naive mean)", "#F39C12"),
             ("esa", "ESA region-trust\n(ours)", "#6C3483")]
    scores = {}
    for k, _, _ in kinds:
        print(f"training {k} ...", flush=True)
        m = train(k, args.timesteps, args.seed)
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
    ax.set_title("Robustness vs imitation: IRL/GAIL reproduces the biased objective,\n"
                 "ESA recovers the latent one (Hopper, 80% lazy majority)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    import os; os.makedirs("paper/figures", exist_ok=True)
    fig.savefig("paper/figures/q1_mdp_hopper_irl.png", dpi=200)
    print("saved paper/figures/q1_mdp_hopper_irl.png")
