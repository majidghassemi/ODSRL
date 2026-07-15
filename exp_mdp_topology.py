"""
v2 rewrite of social_topology_experiment.py
(social_topology_dynamics.png + social_topology_graph.png).

A sycophancy "contagion" spreads over a scale-free (Barabasi-Albert) evaluator
network from Patient Zero (the highest-degree influencer). The v1 version used the
global TrustMechanism; here we use esa_mdp.ContextualTrust (per-region trust +
active audit + fail-safe). Per-node trust is the region-averaged weight w.mean(0).
ESA "quarantines" infected nodes: their trust collapses while healthy nodes keep
theirs, so the agent tracks the latent reward despite a spreading majority.

Outputs: q9a_topology_dynamics.png (infection rate vs trust in Patient Zero vs avg
trust in healthy nodes) and q9b_topology_graph.png (final trust-colored network).
MuJoCo + SB3 + networkx.
"""
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from esa_mdp import ContextualTrust


class SocialGraphWrapper(gym.Wrapper):
    def __init__(self, env, stats, M=20, method="esa"):
        super().__init__(env)
        self.M = M
        self.method = method
        self.stats = stats
        self.G = nx.barabasi_albert_graph(n=M, m=2, seed=42)
        self.patient_zero = max(dict(self.G.degree()), key=dict(self.G.degree()).get)
        self.infection = np.zeros(M)
        self.infection[self.patient_zero] = 1.0
        self.infection_prob = 0.05
        self.spread_interval = 1000
        self.t = 0
        self.trust = ContextualTrust(M, n_regions=8, eta=0.2, tau=1.0)
        self._last_obs = np.zeros(env.observation_space.shape[0])

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs
        return obs, info

    def _node_trust(self):
        return self.trust.w.mean(0)               # region-averaged per-evaluator trust

    def step(self, action):
        obs, latent, term, trunc, info = self.env.step(action)
        self.t += 1
        xv = info.get("x_velocity", latent)
        region_obs = self._last_obs
        if self.t % self.spread_interval == 0:
            self._spread()

        signals = np.zeros(self.M)
        for i in range(self.M):
            noise = np.random.normal(0, 0.1)
            signals[i] = latent - (2.0 * xv) + noise if self.infection[i] else latent + noise

        if self.method == "standard":
            perceived = float(np.mean(signals))
        else:
            internal = latent if np.random.rand() < 0.10 else None
            g = self.trust.region(region_obs)
            if internal is not None and not self.trust.resolved[g]:
                self.trust.update(region_obs, signals, internal)
            perceived = self.trust.trusted_reward(region_obs, signals, reference=internal)

        w = self._node_trust()
        self.stats["rewards"].append(latent)
        self.stats["infection_rate"].append(float(self.infection.mean()))
        self.stats["trust_zero"].append(float(w[self.patient_zero]))
        healthy = self.infection == 0
        self.stats["trust_healthy"].append(float(w[healthy].mean()) if healthy.any() else 0.0)
        self._last_obs = obs
        return obs, perceived, term, trunc, info

    def _spread(self):
        new = self.infection.copy()
        for node in self.G.nodes():
            if self.infection[node] == 0:
                inf_nb = sum(self.infection[n] for n in self.G.neighbors(node))
                if np.random.rand() < (1 - (1 - self.infection_prob) ** inf_nb):
                    new[node] = 1.0
        self.infection = new


def run(timesteps, seed):
    stats = {"rewards": [], "infection_rate": [], "trust_zero": [], "trust_healthy": []}
    holder = {}
    def make():
        w = SocialGraphWrapper(gym.make("Hopper-v4", render_mode=None), stats, method="esa")
        holder["w"] = w
        return w
    env = VecMonitor(DummyVecEnv([make]))
    model = PPO("MlpPolicy", env, verbose=0, device="auto", seed=seed)
    model.learn(total_timesteps=timesteps)
    return stats, holder["w"]


def roll(a, w=300):
    a = np.asarray(a, float)
    return np.convolve(a, np.ones(w) / w, mode="valid") if len(a) >= w else a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats, wrap = run(args.timesteps, args.seed)
    print(f"final infection rate: {stats['infection_rate'][-1]:.2f}  "
          f"trust in patient zero: {stats['trust_zero'][-1]:.4f}  "
          f"avg healthy trust: {stats['trust_healthy'][-1]:.4f}")

    # dynamics
    plt.figure(figsize=(10, 6))
    x = np.arange(len(stats["infection_rate"]))
    plt.fill_between(x, stats["infection_rate"], color="#E74C3C", alpha=0.12,
                     label="network infection rate")
    plt.plot(roll(stats["trust_zero"]), color="#C0392B", ls="--", lw=2.5,
             label="trust in Patient Zero (influencer)")
    plt.plot(roll(stats["trust_healthy"]), color="#6C3483", lw=2.5,
             label="avg trust in healthy nodes")
    plt.xlabel("environment step"); plt.ylabel("trust weight / infection fraction")
    plt.legend(loc="upper left"); plt.grid(alpha=0.25)
    import os; os.makedirs("paper/figures", exist_ok=True)
    plt.tight_layout(); plt.savefig("paper/figures/q9a_topology_dynamics.png", dpi=200)
    print("saved paper/figures/q9a_topology_dynamics.png")

    # final graph
    plt.figure(figsize=(9, 8))
    G, w = wrap.G, wrap._node_trust()
    pos = nx.spring_layout(G, seed=42)
    node_colors = plt.cm.RdYlGn(w / w.max())
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600,
                           edgecolors="black", linewidths=1.5)
    px, py = pos[wrap.patient_zero]
    plt.text(px, py + 0.09, "Patient Zero", ha="center", fontweight="bold",
             color="#C0392B", fontsize=13)
    plt.axis("off"); plt.tight_layout()
    plt.savefig("paper/figures/q9b_topology_graph.png", dpi=200)
    print("saved paper/figures/q9b_topology_graph.png")
