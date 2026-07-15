"""
Q1 (MDP, Hopper) -- Lazy/strategic majority with PER-REGION (contextual) trust.

v2 rewrite of advanced_experiments.py: the StrategicHopperWrapper below is identical
to the original except the global TrustMechanism is replaced by
`esa_mdp.ContextualTrust`, exactly as prescribed in that file's USAGE block
(per-region trust keyed by the obs sign-pattern, active auditing gated on
`resolved`, absolute-distrust fail-safe via tau).

Scenario: M=10 evaluators, 80% are liars that penalize forward velocity
(y = latent - bias * x_velocity); a strategic adversary anneals its lying magnitude
against the agent's trust so it sits at the edge of detection. Standard PPO chases
the majority and stops hopping; ESA audits the source and recovers v_x > 0.

Needs MuJoCo + Stable-Baselines3 (installed).  Run: python3 exp_mdp_hopper.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from esa_mdp import ContextualTrust


class StrategicHopperWrapper(gym.Wrapper):
    def __init__(self, env, stats_buffer, num_evaluators=10, bias_ratio=0.8,
                 method='esa'):
        super().__init__(env)
        self.M = num_evaluators
        self.method = method
        self.stats = stats_buffer

        self.num_liars = int(self.M * bias_ratio)
        self.is_liar = np.array([True] * self.num_liars +
                                [False] * (self.M - self.num_liars))
        np.random.shuffle(self.is_liar)

        self.current_bias_magnitude = 2.0
        self.target_trust = 0.05

        # --- the two-line wiring from esa_mdp.py USAGE ---
        self.trust_model = ContextualTrust(self.M, n_regions=8, eta=0.05, tau=1.0)
        self._last_obs = np.zeros(self.observation_space.shape[0])

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, latent_reward, terminated, truncated, info = self.env.step(action)
        x_velocity = info.get("x_velocity", latent_reward)
        region_obs = self._last_obs        # trust keyed by the state we acted in

        # strategic adversary: anneal lying magnitude against detected trust
        liar_influence = self.trust_model.liar_weight(self.is_liar)
        if liar_influence < self.target_trust:
            self.current_bias_magnitude *= 0.999
        else:
            self.current_bias_magnitude = min(2.0, self.current_bias_magnitude * 1.005)

        social_signals = np.zeros(self.M)
        for i in range(self.M):
            noise = np.random.normal(0, 0.1)
            if self.is_liar[i]:
                social_signals[i] = latent_reward - self.current_bias_magnitude * x_velocity + noise
            else:
                social_signals[i] = latent_reward + noise

        if self.method == 'standard':
            perceived_reward = float(np.mean(social_signals))
        else:
            internal = latent_reward if np.random.rand() < 0.10 else None
            g = self.trust_model.region(region_obs)
            if internal is not None and not self.trust_model.resolved[g]:
                self.trust_model.update(region_obs, social_signals, internal)
            perceived_reward = self.trust_model.trusted_reward(
                region_obs, social_signals, reference=internal)

        self.stats['rewards'].append(latent_reward)
        self.stats['liar_weight'].append(liar_influence)
        self.stats['bias_mag'].append(self.current_bias_magnitude)
        self.stats['audits'].append(self.trust_model.audits)
        self._last_obs = obs
        return obs, perceived_reward, terminated, truncated, info


def run_experiment(method, timesteps, seed=0):
    print(f"  running Hopper method={method} for {timesteps} steps ...")
    stats = {'rewards': [], 'liar_weight': [], 'bias_mag': [], 'audits': []}
    def make():
        e = gym.make("Hopper-v4", render_mode=None)
        return StrategicHopperWrapper(e, stats, method=method)
    env = VecMonitor(DummyVecEnv([make]))
    model = PPO("MlpPolicy", env, verbose=0, device='auto', seed=seed)
    model.learn(total_timesteps=timesteps)
    return {k: np.array(v) for k, v in stats.items()}


def rolling(a, w=500):
    if len(a) < w:
        return np.asarray(a, dtype=float)
    return np.convolve(a, np.ones(w) / w, mode='valid')


def rolled_band(runs, key, w=500):
    """Stack per-seed rolling curves -> (x, mean, std) across seeds.
    Trims to the shortest seed so cross-seed std is well defined."""
    curves = [rolling(r[key], w) for r in runs]
    L = min(len(c) for c in curves)
    M = np.stack([c[:L] for c in curves])          # [seeds, L]
    return np.arange(L), M.mean(0), M.std(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    print(f"Q1 Hopper: {args.seeds} seeds x {args.timesteps} steps, per method")
    std_runs = [run_experiment('standard', args.timesteps, s) for s in seeds]
    esa_runs = [run_experiment('esa', args.timesteps, s) for s in seeds]

    # --- summary stats across seeds (last 1k latent reward) ---
    std_final = np.array([r['rewards'][-1000:].mean() for r in std_runs])
    esa_final = np.array([r['rewards'][-1000:].mean() for r in esa_runs])
    liar_final = np.array([r['liar_weight'][-1] for r in esa_runs])
    print(f"  Standard PPO latent (last 1k): {std_final.mean():.3f} ± {std_final.std():.3f}  (n={args.seeds})")
    print(f"  ESA region   latent (last 1k): {esa_final.mean():.3f} ± {esa_final.std():.3f}")
    print(f"  ESA trust-in-liars (final):    {liar_final.mean():.3f} ± {liar_final.std():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # left: latent reward mean ± std band across seeds
    for runs, color, lab in [(std_runs, '#F39C12', 'Standard PPO'),
                             (esa_runs, '#6C3483', 'ESA region-trust (ours)')]:
        x, m, sd = rolled_band(runs, 'rewards')
        ax[0].plot(x, m, color=color, lw=2.5, label=lab)
        ax[0].fill_between(x, m - sd, m + sd, color=color, alpha=0.18)
    ax[0].axhline(0, color='grey', ls=':', lw=1)
    ax[0].set_xlabel("environment step"); ax[0].set_ylabel("latent (true) reward")
    ax[0].set_title(f"Hopper latent reward under 80% lazy majority\n(mean ± std over {args.seeds} seeds)")
    ax[0].legend(loc='upper left'); ax[0].grid(alpha=0.2)

    # right: trust-in-liars mean ± std band + adversary bias
    xt, mt, sdt = rolled_band(esa_runs, 'liar_weight')
    ax[1].plot(xt, mt, color='#C0392B', lw=2.5, label='trust in liars')
    ax[1].fill_between(xt, mt - sdt, mt + sdt, color='#C0392B', alpha=0.18)
    xb, mb, sdb = rolled_band(esa_runs, 'bias_mag')
    ax[1].plot(xb, mb, color='#2E86C1', lw=2.5, ls='--', label='adversary bias magnitude')
    ax[1].fill_between(xb, mb - sdb, mb + sdb, color='#2E86C1', alpha=0.12)
    ax[1].axhline(0.05, color='grey', ls=':', lw=1, label='target trust')
    ax[1].set_xlabel("environment step"); ax[1].set_ylabel("weight / magnitude")
    ax[1].set_title(f"Epistemic source judgment (strategic Nash)\n(mean ± std over {args.seeds} seeds)")
    ax[1].legend(loc='center right'); ax[1].grid(alpha=0.2)

    plt.tight_layout()
    import os; os.makedirs("paper/figures", exist_ok=True)
    out = "paper/figures/q1_mdp_hopper.png"
    plt.savefig(out, dpi=200)
    print(f"  saved {out}")
