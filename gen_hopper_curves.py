"""
Data generator for Fig. C (q1_hopper_latent). NO PLOTTING HERE.

Trains three methods on Hopper-v4 under the strategic 80%-lazy majority and logs the
LATENT (true) reward at every environment step, for every seed:

  standard : PPO on the plain mean of the social signals (majority-captured)
  gail     : PPO on the demonstrator's biased reward (latent - 2*v_x), i.e. the
             IRL/adversarial-imitation proxy from exp_mdp_hopper_irl.py. It optimizes
             the recovered biased objective; we log the latent reward it actually earns.
  esa      : PPO on the region-trust ESA reward (esa_mdp.ContextualTrust)

Writes hopper_latent_curves.npz with, per method, a [seeds, T] matrix of per-step
latent reward. Fig. C reads that file and nothing else.

Run: python3 gen_hopper_curves.py --timesteps 50000 --seeds 5
"""
import argparse, json, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from exp_mdp_hopper import StrategicHopperWrapper

METHODS = ("standard", "gail", "esa")


class GailRewardWrapper(gym.Wrapper):
    """Demonstrator's biased reward (what IRL/GAIL recovers and then optimizes),
    while logging the LATENT reward the resulting behaviour actually earns."""

    def __init__(self, env, stats):
        super().__init__(env)
        self.stats = stats

    def step(self, action):
        obs, latent, term, trunc, info = self.env.step(action)
        xv = info.get("x_velocity", latent)
        self.stats["rewards"].append(latent)
        return obs, latent - 2.0 * xv, term, trunc, info


def run_one(method, timesteps, seed):
    import torch
    torch.set_num_threads(1)
    np.random.seed(seed)                      # wrapper uses module-level np.random
    stats = {"rewards": [], "liar_weight": [], "bias_mag": [], "audits": []}

    def make():
        e = gym.make("Hopper-v4", render_mode=None)
        if method == "gail":
            return GailRewardWrapper(e, stats)
        return StrategicHopperWrapper(e, stats, method=method)

    env = VecMonitor(DummyVecEnv([make]))
    model = PPO("MlpPolicy", env, verbose=0, device="cpu", seed=seed)
    model.learn(total_timesteps=timesteps)
    env.close()
    return method, seed, np.asarray(stats["rewards"], dtype=np.float32)


def _worker(job):
    return run_one(*job)


if __name__ == "__main__":
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="hopper_latent_curves.npz")
    args = ap.parse_args()

    jobs = [(m, args.timesteps, s) for m in METHODS for s in range(args.seeds)]
    print(f"Fig C data: {len(jobs)} runs ({len(METHODS)} methods x {args.seeds} seeds)"
          f" x {args.timesteps} steps", flush=True)

    with mp.get_context("spawn").Pool(processes=min(len(jobs), 15)) as pool:
        results = pool.map(_worker, jobs)

    out, summary = {}, {}
    for m in METHODS:
        curves = [r for meth, _, r in sorted(results, key=lambda t: t[1]) if meth == m]
        L = min(len(c) for c in curves)
        M = np.stack([c[:L] for c in curves])            # [seeds, T]
        out[m] = M
        last = M[:, -1000:].mean(1)                       # latent return, last 1k steps
        summary[m] = {"mean_last1k": float(last.mean()), "std_last1k": float(last.std()),
                      "seeds": int(M.shape[0]), "steps": int(L)}
        print(f"  {m:9s} latent (last 1k) = {last.mean():.3f} +/- {last.std():.3f}", flush=True)

    np.savez_compressed(args.out, **out)
    json.dump(summary, open("hopper_latent_summary.json", "w"), indent=2)
    print(f"saved {args.out} and hopper_latent_summary.json", flush=True)
