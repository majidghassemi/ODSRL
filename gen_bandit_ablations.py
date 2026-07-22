"""
Data generator for Figs. K/L/M (q8a, q8b, q8c ablations). NO PLOTTING HERE.

Reuses the exact experiment from exp_bandit_ablations.py (same env, same seeds, same
horizon) and dumps the per-seed cumulative latent regret traces for the three sweeps:

  A (q8a) bias ratio     -- ESA regret vs the fraction of biased evaluators, with the
                            absolute-distrust fail-safe (tau=0.5) on, so 95-100% (where
                            there is no honest minority) stays bounded.
  B (q8b) eta            -- sensitivity to the trust learning rate at a fixed 80% bias.
  C (q8c) reference noise -- an imperfect reference (sigma_ref); recovery should degrade
                            gracefully while the divergence arm stays covered.

Writes bandit_ablation_curves.npz: a [seeds, STEPS] float32 matrix per setting, keyed
<sweep>_<value index>, plus the swept values in bandit_ablation_summary.json. The three
figures read those files and nothing else.

Run: python3 gen_bandit_ablations.py --seeds 15
"""
import argparse
import json

import numpy as np

from runners import run_esa
from social_bandit import SocialBanditEnv

K, M = 5, 20
OPT, BIAS_ARM, GAP, BIAS_MAG = 0, 1, 0.5, 1.0
STEPS = 5000

RATIOS = [0.5, 0.7, 0.9, 0.95, 1.0]     # sweep A: fail-safe on (tau=0.5)
ETAS = [0.1, 0.5, 2.0]                  # sweep B: 80% biased, no fail-safe
NOISES = [0.0, 0.1, 0.5, 1.0]           # sweep C: 80% biased, no fail-safe


def esa_regret(sycophant_ratio, eta, sigma_ref, tau, seed):
    env = SocialBanditEnv(k=K, M=M, optimal_arm=OPT, bias_arm=BIAS_ARM, arm_gap=GAP,
                          bias_mag=BIAS_MAG, sycophant_ratio=sycophant_ratio,
                          coverage="full", sigma_ref=sigma_ref, seed=seed)
    return run_esa(env, STEPS, eta=eta, audit_mode="active", tau=tau,
                   trust_scope="region", seed=seed)["regret"]


SWEEPS = {
    "a": (RATIOS, lambda v, s: esa_regret(v, 0.5, 0.1, 0.5, s)),
    "b": (ETAS, lambda v, s: esa_regret(0.8, v, 0.1, None, s)),
    "c": (NOISES, lambda v, s: esa_regret(0.8, 0.5, v, None, s)),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    args = ap.parse_args()

    out = {}
    summary = {"steps": STEPS, "seeds": args.seeds, "k": K, "M": M,
               "values": {"a": RATIOS, "b": ETAS, "c": NOISES}, "final": {}}

    for sweep, (values, fn) in SWEEPS.items():
        print(f"sweep {sweep}: {values}", flush=True)
        for i, v in enumerate(values):
            R = np.stack([fn(v, s) for s in range(args.seeds)]).astype(np.float32)
            out[f"{sweep}_{i}"] = R
            summary["final"][f"{sweep}_{i}"] = [float(R.mean(0)[-1]),
                                                float(R.std(0)[-1])]
            print(f"  {v}: final regret {R.mean(0)[-1]:8.1f} +/- {R.std(0)[-1]:5.1f}",
                  flush=True)

    np.savez_compressed("bandit_ablation_curves.npz", **out)
    json.dump(summary, open("bandit_ablation_summary.json", "w"), indent=2)
    print("saved bandit_ablation_curves.npz + bandit_ablation_summary.json")
